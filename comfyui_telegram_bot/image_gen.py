from dataclasses import dataclass
import re
from typing import Tuple, Dict, Any, Optional
import math
import random
from .config import ImageGenerationConfig, ModeConfig
from PIL import Image, ImageDraw, ImageFont
from copy import deepcopy
import json
import websockets
import asyncio
import io
import os
import requests
from . import logger

@dataclass
class GenerationParameters:
    user_id: int
    prompt: str
    batch_size: int
    width: int
    height: int
    is_set_seed: bool
    seed: Optional[int]
    cfg: float
    steps: int
    lora: Optional[str]
    lora_strength: float
    sampler: str
    scheduler: str
    mode: str
    prompt_enhance: Optional[str]
    prompt_template_pre_pe: str
    prompt_template_post_pe: str

    @classmethod
    def from_message(cls, user_id: int, message: str, config: ImageGenerationConfig) -> 'GenerationParameters':
        prompt, params = cls._extract_parameters(message)

        mp_count = params.pop("mp_count", 1)
        if "ratio" in params:
            width, height = cls._calculate_resolution(*params.pop("ratio"), mp_count)
        elif "size" in params:
            width, height = map(lambda x: round(x/64)*64, params.pop("size"))
        else:
            width, height = cls._calculate_resolution(1, 1, mp_count)
        
        batch_size = params.pop("batch_size", 1)

        mode = params.get("mode", "default")
        if mode == "default":
            mode_config = ModeConfig.get_defaults()
        elif mode in config.modes:
            mode_config = config.modes[mode]
        else:
            raise Exception(f"Invalid mode {mode}")
        
        cfg = params.get("cfg", mode_config.cfg)
        steps = params.get("steps", mode_config.steps)
        seed = params.get("seed")
        is_set_seed = seed is not None
        prompt_enhance = params.get("prompt_enhance")

        return cls(
            user_id=user_id,
            prompt=prompt,
            batch_size=batch_size,
            width=width,
            height=height,
            seed=seed,
            cfg=cfg,
            is_set_seed=is_set_seed,
            steps=steps,
            lora=mode_config.lora,
            lora_strength=mode_config.lora_strength,
            sampler=mode_config.sampler,
            scheduler=mode_config.scheduler,
            mode=mode,
            prompt_enhance=prompt_enhance,
            prompt_template_pre_pe=mode_config.prompt_template_pre_pe,
            prompt_template_post_pe=mode_config.prompt_template_post_pe
        )

    @staticmethod
    def _extract_parameters(message: str) -> Tuple[str, Dict[str, Any]]:
        patterns = {
            "size": {
                "pattern": r'(\d+)x(\d+)',
                "type": int,
                "num_params": 2
            },
            "ratio": {
                "pattern": r'(\d+\.?\d*):(\d+\.?\d*)',
                "type": float,
                "num_params": 2
            },
            "mp_count": {
                "pattern": r'(\d+\.?\d*)MP',
                "type": float,
                "num_params": 1
            },
            "batch_size": {
                "pattern": r'(\d+)x',
                "type": int,
                "num_params": 1
            },
            "steps": {
                "pattern": r's=(\d+)',
                "type": int,
                "num_params": 1
            },
            "cfg": {
                "pattern": r'cfg=(\d+\.?\d*)',
                "type": float,
                "num_params": 1
            },
            "mode": {
                "pattern": r'm=([a-zA-Z]+)',
                "type": str,
                "num_params": 1
            },
            "seed": {
                "pattern": r'seed=(\d+)',
                "type": int,
                "num_params": 1
            },
            "prompt_enhance": {
                "pattern": r'pe=([\w]+)',
                "type": str,
                "num_params": 1
            }
        }

        combined_pattern = '|'.join(fr"\b{info['pattern']}\b" for info in patterns.values())
        
        all_matches = list(re.finditer(combined_pattern, message))
        if not all_matches:
            return message.strip(), {}
        
        params_start = all_matches[0].start()
        
        prompt = message[:params_start].strip()
        params = message[params_start:]
        
        result_params = {}
        for param_name, info in patterns.items():
            param_matches = list(re.finditer(fr"\b{info['pattern']}\b", params))
            for param_match in param_matches:
                match info['num_params']:
                    case 1:
                        result_params[param_name] = info['type'](param_match.group(1))
                    case 2:
                        result_params[param_name] = tuple(map(info['type'], param_match.groups()))
                    case _:
                        raise ValueError(f"Unexpected number of parameters for {param_name}")
        
        # remove matches from prompt
        for match in all_matches:
            message = message.replace(match.group(), '', 1)
        prompt = message.strip()
        
        return prompt, result_params
    
    @staticmethod
    def _calculate_resolution(width_ratio, height_ratio, area_mp):
        area_pixels = area_mp * 1024 * 1024

        width = math.sqrt((area_pixels * width_ratio) / height_ratio)
        height = width * (height_ratio / width_ratio)

        width = round(width / 64) * 64
        height = round(height / 64) * 64

        while width * height > area_pixels:
            if width > height:
                width -= 64
            else:
                height -= 64

        while (width + 64) * (height + 64) <= area_pixels:
            if width < height:
                width += 64
            else:
                height += 64

        return int(width), int(height)
    
    def update_prompt(self, prompt):
        self.prompt = prompt

    def update_before_generation(self):
        if not self.is_set_seed: # randomize seed if not set by user
            self.seed = random.randint(0, 2**32 - 1)
    
    def create_description(self):
        desc = f"Size: {self.width}x{self.height}\nSeed: `{self.seed}`\nGuidance: {self.cfg}\nSteps: {self.steps}"
        if self.lora:
            desc += f"\nLora: `{self.lora}`\nLora strength: {self.lora_strength}"
        if self.mode != "default":
            desc += f"\nMode: {self.mode}"
        return desc


class ComfyUIImageGeneration:
    def __init__(self, config: ImageGenerationConfig):
        self.config = config
        with open(self.config.workflow_filepath, "r") as file:
            self.workflow = json.load(file)

    def create_placeholder_image(self, gp: GenerationParameters):
        width, height = gp.width, gp.height
        placeholder_image = Image.new('RGB', (width, height), color='black')
        draw = ImageDraw.Draw(placeholder_image)
        text = f"{width}x{height}"
        try:
            font = ImageFont.truetype(str(self.config.placeholder_image_font_filepath), width//10)
        except IOError:
            font = ImageFont.load_default(width//10)
        text_left, text_top, text_right, text_bottom = draw.textbbox((width // 2, height // 2), text, font=font, anchor="mm")
        draw.text((text_left, text_top), text, font=font, fill='white')
        placeholder_path = f"placeholder_{gp.user_id}.jpg"
        placeholder_image.save(placeholder_path)
        return placeholder_path

    def _prepare_workflow(self, gp: GenerationParameters) -> dict:
        workflow = deepcopy(self.workflow)
        if gp.lora is None:
            workflow.pop("55")
            workflow["22"]["inputs"]["model"][0] = "54" # disable lora (reroute node)
        else:
            workflow["55"]["inputs"]["lora_name"] = gp.lora
            workflow["55"]["inputs"]["strength_model"] = gp.lora_strength

        if self.config.save_images:
            workflow.pop("59")
        else:
            workflow.pop("52")

        workflow["54"]["inputs"]["unet_name"] = self.config.model
        workflow["10"]["inputs"]["vae_name"] = self.config.vae
        workflow["11"]["inputs"]["clip_name1"] = self.config.clip_t5
        workflow["11"]["inputs"]["clip_name2"] = self.config.clip_l

        workflow["6"]["inputs"]["text"] = gp.prompt

        workflow["25"]["inputs"]["noise_seed"] = gp.seed

        workflow["27"]["inputs"]["width"] = gp.width
        workflow["27"]["inputs"]["height"] = gp.height

        workflow["26"]["inputs"]["guidance"] = gp.cfg
        workflow["16"]["inputs"]["sampler_name"] = gp.sampler
        workflow["17"]["inputs"]["scheduler"] = gp.scheduler
        workflow["17"]["inputs"]["steps"] = gp.steps
        return workflow
    
    async def generate_image(self, gp: GenerationParameters, edit_caption_callback, edit_media_callback, user_queues):
        logger.info(f"Generation started for user {gp.user_id}")
        gp.update_before_generation()
        workflow = self._prepare_workflow(gp)
        
        comfy_prompt = {
            "prompt": workflow,
            "client_id": str(gp.user_id)
        }

        async with websockets.connect(f"{self.config.websocket_url}?clientId={gp.user_id}") as websocket:
            response = requests.post(f"{self.config.server_url}/prompt", json=comfy_prompt)
            prompt_id = response.json()['prompt_id']
            websocket_task = asyncio.create_task(self._process_websocket_messages(gp.user_id, workflow, websocket, user_queues, prompt_id, edit_caption_callback, edit_media_callback))
            ok = await websocket_task
        if not ok:
            return
        
        logger.info("Generation complete")
        await edit_caption_callback("Image generation complete. Fetching final result...")

        history_response = requests.get(f"{self.config.server_url}/history/{prompt_id}")
        history = history_response.json()

        output_data = history[prompt_id]['outputs']
        image_filename = None
        for node_id, node_output in output_data.items():
            if 'images' in node_output:
                image_filename = node_output['images'][0]['filename']
                break

        if image_filename:
            image_url = f"{self.config.server_url}/view?filename={image_filename}"
            if not self.config.save_images:
                image_url += "&type=temp"
            image_response = requests.get(image_url)
            image = Image.open(io.BytesIO(image_response.content))
            temp_image_path = f"temp_image_{gp.user_id}.png"
            image.save(temp_image_path)
            final_caption = f"Final image generated with settings:\n{gp.create_description().replace('.', '\\.')}"
            logger.debug("Sending final image")
            await edit_media_callback(caption=final_caption, media_path=temp_image_path, parse_mode='MarkdownV2')
            os.remove(temp_image_path)
        else:
            logger.error("Failed to find result image")
            await edit_caption_callback("Failed to generate image.")

    async def _process_websocket_messages(self, user_id, workflow, websocket, user_queues, prompt_id, edit_caption_callback, edit_media_callback):
        logger.debug("Start websocket communication")
        show_preview = False
        current_caption = None

        async for message in websocket:
            if user_queues[user_id][0]['cancel']:
                logger.info("Generation cancelled")
                if user_queues[user_id][0]["running"]:
                    response = requests.post(f"{self.config.server_url}/interrupt")
                    logger.debug("ws Response:", response.text)
                else:
                    response = requests.post(f"{self.config.server_url}/queue", json={"delete": [prompt_id]})
                    logger.debug("ws Response:", response.text)
                await edit_caption_callback("Image generation cancelled.")
                return False

            if isinstance(message, str):
                logger.debug(f"ws Message: {message}")
                show_preview = False
                data = json.loads(message)
                if 'data' in data and data['data'].get('prompt_id') != prompt_id:
                    continue
                elif data['type'] == 'executing':
                    user_queues[user_id][0]["running"] = True
                    node_id = data['data']['node']
                    if node_id:
                        node_title = workflow[node_id]["_meta"]["title"]
                    else:
                        node_title = ""

                    current_caption = f"Executing node {node_title}..."
                elif data['type'] == 'progress':
                    value = data['data']['value']
                    max_value = data['data']['max']
                    if (value-1) % self.config.update_preview_every_n_steps == 0:
                        show_preview = True
                    logger.info(f"Generation progress {value}/{max_value}")
                    current_caption = f"Progress: {value}/{max_value}"
                elif data['type'] == 'executed':
                    node_id = data['data']['node']
                    current_caption = f"Completed node {node_id}"
                elif data['type'] == 'execution_cached':
                    current_caption = "Using cached execution..."
                elif data['type'] == 'execution_success':
                    break
                elif data['type'] == 'execution_error':
                    logger.info(f"Generation failed {data['data']}")
                    caption = "Failed to generate image, an error has occured. Try again."
                    if "data" in data:
                        caption += f" (Error type: {data['data'].get('exception_type')}, Error message: {data['data'].get('exception_message')})"
                    await edit_caption_callback(caption)
                else:
                    continue
                await edit_caption_callback(current_caption)

            elif show_preview:
                logger.debug(f"Updating preview")
                image = Image.open(io.BytesIO(message[8:]))
                temp_preview_path = f"temp_preview_{user_id}.png"
                image.save(temp_preview_path, "PNG")
                await edit_media_callback(caption=current_caption, media_path=temp_preview_path)
                os.remove(temp_preview_path)
        return True

