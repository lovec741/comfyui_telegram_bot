import os
import asyncio
from telegram import Update, InputMediaPhoto, BotCommand
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from collections import deque
from telegram.error import NetworkError, TimedOut
from functools import wraps
import traceback

from .prompt_enhance import PromptEnhanceServiceFactory, PromptEnhanceService
from .image_gen import ComfyUIImageGeneration, GenerationParameters
from .config import ModeConfig

from . import logger, config

img_gen = ComfyUIImageGeneration(config.image_generation)
pe_service: PromptEnhanceService = PromptEnhanceServiceFactory.create(config.prompt_enhancement.service, config=config.prompt_enhancement)

user_queues = {}

def async_retry(max_retries=3, initial_delay=1, backoff_factor=2):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            delay = initial_delay
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except (NetworkError, TimedOut) as e:
                    if attempt == max_retries - 1:
                        logger.error("Async Retry Failed:", e)
                        return None
                    await asyncio.sleep(delay)
                    delay *= backoff_factor
        return wrapper
    return decorator

@async_retry()
async def edit_caption_with_retry(message, **kwargs):
    return await message.edit_caption(**kwargs)

@async_retry()
async def edit_media_with_retry(message, **kwargs):
    return await message.edit_media(**kwargs)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text('Hi! Send me a prompt and I\'ll generate an image using ComfyUI. You can queue multiple requests. For more information run /help')

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    if user_id in user_queues and user_queues[user_id]:
        logger.info(f"Cancelled the current generation for user {user_id}")
        task = user_queues[user_id][0]
        task['cancel'] = True
        await update.message.reply_text("Cancelling the current image generation...")
    else:
        await update.message.reply_text("No active image generation to cancel.")

async def cancelall(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    if user_id in user_queues and user_queues[user_id]:
        logger.info(f"Cancelled all generations for user {user_id}")
        task = user_queues[user_id][0]
        task['cancel'] = True
        user_queues[user_id] = deque([task])
        await update.message.reply_text("Cancelled all image generations!")
    else:
        await update.message.reply_text("No active image generation to cancel.")

async def queue_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    if user_id in user_queues:
        queue = user_queues[user_id]
        queue_length = len(queue)

        if queue_length == 0:
            await update.message.reply_text("Your queue is currently empty.")
            return

        status_message = f"You have {queue_length} task(s) in your queue:\n\n"

        for i, task in enumerate(queue, start=1):
            params: GenerationParameters = task["params"]
            prompt = params.prompt
            status = "Running" if task['running'] else "Pending"

            max_prompt_length = 50
            if len(prompt) > max_prompt_length:
                prompt = prompt[:max_prompt_length] + "..."

            status_message += (f"{i}. Status: {status}\n"
                               f"   Prompt: {prompt}\n"
                               f"   Dimensions: {params.width}x{params.height}\n\n")

        if queue[0]['running']:
            status_message += "The first task in your queue is currently being processed."
        else:
            status_message += "Your tasks are queued and will be processed in order."

        await update.message.reply_text(status_message)
    else:
        await update.message.reply_text("You don't have any tasks in your queue.")

async def generate_image(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    if user_id not in user_queues:
        user_queues[user_id] = deque()

    try:
        params: GenerationParameters = GenerationParameters.from_message(user_id, update.message.text, config.image_generation)
    except Exception as e:
        logger.info(f"User {user_id} sent invalid generation parameters - {e}")
        await update.message.reply_text(f"Invalid generation parameters: {e}")
        return
    
    logger.info(f"Parsed message from {user_id} - {params}")

    for _ in range(params.batch_size):
        task = {
            'update': update,
            'context': context,
            'cancel': False,
            'running': False,
            'params': params,
        }
        user_queues[user_id].append(task)

    await update.message.reply_text(f"Your request has been queued. {params.batch_size} image(s) added to the queue. Total tasks in queue: {len(user_queues[user_id])}")

    if len(user_queues[user_id]) == params.batch_size:
        asyncio.create_task(process_user_queue(user_id))

async def process_user_queue(user_id):
    while user_queues[user_id]:
        task = user_queues[user_id][0]
        logger.info(f"Processing user queue for user {user_id}. Remaining tasks in queue: {len(user_queues[user_id])-1}")
        await generate_image_task(task['update'], task['context'], task['params'])
        user_queues[user_id].popleft()

async def generate_image_task(update: Update, context: ContextTypes.DEFAULT_TYPE, params: GenerationParameters) -> None:
    try:
        logger.info("Starting generate image task")
        prompt = params.prompt_template_pre_pe.format(params.prompt)
        
        if params.prompt_enhance:
            logger.info("Waiting for enhanced prompt")
            waiting_message = await update.message.reply_text(text="Waiting for enhanced prompt...")
            try:
                prompt = pe_service.enhance_prompt(params.prompt, params.prompt_enhance)
            except Exception as e:
                logger.error(f"Failed to enhance prompt: {e}")
                await waiting_message.edit_text(text=str(e))
                return
            logger.info(f"Received enhanced prompt: {prompt}")
            await waiting_message.edit_text(text=f"Enhanced prompt:\n```\n{prompt}\n```", parse_mode='MarkdownV2')
        
        prompt = params.prompt_template_post_pe.format(prompt)
        params.update_prompt(prompt)

        logger.debug("Creating placeholder image")
        placeholder_path = img_gen.create_placeholder_image(params)
        
        with open(placeholder_path, 'rb') as photo:
            status_message = await update.message.reply_photo(photo=photo, caption="Preparing to generate image...")

        os.remove(placeholder_path)

        async def edit_caption_callback(caption: str, **kwargs):
            return await edit_caption_with_retry(status_message, caption=caption, **kwargs)
        
        async def edit_media_callback(caption: str, media_path: str, **kwargs):
            with open(media_path, 'rb') as photo:
                await edit_media_with_retry(status_message, media=InputMediaPhoto(media=photo, caption=caption, **kwargs))

        await img_gen.generate_image(params, edit_caption_callback, edit_media_callback, user_queues)

    except Exception as e:
        logger.error("While generating image an error occurred:", exc_info=e)
        await edit_caption_with_retry(status_message, caption=f"An error occurred: {e}")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    mode_descriptions = []
    for mode_name, mode_config in [("default", ModeConfig.get_defaults())] + list(config.image_generation.modes.items()):
        desc = f"      <code>m={mode_name}</code>"
        if mode_config.description:
            desc += f": {mode_config.description}"
        params = []
        params.append(f"cfg={mode_config.cfg}")
        params.append(f"steps={mode_config.steps}")
        if mode_config.lora:
            params.append(f"lora={mode_config.lora}")
            params.append(f"lora_strength={mode_config.lora_strength}")
        params.append(f"sampler={mode_config.sampler}")
        params.append(f"scheduler={mode_config.scheduler}")
        if params:
            desc += "".join(["\n          "+i for i in params])
        mode_descriptions.append(desc)

    modes_text = "\n".join(mode_descriptions)
    pe_types_text = "\n".join([f"      <code>pe={pe_type}</code>: {config.prompt_enhancement.types[pe_type].description}" 
                         for pe_type in config.prompt_enhancement.types.keys()])
    help_text = """
Available commands:
/start - Display the welcome message
/help - Show this help message
/cancel - Cancel the current image generation
/cancelall - Cancel all generations in queue
/status - Check your queue status

To generate an image, simply send a text message with your prompt.
You can add parameters at the end of your prompt.

<u>Full list of parameters</u>:
<b>Resolution</b>: <code>1920x1080</code>
<b>Aspect ratio</b>: <code>16:9</code>
<b>Megapixel size</b>: <code>4MP</code> (gets overwritten by resolution)
<b>Batch size</b>: <code>3x</code>
<b>Modes</b>:
{}
<b>Guidance</b>: <code>cfg=3.5</code> 
    1-2 for paintings
    1.5-2.5 realistic
    3.5 balanced (default)
    4 flat colors/cartoon
<b>Steps</b>: <code>s=20</code>
<b>Seed</b>: <code>seed=123456789</code> (random by default)
<b>Prompt enhancement</b>:
{}

<u>Usage example</u>: 
"Create a landscape with mountains <code>1920x1080 2x</code>"
"A man in a red t-shirt <code>2MP pe=default m=real</code>"
    """.format(modes_text, pe_types_text)
    await update.message.reply_text(help_text, parse_mode="HTML")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message.text.startswith('/'):
        await update.message.reply_text("Unknown command. Available commands are /start, /cancel, and /status.")
    else:
        await generate_image(update, context)

async def post_init(application: Application) -> None:
    commands = [
        BotCommand("help", "Show help message"),
        BotCommand("cancel", "Cancel current generation"),
        BotCommand("cancelall", "Cancel all generations in queue"),
        BotCommand("status", "Check queue status"),
    ]
    await application.bot.set_my_commands(commands)

def main() -> None:
    application = Application.builder().token(config.telegram_bot.token).post_init(post_init).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("cancel", cancel))
    application.add_handler(CommandHandler("cancelall", cancelall))
    application.add_handler(CommandHandler("status", queue_status))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    logger.info("Started telegram bot")
    application.run_polling(allowed_updates=Update.ALL_TYPES)