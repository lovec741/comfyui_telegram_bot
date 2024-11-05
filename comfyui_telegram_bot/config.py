from dataclasses import dataclass, field
from typing import Dict, Optional, Any, Type, TypeVar, get_type_hints
from pathlib import Path
import yaml

T = TypeVar('T')

@dataclass
class BaseConfig:
    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        field_types = get_type_hints(cls)
        kwargs = {}
        
        for field_name, field_value in data.items():
            if field_name not in field_types:
                continue
                
            field_type = field_types[field_name]
            
            if field_type == Path and isinstance(field_value, str):
                kwargs[field_name] = Path(field_value)
            elif isinstance(field_value, dict) and isinstance(field_type, type) and issubclass(field_type, BaseConfig):
                kwargs[field_name] = field_type.from_dict(field_value)
            elif field_name == "modes" and isinstance(field_value, dict):
                kwargs[field_name] = {
                    k: ModeConfig.from_dict(v) for k, v in field_value.items()
                }
            elif field_name == "types" and isinstance(field_value, dict):
                kwargs[field_name] = {
                    k: PromptEnhanceTypeConfig.from_dict(v) for k, v in field_value.items()
                }
            else:
                kwargs[field_name] = field_value
                
        return cls(**kwargs)

@dataclass
class PromptEnhanceTypeConfig(BaseConfig):
    description: str
    system_prompt: str


@dataclass
class PromptEnhanceConfig(BaseConfig):
    service: str
    api_key: str
    model: str
    max_tokens: int
    temperature: float
    types: Dict[str, PromptEnhanceTypeConfig]
    retries: int

@dataclass
class TelegramBotConfig(BaseConfig):
    token: str

@dataclass
class ModeConfig(BaseConfig):
    cfg: float = 3.5
    steps: int = 20
    lora: Optional[str] = None
    lora_strength: float = 1.0
    sampler: str = "euler"
    scheduler: str = "simple"
    prompt_template_pre_pe: str = "{}"
    prompt_template_post_pe: str = "{}"
    description: Optional[str] = None
    
    @classmethod
    def get_defaults(cls) -> 'ModeConfig':
        return cls()

@dataclass
class ImageGenerationConfig(BaseConfig):
    model: str
    vae: str
    clip_t5: str
    clip_l: str
    server_url: str
    websocket_url: str
    workflow_filepath: Path
    save_images: bool
    placeholder_image_font_filepath: Path
    update_preview_every_n_steps: int
    modes: Dict[str, ModeConfig] = field(default_factory=dict)

@dataclass
class Config(BaseConfig):
    prompt_enhancement: PromptEnhanceConfig
    telegram_bot: TelegramBotConfig
    image_generation: ImageGenerationConfig

    @classmethod
    def from_yaml(cls, path: Path | str) -> 'Config':
        path = Path(path)
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)
