from importlib import import_module
import inspect
import os
from typing import Type, Dict, Any, Optional

from .config import PromptEnhanceConfig

class PromptEnhanceService:
    def __init__(self, config: PromptEnhanceConfig):
        self.config = config

    def enhance_prompt(self, user_prompt: str, pe_type: str) -> Optional[str]:
        raise Exception("Not implemented")
    

class PromptEnhanceServiceFactory:
    _service_cache: Dict[str, Type] = {}
    
    @classmethod
    def _discover_services(cls) -> None:
        if cls._service_cache:
            return
        
        services_dir = "comfyui_telegram_bot.services"
        
        for filename in os.listdir(services_dir.replace(".", "/")):
            if filename.endswith('.py') and not filename.startswith('__'):
                module_name = filename[:-3]
                module = import_module(f"{services_dir}.{module_name}")
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if obj.__module__ == f"{services_dir}.{module_name}":
                        cls._service_cache[module_name] = obj
                        break
    
    @classmethod
    def create(cls, service_name: str, **kwargs: Any) -> Any:
        cls._discover_services()
        
        if service_name not in cls._service_cache:
            available = ', '.join(sorted(cls._service_cache.keys()))
            raise ValueError(
                f"Service '{service_name}' not found. Available services: {available}"
            )
            
        return cls._service_cache[service_name](**kwargs)