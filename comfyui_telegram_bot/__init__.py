import logging
from .config import Config

config = Config.from_yaml("config.yaml")

level = logging.DEBUG if config.logger.debug else logging.INFO

logger = logging.getLogger(__name__)
logger.setLevel(level)

console_handler = logging.StreamHandler()
console_handler.setLevel(level)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(filename)s - %(message)s')
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)
