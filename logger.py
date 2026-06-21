from loguru import logger
from config import settings
import sys
import os

os.makedirs(settings.log_dir, exist_ok=True)

logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss}|{level:<8}|{module}:{function}:{line}-{message}")
logger.add(
    os.path.join(settings.log_dir, "app.log"),
    rotation="5 MB",
    retention="10 days",
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss}|{level:<8}|{module}:{function}:{line}-{message}",
)

__all__ = ["logger"]