from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger as _logger

from ..config import settings


LOG_FILE_PATH = Path(settings.log_file)
LOG_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)

_logger.remove()

_logger.add(
    sys.stdout,
    colorize=True,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>",
    level=settings.log_level.upper(),
)

_logger.add(
    LOG_FILE_PATH,
    rotation="10 MB",
    enqueue=True,
    level=settings.log_level.upper(),
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
)

logger = _logger

