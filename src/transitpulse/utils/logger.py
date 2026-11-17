"""Structured Logging Module for TransitPulse-AI."""

import logging
import sys
from pathlib import Path

from transitpulse.config import settings


def get_logger(name: str, log_file: Path | None = None) -> logging.Logger:
    """Configures and returns a structured logger.

    Args:
        name: Name of the logger, typically __name__.
        log_file: Optional path to log file. If omitted, uses default logs directory.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] [%(name)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Stream Handler (stdout)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)

    # File Handler
    if log_file is None:
        log_file = settings.LOGS_DIR / "transitpulse.log"

    try:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)
    except Exception as exc:
        logger.warning("Could not attach file handler to %s: %s", log_file, exc)

    return logger
