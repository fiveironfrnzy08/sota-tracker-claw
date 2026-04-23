"""Logging configuration for SOTA Tracker."""

import logging
import sys
from pathlib import Path
from typing import Optional

LOG_DIR = Path(__file__).parent.parent.parent / ".logs"


def setup_logging(
    name: str = "sota_tracker",
    level: int = logging.INFO,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Configure and return a logger.

    Args:
        name: Logger name (default: "sota_tracker")
        level: Logging level (default: INFO)
        format_string: Custom format string (optional)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        fmt = logging.Formatter(
            format_string or "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # stdout handler
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setFormatter(fmt)
        logger.addHandler(stdout_handler)

        # persistent file handler (writes to project .logs/ directory)
        try:
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(LOG_DIR / "sota-tracker-app.log")
            file_handler.setFormatter(fmt)
            file_handler.setLevel(logging.WARNING)
            logger.addHandler(file_handler)
        except OSError:
            pass  # don't fail startup if log dir is unwritable

        logger.setLevel(level)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a child logger with the given name.

    Creates a logger under the sota_tracker namespace.

    Args:
        name: Logger name (will be prefixed with "sota_tracker.")

    Returns:
        Logger instance
    """
    return logging.getLogger(f"sota_tracker.{name}")


# Initialize default logger on import
logger = setup_logging()
