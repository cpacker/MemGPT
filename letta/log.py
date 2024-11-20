import logging
from logging.config import dictConfig
from pathlib import Path
from sys import stdout
from typing import Optional

from letta.settings import settings

selected_log_level = logging.DEBUG if settings.debug else logging.INFO


def _setup_logfile() -> "Path":
    """ensure the logger filepath is in place

    Returns: the logfile Path
    """
    logfile = Path(settings.letta_dir / "logs" / "Letta.log")
    logfile.parent.mkdir(parents=True, exist_ok=True)
    logfile.touch(exist_ok=True)
    return logfile


# TODO: production logging should be much less invasive
DEVELOPMENT_LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,  # Allow capturing from all loggers
    "formatters": {
        "standard": {"format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"},
        "no_datetime": {"format": "%(name)s - %(levelname)s - %(message)s"},
    },
    "handlers": {
        "console": {
            "level": selected_log_level,
            "class": "logging.StreamHandler",
            "stream": stdout,
            "formatter": "no_datetime",
        },
        "file": {
            "level": "DEBUG",
            "class": "logging.handlers.RotatingFileHandler",
            "filename": _setup_logfile(),
            "maxBytes": 1024**2 * 10,
            "backupCount": 3,
            "formatter": "standard",
        },
    },
    "root": {  # Root logger handles all logs
        "level": logging.DEBUG if settings.debug else logging.INFO,
        "handlers": ["console", "file"],
    },
    "loggers": {
        "Letta": {
            "level": logging.DEBUG if settings.debug else logging.INFO,
            "propagate": True,  # Let logs bubble up to root
        },
        "uvicorn": {
            "level": "CRITICAL",
            "handlers": ["console"],
            "propagate": False,
        },
    },
}


def get_logger(name: Optional[str] = None) -> "logging.Logger":
    """returns the project logger, scoped to a child name if provided
    Args:
        name: will define a child logger
    """
    dictConfig(DEVELOPMENT_LOGGING)
    parent_logger = logging.getLogger("Letta")
    if name:
        return parent_logger.getChild(name)
    return parent_logger
