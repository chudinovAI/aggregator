"""
Centralized logging utilities that respect dynamic settings.
"""

from __future__ import annotations

import logging
from logging.config import dictConfig
from pathlib import Path
from typing import Any

from .config import Settings, get_settings

_LOGGER_CONFIGURED = False


def _resolve_log_level(level_name: str) -> int:
    """Return a logging level constant from a case-insensitive string."""

    numeric_level = logging.getLevelName(level_name.upper())
    if isinstance(numeric_level, int):
        return numeric_level
    raise ValueError(f"Unsupported log level: {level_name}")


def _build_logging_config(settings: Settings) -> dict[str, Any]:
    """Construct a logging dictConfig payload based on runtime settings."""

    log_settings = settings.logging
    log_dir: Path = log_settings.directory
    log_file = log_dir / log_settings.file_name
    log_dir.mkdir(parents=True, exist_ok=True)

    level = _resolve_log_level(log_settings.level)

    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": log_settings.format,
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": level,
                "formatter": "standard",
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": level,
                "formatter": "standard",
                "filename": str(log_file),
                "encoding": "utf-8",
                "maxBytes": log_settings.max_bytes,
                "backupCount": log_settings.backup_count,
            },
        },
        "root": {
            "level": level,
            "handlers": ["console", "file"],
        },
    }


def setup_logging(settings: Settings | None = None) -> logging.Logger:
    """
    Configure the global logging stack and return the application logger.

    Calling this function multiple times is idempotent; the logging config
    will only be applied during the first invocation.
    """

    global _LOGGER_CONFIGURED

    runtime_settings = settings or get_settings()
    logger_name = runtime_settings.app.name

    if not _LOGGER_CONFIGURED:
        dictConfig(_build_logging_config(runtime_settings))
        _LOGGER_CONFIGURED = True

    logger = logging.getLogger(logger_name)
    logger.setLevel(_resolve_log_level(runtime_settings.logging.level))
    return logger


__all__ = ["setup_logging"]
