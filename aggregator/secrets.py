from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Mapping

LOGGER = logging.getLogger(__name__)


def load_runtime_secrets(
    *,
    secrets_path: str | None = None,
    required_keys: Mapping[str, str] | None = None,
) -> None:
    """Load runtime secrets from a JSON file into environment variables.

    Keys already present in os.environ are left untouched. The JSON file must
    contain simple key/value pairs (strings or scalars). The path defaults to
    the AGG_SECRETS_PATH environment variable when not provided explicitly.
    """

    path_value = secrets_path or os.environ.get("AGG_SECRETS_PATH")
    if not path_value:
        LOGGER.info("AGG_SECRETS_PATH not set; relying on existing env vars.")
        _warn_missing(required_keys)
        return

    path = Path(path_value).expanduser()
    if not path.exists():
        LOGGER.warning("Secrets file not found at %s. Skipping load.", path)
        _warn_missing(required_keys)
        return

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        LOGGER.error("Failed to load secrets file %s: %s", path, exc)
        _warn_missing(required_keys)
        return

    if not isinstance(data, dict):
        LOGGER.error("Secrets file %s must contain a top-level JSON object.", path)
        _warn_missing(required_keys)
        return

    loaded = 0
    for key, value in data.items():
        if not isinstance(key, str):
            continue
        env_value = str(value)
        if key in os.environ:
            continue
        os.environ[key] = env_value
        loaded += 1

    LOGGER.info(
        "Loaded %d secrets from %s (existing env vars preserved).", loaded, path
    )
    _warn_missing(required_keys)


def _warn_missing(required_keys: Mapping[str, str] | None) -> None:
    if not required_keys:
        return
    missing = [key for key in required_keys if key not in os.environ]
    if missing:
        LOGGER.warning(
            "Missing required secrets: %s. Provide them via environment or AGG_SECRETS_PATH.",
            ", ".join(sorted(missing)),
        )
