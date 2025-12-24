from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional


class FileCache:
    """Simple file-based cache with TTL in seconds."""

    def __init__(self, path: Path, ttl_seconds: int) -> None:
        self._path = path
        self._ttl = ttl_seconds
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            self._cache = json.loads(self._path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            self._cache = {}

    def _persist(self) -> None:
        self._path.write_text(json.dumps(self._cache), encoding="utf-8")

    def get(self, key: str) -> Optional[Any]:
        entry = self._cache.get(key)
        if not entry:
            return None
        expires_at = entry.get("expires_at", 0)
        if expires_at < time.time():
            self._cache.pop(key, None)
            self._persist()
            return None
        return entry.get("value")

    def set(self, key: str, value: Any) -> None:
        self._cache[key] = {"expires_at": time.time() + self._ttl, "value": value}
        self._persist()
