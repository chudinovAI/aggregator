from __future__ import annotations

import logging
from typing import Sequence
from urllib.parse import urlparse

LOGGER = logging.getLogger(__name__)


class InputValidator:
    """Centralized validation helpers for external inputs and URLs."""

    def __init__(self, allowed_schemes: Sequence[str]) -> None:
        self._allowed_schemes = tuple(scheme.lower() for scheme in allowed_schemes)

    def is_safe_url(self, url: str) -> bool:
        if not url:
            return False
        parsed = urlparse(url.strip())
        if not parsed.scheme or not parsed.netloc:
            return False
        scheme = parsed.scheme.lower()
        if scheme not in self._allowed_schemes:
            return False
        if any(char.isspace() for char in url):
            return False
        return True

    def ensure_safe_url(self, url: str) -> str:
        if self.is_safe_url(url):
            return url
        raise ValueError(f"Unsafe URL detected: {url!r}")

    def filter_safe_url(self, url: str) -> str:
        return url if self.is_safe_url(url) else ""
