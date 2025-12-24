from __future__ import annotations

import logging
from functools import lru_cache
from typing import Iterable, Optional, Set

from langdetect import DetectorFactory, LangDetectException, detect

LOGGER = logging.getLogger(__name__)

DetectorFactory.seed = 42


class LanguageDetector:
    """Thin wrapper around langdetect with caching and allowlist filtering."""

    def __init__(self, allowed_languages: Iterable[str], enabled: bool = True) -> None:
        self._allowed: Set[str] = {lang.lower() for lang in allowed_languages}
        self._enabled = enabled

    @lru_cache(maxsize=2048)
    def detect_language(self, text: str) -> Optional[str]:
        if not self._enabled:
            return None
        try:
            return detect(text).lower()
        except LangDetectException:
            LOGGER.debug("Language detection failed for text length=%d", len(text))
            return None

    def is_allowed(self, text: str) -> bool:
        if not self._enabled:
            return True
        language = self.detect_language(text)
        if language is None:
            return False
        return language in self._allowed
