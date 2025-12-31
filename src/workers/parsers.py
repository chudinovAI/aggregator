"""
Parser factory and source registry for background workers.

This module provides:
- REGISTERED_SOURCES: Map of available source names
- create_parser: Factory function to create parser instances
"""

from __future__ import annotations

import logging

from ..aggregator.parsers import HackerNewsParser, RedditParser
from ..aggregator.parsers.base import BaseParser
from ..config import Settings

LOGGER = logging.getLogger(__name__)


# Map source names to their parser classes/factories
# Parsers are instantiated dynamically based on settings
REGISTERED_SOURCES: dict[str, str] = {
    "reddit": "reddit",
    "hackernews": "hackernews",
}


def create_parser(source_name: str, settings: Settings) -> BaseParser | None:
    """
    Factory function to create a parser instance based on source name and settings.

    Returns None if the source is disabled or misconfigured.
    """
    sources = settings.sources
    parsing = settings.parsing

    if source_name == "reddit":
        if not sources.reddit.enabled:
            LOGGER.debug("Reddit source is disabled")
            return None
        return RedditParser(
            subreddits=sources.reddit.subreddits,
            request_delay_seconds=parsing.request_delay_seconds,
            user_agent=sources.reddit.user_agent,
        )

    elif source_name == "hackernews":
        if not sources.hackernews.enabled:
            LOGGER.debug("HackerNews source is disabled")
            return None
        return HackerNewsParser(
            request_delay_seconds=parsing.request_delay_seconds,
            max_concurrent_requests=sources.hackernews.max_concurrent_requests,
        )

    LOGGER.warning("Unknown source: %s", source_name)
    return None


__all__ = [
    "REGISTERED_SOURCES",
    "create_parser",
]
