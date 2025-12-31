"""
Shared parser abstractions and data structures.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol

from ..exceptions import FetchError, ParseError, ParserError, ValidationError


@dataclass(slots=True)
class ParsedPost:
    """
    Standard representation of parsed content across parsers.
    """

    id: str
    title: str
    content: str
    source_url: str
    source_name: str
    published_at: datetime
    fetched_at: datetime
    raw_data: dict[str, Any] = field(default_factory=dict)


UrlValidator = Callable[[str], bool]


class RateLimiter(Protocol):
    """Protocol describing an asynchronous rate limiter."""

    async def acquire(self, key: str) -> None:
        """Acquire a rate limit token for the specified key."""


class BaseParser(ABC):
    """Abstract base class that encapsulates parser workflow and logging."""

    def __init__(self, base_url: str, *, logger: logging.Logger | None = None) -> None:
        """Initialize the parser with a base URL and optional logger."""

        if not base_url:
            raise ValueError("base_url must be provided.")
        self.base_url = base_url.rstrip("/")
        self._logger = logger or logging.getLogger(self.__class__.__name__)

    @property
    def logger(self) -> logging.Logger:
        """Return the parser-specific logger instance."""

        return self._logger

    async def get_posts(self, topic: str, limit: int) -> list[ParsedPost]:
        """
        Fetch, parse, and validate posts for a topic, enforcing the provided limit.
        """

        if limit <= 0:
            raise ValueError("limit must be a positive integer.")
        if not topic.strip():
            raise ValueError("topic must be a non-empty string.")

        self.logger.info(
            "Fetching up to %d posts for topic '%s' via %s.",
            limit,
            topic,
            self.__class__.__name__,
        )

        url = self._safe_build_url(topic, limit)
        content = await self._safe_fetch(url)
        posts = await self._safe_extract(content, topic)

        validated: list[ParsedPost] = []
        for post in posts:
            try:
                if self.validate_post(post):
                    validated.append(post)
                else:
                    self.logger.debug("Post %s failed validation.", post.id)
            except ValidationError as exc:
                self.logger.warning(
                    "Validation raised for post %s (%s).", post.id, exc, exc_info=True
                )
            except Exception as exc:  # noqa: BLE001
                self.logger.exception(
                    "Unexpected validation failure for post %s.",
                    post.id,
                    exc_info=exc,
                )

        trimmed = validated[:limit]
        self.logger.info(
            "Returning %d post(s) for topic '%s' after validation.",
            len(trimmed),
            topic,
        )
        return trimmed

    def build_topic_url(self, topic: str, limit: int) -> str:
        """
        Build the remote URL for a topic.

        Subclasses should override this when request URLs need topic-specific
        parameters. The default implementation simply returns the base URL.
        """

        return self.base_url

    async def _safe_fetch(self, url: str) -> str:
        """Wrap fetch_content with consistent exception handling."""

        try:
            return await self.fetch_content(url)
        except FetchError:
            self.logger.exception("Failed to fetch content from %s.", url)
            raise
        except Exception as exc:  # noqa: BLE001
            self.logger.exception("Unexpected fetch failure for %s.", url, exc_info=exc)
            raise FetchError("Unhandled fetch error.") from exc

    async def _safe_extract(self, content: str, topic: str) -> list[ParsedPost]:
        """Wrap extract_posts with consistent exception handling."""

        try:
            return await self.extract_posts(content, topic)
        except ParseError:
            self.logger.exception("Failed to parse content for topic '%s'.", topic)
            raise
        except Exception as exc:  # noqa: BLE001
            self.logger.exception("Unexpected parse failure for topic '%s'.", topic, exc_info=exc)
            raise ParseError("Unhandled parse error.") from exc

    def _safe_build_url(self, topic: str, limit: int) -> str:
        """Wrap build_topic_url to normalize raised errors."""

        try:
            return self.build_topic_url(topic, limit)
        except ParserError:
            self.logger.exception("Failed to build topic URL for '%s'.", topic, exc_info=True)
            raise
        except Exception as exc:  # noqa: BLE001
            self.logger.exception(
                "Unexpected error while building URL for '%s'.", topic, exc_info=exc
            )
            raise ParserError("Unhandled URL construction error.") from exc

    @abstractmethod
    async def fetch_content(self, url: str) -> str:
        """Retrieve raw content from the upstream service."""

    @abstractmethod
    async def extract_posts(self, content: str, topic: str) -> list[ParsedPost]:
        """Transform the raw payload into normalized ParsedPost objects."""

    @abstractmethod
    def validate_post(self, post: ParsedPost) -> bool:
        """Return True if the parsed post is acceptable."""


__all__ = ["BaseParser", "ParsedPost", "RateLimiter", "UrlValidator"]
