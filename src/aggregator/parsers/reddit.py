"""
Reddit parser built on the BaseParser abstraction.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlencode

import httpx

from ..exceptions import FetchError, ParseError, ValidationError
from .base import BaseParser, ParsedPost, RateLimiter, UrlValidator


class RedditParser(BaseParser):
    """Fetches top posts from configured subreddits via Reddit's JSON API."""

    def __init__(
        self,
        *,
        subreddits: Sequence[str],
        rate_limiter: RateLimiter | None = None,
        validator: UrlValidator | None = None,
        client: httpx.AsyncClient | None = None,
        request_delay_seconds: float = 0.2,
        max_limit: int = 100,
        user_agent: str = "news-aggregator/0.1",
    ) -> None:
        """Initialize the parser with subreddit targeting and safety controls."""

        if not subreddits:
            raise ValueError("At least one subreddit must be provided.")

        super().__init__("https://www.reddit.com")

        normalized = tuple(filter(None, (self._normalize(name) for name in subreddits)))
        if not normalized:
            raise ValueError("At least one valid subreddit must be provided.")

        self._subreddits = normalized
        self._rate_limiter = rate_limiter
        self._validator = validator
        self._request_delay = max(0.0, request_delay_seconds)
        self._max_limit = max(1, max_limit)
        self._user_agent = user_agent
        self._client = client or httpx.AsyncClient(
            headers={
                "User-Agent": user_agent,
                "Accept": "application/json",
            },
            timeout=httpx.Timeout(10.0, connect=5.0),
        )
        self._owns_client = client is None

    async def aclose(self) -> None:
        """Close the owned HTTP client."""

        if self._owns_client:
            await self._client.aclose()

    def build_topic_url(self, topic: str, limit: int) -> str:
        """Construct the subreddit listing URL for the provided topic."""

        subreddit = self._normalize(topic) or self._subreddits[0]
        per_page = min(self._max_limit, limit)
        params = urlencode({"t": "week", "limit": per_page})
        return f"{self.base_url}/r/{subreddit}/top.json?{params}"

    async def fetch_content(self, url: str) -> str:
        """Fetch subreddit listings while honoring the optional rate limiter."""

        if self._rate_limiter:
            await self._rate_limiter.acquire("reddit")

        try:
            response = await self._client.get(url)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise FetchError(f"Reddit request failed for {url}") from exc

        if self._request_delay:
            await asyncio.sleep(self._request_delay)

        return response.text

    async def extract_posts(self, content: str, topic: str) -> list[ParsedPost]:
        """Convert the Reddit JSON payload into ParsedPost objects."""

        try:
            payload = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ParseError("Reddit response was not valid JSON.") from exc

        data = payload.get("data") or {}
        children = data.get("children") or []
        fetched_at = datetime.now(UTC)
        posts: list[ParsedPost] = []

        for record in children:
            raw: dict[str, Any] = record.get("data") or {}
            post_id = raw.get("id") or raw.get("name")
            if not post_id:
                continue
            post = ParsedPost(
                id=post_id,
                title=(raw.get("title") or "").strip(),
                content=self._build_content(raw),
                source_url=self._resolve_url(raw),
                source_name=f"reddit/r/{raw.get('subreddit', topic)}",
                published_at=self._to_datetime(raw.get("created_utc")),
                fetched_at=fetched_at,
                raw_data=raw,
            )
            posts.append(post)

        return posts

    def validate_post(self, post: ParsedPost) -> bool:
        """Ensure the parsed post has required fields and safe URLs."""

        if not post.title or not post.source_url:
            return False

        if self._validator and not self._validator(post.source_url):
            raise ValidationError("Unsafe Reddit URL detected.", context={"url": post.source_url})

        return True

    @staticmethod
    def _normalize(topic: str) -> str:
        """Normalize subreddit identifiers to a consistent format."""

        return topic.strip().lstrip("r/").lower().replace(" ", "_")

    @staticmethod
    def _to_datetime(timestamp: Any) -> datetime:
        """Convert Reddit timestamps into timezone-aware datetimes."""

        if isinstance(timestamp, (float, int)):
            return datetime.fromtimestamp(float(timestamp), tz=UTC)
        return datetime.now(UTC)

    @staticmethod
    def _build_content(raw: dict[str, Any]) -> str:
        """Return a concise textual summary for the Reddit post."""

        text = (raw.get("selftext") or "").strip()
        if text:
            return text[:600]
        return (raw.get("title") or "").strip()

    @staticmethod
    def _resolve_url(raw: dict[str, Any]) -> str:
        """Resolve the best-available URL for a Reddit submission."""

        url = (
            raw.get("url_overridden_by_dest")
            or raw.get("url")
            or raw.get("permalink", "")
        )
        # Ensure we always return an absolute URL
        if url.startswith("/"):
            return f"https://www.reddit.com{url}"
        return url or f"https://www.reddit.com{raw.get('permalink', '')}"


__all__ = ["RedditParser"]
