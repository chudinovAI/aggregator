"""
Hacker News parser implementation using the BaseParser abstraction.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from typing import Any

import httpx

from ..exceptions import FetchError, ParseError
from .base import BaseParser, ParsedPost, RateLimiter, UrlValidator

# HN Firebase API endpoints
TOPSTORIES_URL = "https://hacker-news.firebaseio.com/v0/topstories.json"
ITEM_URL_TEMPLATE = "https://hacker-news.firebaseio.com/v0/item/{item_id}.json"


class HackerNewsParser(BaseParser):
    """Fetches top stories from Hacker News via the Firebase API."""

    def __init__(
        self,
        *,
        rate_limiter: RateLimiter | None = None,
        validator: UrlValidator | None = None,
        client: httpx.AsyncClient | None = None,
        request_delay_seconds: float = 0.1,
        max_concurrent_requests: int = 10,
        user_agent: str = "news-aggregator/0.1",
    ) -> None:
        """Initialize the HackerNews parser with rate limiting and safety controls."""

        super().__init__(TOPSTORIES_URL)

        self._rate_limiter = rate_limiter
        self._validator = validator
        self._request_delay = max(0.0, request_delay_seconds)
        self._semaphore = asyncio.Semaphore(max_concurrent_requests)
        self._user_agent = user_agent
        self._client = client or httpx.AsyncClient(
            headers={
                "User-Agent": user_agent,
                "Accept": "application/json",
            },
            timeout=httpx.Timeout(15.0, connect=5.0),
        )
        self._owns_client = client is None

    async def aclose(self) -> None:
        """Close the owned HTTP client."""

        if self._owns_client:
            await self._client.aclose()

    def build_topic_url(self, topic: str, limit: int) -> str:
        """
        Return the top stories endpoint.

        HN doesn't support topic-based queries natively, so we fetch top stories
        and filter client-side if needed.
        """
        # HN API doesn't support topic filtering - we return top stories
        return TOPSTORIES_URL

    async def fetch_content(self, url: str) -> str:
        """Fetch top story IDs from Hacker News."""

        if self._rate_limiter:
            await self._rate_limiter.acquire("hackernews")

        try:
            response = await self._client.get(url)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise FetchError(f"HackerNews request failed for {url}") from exc

        if self._request_delay:
            await asyncio.sleep(self._request_delay)

        return response.text

    async def extract_posts(self, content: str, topic: str) -> list[ParsedPost]:
        """
        Fetch individual story details for each story ID.

        This method:
        1. Parses the top story IDs from the initial response
        2. Fetches each story's details concurrently
        3. Filters by topic keywords if provided
        4. Converts to ParsedPost objects
        """

        try:
            story_ids = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ParseError("HackerNews response was not valid JSON.") from exc

        if not isinstance(story_ids, list):
            raise ParseError("Expected a list of story IDs from HackerNews.")

        fetched_at = datetime.now(UTC)
        posts: list[ParsedPost] = []

        # Fetch stories concurrently with semaphore limiting
        tasks = [self._fetch_story(story_id) for story_id in story_ids[:100]]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        topic_lower = topic.lower().strip() if topic else ""

        for result in results:
            if isinstance(result, BaseException):
                self.logger.debug("Skipping story due to error: %s", result)
                continue

            if result is None:
                continue

            story: dict[str, Any] = result

            # Filter by topic if specified
            if topic_lower and topic_lower not in ("top", "all", "hackernews"):
                title_lower = str(story.get("title", "")).lower()
                text_lower = str(story.get("text", "")).lower() if story.get("text") else ""
                if topic_lower not in title_lower and topic_lower not in text_lower:
                    continue

            post = self._story_to_post(story, fetched_at)
            if post:
                posts.append(post)

        return posts

    async def _fetch_story(self, story_id: int) -> dict[str, Any] | None:
        """Fetch a single story's details from the HN API."""

        async with self._semaphore:
            if self._rate_limiter:
                await self._rate_limiter.acquire("hackernews")

            url = ITEM_URL_TEMPLATE.format(item_id=story_id)

            try:
                response = await self._client.get(url)
                response.raise_for_status()
                data = response.json()
            except httpx.HTTPError as exc:
                self.logger.debug("Failed to fetch story %d: %s", story_id, exc)
                return None
            except json.JSONDecodeError:
                self.logger.debug("Invalid JSON for story %d", story_id)
                return None

            if self._request_delay:
                await asyncio.sleep(self._request_delay)

            return data

    def _story_to_post(self, story: dict[str, Any], fetched_at: datetime) -> ParsedPost | None:
        """Convert a HN story dict to a ParsedPost."""

        if not isinstance(story, dict):
            return None

        story_id = story.get("id")
        title = story.get("title", "").strip()

        if not story_id or not title:
            return None

        # Skip non-story items (comments, polls, etc.)
        item_type = story.get("type", "story")
        if item_type not in ("story", "job"):
            return None

        # Build content from text or use title
        content = story.get("text", "").strip() if story.get("text") else title

        # Get URL - fall back to HN discussion page
        source_url = story.get("url") or f"https://news.ycombinator.com/item?id={story_id}"

        return ParsedPost(
            id=str(story_id),
            title=title,
            content=content[:600] if content else title,
            source_url=source_url,
            source_name="hackernews",
            published_at=self._to_datetime(story.get("time")),
            fetched_at=fetched_at,
            raw_data={
                "score": story.get("score", 0),
                "by": story.get("by", "unknown"),
                "descendants": story.get("descendants", 0),
                "type": item_type,
            },
        )

    def validate_post(self, post: ParsedPost) -> bool:
        """Ensure the parsed post has required fields and safe URLs."""

        if not post.title or not post.source_url:
            return False

        if self._validator and not self._validator(post.source_url):
            # Replace unsafe URL with HN discussion link
            self.logger.warning(
                "Replacing unsafe URL for post %s with HN discussion link.",
                post.id,
            )
            # Note: Can't modify post here, validation should just return False
            return False

        return True

    @staticmethod
    def _to_datetime(timestamp: Any) -> datetime:
        """Convert Unix timestamp to timezone-aware datetime."""

        if isinstance(timestamp, (float, int)) and timestamp > 0:
            return datetime.fromtimestamp(float(timestamp), tz=UTC)
        return datetime.now(UTC)


__all__ = ["HackerNewsParser"]
