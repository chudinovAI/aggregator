from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Sequence

import aiohttp

from ..ratelimiter import AsyncRateLimiter
from ..retry import async_retry
from ..validators import InputValidator
from ..cache import FileCache
from ..config import AggregatorConfig
from ..types import Post

LOGGER = logging.getLogger(__name__)

TOPSTORIES_ENDPOINT = "https://hacker-news.firebaseio.com/v0/topstories.json"
ITEM_ENDPOINT_TEMPLATE = "https://hacker-news.firebaseio.com/v0/item/{story_id}.json"


class HackerNewsCollector:
    def __init__(
        self,
        config: AggregatorConfig,
        cache: FileCache | None = None,
        semaphore: asyncio.Semaphore | None = None,
        rate_limiter: AsyncRateLimiter | None = None,
        validator: InputValidator | None = None,
    ) -> None:
        self._config = config
        self._cache = cache
        self._semaphore = semaphore or asyncio.Semaphore(config.api_concurrency_limit)
        self._rate_limiter = rate_limiter
        self._validator = validator

    def collect(self) -> List[Post]:
        try:
            return asyncio.run(self.collect_async())
        except RuntimeError:
            LOGGER.warning("Event loop already running, falling back to sync requests.")
            return []

    async def collect_async(self) -> List[Post]:
        timeout = aiohttp.ClientTimeout(total=self._config.request_timeout_seconds)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            story_ids = await self._fetch_top_story_ids(session)
            if not story_ids:
                return []

            cutoff = datetime.now() - timedelta(days=self._config.collection_days)
            tasks = [
                self._load_story(session, story_id, cutoff)
                for story_id in story_ids[: self._config.hackernews_top_limit]
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        posts: List[Post] = []
        for result in results:
            if isinstance(result, Exception):
                LOGGER.debug("Skipping story due to error: %s", result)
                continue
            if result:
                posts.append(result)

        LOGGER.info("Collected %d Hacker News posts.", len(posts))
        return posts

    async def _fetch_top_story_ids(
        self, session: aiohttp.ClientSession
    ) -> Sequence[int]:
        cached = self._cache.get("hn_topstories") if self._cache else None
        if cached:
            return cached

        async def _op():
            async with self._semaphore:
                await self._throttle("hackernews")
                await asyncio.sleep(self._config.api_request_delay_seconds)
                async with session.get(TOPSTORIES_ENDPOINT) as response:
                    response.raise_for_status()
                    return await response.json()

        try:
            data = await async_retry(
                _op,
                retries=self._config.api_retries,
                base_delay=self._config.api_retry_base_delay,
                retry_exceptions=(
                    aiohttp.ClientError,
                    asyncio.TimeoutError,
                    ValueError,
                ),
            )
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError):
            LOGGER.exception("Failed to request Hacker News top stories after retries.")
            return []
        if self._cache:
            self._cache.set("hn_topstories", data)
        if not isinstance(data, list):
            LOGGER.warning("Unexpected response format for Hacker News top stories.")
            return []
        return data

    async def _load_story(
        self, session: aiohttp.ClientSession, story_id: int, cutoff: datetime
    ) -> Post | None:
        cache_key = f"hn_story_{story_id}"
        cached = self._cache.get(cache_key) if self._cache else None
        if cached:
            story = cached
        else:

            async def _op():
                async with self._semaphore:
                    await self._throttle("hackernews")
                    await asyncio.sleep(self._config.api_request_delay_seconds)
                    async with session.get(
                        ITEM_ENDPOINT_TEMPLATE.format(story_id=story_id)
                    ) as response:
                        response.raise_for_status()
                        return await response.json()

            try:
                story = await async_retry(
                    _op,
                    retries=self._config.api_retries,
                    base_delay=self._config.api_retry_base_delay,
                    retry_exceptions=(
                        aiohttp.ClientError,
                        asyncio.TimeoutError,
                        ValueError,
                    ),
                )
            except (aiohttp.ClientError, asyncio.TimeoutError, ValueError):
                LOGGER.debug(
                    "Skipping Hacker News story %s due to request or parsing error.",
                    story_id,
                )
                return None
            if self._cache:
                self._cache.set(cache_key, story)

        if not isinstance(story, dict) or "title" not in story:
            return None

        story_time = datetime.fromtimestamp(story.get("time", 0))
        if story_time < cutoff:
            return None

        story_url = story.get(
            "url",
            f"https://news.ycombinator.com/item?id={story_id}",
        )
        if self._validator and story_url:
            if not self._validator.is_safe_url(story_url):
                LOGGER.debug(
                    "Replacing unsafe Hacker News URL for story %s with canonical link.",
                    story_id,
                )
                story_url = f"https://news.ycombinator.com/item?id={story_id}"

        return {
            "title": story["title"],
            "url": story_url,
            "score": int(story.get("score", 0) or 0),
            "created_utc": story_time,
            "num_comments": int(story.get("descendants", 0) or 0),
            "author": story.get("by", "unknown"),
            "source": "hackernews",
            "hn_id": story_id,
            "selftext": "",
        }

    async def _throttle(self, key: str) -> None:
        if self._rate_limiter:
            await self._rate_limiter.acquire(key)
