from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import List

import asyncpraw
from asyncpraw import Reddit
from asyncprawcore import PrawcoreException

from ..config import AggregatorConfig
from ..ratelimiter import AsyncRateLimiter
from ..retry import async_retry
from ..validators import InputValidator
from ..types import Post
from ..utils import sanitize_text

LOGGER = logging.getLogger(__name__)


class RedditCollector:
    def __init__(
        self,
        client: Reddit,
        config: AggregatorConfig,
        rate_limiter: AsyncRateLimiter | None = None,
        validator: InputValidator | None = None,
    ) -> None:
        self._client = client
        self._config = config
        self._validator = validator
        self._rate_limiter = rate_limiter

    def collect(self) -> List[Post]:
        try:
            return asyncio.run(self.collect_async())
        except RuntimeError:
            LOGGER.warning(
                "Event loop already running; skipping async Reddit collection."
            )
            return []

    async def collect_async(self) -> List[Post]:
        posts: List[Post] = []
        try:
            for subreddit_name in self._config.subreddits:
                posts.extend(
                    await self._collect_subreddit(
                        subreddit_name, self._config.reddit_top_limit
                    )
                )
        except (asyncpraw.exceptions.PRAWException, PrawcoreException):
            LOGGER.exception("Failed to collect Reddit posts.")
        finally:
            try:
                await self._client.close()
            except Exception:  # noqa: BLE001
                LOGGER.debug("Failed to close Reddit client cleanly.")

        LOGGER.info("Collected %d Reddit posts.", len(posts))
        return posts

    async def _collect_subreddit(self, subreddit_name: str, limit: int) -> List[Post]:
        async def _op():
            await self._throttle()
            subreddit = await self._client.subreddit(subreddit_name, fetch=True)
            items: List[Post] = []
            async for submission in subreddit.top("week", limit=limit):
                await self._throttle()
                await asyncio.sleep(self._config.api_request_delay_seconds)
                post_url = submission.url
                if self._validator and not self._validator.is_safe_url(post_url):
                    LOGGER.debug("Skipping Reddit post due to unsafe URL: %s", post_url)
                    continue
                items.append(
                    {
                        "title": submission.title,
                        "url": post_url,
                        "score": submission.score,
                        "created_utc": datetime.fromtimestamp(submission.created_utc),
                        "num_comments": submission.num_comments,
                        "subreddit": subreddit_name,
                        "author": str(submission.author),
                        "selftext": sanitize_text((submission.selftext or "")[:300]),
                        "source": "reddit",
                        "permalink": f"https://reddit.com{submission.permalink}",
                    }
                )
            return items

        try:
            return await async_retry(
                _op,
                retries=self._config.api_retries,
                base_delay=self._config.api_retry_base_delay,
                retry_exceptions=(
                    asyncpraw.exceptions.PRAWException,
                    PrawcoreException,
                    asyncio.TimeoutError,
                ),
            )
        except (
            asyncpraw.exceptions.PRAWException,
            PrawcoreException,
            asyncio.TimeoutError,
        ):
            LOGGER.exception(
                "Failed to collect subreddit r/%s after retries.", subreddit_name
            )
            return []

    async def _throttle(self) -> None:
        if self._rate_limiter:
            await self._rate_limiter.acquire("reddit")
