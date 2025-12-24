from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Sequence

import aiohttp

from ..cache import FileCache
from ..config import AggregatorConfig
from ..ratelimiter import AsyncRateLimiter
from ..retry import async_retry
from ..validators import InputValidator
from ..types import Post
from ..utils import parse_iso8601_duration, sanitize_text

LOGGER = logging.getLogger(__name__)

YOUTUBE_SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"
YOUTUBE_VIDEOS_URL = "https://www.googleapis.com/youtube/v3/videos"


class TedCollector:
    def __init__(
        self,
        youtube_api_key: str,
        config: AggregatorConfig,
        cache: FileCache | None = None,
        semaphore: asyncio.Semaphore | None = None,
        rate_limiter: AsyncRateLimiter | None = None,
        validator: InputValidator | None = None,
    ) -> None:
        self._api_key = youtube_api_key
        self._config = config
        self._ted_topics = config.lowercased_ted_topics()
        self._cache = cache
        self._semaphore = semaphore or asyncio.Semaphore(config.api_concurrency_limit)
        self._validator = validator
        self._rate_limiter = rate_limiter

    def collect(self) -> List[Post]:
        try:
            return asyncio.run(self.collect_async())
        except RuntimeError:
            LOGGER.warning("Event loop already running; skipping TED async collection.")
            return []

    async def collect_async(self) -> List[Post]:
        timeout = aiohttp.ClientTimeout(total=self._config.request_timeout_seconds)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            cutoff = datetime.now() - timedelta(days=self._config.collection_days)
            videos: List[Post] = []

            for query in self._config.ted_search_queries:
                items = await self._search_videos(session, query, cutoff)
                if not items:
                    continue
                details = await self._load_video_details(session, items)
                videos.extend(self._parse_videos(details))

        LOGGER.info("Collected %d TED videos.", len(videos))
        return videos

    async def _search_videos(
        self, session: aiohttp.ClientSession, query: str, cutoff: datetime
    ) -> Sequence[Dict]:
        params = {
            "q": query,
            "type": "video",
            "part": "id,snippet",
            "maxResults": 30,
            "publishedAfter": cutoff.isoformat() + "Z",
            "order": "relevance",
            "key": self._api_key,
        }

        cache_key = f"ted_search_{query}"
        cached = self._cache.get(cache_key) if self._cache else None
        if cached:
            return cached

        async def _op():
            async with self._semaphore:
                await self._throttle("youtube")
                await asyncio.sleep(self._config.api_request_delay_seconds)
                async with session.get(YOUTUBE_SEARCH_URL, params=params) as response:
                    response.raise_for_status()
                    return await response.json()

        try:
            data = await async_retry(
                _op,
                retries=self._config.api_retries,
                base_delay=self._config.api_retry_base_delay,
                retry_exceptions=(aiohttp.ClientError, asyncio.TimeoutError),
            )
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError):
            LOGGER.exception("YouTube search failed for query '%s'.", query)
            return []

        items = data.get("items", [])
        if self._cache:
            self._cache.set(cache_key, items)
        return items

    async def _load_video_details(
        self, session: aiohttp.ClientSession, items: Sequence[Dict]
    ) -> Sequence[Dict]:
        video_ids = [
            item["id"]["videoId"]
            for item in items
            if isinstance(item, dict)
            and isinstance(item.get("id"), dict)
            and item["id"].get("videoId")
        ]
        if not video_ids:
            return []

        params = {
            "part": "snippet,contentDetails,statistics",
            "id": ",".join(video_ids),
            "key": self._api_key,
        }

        details_key = f"ted_details_{hash(tuple(sorted(video_ids)))}"
        cached = self._cache.get(details_key) if self._cache else None
        if cached:
            return cached

        async def _op():
            async with self._semaphore:
                await self._throttle("youtube")
                await asyncio.sleep(self._config.api_request_delay_seconds)
                async with session.get(YOUTUBE_VIDEOS_URL, params=params) as response:
                    response.raise_for_status()
                    return await response.json()

        try:
            data = await async_retry(
                _op,
                retries=self._config.api_retries,
                base_delay=self._config.api_retry_base_delay,
                retry_exceptions=(aiohttp.ClientError, asyncio.TimeoutError),
            )
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError):
            LOGGER.exception("Failed to load video details for %s", video_ids)
            return []

        items = data.get("items", [])
        if self._cache:
            self._cache.set(details_key, items)
        return items

    def _parse_videos(self, items: Sequence[Dict]) -> List[Post]:
        videos: List[Post] = []
        for video in items:
            snippet = video.get("snippet", {})
            content_details = video.get("contentDetails", {})
            statistics = video.get("statistics", {})

            try:
                duration = parse_iso8601_duration(content_details.get("duration", ""))
            except ValueError:
                LOGGER.debug(
                    "Skipping video %s due to invalid duration format.",
                    video.get("id"),
                )
                continue

            if duration < self._config.ted_min_duration_seconds:
                continue
            if not self._is_relevant(snippet):
                continue

            published_at = snippet.get("publishedAt")
            created = (
                datetime.fromisoformat(published_at.replace("Z", "+00:00")).replace(
                    tzinfo=None
                )
                if published_at
                else None
            )

            description = sanitize_text((snippet.get("description") or "")[:300])
            url = f"https://www.youtube.com/watch?v={video.get('id')}"
            if self._validator and not self._validator.is_safe_url(url):
                LOGGER.debug("Skipping TED video due to unsafe URL: %s", url)
                continue
            videos.append(
                {
                    "title": sanitize_text(snippet.get("title", "")),
                    "url": url,
                    "created_utc": created,
                    "duration": duration,
                    "view_count": int(statistics.get("viewCount", 0) or 0),
                    "description": description,
                    "source": "ted_youtube",
                    "channel": sanitize_text(snippet.get("channelTitle", "")),
                    "score": int(statistics.get("viewCount", 0) or 0) // 1000,
                    "selftext": description,
                }
            )
        return videos

    def _is_relevant(self, snippet: Dict[str, object]) -> bool:
        title = snippet.get("title", "")
        description = snippet.get("description", "")
        text = f"{title} {description}".lower()
        return any(topic in text for topic in self._ted_topics)

    async def _throttle(self, key: str) -> None:
        if self._rate_limiter:
            await self._rate_limiter.acquire(key)
