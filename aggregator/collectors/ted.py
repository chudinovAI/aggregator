from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Sequence

from googleapiclient.errors import HttpError

from ..config import AggregatorConfig
from ..types import Post
from ..utils import parse_iso8601_duration

LOGGER = logging.getLogger(__name__)


class TedCollector:
    def __init__(self, youtube_client: Any, config: AggregatorConfig) -> None:
        self._youtube = youtube_client
        self._config = config
        self._ted_topics = config.lowercased_ted_topics()

    def collect(self) -> List[Post]:
        videos: List[Post] = []
        cutoff = datetime.now() - timedelta(days=self._config.collection_days)

        for query in self._config.ted_search_queries:
            items = self._search_videos(query, cutoff)
            if not items:
                continue
            video_details = self._load_video_details(items)
            videos.extend(self._parse_videos(video_details))

        LOGGER.info("Collected %d TED videos.", len(videos))
        return videos

    def _search_videos(self, query: str, cutoff: datetime) -> Sequence[Dict[str, Any]]:
        try:
            response = (
                self._youtube.search()
                .list(
                    q=query,
                    type="video",
                    part="id,snippet",
                    maxResults=30,
                    publishedAfter=cutoff.isoformat() + "Z",
                    order="relevance",
                )
                .execute()
            )
        except HttpError:
            LOGGER.exception("YouTube search failed for query '%s'.", query)
            return []
        return response.get("items", [])

    def _load_video_details(self, items: Sequence[Dict[str, Any]]):
        video_ids = [
            item["id"]["videoId"]
            for item in items
            if isinstance(item, dict)
            and isinstance(item.get("id"), dict)
            and item["id"].get("videoId")
        ]
        if not video_ids:
            return []
        try:
            response = (
                self._youtube.videos()
                .list(
                    part="snippet,contentDetails,statistics",
                    id=",".join(video_ids),
                )
                .execute()
            )
        except HttpError:
            LOGGER.exception("Failed to load video details for %s", video_ids)
            return []
        return response.get("items", [])

    def _parse_videos(self, items: Sequence[Dict[str, Any]]) -> List[Post]:
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

            description = (snippet.get("description") or "")[:300]
            videos.append(
                {
                    "title": snippet.get("title", ""),
                    "url": f"https://www.youtube.com/watch?v={video.get('id')}",
                    "created_utc": created,
                    "duration": duration,
                    "view_count": int(statistics.get("viewCount", 0) or 0),
                    "description": description,
                    "source": "ted_youtube",
                    "channel": snippet.get("channelTitle", ""),
                    "score": int(statistics.get("viewCount", 0) or 0) // 1000,
                    "selftext": description,
                }
            )
        return videos

    def _is_relevant(self, snippet: Dict[str, Any]) -> bool:
        title = snippet.get("title", "")
        description = snippet.get("description", "")
        text = f"{title} {description}".lower()
        return any(topic in text for topic in self._ted_topics)
