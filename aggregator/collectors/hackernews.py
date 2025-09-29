from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import List

import requests

from ..config import AggregatorConfig
from ..types import Post

LOGGER = logging.getLogger(__name__)

TOPSTORIES_ENDPOINT = "https://hacker-news.firebaseio.com/v0/topstories.json"
ITEM_ENDPOINT_TEMPLATE = "https://hacker-news.firebaseio.com/v0/item/{story_id}.json"


class HackerNewsCollector:
    def __init__(self, config: AggregatorConfig) -> None:
        self._config = config

    def collect(self) -> List[Post]:
        try:
            response = requests.get(
                TOPSTORIES_ENDPOINT,
                timeout=self._config.request_timeout_seconds,
            )
            response.raise_for_status()
            story_ids = response.json()
        except (requests.RequestException, ValueError):
            LOGGER.exception("Failed to request Hacker News top stories.")
            return []

        if not isinstance(story_ids, list):
            LOGGER.warning("Unexpected response format for Hacker News top stories.")
            return []

        posts: List[Post] = []
        cutoff = datetime.now() - timedelta(days=self._config.collection_days)
        for story_id in story_ids[: self._config.hackernews_top_limit]:
            post = self._load_story(story_id, cutoff)
            if post:
                posts.append(post)

        LOGGER.info("Collected %d Hacker News posts.", len(posts))
        return posts

    def _load_story(self, story_id: int, cutoff: datetime) -> Post | None:
        try:
            story_response = requests.get(
                ITEM_ENDPOINT_TEMPLATE.format(story_id=story_id),
                timeout=self._config.request_timeout_seconds,
            )
            story_response.raise_for_status()
            story = story_response.json()
        except (requests.RequestException, ValueError):
            LOGGER.debug(
                "Skipping Hacker News story %s due to request or parsing error.",
                story_id,
            )
            return None

        if not isinstance(story, dict) or "title" not in story:
            return None

        story_time = datetime.fromtimestamp(story.get("time", 0))
        if story_time < cutoff:
            return None

        return {
            "title": story["title"],
            "url": story.get(
                "url",
                f"https://news.ycombinator.com/item?id={story_id}",
            ),
            "score": int(story.get("score", 0) or 0),
            "created_utc": story_time,
            "num_comments": int(story.get("descendants", 0) or 0),
            "author": story.get("by", "unknown"),
            "source": "hackernews",
            "hn_id": story_id,
            "selftext": "",
        }
