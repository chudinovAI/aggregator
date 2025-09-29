from __future__ import annotations

import logging
from datetime import datetime
from typing import Iterable, List

import praw
import prawcore

from ..config import AggregatorConfig
from ..types import Post

LOGGER = logging.getLogger(__name__)


class RedditCollector:
    def __init__(self, client: praw.Reddit, config: AggregatorConfig) -> None:
        self._client = client
        self._config = config

    def collect(self) -> List[Post]:
        posts: List[Post] = []
        for subreddit_name in self._config.subreddits:
            try:
                subreddit = self._client.subreddit(subreddit_name)
                posts.extend(self._collect_subreddit(subreddit_name, subreddit))
            except (
                praw.exceptions.PRAWException,
                prawcore.exceptions.PrawcoreException,
            ):
                LOGGER.exception("Failed to collect subreddit r/%s.", subreddit_name)
        LOGGER.info("Collected %d Reddit posts.", len(posts))
        return posts

    def _collect_subreddit(
        self, subreddit_name: str, subreddit: praw.models.Subreddit
    ) -> Iterable[Post]:
        for submission in subreddit.top("week", limit=self._config.reddit_top_limit):
            yield {
                "title": submission.title,
                "url": submission.url,
                "score": submission.score,
                "created_utc": datetime.fromtimestamp(submission.created_utc),
                "num_comments": submission.num_comments,
                "subreddit": subreddit_name,
                "author": str(submission.author),
                "selftext": (submission.selftext or "")[:300],
                "source": "reddit",
                "permalink": f"https://reddit.com{submission.permalink}",
            }
