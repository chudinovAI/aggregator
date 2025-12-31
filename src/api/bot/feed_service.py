"""
Real-time feed service that fetches and scores posts for user topics.

This service:
1. Fetches posts from HN and Reddit based on user topics
2. Scores them using semantic similarity to topics
3. Returns top-N most relevant posts
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

LOGGER = logging.getLogger(__name__)

# Config path
CONFIG_PATH = Path(__file__).parents[3] / "config"


@dataclass
class ScoredPost:
    """Post with relevance score."""

    title: str
    url: str
    source: str
    score: float  # 0.0 to 1.0 relevance score
    published_at: datetime
    engagement: int  # upvotes/points
    raw_data: dict[str, Any] | None = None


# Topic to subreddit mapping
TOPIC_SUBREDDITS: dict[str, list[str]] = {
    "python": ["Python", "django", "flask", "FastAPI", "learnpython"],
    "machine_learning": [
        "MachineLearning",
        "deeplearning",
        "LocalLLaMA",
        "ChatGPT",
        "OpenAI",
        "ClaudeAI",
        "ollama",
        "LangChain",
    ],
    "web_development": [
        "webdev",
        "javascript",
        "reactjs",
        "vuejs",
        "Frontend",
        "nextjs",
        "sveltejs",
    ],
    "devops": ["devops", "docker", "kubernetes", "aws", "terraform", "sre"],
    "rust": ["rust", "learnrust", "rust_gamedev"],
    "golang": ["golang", "learngolang"],
    "security": ["netsec", "cybersecurity", "hacking", "privacy"],
    "databases": ["PostgreSQL", "mysql", "mongodb", "redis", "Database"],
    "linux": ["linux", "linuxadmin", "sysadmin", "selfhosted", "homelab"],
    "startups": ["startups", "SaaS", "Entrepreneur", "indiehackers"],
    "career": ["cscareerquestions", "ExperiencedDevs", "leetcode"],
    "open_source": ["opensource", "github", "selfhosted"],
}


def _get_subreddits_for_topics(topics: list[str]) -> list[str]:
    """Get relevant subreddits for user topics."""
    subreddits = set()
    for topic in topics:
        topic_key = topic.lower().replace(" ", "_")
        if topic_key in TOPIC_SUBREDDITS:
            subreddits.update(TOPIC_SUBREDDITS[topic_key])
        else:
            # Try to use topic as subreddit name
            subreddits.add(topic.replace(" ", ""))
    return list(subreddits)[:15]  # Limit to avoid too many requests


def _load_topic_keywords() -> dict[str, list[str]]:
    """Load topic keywords from config."""
    config_path = CONFIG_PATH / "topics.json"
    if not config_path.exists():
        return {}
    try:
        with open(config_path) as f:
            data = json.load(f)
            topics = data.get("topics", {})
            return {k: v.get("keywords", []) for k, v in topics.items()}
    except Exception:
        return {}


class FeedService:
    """Service for fetching and scoring posts in real-time."""

    def __init__(self) -> None:
        self._client: httpx.AsyncClient | None = None
        self._topic_keywords = _load_topic_keywords()
        self._scorer: Any = None  # SemanticScorer if available

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                headers={
                    "User-Agent": "NewsAggregator/1.0 (telegram bot)",
                    "Accept": "application/json",
                },
                timeout=httpx.Timeout(15.0, connect=5.0),
                follow_redirects=True,
            )
        return self._client

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    def _init_scorer(self) -> None:
        """Initialize semantic scorer if available."""
        if self._scorer is not None:
            return
        try:
            # Dynamic import to avoid hard dependency
            import importlib

            module = importlib.import_module("src.ml.semantic_scorer")
            SemanticScorer = getattr(module, "SemanticScorer")

            self._scorer = SemanticScorer()
            if not self._scorer.initialize():
                self._scorer = None
                LOGGER.info("SemanticScorer not available, using keyword matching")
        except Exception as e:
            LOGGER.warning("Failed to init SemanticScorer: %s", e)
            self._scorer = None

    async def get_top_posts(
        self,
        topics: list[str],
        sources: list[str],
        limit: int = 10,
    ) -> list[ScoredPost]:
        """
        Fetch and score posts for user topics.

        Args:
            topics: User's interest topics
            sources: Enabled sources (reddit, hackernews)
            limit: Number of top posts to return

        Returns:
            List of scored posts, sorted by relevance
        """
        if not topics:
            return []

        self._init_scorer()

        # Fetch posts from all sources concurrently
        tasks = []
        if "hackernews" in sources:
            tasks.append(self._fetch_hackernews(topics))
        if "reddit" in sources:
            subreddits = _get_subreddits_for_topics(topics)
            if subreddits:
                tasks.append(self._fetch_reddit(subreddits))

        if not tasks:
            return []

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect all posts
        all_posts: list[ScoredPost] = []
        for result in results:
            if isinstance(result, BaseException):
                LOGGER.warning("Fetch error: %s", result)
                continue
            if isinstance(result, list):
                all_posts.extend(result)

        if not all_posts:
            return []

        # Score posts
        scored_posts = self._score_posts(all_posts, topics)

        # Sort by score and return top N
        scored_posts.sort(key=lambda p: p.score, reverse=True)
        return scored_posts[:limit]

    async def _fetch_hackernews(self, topics: list[str]) -> list[ScoredPost]:
        """Fetch top stories from HN."""
        client = await self._get_client()
        posts: list[ScoredPost] = []

        try:
            # Get top story IDs
            resp = await client.get(
                "https://hacker-news.firebaseio.com/v0/topstories.json"
            )
            resp.raise_for_status()
            story_ids = resp.json()[:50]  # Top 50

            # Fetch stories concurrently (batch of 20)
            semaphore = asyncio.Semaphore(20)

            async def fetch_story(story_id: int) -> dict[str, Any] | None:
                async with semaphore:
                    try:
                        r = await client.get(
                            f"https://hacker-news.firebaseio.com/v0/item/{story_id}.json"
                        )
                        r.raise_for_status()
                        return r.json()
                    except Exception:
                        return None

            stories = await asyncio.gather(
                *[fetch_story(sid) for sid in story_ids],
                return_exceptions=True,
            )

            for story in stories:
                if not isinstance(story, dict) or not story:
                    continue
                if story.get("type") not in ("story", "job"):
                    continue

                title = story.get("title", "")
                if not title:
                    continue

                url = story.get("url") or f"https://news.ycombinator.com/item?id={story.get('id')}"
                timestamp = story.get("time", 0)
                published = (
                    datetime.fromtimestamp(timestamp, tz=UTC)
                    if timestamp
                    else datetime.now(UTC)
                )

                posts.append(
                    ScoredPost(
                        title=title,
                        url=url,
                        source="HN",
                        score=0.0,  # Will be scored later
                        published_at=published,
                        engagement=story.get("score", 0),
                        raw_data={"text": story.get("text", "")},
                    )
                )

        except Exception as e:
            LOGGER.warning("HN fetch error: %s", e)

        return posts

    async def _fetch_reddit(self, subreddits: list[str]) -> list[ScoredPost]:
        """Fetch top posts from subreddits."""
        client = await self._get_client()
        posts: list[ScoredPost] = []

        # Fetch subreddits concurrently
        semaphore = asyncio.Semaphore(5)

        async def fetch_subreddit(subreddit: str) -> list[ScoredPost]:
            async with semaphore:
                try:
                    await asyncio.sleep(0.1)  # Rate limiting
                    r = await client.get(
                        f"https://www.reddit.com/r/{subreddit}/hot.json",
                        params={"limit": 25},
                    )
                    r.raise_for_status()
                    data = r.json()
                    result = []

                    for child in data.get("data", {}).get("children", []):
                        raw = child.get("data", {})
                        if not raw.get("title"):
                            continue

                        # Skip stickied posts
                        if raw.get("stickied"):
                            continue

                        url = (
                            raw.get("url_overridden_by_dest")
                            or raw.get("url")
                            or f"https://reddit.com{raw.get('permalink', '')}"
                        )
                        timestamp = raw.get("created_utc", 0)
                        published = (
                            datetime.fromtimestamp(timestamp, tz=UTC)
                            if timestamp
                            else datetime.now(UTC)
                        )

                        result.append(
                            ScoredPost(
                                title=raw.get("title", ""),
                                url=url,
                                source=f"r/{subreddit}",
                                score=0.0,
                                published_at=published,
                                engagement=raw.get("ups", 0),
                                raw_data={"selftext": raw.get("selftext", "")},
                            )
                        )
                    return result
                except Exception as e:
                    LOGGER.debug("Reddit fetch error for r/%s: %s", subreddit, e)
                    return []

        results = await asyncio.gather(
            *[fetch_subreddit(sub) for sub in subreddits],
            return_exceptions=True,
        )

        for result in results:
            if isinstance(result, list):
                posts.extend(result)

        return posts

    def _score_posts(
        self,
        posts: list[ScoredPost],
        topics: list[str],
    ) -> list[ScoredPost]:
        """Score posts by relevance to topics."""
        if not posts:
            return []

        # Get all keywords for topics
        all_keywords: set[str] = set()
        for topic in topics:
            topic_key = topic.lower().replace(" ", "_")
            if topic_key in self._topic_keywords:
                all_keywords.update(
                    kw.lower() for kw in self._topic_keywords[topic_key]
                )
            all_keywords.add(topic.lower())

        # Try semantic scoring first
        if self._scorer and self._scorer.is_available:
            return self._score_semantic(posts, topics)

        # Fallback to keyword scoring
        return self._score_keywords(posts, all_keywords)

    def _score_semantic(
        self,
        posts: list[ScoredPost],
        topics: list[str],
    ) -> list[ScoredPost]:
        """Score posts using semantic similarity."""
        post_dicts = [
            {
                "title": p.title,
                "content": (p.raw_data or {}).get("selftext", "")
                or (p.raw_data or {}).get("text", ""),
            }
            for p in posts
        ]

        scores = self._scorer.score_posts_batch(post_dicts, topics)

        for post, score in zip(posts, scores):
            # Combine semantic score with engagement (log scale)
            import math

            engagement_bonus = min(0.2, math.log10(max(1, post.engagement)) / 20)
            post.score = min(1.0, score * 0.85 + engagement_bonus)

        return posts

    def _score_keywords(
        self,
        posts: list[ScoredPost],
        keywords: set[str],
    ) -> list[ScoredPost]:
        """Score posts using keyword matching."""
        import math

        for post in posts:
            title_lower = post.title.lower()
            content = ""
            if post.raw_data:
                content = (
                    post.raw_data.get("selftext", "")
                    or post.raw_data.get("text", "")
                ).lower()

            # Count keyword matches
            matches = 0
            for kw in keywords:
                if kw in title_lower:
                    matches += 2  # Title match worth more
                if kw in content:
                    matches += 1

            # Normalize score (0-1)
            keyword_score = min(1.0, matches / max(1, len(keywords) * 0.5))

            # Add engagement bonus
            engagement_bonus = min(0.3, math.log10(max(1, post.engagement)) / 15)

            post.score = min(1.0, keyword_score * 0.7 + engagement_bonus)

        return posts


# Global instance
_feed_service: FeedService | None = None


def get_feed_service() -> FeedService:
    """Get or create global feed service instance."""
    global _feed_service
    if _feed_service is None:
        _feed_service = FeedService()
    return _feed_service


__all__ = ["FeedService", "ScoredPost", "get_feed_service"]
