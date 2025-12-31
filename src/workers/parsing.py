"""
Source parsing task for background workers.

This module provides the parse_all_sources task that collects posts
from all configured news sources.
"""

from __future__ import annotations

import asyncio
import logging
import math
from datetime import UTC, datetime

from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from ..aggregator.parsers import ParsedPost
from ..aggregator.parsers.base import BaseParser
from ..config import Settings
from ..db.cache import PostsCache
from ..db.repository import PostRepository
from ..ml.classifier import ClassifierConfig, TextClassifier
from ..ml.semantic_scorer import SemanticScorer, get_semantic_scorer
from .parsers import REGISTERED_SOURCES, create_parser
from .types import ParseResult, TaskResult

LOGGER = logging.getLogger(__name__)

# Global semantic scorer instance
_semantic_scorer: SemanticScorer | None = None


def _get_semantic_scorer() -> SemanticScorer | None:
    """Get or initialize the global semantic scorer."""
    global _semantic_scorer
    if _semantic_scorer is None:
        try:
            scorer = get_semantic_scorer()
            if scorer.is_available:
                _semantic_scorer = scorer
                LOGGER.info("Semantic scorer initialized successfully")
            else:
                LOGGER.info("Semantic scorer not available, using heuristics")
        except Exception as e:
            LOGGER.warning("Failed to initialize semantic scorer: %s", e)
    return _semantic_scorer


def _compute_heuristic_score(post: ParsedPost) -> float:
    """
    Compute a heuristic score based on engagement metrics when no ML model is available.

    Uses upvotes, comments, and recency to estimate post "interestingness".
    Returns a score between 0.0 and 1.0.
    """
    raw = post.raw_data or {}
    score = 0.5  # Base score

    # Reddit scoring (upvotes, comments ratio)
    if "ups" in raw or "score" in raw:
        upvotes = raw.get("ups") or raw.get("score") or 0
        comments = raw.get("num_comments", 0)

        # Normalize upvotes (log scale, max reasonable ~10000)
        if upvotes > 0:
            upvote_factor = min(math.log10(upvotes + 1) / 4.0, 1.0)  # 0-1 scale
            score += upvote_factor * 0.3

        # Comments indicate engagement
        if comments > 0:
            comment_factor = min(math.log10(comments + 1) / 3.0, 1.0)
            score += comment_factor * 0.15

        # Upvote ratio (Reddit specific)
        upvote_ratio = raw.get("upvote_ratio", 0.5)
        score += (upvote_ratio - 0.5) * 0.1

    # HackerNews scoring
    if "points" in raw or ("score" in raw and "descendants" in raw):
        points = raw.get("points") or raw.get("score") or 0
        descendants = raw.get("descendants", 0)

        if points > 0:
            points_factor = min(math.log10(points + 1) / 3.0, 1.0)
            score += points_factor * 0.3

        if descendants > 0:
            comment_factor = min(math.log10(descendants + 1) / 3.0, 1.0)
            score += comment_factor * 0.15

    # Clamp to valid range
    return max(0.0, min(1.0, score))


def _get_topics_for_source(source_name: str, settings: Settings) -> list[str]:
    """Get list of topics/subreddits to query for a source."""
    if source_name == "reddit":
        # Parse up to 50 subreddits to include language-specific subs (golang, rust, etc.)
        subreddits = settings.sources.reddit.subreddits[:50]
        return subreddits if subreddits else ["technology"]
    elif source_name == "hackernews":
        # HN doesn't support topics, just return "top"
        return ["top"]
    return ["technology"]


async def parse_all_sources(
    session_factory: async_sessionmaker[AsyncSession],
    redis_client: Redis | None,
    settings: Settings,
    *,
    sources: list[str] | None = None,
) -> TaskResult:
    """
    Collect new posts from all configured sources.

    This task:
    1. Iterates through all registered sources (or specified subset)
    2. Fetches posts from each source
    3. Runs classifier to score posts
    4. Saves posts to database
    5. Updates the Redis cache

    Args:
        session_factory: SQLAlchemy async session factory
        redis_client: Optional Redis client for caching
        settings: Application settings
        sources: Optional list of source names to parse (defaults to all)

    Returns:
        TaskResult with parsing statistics
    """
    started_at = datetime.now(UTC)
    LOGGER.info("Starting parse_all_sources task")

    sources_to_parse = sources or list(REGISTERED_SOURCES.keys())

    # Filter to only enabled sources
    enabled_sources: list[str] = []
    for source_name in sources_to_parse:
        check_parser = create_parser(source_name, settings)
        if check_parser:
            enabled_sources.append(source_name)

    if not enabled_sources:
        LOGGER.warning("No sources enabled for parsing")
        return TaskResult(
            task_name="parse_all_sources",
            success=True,
            message="No sources enabled",
            details={"sources_parsed": 0},
            started_at=started_at,
            finished_at=datetime.now(UTC),
        )

    results: list[ParseResult] = []
    total_fetched = 0
    total_saved = 0
    total_errors: list[str] = []

    # Initialize classifier for scoring
    classifier: TextClassifier | None = None
    try:
        if settings.ml.model_path.exists():
            config = ClassifierConfig(
                model_path=settings.ml.model_path,
                confidence_threshold=settings.ml.threshold,
            )
            classifier = TextClassifier(config)
            LOGGER.info("Classifier loaded for post scoring")
    except Exception as exc:
        LOGGER.warning("Could not load classifier: %s", exc)

    # Initialize cache if Redis is available
    cache: PostsCache | None = None
    if redis_client:
        cache = PostsCache(redis_client, namespace="posts")

    for source_name in enabled_sources:
        source_start = datetime.now(UTC)
        source_errors: list[str] = []
        posts_fetched = 0
        posts_saved = 0
        parser: BaseParser | None = None

        try:
            # Create parser instance
            parser = create_parser(source_name, settings)
            if not parser:
                LOGGER.warning("Could not create parser for source: %s", source_name)
                source_errors.append(f"Parser creation failed for {source_name}")
                continue

            # Get list of topics/subreddits to fetch
            topics_to_fetch = _get_topics_for_source(source_name, settings)
            limit_per_topic = max(10, settings.parsing.max_articles_per_source // len(topics_to_fetch))

            # Fetch posts from all topics
            all_posts: list[ParsedPost] = []
            for topic in topics_to_fetch:
                try:
                    topic_posts = await parser.get_posts(topic=topic, limit=limit_per_topic)
                    all_posts.extend(topic_posts)
                    if source_name == "reddit":
                        await asyncio.sleep(0.3)  # Rate limiting between subreddits
                except Exception as topic_exc:
                    LOGGER.debug("Failed to fetch %s/%s: %s", source_name, topic, topic_exc)

            posts = all_posts
            posts_fetched = len(posts)

            if not posts:
                LOGGER.info("No posts fetched from %s", source_name)
                continue

            # Score posts with classifier, semantic scorer, or heuristics
            if classifier and classifier.is_loaded:
                # Priority 1: Trained classifier
                texts = [f"{p.title} {p.content}" for p in posts]
                try:
                    predictions = classifier.batch_predict(texts)
                    scores = [preds[0].confidence if preds else 0.0 for preds in predictions]
                except Exception as exc:
                    LOGGER.warning("Classifier scoring failed: %s", exc)
                    scores = [_compute_heuristic_score(p) for p in posts]
            else:
                # Priority 2: Semantic scoring with DistilBERT
                semantic_scorer = _get_semantic_scorer()
                if semantic_scorer and semantic_scorer.is_available:
                    post_dicts = [{"title": p.title, "content": p.content} for p in posts]
                    try:
                        semantic_scores = semantic_scorer.score_posts_batch(post_dicts)
                        # Combine semantic score with engagement heuristics
                        heuristic_scores = [_compute_heuristic_score(p) for p in posts]
                        # 70% semantic, 30% engagement
                        scores = [
                            0.7 * sem + 0.3 * heur
                            for sem, heur in zip(semantic_scores, heuristic_scores)
                        ]
                        LOGGER.debug("Using semantic + heuristic scoring for %d posts", len(posts))
                    except Exception as exc:
                        LOGGER.warning("Semantic scoring failed: %s", exc)
                        scores = [_compute_heuristic_score(p) for p in posts]
                else:
                    # Priority 3: Pure heuristic scoring
                    scores = [_compute_heuristic_score(p) for p in posts]
                    LOGGER.debug("Using heuristic scoring for %d posts", len(posts))

            # Save posts to database
            async with session_factory() as session:
                repo = PostRepository(session)
                for post, score in zip(posts, scores):
                    try:
                        saved = await repo.save_post(post, classifier_score=score)
                        if saved:
                            posts_saved += 1
                    except Exception as exc:
                        LOGGER.warning("Failed to save post %s: %s", post.id, exc)
                        source_errors.append(f"Save failed: {post.id}")

                await session.commit()

            # Update cache
            if cache:
                try:
                    await cache.cache_posts(
                        source_name,
                        posts,
                        ttl_seconds=settings.redis.cache_ttl_seconds,
                    )
                except Exception as exc:
                    LOGGER.warning("Cache update failed for %s: %s", source_name, exc)

            LOGGER.info(
                "Parsed %s: fetched=%d, saved=%d",
                source_name,
                posts_fetched,
                posts_saved,
            )

        except Exception as exc:
            LOGGER.exception("Error parsing source %s: %s", source_name, exc)
            source_errors.append(str(exc))
        finally:
            # Close parser if it owns its HTTP client
            if parser is not None:
                close_method = getattr(parser, "aclose", None)
                if close_method is not None:
                    try:
                        result = close_method()
                        if asyncio.iscoroutine(result):
                            await result
                    except Exception:
                        pass

        source_duration = (datetime.now(UTC) - source_start).total_seconds()
        results.append(
            ParseResult(
                source=source_name,
                posts_fetched=posts_fetched,
                posts_saved=posts_saved,
                errors=source_errors,
                duration_seconds=source_duration,
            )
        )
        total_fetched += posts_fetched
        total_saved += posts_saved
        total_errors.extend(source_errors)

    finished_at = datetime.now(UTC)
    success = len(total_errors) == 0

    LOGGER.info(
        "parse_all_sources completed: sources=%d, fetched=%d, saved=%d, errors=%d",
        len(enabled_sources),
        total_fetched,
        total_saved,
        len(total_errors),
    )

    return TaskResult(
        task_name="parse_all_sources",
        success=success,
        message=f"Parsed {len(enabled_sources)} sources",
        details={
            "sources_parsed": len(enabled_sources),
            "total_fetched": total_fetched,
            "total_saved": total_saved,
            "errors": total_errors,
            "results": [
                {
                    "source": r.source,
                    "fetched": r.posts_fetched,
                    "saved": r.posts_saved,
                    "duration": r.duration_seconds,
                }
                for r in results
            ],
        },
        started_at=started_at,
        finished_at=finished_at,
    )


__all__ = [
    "parse_all_sources",
]
