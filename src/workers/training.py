"""
Classifier training task for background workers.

This module provides the retrain_classifier task that retrains
the ML classifier using labeled data from the database.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from ..config import Settings
from ..db.models import Post, UserPostRead
from ..ml.classifier import ClassifierConfig, TextClassifier, TrainingDataset
from .types import TaskResult

LOGGER = logging.getLogger(__name__)


async def retrain_classifier(
    session_factory: async_sessionmaker[AsyncSession],
    settings: Settings,
    *,
    min_samples: int = 100,
) -> TaskResult:
    """
    Retrain the ML classifier using labeled data from the database.

    This task:
    1. Loads labeled posts from the database (posts marked as read/interesting)
    2. Prepares training dataset
    3. Retrains the classifier model
    4. Saves the updated model

    Args:
        session_factory: SQLAlchemy async session factory
        settings: Application settings
        min_samples: Minimum samples required for retraining

    Returns:
        TaskResult with training statistics
    """
    started_at = datetime.now(UTC)
    LOGGER.info("Starting retrain_classifier task")

    async with session_factory() as session:
        # Query posts with user feedback (read = positive, not read = negative)
        # Using read status as implicit feedback
        read_posts_query = (
            select(Post).join(UserPostRead, UserPostRead.post_id == Post.id).distinct()
        )
        read_result = await session.scalars(read_posts_query)
        positive_posts = list(read_result)

        # Get unread posts as negative samples
        read_ids_subquery = select(UserPostRead.post_id).distinct().scalar_subquery()
        unread_posts_query = (
            select(Post)
            .where(Post.id.notin_(read_ids_subquery))
            .order_by(func.random())
            .limit(len(positive_posts))  # Balance classes
        )
        unread_result = await session.scalars(unread_posts_query)
        negative_posts = list(unread_result)

    total_samples = len(positive_posts) + len(negative_posts)
    LOGGER.info(
        "Training data: positive=%d, negative=%d, total=%d",
        len(positive_posts),
        len(negative_posts),
        total_samples,
    )

    if total_samples < min_samples:
        message = f"Insufficient training data: {total_samples} < {min_samples}"
        LOGGER.warning(message)
        return TaskResult(
            task_name="retrain_classifier",
            success=False,
            message=message,
            details={
                "positive_samples": len(positive_posts),
                "negative_samples": len(negative_posts),
                "min_required": min_samples,
            },
            started_at=started_at,
            finished_at=datetime.now(UTC),
        )

    # Prepare training data
    texts: list[str] = []
    labels: list[str] = []

    for post in positive_posts:
        texts.append(f"{post.title} {post.content}")
        labels.append("interesting")

    for post in negative_posts:
        texts.append(f"{post.title} {post.content}")
        labels.append("boring")

    # Create and train classifier
    try:
        config = ClassifierConfig(
            model_path=settings.ml.model_path,
            confidence_threshold=settings.ml.threshold,
        )
        classifier = TextClassifier(config)
        training_data = TrainingDataset(texts=texts, labels=labels)

        await classifier.retrain(training_data)

        finished_at = datetime.now(UTC)
        LOGGER.info("Classifier retrained successfully")

        return TaskResult(
            task_name="retrain_classifier",
            success=True,
            message="Classifier retrained successfully",
            details={
                "positive_samples": len(positive_posts),
                "negative_samples": len(negative_posts),
                "total_samples": total_samples,
                "model_path": str(settings.ml.model_path),
            },
            started_at=started_at,
            finished_at=finished_at,
        )

    except Exception as exc:
        LOGGER.exception("Classifier retraining failed: %s", exc)
        return TaskResult(
            task_name="retrain_classifier",
            success=False,
            message=f"Retraining failed: {exc}",
            details={"error": str(exc)},
            started_at=started_at,
            finished_at=datetime.now(UTC),
        )


__all__ = [
    "retrain_classifier",
]
