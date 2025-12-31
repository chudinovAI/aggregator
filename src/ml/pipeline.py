"""
Classifier pipelines that integrate with repositories and caching.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable, Sequence

from ..aggregator.parsers.base import ParsedPost
from ..config import Settings, get_settings
from ..db.cache import PostsCache
from ..db.models import Post
from ..db.repository import PostRepository
from .classifier import (
    ClassifierConfig,
    TextClassifier,
    TrainingDataset,
)

LOGGER = logging.getLogger(__name__)


class PredictionPipeline:
    """
    Executes batch predictions and persists the best posts into the database.
    """

    def __init__(
        self,
        classifier: TextClassifier,
        post_repository: PostRepository,
        *,
        cache: PostsCache | None = None,
        batch_size: int = 32,
        confidence_threshold: float | None = None,
    ) -> None:
        self._classifier = classifier
        self._post_repository = post_repository
        self._cache = cache
        self._batch_size = max(1, batch_size)
        self._threshold = (
            confidence_threshold
            if confidence_threshold is not None
            else classifier.config.confidence_threshold
        )

    async def run(
        self,
        source: str,
        posts: Sequence[ParsedPost],
        *,
        cache_ttl: int = 900,
    ) -> list[Post]:
        """
        Classify posts, persist the relevant ones, and refresh the cache.
        """

        dataset = list(posts)
        if not dataset and self._cache:
            cached = await self._cache.get_posts(source)
            if cached:
                dataset = cached

        if not dataset:
            return []

        loop = asyncio.get_running_loop()
        stored_posts: list[Post] = []

        for chunk in _chunk(dataset, self._batch_size):
            texts = [self._compose_text(post) for post in chunk]
            predictions = await loop.run_in_executor(None, self._classifier.batch_predict, texts)
            for parsed_post, result_list in zip(chunk, predictions):
                if not result_list:
                    continue
                top_prediction = result_list[0]
                if (
                    top_prediction.label != "interesting"
                    or top_prediction.confidence < self._threshold
                ):
                    continue
                saved_post = await self._post_repository.save_post(
                    parsed_post, classifier_score=top_prediction.confidence
                )
                if saved_post:
                    stored_posts.append(saved_post)

        if stored_posts:
            await self._post_repository.session.commit()
            LOGGER.info("Stored %d interesting posts for source '%s'.", len(stored_posts), source)

        if self._cache:
            await self._cache.cache_posts(source, dataset, ttl_seconds=cache_ttl)

        return stored_posts

    @staticmethod
    def _compose_text(post: ParsedPost) -> str:
        return f"{post.title.strip()} {post.content.strip()}".strip()


class TrainingPipeline:
    """
    Orchestrates classifier retraining to keep the model up to date.
    """

    def __init__(self, classifier: TextClassifier) -> None:
        self._classifier = classifier

    async def run(self, dataset: TrainingDataset) -> None:
        """Trigger asynchronous retraining with the supplied dataset."""

        await self._classifier.retrain(dataset)
        LOGGER.info("Training pipeline completed for %d samples.", len(dataset.texts))


def build_classifier_from_settings(settings: Settings | None = None) -> TextClassifier:
    """
    Build a classifier instance using defaults from the global settings.
    """

    settings = settings or get_settings()
    ml_settings = settings.ml
    config = ClassifierConfig(
        model_path=ml_settings.model_path,
        confidence_threshold=ml_settings.threshold,
        vectorizer_type="tfidf",
        embedding_model=None,
    )
    return TextClassifier(config)


def _chunk(items: Sequence[ParsedPost], size: int) -> Iterable[list[ParsedPost]]:
    """Yield fixed-size chunks from the provided sequence."""

    for index in range(0, len(items), size):
        yield list(items[index : index + size])


__all__ = [
    "PredictionPipeline",
    "TrainingPipeline",
    "build_classifier_from_settings",
]
