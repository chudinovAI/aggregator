"""
Semantic scoring using sentence transformers (DistilBERT-based).

This module provides zero-shot topic relevance scoring without requiring
a trained classifier. It uses pre-trained sentence embeddings to compute
similarity between posts and user topics.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

LOGGER = logging.getLogger(__name__)

# Default model - good balance of speed and quality
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Cache for topic embeddings
_topic_embeddings_cache: dict[str, NDArray[np.float32]] = {}


class SemanticScorer:
    """
    Scores posts based on semantic similarity to user topics.

    Uses sentence-transformers to create embeddings and compute
    cosine similarity between posts and topic keywords.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        topics_config_path: Path | None = None,
    ) -> None:
        """
        Initialize the semantic scorer.

        Args:
            model_name: HuggingFace model name for sentence embeddings
            topics_config_path: Path to topics.json config file
        """
        self._model_name = model_name
        self._model: SentenceTransformer | None = None
        self._topics_config_path = topics_config_path or (
            Path(__file__).parents[2] / "config" / "topics.json"
        )
        self._topic_keywords: dict[str, list[str]] = {}
        self._is_available = False

    @property
    def is_available(self) -> bool:
        """Check if sentence-transformers is installed and model is loaded."""
        return self._is_available and self._model is not None

    def initialize(self) -> bool:
        """
        Initialize the model and load topic embeddings.

        Returns:
            True if initialization succeeded, False otherwise.
        """
        try:
            from sentence_transformers import SentenceTransformer

            LOGGER.info("Loading sentence-transformer model: %s", self._model_name)
            self._model = SentenceTransformer(self._model_name)
            self._load_topic_keywords()
            self._precompute_topic_embeddings()
            self._is_available = True
            LOGGER.info("SemanticScorer initialized successfully")
            return True

        except ImportError:
            LOGGER.warning(
                "sentence-transformers not installed. "
                "Install with: uv add sentence-transformers torch"
            )
            self._is_available = False
            return False
        except Exception as e:
            LOGGER.exception("Failed to initialize SemanticScorer: %s", e)
            self._is_available = False
            return False

    def _load_topic_keywords(self) -> None:
        """Load topic keywords from config file."""
        if not self._topics_config_path.exists():
            LOGGER.warning("Topics config not found: %s", self._topics_config_path)
            return

        try:
            with open(self._topics_config_path) as f:
                data = json.load(f)
                topics = data.get("topics", {})

            for topic_key, topic_data in topics.items():
                keywords = topic_data.get("keywords", [])
                display_name = topic_data.get("display_name", topic_key)
                # Include both the display name and keywords
                all_keywords = [display_name] + keywords
                self._topic_keywords[topic_key] = all_keywords

            LOGGER.info("Loaded %d topics with keywords", len(self._topic_keywords))

        except Exception as e:
            LOGGER.warning("Failed to load topics config: %s", e)

    def _precompute_topic_embeddings(self) -> None:
        """Pre-compute embeddings for all topic keywords."""
        global _topic_embeddings_cache

        if not self._model or not self._topic_keywords:
            return

        for topic_key, keywords in self._topic_keywords.items():
            if topic_key in _topic_embeddings_cache:
                continue

            # Create a combined representation of the topic
            # by averaging embeddings of all keywords
            try:
                embeddings = self._model.encode(
                    keywords,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
                # Average the keyword embeddings
                topic_embedding = np.mean(embeddings, axis=0).astype(np.float32)
                _topic_embeddings_cache[topic_key] = topic_embedding
            except Exception as e:
                LOGGER.warning("Failed to embed topic %s: %s", topic_key, e)

        LOGGER.info("Pre-computed embeddings for %d topics", len(_topic_embeddings_cache))

    def score_post(
        self,
        title: str,
        content: str,
        user_topics: list[str] | None = None,
    ) -> float:
        """
        Score a single post based on semantic similarity to topics.

        Args:
            title: Post title
            content: Post content/summary
            user_topics: List of user's selected topics (uses all if None)

        Returns:
            Score between 0.0 and 1.0
        """
        if not self.is_available:
            return 0.5  # Default neutral score

        # Combine title and content (title weighted more)
        text = f"{title}. {title}. {content[:500] if content else ''}"

        try:
            post_embedding = self._model.encode(  # type: ignore
                text,
                convert_to_numpy=True,
                show_progress_bar=False,
            ).astype(np.float32)

            return self._compute_topic_similarity(post_embedding, user_topics)

        except Exception as e:
            LOGGER.warning("Failed to score post: %s", e)
            return 0.5

    def score_posts_batch(
        self,
        posts: list[dict[str, Any]],
        user_topics: list[str] | None = None,
    ) -> list[float]:
        """
        Score multiple posts efficiently using batch encoding.

        Args:
            posts: List of dicts with 'title' and 'content' keys
            user_topics: List of user's selected topics

        Returns:
            List of scores between 0.0 and 1.0
        """
        if not self.is_available or not posts:
            return [0.5] * len(posts)

        # Prepare texts (title weighted by repetition)
        texts = [
            f"{p.get('title', '')}. {p.get('title', '')}. {(p.get('content', '') or '')[:500]}"
            for p in posts
        ]

        try:
            embeddings = self._model.encode(  # type: ignore
                texts,
                convert_to_numpy=True,
                show_progress_bar=False,
                batch_size=32,
            ).astype(np.float32)

            scores = [self._compute_topic_similarity(emb, user_topics) for emb in embeddings]
            return scores

        except Exception as e:
            LOGGER.warning("Failed to batch score posts: %s", e)
            return [0.5] * len(posts)

    def _compute_topic_similarity(
        self,
        post_embedding: NDArray[np.float32],
        user_topics: list[str] | None = None,
    ) -> float:
        """
        Compute similarity between post embedding and topic embeddings.

        Uses cosine similarity and returns the max similarity to any
        of the user's topics (or all topics if none specified).
        """
        global _topic_embeddings_cache

        if not _topic_embeddings_cache:
            return 0.5

        # Normalize user topic names
        if user_topics:
            topic_keys = [t.lower().replace(" ", "_") for t in user_topics]
            topic_keys = [k for k in topic_keys if k in _topic_embeddings_cache]
            if not topic_keys:
                # User topics not found in cache, use all
                topic_keys = list(_topic_embeddings_cache.keys())
        else:
            topic_keys = list(_topic_embeddings_cache.keys())

        # Compute cosine similarities
        similarities = []
        post_norm = post_embedding / (np.linalg.norm(post_embedding) + 1e-8)

        for topic_key in topic_keys:
            topic_emb = _topic_embeddings_cache[topic_key]
            topic_norm = topic_emb / (np.linalg.norm(topic_emb) + 1e-8)
            sim = float(np.dot(post_norm, topic_norm))
            similarities.append(sim)

        if not similarities:
            return 0.5

        # Use max similarity, scaled to 0-1 range
        # Cosine similarity is in [-1, 1], but typically positive for related content
        max_sim = max(similarities)
        # Scale from typical range [0.2, 0.8] to [0, 1]
        score = (max_sim - 0.2) / 0.6
        return max(0.0, min(1.0, score))


# Global singleton instance
_scorer_instance: SemanticScorer | None = None


def get_semantic_scorer() -> SemanticScorer:
    """Get or create the global SemanticScorer instance."""
    global _scorer_instance
    if _scorer_instance is None:
        _scorer_instance = SemanticScorer()
        _scorer_instance.initialize()
    return _scorer_instance


def score_post_semantically(
    title: str,
    content: str,
    user_topics: list[str] | None = None,
) -> float:
    """
    Convenience function to score a single post.

    Falls back to 0.5 if semantic scoring is not available.
    """
    scorer = get_semantic_scorer()
    return scorer.score_post(title, content, user_topics)


__all__ = [
    "SemanticScorer",
    "get_semantic_scorer",
    "score_post_semantically",
]
