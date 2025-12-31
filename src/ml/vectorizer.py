"""
Text vectorization utilities with caching support.
"""

from __future__ import annotations

import logging
import pickle
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray
from sklearn.feature_extraction.text import TfidfVectorizer

if TYPE_CHECKING:  # pragma: no cover - optional dependency typing aid
    from sentence_transformers import SentenceTransformer

VectorizerType = Literal["tfidf", "embedding"]
LOGGER = logging.getLogger(__name__)


class TextVectorizer:
    """
    Vectorizes text using either TF-IDF or transformer embeddings with result caching.
    """

    def __init__(
        self,
        *,
        vectorizer_type: VectorizerType = "tfidf",
        embedding_model: str | None = None,
        max_features: int = 5000,
    ) -> None:
        if vectorizer_type == "embedding" and not embedding_model:
            raise ValueError("embedding_model is required when vectorizer_type='embedding'.")

        self._type = vectorizer_type
        self._embedding_model_name = embedding_model
        self._max_features = max_features
        self._tfidf: TfidfVectorizer | None = None
        self._embedding_model: SentenceTransformer | None = None
        self._cache: dict[str, NDArray[np.float32]] = {}
        self._feature_dim: int | None = None
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        """Return True when the vectorizer has been fitted or initialized."""

        return self._is_fitted

    def fit_transform(self, texts: Sequence[str]) -> NDArray[np.float32]:
        """
        Fit the vectorizer on the provided texts and return the resulting features.
        """

        sanitized = self._sanitize_texts(texts)
        if not sanitized:
            raise ValueError("Cannot fit vectorizer with an empty dataset.")

        self._cache.clear()

        if self._type == "tfidf":
            self._tfidf = TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=self._max_features,
                min_df=2,
                max_df=0.95,
                lowercase=True,
                sublinear_tf=True,
            )
            matrix = self._tfidf.fit_transform(sanitized)
            dense = matrix.astype(np.float32).toarray()
        else:
            dense = self._encode_embeddings(sanitized)

        self._feature_dim = dense.shape[1]
        self._cache.update({text: vector for text, vector in zip(sanitized, dense)})
        self._is_fitted = True
        return dense

    def transform(self, texts: Sequence[str]) -> NDArray[np.float32]:
        """
        Vectorize texts using cached results whenever possible.
        """

        sanitized = self._sanitize_texts(texts)
        if not sanitized:
            return np.empty((0, self._feature_dim or 0), dtype=np.float32)

        missing = [text for text in sanitized if text not in self._cache]
        if missing:
            fresh_vectors = self._transform_backend(missing)
            self._cache.update({text: vector for text, vector in zip(missing, fresh_vectors)})

        stacked = np.vstack([self._cache[text] for text in sanitized])
        return stacked

    def get_state(self) -> dict[str, Any]:
        """
        Serialize the vectorizer state for persistence.
        """

        state: dict[str, Any] = {
            "type": self._type,
            "embedding_model": self._embedding_model_name,
            "max_features": self._max_features,
            "feature_dim": self._feature_dim,
        }
        if self._type == "tfidf" and self._tfidf is not None:
            state["tfidf"] = pickle.dumps(self._tfidf)
            state["is_fitted"] = self._is_fitted
        else:
            state["is_fitted"] = True
        return state

    def load_state(self, state: dict[str, Any]) -> None:
        """
        Restore the vectorizer from serialized state.
        """

        state_type = state.get("type")
        if state_type not in {"tfidf", "embedding"}:
            raise ValueError("Invalid vectorizer state provided.")

        self._type = state_type  # type: ignore[assignment]
        self._embedding_model_name = state.get("embedding_model")
        self._max_features = int(state.get("max_features", self._max_features))
        self._feature_dim = state.get("feature_dim")
        self._cache.clear()
        self._is_fitted = bool(state.get("is_fitted", False))

        if self._type == "tfidf":
            blob = state.get("tfidf")
            if blob is None:
                raise ValueError("TF-IDF state missing serialized payload.")
            self._tfidf = pickle.loads(blob)
        else:
            self._tfidf = None
            self._is_fitted = True

    def reset_cache(self) -> None:
        """Clear the cached vectors."""

        self._cache.clear()

    def _transform_backend(self, texts: Iterable[str]) -> NDArray[np.float32]:
        if self._type == "tfidf":
            if not self._tfidf or not self._is_fitted:
                raise ValueError("TF-IDF vectorizer has not been fitted yet.")
            matrix = self._tfidf.transform(list(texts))
            vectors = matrix.astype(np.float32).toarray()
            self._feature_dim = vectors.shape[1]
            return vectors

        vectors = self._encode_embeddings(list(texts))
        self._feature_dim = vectors.shape[1]
        return vectors

    def _encode_embeddings(self, texts: Sequence[str]) -> NDArray[np.float32]:
        model = self._ensure_embedding_model()
        embeddings = model.encode(list(texts), convert_to_numpy=True, show_progress_bar=False)
        embeddings = np.asarray(embeddings, dtype=np.float32)
        self._feature_dim = embeddings.shape[1]
        return embeddings

    def _ensure_embedding_model(self) -> SentenceTransformer:
        if self._embedding_model:
            return self._embedding_model

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "sentence-transformers is required for embedding vectorization."
            ) from exc

        model_name = self._embedding_model_name or "sentence-transformers/all-MiniLM-L6-v2"
        LOGGER.info("Loading embedding model: %s", model_name)
        self._embedding_model = SentenceTransformer(model_name)
        return self._embedding_model

    @staticmethod
    def _sanitize_texts(texts: Sequence[str]) -> list[str]:
        return [text.strip() if text else "" for text in texts]


__all__ = ["TextVectorizer", "VectorizerType"]
