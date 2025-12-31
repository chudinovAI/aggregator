"""
Async-friendly text classifier with batch inference support.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, NamedTuple

import joblib
import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression

from .vectorizer import TextVectorizer, VectorizerType

LabelLiteral = Literal["interesting", "boring"]
LOGGER = logging.getLogger(__name__)


class PredictionResult(NamedTuple):
    """
    Represents a single prediction with probability and explanation details.
    """

    label: LabelLiteral
    confidence: float
    explanation: dict[str, float]


@dataclass(slots=True)
class ClassifierConfig:
    """
    Runtime configuration required to load or train the classifier.
    """

    model_path: Path
    confidence_threshold: float = 0.5
    vectorizer_type: VectorizerType = "tfidf"
    embedding_model: str | None = None
    max_features: int = 5000

    def __post_init__(self) -> None:
        if isinstance(self.model_path, str):
            object.__setattr__(self, "model_path", Path(self.model_path))
        if not (0.0 <= self.confidence_threshold <= 1.0):
            raise ValueError("confidence_threshold must be within [0.0, 1.0].")
        if self.vectorizer_type == "embedding" and not self.embedding_model:
            raise ValueError("embedding_model must be provided when using embeddings.")


@dataclass(slots=True)
class TrainingDataset:
    """
    Container for supervised training data.
    """

    texts: Sequence[str]
    labels: Sequence[Any]

    def __post_init__(self) -> None:
        if len(self.texts) != len(self.labels):
            raise ValueError("texts and labels must have identical length.")
        if not self.texts:
            raise ValueError("Training dataset cannot be empty.")


class TextClassifier:
    """
    Binary classifier that predicts whether a post is interesting.
    """

    def __init__(
        self,
        config: ClassifierConfig,
        *,
        vectorizer: TextVectorizer | None = None,
        estimator: LogisticRegression | None = None,
    ) -> None:
        self.config = config
        self._vectorizer = vectorizer or TextVectorizer(
            vectorizer_type=config.vectorizer_type,
            embedding_model=config.embedding_model,
            max_features=config.max_features,
        )
        self._model = estimator

    @property
    def is_loaded(self) -> bool:
        """Return True if the classifier has a trained model in memory."""

        return self._model is not None

    async def predict(self, text: str, top_k: int = 1) -> list[PredictionResult]:
        """
        Predict labels for a single text asynchronously.
        """

        if not text.strip():
            return []

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._predict_sync, text, top_k)

    def batch_predict(self, texts: list[str]) -> list[list[PredictionResult]]:
        """
        Predict labels for multiple texts using vectorized operations.

        Time Complexity: O(n * vocab_size) for TF-IDF transform + O(n * features) for prediction
        Space Complexity: O(n * vocab_size) for feature matrix

        Optimizations applied:
        - Vectorized probability processing using numpy argsort
        - Pre-computed class labels to avoid repeated lookups
        - Batch explanation dict creation
        """

        if not texts:
            return []

        self._ensure_model_loaded()

        try:
            features = self._vectorizer.transform(texts)
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Failed to vectorize texts.", exc_info=exc)
            raise RuntimeError("Vectorization failed.") from exc

        if features.size == 0:
            return [[] for _ in texts]

        if self._model is None:
            raise RuntimeError("Model must be loaded before prediction")
        probabilities = self._model.predict_proba(features)
        return self._batch_rows_to_predictions(probabilities)

    async def retrain(self, training_data: TrainingDataset) -> None:
        """
        Retrain the classifier asynchronously using the provided dataset.
        """

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._train_sync, training_data)

    def _predict_sync(self, text: str, top_k: int) -> list[PredictionResult]:
        predictions = self.batch_predict([text])
        if not predictions:
            return []
        ranked = predictions[0]
        threshold = self.config.confidence_threshold
        filtered = [result for result in ranked if result.confidence >= threshold]
        if not filtered:
            filtered = ranked[:1]
        return filtered[: max(1, top_k)]

    def _train_sync(self, training_data: TrainingDataset) -> None:
        LOGGER.info("Starting classifier training with %d samples.", len(training_data.texts))
        texts = [text.strip() if text else "" for text in training_data.texts]
        labels = [self._normalize_label(label) for label in training_data.labels]

        try:
            features = self._vectorizer.fit_transform(texts)
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Failed to vectorize training data.", exc_info=exc)
            raise RuntimeError("Vectorization failed.") from exc

        self._model = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
        )
        self._model.fit(features, labels)
        self._save_model()
        LOGGER.info("Classifier retraining finished successfully.")

    def _ensure_model_loaded(self) -> None:
        if self._model is not None:
            return

        if not self.config.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found at {self.config.model_path}. Retrain the classifier first."
            )

        LOGGER.info("Loading classifier from %s", self.config.model_path)
        payload = joblib.load(self.config.model_path)
        model = payload.get("model")
        vectorizer_state = payload.get("vectorizer_state")

        if not isinstance(model, LogisticRegression):
            raise ValueError("Persisted model is invalid or corrupted.")
        if not isinstance(vectorizer_state, dict):
            raise ValueError("Persisted vectorizer state is invalid.")

        self._model = model
        self._vectorizer.load_state(vectorizer_state)

    def _save_model(self) -> None:
        if not self._model:
            raise RuntimeError("Cannot save classifier before training.")

        payload = {
            "model": self._model,
            "config": asdict(self.config),
            "vectorizer_state": self._vectorizer.get_state(),
        }
        self.config.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(payload, self.config.model_path)

    def _batch_rows_to_predictions(
        self, probabilities: NDArray[np.float64]
    ) -> list[list[PredictionResult]]:
        """
        Vectorized conversion of probability matrix to predictions.

        Optimizations:
        - Pre-compute normalized class labels once
        - Use numpy argsort for batch sorting instead of Python sorted()
        - Minimize per-row Python object creation
        """
        if self._model is None:
            raise RuntimeError("Model must be loaded before prediction")
        classes = self._model.classes_
        n_classes = len(classes)

        # Normalize labels once (typically just 2 classes)
        normalized_labels: list[LabelLiteral] = [self._normalize_label(label) for label in classes]

        # Clip all probabilities at once
        clipped = np.clip(probabilities, 0.0, 1.0)

        # Get sort order for all rows at once (descending by negating)
        # Shape: (n_samples, n_classes)
        sort_indices = np.argsort(-clipped, axis=1)

        results: list[list[PredictionResult]] = []
        for i, row in enumerate(clipped):
            # Build explanation dict for this row
            explanation: dict[str, float] = {
                normalized_labels[j]: float(row[j]) for j in range(n_classes)
            }

            # Build predictions in sorted order
            predictions = [
                PredictionResult(
                    label=normalized_labels[idx],
                    confidence=float(row[idx]),
                    explanation=explanation,
                )
                for idx in sort_indices[i]
            ]
            results.append(predictions)

        return results

    def _row_to_predictions(self, row: NDArray[np.float64]) -> list[PredictionResult]:
        """
        Convert a single probability row to predictions (kept for backward compatibility).
        """
        if self._model is None:
            raise RuntimeError("Model must be loaded before prediction")
        classes = self._model.classes_
        probs = np.clip(row, 0.0, 1.0)
        explanation: dict[str, float] = {
            self._normalize_label(label): float(prob) for label, prob in zip(classes, probs)
        }

        ordered = sorted(explanation.items(), key=lambda item: item[1], reverse=True)
        return [
            PredictionResult(
                label=self._normalize_label(label),
                confidence=score,
                explanation=explanation,
            )
            for label, score in ordered
        ]

    @staticmethod
    def _normalize_label(raw_label: Any) -> LabelLiteral:
        value = str(raw_label).strip().lower()
        if value in {"1", "true", "relevant", "interesting", "positive"}:
            return "interesting"
        return "boring"


__all__ = [
    "ClassifierConfig",
    "PredictionResult",
    "TextClassifier",
    "TrainingDataset",
]
