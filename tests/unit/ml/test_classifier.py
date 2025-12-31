"""
Unit tests for TextClassifier.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from src.ml.classifier import (
    ClassifierConfig,
    PredictionResult,
    TextClassifier,
    TrainingDataset,
)


class TestClassifierConfig:
    """Tests for ClassifierConfig validation."""

    def test_valid_config(self) -> None:
        config = ClassifierConfig(
            model_path="/tmp/model.pkl",
            confidence_threshold=0.7,
            vectorizer_type="tfidf",
            max_features=1000,
        )
        assert config.confidence_threshold == 0.7
        assert config.max_features == 1000

    def test_invalid_threshold_below_zero(self) -> None:
        with pytest.raises(ValueError, match="confidence_threshold"):
            ClassifierConfig(model_path="/tmp/model.pkl", confidence_threshold=-0.1)

    def test_invalid_threshold_above_one(self) -> None:
        with pytest.raises(ValueError, match="confidence_threshold"):
            ClassifierConfig(model_path="/tmp/model.pkl", confidence_threshold=1.5)

    def test_embedding_requires_model_name(self) -> None:
        with pytest.raises(ValueError, match="embedding_model"):
            ClassifierConfig(
                model_path="/tmp/model.pkl",
                vectorizer_type="embedding",
                embedding_model=None,
            )


class TestTrainingDataset:
    """Tests for TrainingDataset validation."""

    def test_valid_dataset(self) -> None:
        dataset = TrainingDataset(
            texts=["text1", "text2"],
            labels=["interesting", "boring"],
        )
        assert len(dataset.texts) == 2
        assert len(dataset.labels) == 2

    def test_empty_dataset_raises(self) -> None:
        with pytest.raises(ValueError, match="cannot be empty"):
            TrainingDataset(texts=[], labels=[])

    def test_mismatched_lengths_raises(self) -> None:
        with pytest.raises(ValueError, match="identical length"):
            TrainingDataset(texts=["a", "b", "c"], labels=["x", "y"])


class TestTextClassifier:
    """Tests for TextClassifier prediction and training."""

    @pytest.fixture
    def temp_model_path(self) -> Path:
        """Create a temporary model path."""
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            return Path(f.name)

    @pytest.fixture
    def trained_classifier(self, temp_model_path: Path) -> TextClassifier:
        """Create and train a classifier with sample data."""
        config = ClassifierConfig(
            model_path=temp_model_path,
            confidence_threshold=0.5,
            vectorizer_type="tfidf",
            max_features=100,
        )
        classifier = TextClassifier(config)

        # Train with sample data
        training_data = TrainingDataset(
            texts=[
                "Python machine learning tutorial",
                "Deep learning neural networks AI",
                "Cooking recipes for dinner",
                "Sports news football match",
                "Data science pandas numpy",
                "Travel vacation beach holiday",
            ]
            * 10,  # Repeat for min_df in TF-IDF
            labels=["interesting", "interesting", "boring", "boring", "interesting", "boring"] * 10,
        )
        classifier._train_sync(training_data)
        return classifier

    def test_is_loaded_false_initially(self, temp_model_path: Path) -> None:
        config = ClassifierConfig(model_path=temp_model_path)
        classifier = TextClassifier(config)
        assert classifier.is_loaded is False

    def test_is_loaded_true_after_training(self, trained_classifier: TextClassifier) -> None:
        assert trained_classifier.is_loaded is True

    def test_batch_predict_empty_list(self, trained_classifier: TextClassifier) -> None:
        result = trained_classifier.batch_predict([])
        assert result == []

    def test_batch_predict_returns_predictions(self, trained_classifier: TextClassifier) -> None:
        texts = ["machine learning python", "vacation travel beach"]
        results = trained_classifier.batch_predict(texts)

        assert len(results) == 2
        assert all(isinstance(r, list) for r in results)
        assert all(isinstance(p, PredictionResult) for r in results for p in r)

    def test_batch_predict_has_confidence(self, trained_classifier: TextClassifier) -> None:
        texts = ["AI neural network deep learning"]
        results = trained_classifier.batch_predict(texts)

        assert len(results) == 1
        assert len(results[0]) > 0
        prediction = results[0][0]
        assert 0.0 <= prediction.confidence <= 1.0

    def test_batch_predict_has_explanation(self, trained_classifier: TextClassifier) -> None:
        texts = ["data science analysis"]
        results = trained_classifier.batch_predict(texts)

        prediction = results[0][0]
        assert "interesting" in prediction.explanation or "boring" in prediction.explanation

    def test_normalize_label_positive_values(self) -> None:
        assert TextClassifier._normalize_label("1") == "interesting"
        assert TextClassifier._normalize_label("true") == "interesting"
        assert TextClassifier._normalize_label("positive") == "interesting"
        assert TextClassifier._normalize_label("interesting") == "interesting"
        assert TextClassifier._normalize_label("relevant") == "interesting"

    def test_normalize_label_negative_values(self) -> None:
        assert TextClassifier._normalize_label("0") == "boring"
        assert TextClassifier._normalize_label("false") == "boring"
        assert TextClassifier._normalize_label("negative") == "boring"
        assert TextClassifier._normalize_label("boring") == "boring"
        assert TextClassifier._normalize_label("irrelevant") == "boring"

    @pytest.mark.asyncio
    async def test_predict_empty_text_returns_empty(
        self, trained_classifier: TextClassifier
    ) -> None:
        result = await trained_classifier.predict("")
        assert result == []

    @pytest.mark.asyncio
    async def test_predict_with_text(self, trained_classifier: TextClassifier) -> None:
        result = await trained_classifier.predict("python machine learning")
        assert len(result) >= 1
        assert isinstance(result[0], PredictionResult)

    @pytest.mark.asyncio
    async def test_retrain_updates_model(self, temp_model_path: Path) -> None:
        config = ClassifierConfig(model_path=temp_model_path)
        classifier = TextClassifier(config)

        training_data = TrainingDataset(
            texts=["good content", "bad content"] * 30,
            labels=["interesting", "boring"] * 30,
        )

        await classifier.retrain(training_data)
        assert classifier.is_loaded is True
        assert temp_model_path.exists()

    def test_model_file_not_found_raises(self, temp_model_path: Path) -> None:
        config = ClassifierConfig(model_path=Path("/nonexistent/model.pkl"))
        classifier = TextClassifier(config)

        with pytest.raises(FileNotFoundError):
            classifier._ensure_model_loaded()
