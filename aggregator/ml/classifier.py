from __future__ import annotations

import logging
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "news_classifier.pkl"
DEFAULT_DATASET_PATH = PROJECT_ROOT / "data" / "news_classification_dataset.csv"

DatasetPath = Path | str
Posts = Sequence[Dict[str, Any]]


class NewsClassifier:
    def __init__(self, model_path: Path | str = DEFAULT_MODEL_PATH) -> None:
        self.model_path = Path(model_path)
        self.pipeline: Pipeline | None = None
        self.is_trained = False

        if self.model_path.exists():
            self.load_model()

    def preprocess_text(self, text: str) -> str:
        if not text:
            return ""
        normalized = unicodedata.normalize("NFKC", text)
        cleaned_chars: List[str] = []
        for char in normalized:
            category = unicodedata.category(char)
            if category.startswith("L") or category.startswith("N"):
                cleaned_chars.append(char.lower())
            elif char.isspace():
                cleaned_chars.append(" ")
        cleaned = "".join(cleaned_chars)
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned.strip()

    def train_model(self, dataset_path: DatasetPath = DEFAULT_DATASET_PATH) -> Pipeline:
        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
        LOGGER.info("Training classifier using dataset: %s", dataset_path)
        data = pd.read_csv(dataset_path, encoding="utf-8")

        data["text_clean"] = data["text"].apply(self.preprocess_text)
        features = data["text_clean"]
        labels = data["label"]

        x_train, x_test, y_train, y_test = train_test_split(
            features,
            labels,
            test_size=0.2,
            random_state=42,
            stratify=labels,
        )

        self.pipeline = Pipeline(
            steps=[
                (
                    "tfidf",
                    TfidfVectorizer(
                        ngram_range=(1, 2),
                        max_features=1000,
                        min_df=2,
                        max_df=0.95,
                        lowercase=True,
                        stop_words=None,
                        sublinear_tf=True,
                    ),
                ),
                (
                    "classifier",
                    LogisticRegression(
                        random_state=42,
                        class_weight="balanced",
                        max_iter=1000,
                        C=1.0,
                    ),
                ),
            ]
        )

        self.pipeline.fit(x_train, y_train)
        predictions = self.pipeline.predict(x_test)
        probabilities = self.pipeline.predict_proba(x_test)[:, 1]

        LOGGER.info(
            "Classification report:\n%s", classification_report(y_test, predictions)
        )
        LOGGER.info("Confusion matrix:\n%s", confusion_matrix(y_test, predictions))
        LOGGER.info(
            "Average probability for positive class: %.3f",
            probabilities[y_test == 1].mean(),
        )
        LOGGER.info(
            "Average probability for negative class: %.3f",
            probabilities[y_test == 0].mean(),
        )

        self.show_feature_importance()
        self.save_model()
        self.is_trained = True
        LOGGER.info("Model trained and saved to %s", self.model_path)
        return self.pipeline

    def show_feature_importance(self, top_n: int = 20) -> None:
        if not self.pipeline:
            LOGGER.warning("Cannot show feature importance before training.")
            return

        vectorizer: TfidfVectorizer = self.pipeline.named_steps["tfidf"]
        classifier: LogisticRegression = self.pipeline.named_steps["classifier"]

        feature_names = vectorizer.get_feature_names_out()
        coefficients = classifier.coef_[0]

        top_positive = np.argsort(coefficients)[-top_n:][::-1]
        LOGGER.info("Top %d positive features:", top_n)
        for index in top_positive:
            LOGGER.info("  %s (%.3f)", feature_names[index], coefficients[index])

        top_negative = np.argsort(coefficients)[:top_n]
        LOGGER.info("Top %d negative features:", top_n)
        for index in top_negative:
            LOGGER.info("  %s (%.3f)", feature_names[index], coefficients[index])

    def predict_relevance(
        self, text: str, threshold: float = 0.5
    ) -> Tuple[bool, float]:
        if not self.pipeline:
            raise ValueError(
                "Classifier is not trained. Call train_model() or load_model()."
            )

        clean_text = self.preprocess_text(text)
        probabilities = self.pipeline.predict_proba([clean_text])[0, 1]
        return probabilities > threshold, probabilities

    def filter_posts(
        self, posts: Posts, threshold: float = 0.6
    ) -> List[Dict[str, Any]]:
        if not self.pipeline:
            raise ValueError("Classifier is not trained.")

        relevant: List[Dict[str, Any]] = []
        for post in posts:
            title = post.get("title", "")
            content = post.get("selftext", "") or post.get("description", "")
            text = f"{title} {content}"
            is_relevant, probability = self.predict_relevance(text, threshold)
            if is_relevant:
                post_copy = post.copy()
                post_copy["ml_score"] = probability
                relevant.append(post_copy)
        relevant.sort(key=lambda entry: entry["ml_score"], reverse=True)
        return relevant

    def save_model(self) -> None:
        if self.pipeline:
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self.pipeline, self.model_path)

    def load_model(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        self.pipeline = joblib.load(self.model_path)
        self.is_trained = True
        LOGGER.info("Model loaded from %s", self.model_path)

    def evaluate_on_new_data(self, posts: Posts, manual_labels: Sequence[int]) -> None:
        if len(posts) != len(manual_labels):
            raise ValueError("Posts and labels must have equal length.")
        if not self.pipeline:
            raise ValueError("Classifier is not trained.")

        texts = []
        for post in posts:
            title = post.get("title", "")
            content = post.get("selftext", "") or post.get("description", "")
            texts.append(self.preprocess_text(f"{title} {content}"))

        predictions = self.pipeline.predict(texts)
        probabilities = self.pipeline.predict_proba(texts)[:, 1]

        LOGGER.info(
            "Evaluation report:\n%s",
            classification_report(manual_labels, predictions),
        )
        LOGGER.info("Misclassified samples:")
        for index, (post, true_label, pred_label, prob) in enumerate(
            zip(posts, manual_labels, predictions, probabilities),
            start=1,
        ):
            if true_label == pred_label:
                continue
            LOGGER.info(
                "%02d. prob=%.3f expected=%s predicted=%s title=%s",
                index,
                prob,
                "relevant" if true_label else "irrelevant",
                "relevant" if pred_label else "irrelevant",
                (post.get("title", "") or "")[:60],
            )


def test_classifier() -> NewsClassifier:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    classifier = NewsClassifier()
    if not classifier.is_trained:
        classifier.train_model()

    samples = [
        "New Python 3.12 features for machine learning developers",
        "CUDA programming tutorial for beginners",
        "Trump wins election latest political news",
        "LGBTQ rights in modern society discussion",
        "Quantum computing breakthrough in IBM research",
        "War in Ukraine latest military updates",
        "Best books for learning computer science algorithms",
        "Feminist movement in the technology industry",
    ]

    for sample in samples:
        is_relevant, probability = classifier.predict_relevance(sample)
        label = "relevant" if is_relevant else "irrelevant"
        LOGGER.info("%s (%.3f) -> %s", sample, probability, label)

    return classifier


if __name__ == "__main__":
    test_classifier()
