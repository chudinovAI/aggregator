"""
Machine learning utilities for the news aggregator.
"""

from .classifier import (
    ClassifierConfig,
    PredictionResult,
    TextClassifier,
    TrainingDataset,
)
from .pipeline import (
    PredictionPipeline,
    TrainingPipeline,
    build_classifier_from_settings,
)
from .semantic_scorer import (
    SemanticScorer,
    get_semantic_scorer,
    score_post_semantically,
)
from .vectorizer import TextVectorizer

__all__ = [
    "ClassifierConfig",
    "PredictionPipeline",
    "PredictionResult",
    "SemanticScorer",
    "TextClassifier",
    "TextVectorizer",
    "TrainingDataset",
    "TrainingPipeline",
    "build_classifier_from_settings",
    "get_semantic_scorer",
    "score_post_semantically",
]
