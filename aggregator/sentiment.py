from __future__ import annotations

import logging
from typing import List, Sequence

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

from .config import AggregatorConfig
from .types import Post

LOGGER = logging.getLogger(__name__)


class SentimentAnalyzer:
    """Adds sentiment scores to posts using a transformer pipeline."""

    def __init__(self, config: AggregatorConfig) -> None:
        model_name = config.sentiment_model_name
        device = 0 if torch.cuda.is_available() else -1
        self._pipe = pipeline(
            "sentiment-analysis",
            model=AutoModelForSequenceClassification.from_pretrained(model_name),
            tokenizer=AutoTokenizer.from_pretrained(model_name),
            device=device,
        )

    def annotate(self, posts: Sequence[Post]) -> List[Post]:
        texts: List[str] = []
        for post in posts:
            title = post.get("title", "") or ""
            content = post.get("selftext", "") or post.get("description", "") or ""
            texts.append(f"{title} {content}"[:512])

        results = self._pipe(texts, truncation=True, max_length=256)
        enriched: List[Post] = []
        for post, result in zip(posts, results):
            post_copy = post.copy()
            label = (result.get("label") or "").lower()
            score = float(result.get("score", 0.0))
            # Normalize to signed sentiment: positive -> +score, negative -> -score, neutral -> 0
            if "neg" in label:
                sentiment_score = -score
                sentiment_label = "negative"
            elif "pos" in label:
                sentiment_score = score
                sentiment_label = "positive"
            else:
                sentiment_score = 0.0
                sentiment_label = "neutral"
            post_copy["sentiment_label"] = sentiment_label
            post_copy["sentiment_score"] = sentiment_score
            enriched.append(post_copy)
        return enriched
