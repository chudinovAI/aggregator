from __future__ import annotations

import math
from datetime import datetime
from typing import Dict

from .config import AggregatorConfig
from .types import Post


class FeatureEngineer:
    """Derives structured features for ranking and model training."""

    def __init__(self, config: AggregatorConfig) -> None:
        self._recency_halflife_hours = config.feature_recency_halflife_hours
        self._engagement_norm = config.feature_engagement_normalizer
        self._source_weights = {
            source.lower(): float(weight)
            for source, weight in config.feature_source_weights.items()
        }

    def compute(self, post: Post) -> Dict[str, float]:
        created = post.get("created_utc")
        now = datetime.utcnow()
        if isinstance(created, datetime):
            age_hours = max((now - created).total_seconds() / 3600.0, 0.0)
        else:
            age_hours = float("inf")

        recency_score = (
            math.exp(-age_hours / self._recency_halflife_hours)
            if age_hours != float("inf")
            else 0.0
        )

        score = float(post.get("score", 0.0) or 0.0)
        comments = float(post.get("num_comments", 0.0) or 0.0)
        engagement_rate = (score + (2.0 * comments)) / max(age_hours, 1.0)
        engagement_norm = min(engagement_rate / self._engagement_norm, 1.0)

        source = (post.get("source") or "unknown").lower()
        source_weight = self._source_weights.get(source, 1.0)

        return {
            "age_hours": age_hours if age_hours != float("inf") else 1e6,
            "recency_score": recency_score,
            "engagement_rate": engagement_rate,
            "engagement_norm": engagement_norm,
            "source_weight": source_weight,
        }
