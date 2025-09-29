from __future__ import annotations

import logging
from typing import Iterable, List, Sequence

from better_profanity import profanity

from .config import AggregatorConfig
from .types import Post

LOGGER = logging.getLogger(__name__)


class PostFilter:
    def __init__(self, config: AggregatorConfig) -> None:
        self._config = config
        self._included_topics = config.lowercased_included_topics()
        self._excluded_keywords = config.lowercased_excluded_keywords()
        self._high_value_keywords = tuple(
            keyword.lower() for keyword in config.high_value_keywords
        )

    @staticmethod
    def _combine_text(post: Post) -> str:
        title = post.get("title", "")
        body = post.get("selftext") or post.get("description") or ""
        return f"{title} {body}".lower()

    def _is_excluded(self, text: str) -> bool:
        return any(keyword in text for keyword in self._excluded_keywords)

    def _is_included(self, text: str) -> bool:
        return any(topic in text for topic in self._included_topics)

    def _is_profane(self, text: str) -> bool:
        return profanity.contains_profanity(text)

    def basic_filter(self, posts: Sequence[Post]) -> List[Post]:
        filtered: List[Post] = []
        for post in posts:
            text = self._combine_text(post)
            if self._is_excluded(text):
                continue
            if self._is_profane(text):
                continue
            if self._is_included(text):
                post_copy = post.copy()
                post_copy["ml_score"] = None
                filtered.append(post_copy)
        return filtered

    def filter_with_classifier(
        self, posts: Sequence[Post], classifier, threshold: float
    ) -> List[Post]:
        ml_filtered = classifier.filter_posts(list(posts), threshold=threshold)
        LOGGER.info(
            "ML filter reduced posts from %d to %d.", len(posts), len(ml_filtered)
        )
        candidates: List[Post] = []
        for post in ml_filtered:
            text = self._combine_text(post)
            if not self._is_profane(text):
                candidates.append(post)
        LOGGER.info("Profanity filter reduced ML output to %d posts.", len(candidates))
        return candidates

    def calculate_base_score(self, post: Post) -> float:
        text = self._combine_text(post)
        score = 0.0

        score_value = post.get("score")
        if isinstance(score_value, (int, float)):
            score += min(float(score_value) / 100.0, 10.0)

        comments_value = post.get("num_comments")
        if isinstance(comments_value, (int, float)):
            score += min(float(comments_value) / 50.0, 5.0)

        topic_matches = sum(1 for topic in self._included_topics if topic in text)
        score += topic_matches * 3.0

        keyword_bonus = (
            sum(1 for keyword in self._high_value_keywords if keyword in text) * 2.0
        )
        score += keyword_bonus

        return score

    def apply_combined_scores(self, posts: Iterable[Post]) -> List[Post]:
        scored_posts: List[Post] = []
        for post in posts:
            post_copy = post.copy()
            base_score = self.calculate_base_score(post_copy)
            ml_score = post_copy.get("ml_score")
            if ml_score is None:
                post_copy["combined_score"] = base_score / 10.0
            else:
                post_copy["combined_score"] = 0.7 * float(ml_score) + 0.3 * (
                    base_score / 10.0
                )
            scored_posts.append(post_copy)
        return scored_posts
