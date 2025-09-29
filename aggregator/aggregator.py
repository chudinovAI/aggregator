from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, List, Optional

from better_profanity import profanity

from .clients import build_reddit_client, build_youtube_client
from .collectors import HackerNewsCollector, RedditCollector, TedCollector
from .config import AggregatorConfig
from .filtering import PostFilter
from .html_report import HtmlReporter
from .types import Post

LOGGER = logging.getLogger(__name__)

try:
    from .ml import NewsClassifier
except ImportError:
    NewsClassifier = None


@dataclass
class AggregatorResult:
    posts: List[Post]
    report_path: str


class AdvancedNewsAggregator:
    def __init__(
        self,
        config: Optional[AggregatorConfig] = None,
        use_ml: bool = True,
        reddit_client=None,
        youtube_client=None,
    ) -> None:
        self._config = config or AggregatorConfig()
        self._reddit = reddit_client or build_reddit_client()
        self._youtube = youtube_client or build_youtube_client()
        self._post_filter = PostFilter(self._config)
        self._reporter = HtmlReporter(self._config)

        profanity.load_censor_words()

        self._use_ml = bool(use_ml and NewsClassifier is not None)
        self._classifier = None
        if self._use_ml and NewsClassifier is not None:
            model_path = self._config.model_path
            self._classifier = NewsClassifier(model_path=model_path)
            LOGGER.info("ML classifier enabled (model path: %s).", model_path)
        elif use_ml:
            LOGGER.info(
                "ML classifier requested but unavailable. Using keyword filtering."
            )
        else:
            LOGGER.info("Keyword-based filtering enabled.")

    def collect_posts(self) -> List[Post]:
        collectors = (
            RedditCollector(self._reddit, self._config),
            HackerNewsCollector(self._config),
            TedCollector(self._youtube, self._config),
        )

        posts: List[Post] = []
        for collector in collectors:
            posts.extend(collector.collect())
        LOGGER.info("Total posts collected: %d", len(posts))
        return posts

    def filter_posts(self, posts: Iterable[Post]) -> List[Post]:
        posts_list = list(posts)
        LOGGER.info("Beginning filtering step for %d posts.", len(posts_list))

        if not posts_list:
            return []

        if self._classifier:
            candidates = self._post_filter.filter_with_classifier(
                posts_list, self._classifier, self._config.ml_threshold
            )
        else:
            LOGGER.info("Applying keyword-based filtering.")
            candidates = self._post_filter.basic_filter(posts_list)
            LOGGER.info(
                "Keyword filter reduced posts from %d to %d.",
                len(posts_list),
                len(candidates),
            )

        scored = self._post_filter.apply_combined_scores(candidates)
        LOGGER.info("Scored %d posts after filtering.", len(scored))
        return scored

    def create_report(
        self, posts: Iterable[Post], filename: Optional[str] = None
    ) -> str:
        return self._reporter.render(list(posts), filename=filename)

    def run(self) -> Optional[AggregatorResult]:
        posts = self.collect_posts()
        if not posts:
            LOGGER.warning(
                "No posts were collected. Check API credentials and network access."
            )
            return None

        filtered = self.filter_posts(posts)
        if not filtered:
            LOGGER.warning("No posts remained after filtering.")
            return None

        ranked_posts = sorted(
            filtered,
            key=lambda post: float(post.get("combined_score", 0.0)),
            reverse=True,
        )
        top_posts = ranked_posts[: self._config.top_posts_limit]
        LOGGER.info("Top posts selected for the report: %d", len(top_posts))

        for index, post in enumerate(top_posts[:10], start=1):
            ml_info = ""
            if post.get("ml_score") is not None:
                ml_info = f", ml_score={float(post['ml_score']):.3f}"
            LOGGER.info(
                "%02d. source=%s score=%.3f%s title=%s",
                index,
                post.get("source", "unknown"),
                float(post.get("combined_score", 0.0)),
                ml_info,
                (post.get("title", "") or "")[:80],
            )
            LOGGER.info("    %s", post.get("url", ""))

        report_path = self.create_report(top_posts)
        LOGGER.info("Aggregation finished successfully. Report path: %s", report_path)
        return AggregatorResult(posts=top_posts, report_path=report_path)
