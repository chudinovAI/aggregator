from __future__ import annotations

import asyncio
import logging
from time import perf_counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from better_profanity import profanity

from .cache import FileCache
from .clients import build_reddit_client, build_youtube_client
from .collectors import HackerNewsCollector, RedditCollector, TedCollector
from .config import AggregatorConfig
from .dedup import deduplicate_posts
from .feedback import FeedbackLoop, FeedbackMetrics
from .filtering import PostFilter
from .html_report import HtmlReporter
from .notifications import EmailNotifier
from .personalization import PersonalizationManager
from .ratelimiter import AsyncRateLimiter
from .sentiment import SentimentAnalyzer
from .storage import Storage, StorageSummary
from .validators import InputValidator
from .types import Post

LOGGER = logging.getLogger(__name__)

try:
    from .ml import NewsClassifier, TransformerConfig, TransformerNewsClassifier
except ImportError:
    NewsClassifier = None
    TransformerNewsClassifier = None
    TransformerConfig = None


@dataclass
class AggregatorResult:
    posts: List[Post]
    report_path: str
    json_path: Optional[str] = None
    markdown_path: Optional[str] = None
    rss_path: Optional[str] = None


class AdvancedNewsAggregator:
    def __init__(
        self,
        config: Optional[AggregatorConfig] = None,
        use_ml: bool = True,
        reddit_client=None,
        youtube_client=None,
    ) -> None:
        self._config = config or AggregatorConfig()
        self._reddit = reddit_client or build_reddit_client(self._config)
        self._youtube = youtube_client or build_youtube_client(self._config)
        self._post_filter = PostFilter(self._config)
        self._reporter = HtmlReporter(self._config)
        self._cache = (
            FileCache(self._config.cache_path, self._config.cache_ttl_seconds)
            if self._config.cache_enabled
            else None
        )
        self._validator = InputValidator(self._config.allowed_url_schemes)
        self._semaphore = asyncio.Semaphore(self._config.api_concurrency_limit)
        overrides = self._build_rate_limit_overrides()
        self._rate_limiter = AsyncRateLimiter(
            self._config.api_rate_limit_per_host,
            self._config.api_rate_limit_window_seconds,
            overrides=overrides,
        )
        self._storage = Storage(self._config) if self._config.storage_enabled else None
        self._sentiment = (
            SentimentAnalyzer(self._config) if self._config.sentiment_enabled else None
        )
        self._notifier = EmailNotifier(self._config)
        self._personalization = PersonalizationManager(self._config)
        self._feedback = (
            FeedbackLoop(self._config) if self._config.feedback_enabled else None
        )
        self._transformer_config: TransformerConfig | None = None
        self._last_autotrain_size = 0

        profanity.load_censor_words()

        self._use_ml = bool(use_ml and NewsClassifier is not None)
        self._classifier = None
        if use_ml:
            if self._config.use_transformer_classifier and TransformerNewsClassifier:
                try:
                    self._transformer_config = self._build_transformer_config()
                    if not self._transformer_config:
                        raise RuntimeError("Transformer configuration unavailable.")
                    self._classifier = TransformerNewsClassifier(
                        config=self._transformer_config,
                        model_path=self._config.transformer_model_dir,
                    )
                    self._use_ml = True
                    LOGGER.info(
                        "Transformer classifier enabled (model dir: %s).",
                        self._config.transformer_model_dir,
                    )
                except Exception as exc:  # noqa: BLE001
                    LOGGER.warning(
                        "Transformer classifier init failed, falling back to classic. %s",
                        exc,
                    )
            if self._classifier is None and NewsClassifier is not None:
                model_path = self._config.model_path
                self._classifier = NewsClassifier(model_path=model_path)
                self._use_ml = True
                LOGGER.info("ML classifier enabled (model path: %s).", model_path)
            if self._classifier is None:
                LOGGER.info(
                    "ML classifier requested but unavailable. Using keyword filtering."
                )
        else:
            LOGGER.info("Keyword-based filtering enabled.")

    def collect_posts(self) -> List[Post]:
        collectors = (
            RedditCollector(
                self._reddit,
                self._config,
                rate_limiter=self._rate_limiter,
                validator=self._validator,
            ),
            HackerNewsCollector(
                self._config,
                cache=self._cache,
                semaphore=self._semaphore,
                rate_limiter=self._rate_limiter,
                validator=self._validator,
            ),
            TedCollector(
                self._youtube,
                self._config,
                cache=self._cache,
                semaphore=self._semaphore,
                rate_limiter=self._rate_limiter,
                validator=self._validator,
            ),
        )
        return self._collect_concurrently(collectors)

    def _collect_concurrently(self, collectors: Iterable) -> List[Post]:
        collector_list = list(collectors)

        async def _run():
            tasks = []
            for collector in collector_list:
                if hasattr(collector, "collect_async"):
                    tasks.append(collector.collect_async())
                else:
                    tasks.append(asyncio.to_thread(collector.collect))
            results = await asyncio.gather(*tasks, return_exceptions=True)
            aggregated: List[Post] = []
            for collector, result in zip(collector_list, results):
                if isinstance(result, Exception):
                    LOGGER.exception(
                        "Collector %s failed",
                        collector.__class__.__name__,
                        exc_info=result,
                    )
                    continue
                aggregated.extend(result or [])
            return aggregated

        try:
            posts = asyncio.run(_run())
        except RuntimeError:
            LOGGER.warning("Event loop already running; collecting sequentially.")
            posts = []
            for collector in collector_list:
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
        self,
        posts: Iterable[Post],
        filename: Optional[str] = None,
        analytics: Optional[StorageSummary] = None,
        feedback: Optional[FeedbackMetrics] = None,
    ) -> str:
        return self._reporter.render(
            list(posts),
            filename=filename,
            analytics=analytics,
            feedback=feedback,
        )

    def run(self) -> Optional[AggregatorResult]:
        timings: dict[str, float] = {}
        storage_summary: Optional[StorageSummary] = None
        feedback_metrics: Optional[FeedbackMetrics] = None

        started = perf_counter()
        posts = self.collect_posts()
        collected_count = len(posts)
        timings["collection_seconds"] = perf_counter() - started
        if self._storage and posts:
            self._storage.save_posts(posts, stage="collected")

        if not posts:
            LOGGER.warning(
                "No posts were collected. Check API credentials and network access."
            )
            return None

        dedup_started = perf_counter()
        deduped = deduplicate_posts(
            posts, similarity_threshold=self._config.dedup_similarity_threshold
        )
        deduped_count = len(deduped)
        timings["dedup_seconds"] = perf_counter() - dedup_started
        if self._storage and deduped:
            self._storage.save_posts(deduped, stage="deduplicated")

        filter_started = perf_counter()
        filtered = self.filter_posts(deduped)
        timings["filter_seconds"] = perf_counter() - filter_started
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

        if self._sentiment:
            top_posts = self._sentiment.annotate(top_posts)

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

        if self._feedback:
            self._feedback.enqueue(top_posts, predicted=True)
            negative_sample_start = self._config.top_posts_limit
            negative_sample_end = (
                negative_sample_start + self._config.feedback_negative_sample_size
            )
            negative_sample = ranked_posts[negative_sample_start:negative_sample_end]
            if negative_sample:
                self._feedback.enqueue(negative_sample, predicted=False)
            feedback_metrics = self._feedback.integrate_feedback()
            self._maybe_autotrain_transformer()

        if self._storage and top_posts:
            self._storage.save_posts(top_posts, stage="report")
            storage_summary = self._storage.summarize()
            if storage_summary:
                LOGGER.info(
                    "Storage summary generated for last %d days (total posts=%d).",
                    storage_summary.window_days,
                    storage_summary.total_posts,
                )

        report_started = perf_counter()
        report_path = self.create_report(
            top_posts,
            analytics=storage_summary,
            feedback=feedback_metrics,
        )
        json_path: Optional[str] = None
        markdown_path: Optional[str] = None
        rss_path: Optional[str] = None
        if self._config.export_json:
            json_path = self._reporter.render_json(
                top_posts,
                analytics=storage_summary,
                feedback=feedback_metrics,
            )
        if self._config.export_markdown:
            markdown_path = self._reporter.render_markdown(
                top_posts,
                analytics=storage_summary,
                feedback=feedback_metrics,
            )
        if self._config.export_rss:
            rss_path = self._reporter.render_rss(top_posts)
        if self._notifier.is_configured():
            self._notifier.send_digest(
                top_posts,
                report_path=report_path,
                extra_paths=(json_path, markdown_path, rss_path),
            )
            personalized = self._personalization.personalized_digests(top_posts)
            for digest in personalized:
                self._notifier.send_personalized(
                    recipient=digest.profile.email,
                    posts=digest.posts,
                    report_path=report_path,
                    extra_paths=(json_path, markdown_path),
                    subject=f"{digest.profile.name}'s Personalized Digest",
                )
        timings["report_seconds"] = perf_counter() - report_started
        timings["total_seconds"] = perf_counter() - started
        metrics = {
            "collected_posts": collected_count,
            "deduplicated_posts": deduped_count,
            "filtered_posts": len(filtered),
            "reported_posts": len(top_posts),
        }
        if feedback_metrics:
            metrics["feedback_precision"] = feedback_metrics.precision
            metrics["feedback_recall"] = feedback_metrics.recall
            metrics["feedback_processed"] = feedback_metrics.processed
        LOGGER.info(
            "Aggregation metrics",
            extra={"metrics": metrics, "timings": timings},
        )
        if feedback_metrics:
            LOGGER.info(
                "Feedback quality: processed=%d precision=%s recall=%s",
                feedback_metrics.processed,
                f"{feedback_metrics.precision:.3f}"
                if feedback_metrics.precision is not None
                else "n/a",
                f"{feedback_metrics.recall:.3f}"
                if feedback_metrics.recall is not None
                else "n/a",
            )
        LOGGER.info("Aggregation finished successfully. Report path: %s", report_path)
        return AggregatorResult(
            posts=top_posts,
            report_path=report_path,
            json_path=json_path,
            markdown_path=markdown_path,
            rss_path=rss_path,
        )

    def _build_rate_limit_overrides(self) -> Dict[str, Tuple[int, int]]:
        overrides: Dict[str, Tuple[int, int]] = {}
        for key, profile in self._config.api_rate_limit_profiles.items():
            requests, window = profile
            overrides[key] = (max(1, int(requests)), max(1, int(window)))
        return overrides

    def _build_transformer_config(self) -> TransformerConfig | None:
        if TransformerConfig is None:
            return None
        return TransformerConfig(
            model_name=self._config.transformer_model_name,
            model_dir=self._config.transformer_model_dir,
            base_dataset_path=self._config.transformer_dataset_path,
            feedback_dataset_path=self._config.feedback_dataset_path,
            feedback_weight=self._config.transformer_feedback_weight,
            max_length=self._config.transformer_max_length,
            learning_rate=self._config.transformer_learning_rate,
            batch_size=self._config.transformer_batch_size,
            num_epochs=self._config.transformer_num_epochs,
            weight_decay=self._config.transformer_weight_decay,
        )

    def _maybe_autotrain_transformer(self) -> None:
        if (
            not self._config.feedback_autotrain_enabled
            or not self._config.use_transformer_classifier
            or not self._feedback
            or TransformerNewsClassifier is None
        ):
            return
        dataset_size = self._feedback.dataset_size()
        if dataset_size < self._config.feedback_min_training_batch:
            LOGGER.debug(
                "Feedback dataset too small for fine-tuning (size=%d, min=%d).",
                dataset_size,
                self._config.feedback_min_training_batch,
            )
            return
        if dataset_size <= self._last_autotrain_size:
            LOGGER.debug(
                "Feedback dataset unchanged since last fine-tune (size=%d).",
                dataset_size,
            )
            return
        transformer_config = (
            self._transformer_config or self._build_transformer_config()
        )
        if not transformer_config:
            LOGGER.warning(
                "Transformer config unavailable; skipping feedback fine-tune."
            )
            return
        LOGGER.info(
            "Fine-tuning transformer classifier on %d feedback samples.",
            dataset_size,
        )
        try:
            trainer = TransformerNewsClassifier(config=transformer_config)
            trainer.train(
                dataset_path=self._config.transformer_dataset_path,
                feedback_path=self._config.feedback_dataset_path,
                feedback_weight=self._config.transformer_feedback_weight,
            )
            self._classifier = trainer
            self._last_autotrain_size = dataset_size
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Transformer fine-tuning failed: %s", exc)
