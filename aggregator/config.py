from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Literal, Tuple

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AggregatorConfig(BaseSettings):
    """Runtime configuration loaded from environment or .env file."""

    model_config = SettingsConfigDict(
        env_file=None,
        env_prefix="AGG_",
        extra="ignore",
    )

    reddit_client_id: str = Field(..., env="REDDIT_CLIENT_ID")
    reddit_client_secret: str = Field(..., env="REDDIT_CLIENT_SECRET")
    youtube_api_key: str = Field(..., env="YOUTUBE_API_KEY")

    collection_days: int = 7
    reddit_top_limit: int = 20
    hackernews_top_limit: int = 100
    ted_min_duration_seconds: int = 12 * 60
    top_posts_limit: int = 50
    ml_threshold: float = 0.6
    dedup_similarity_threshold: float = Field(
        default=0.88, ge=0.0, le=1.0, description="Threshold for fuzzy deduplication."
    )
    request_timeout_seconds: int = 10
    api_retries: int = 3
    api_retry_base_delay: float = 0.5
    api_concurrency_limit: int = 5
    api_request_delay_seconds: float = 0.2
    api_rate_limit_per_host: int = 60
    api_rate_limit_window_seconds: int = 60
    api_rate_limit_profiles: Dict[str, Tuple[int, int]] = Field(default_factory=dict)
    export_json: bool = True
    export_markdown: bool = True
    export_rss: bool = True
    use_transformer_classifier: bool = True
    sentiment_enabled: bool = True
    sentiment_model_name: str = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
    storage_enabled: bool = True
    storage_backend: Literal["sqlite", "postgres"] = "sqlite"
    storage_path: Path = Field(default_factory=lambda: Path("data/aggregator.db"))
    storage_pg_dsn: str | None = None
    storage_summary_days: int = 30
    redis_enabled: bool = False
    redis_url: str = "redis://localhost:6379/0"
    redis_cache_ttl_seconds: int = 900
    cache_enabled: bool = True
    cache_path: Path = Field(default_factory=lambda: Path("data/http_cache.json"))
    cache_ttl_seconds: int = 4 * 60 * 60
    enable_language_filter: bool = True
    allowed_languages: Tuple[str, ...] = ("en", "ru")
    allowed_url_schemes: Tuple[str, ...] = ("http", "https")
    stop_words: Tuple[str, ...] = (
        "breaking news",
        "sponsored",
        "giveaway",
        "clickbait",
    )
    stop_topics: Tuple[str, ...] = (
        "politics",
        "war",
        "religion",
        "celebrity gossip",
    )
    email_notifications_enabled: bool = False
    email_smtp_host: str | None = None
    email_smtp_port: int = 587
    email_smtp_username: str | None = None
    email_smtp_password: str | None = None
    email_sender: str | None = None
    email_recipients: Tuple[str, ...] = ()
    personalization_enabled: bool = False
    user_profiles_path: Path = Field(
        default_factory=lambda: Path("data/user_profiles.json")
    )
    feedback_enabled: bool = True
    feedback_queue_path: Path = Field(
        default_factory=lambda: Path("data/feedback_queue.jsonl")
    )
    feedback_events_path: Path = Field(
        default_factory=lambda: Path("data/feedback_events.jsonl")
    )
    feedback_dataset_path: Path = Field(
        default_factory=lambda: Path("data/feedback_dataset.jsonl")
    )
    feedback_negative_sample_size: int = 20
    feedback_min_training_batch: int = 50
    feedback_autotrain_enabled: bool = False

    subreddits: Tuple[str, ...] = (
        "compsci",
        "computerscience",
        "csMajors",
        "datascience",
        "dataisbeautiful",
        "cuda",
        "learnmachinelearning",
        "Python",
    )

    included_topics: Tuple[str, ...] = (
        "books",
        "textbook",
        "useful links",
        "c++",
        "cuda",
        "rust",
        "python",
        "machine learning",
        "data science",
        "artificial intelligence",
        "computer science",
        "ai",
        "computers",
        "software",
        "physics",
        "space",
        "science",
        "astronomy",
        "universe",
        "psychology",
        "brain",
        "education",
        "math",
        "investing",
        "business",
        "tutorial",
        "guide",
        "open source",
    )

    excluded_keywords: Tuple[str, ...] = (
        "lgbtq",
        "lgbt",
        "politics",
        "political",
        "war",
        "feminist",
        "feminism",
        "minority",
        "minorities",
        "transgender",
        "gay",
        "lesbian",
    )

    ted_topics: Tuple[str, ...] = (
        "artificial intelligence",
        "computers",
        "software",
        "physics",
        "space",
        "science",
        "astronomy",
        "universe",
        "personality",
        "memory",
        "psychology",
        "mental health",
        "brain",
        "sleep",
        "personal growth",
        "motivation",
        "mindfulness",
        "depression",
        "education",
        "math",
        "investing",
        "business",
    )

    ted_search_queries: Tuple[str, ...] = (
        "TED talks",
        "TED conference",
        "TEDx talks",
    )

    high_value_keywords: Tuple[str, ...] = (
        "tutorial",
        "guide",
        "research",
        "paper",
        "study",
        "breakthrough",
        "innovation",
        "open source",
        "free",
    )

    html_timestamp_format: str = "%Y-%m-%d %H:%M"
    html_footer_format: str = "%Y-%m-%d %H:%M:%S"
    reports_dir: Path = Field(default_factory=lambda: Path("docs"))
    sorttable_js_path: Path = Field(default_factory=lambda: Path("static/sorttable.js"))
    model_path: Path = Field(default_factory=lambda: Path("models/news_classifier.pkl"))
    transformer_model_dir: Path = Field(
        default_factory=lambda: Path("models/news_transformer")
    )
    transformer_dataset_path: Path = Field(
        default_factory=lambda: Path("data/news_classification_dataset.csv")
    )
    transformer_model_name: str = "distilbert-base-multilingual-cased"
    transformer_max_length: int = 256
    transformer_learning_rate: float = 3e-5
    transformer_batch_size: int = 8
    transformer_num_epochs: int = 2
    transformer_weight_decay: float = 0.01
    transformer_feedback_weight: float = 2.0
    feature_recency_halflife_hours: float = 48.0
    feature_source_weights: Dict[str, float] = Field(
        default_factory=lambda: {"reddit": 1.0, "hackernews": 1.1, "ted_youtube": 0.9}
    )
    feature_engagement_normalizer: float = 50.0

    @field_validator(
        "collection_days",
        "reddit_top_limit",
        "hackernews_top_limit",
        "redis_cache_ttl_seconds",
        "api_rate_limit_per_host",
        "api_rate_limit_window_seconds",
        "feedback_negative_sample_size",
        "feedback_min_training_batch",
        "transformer_max_length",
        "transformer_batch_size",
        "transformer_num_epochs",
    )
    @classmethod
    def _validate_positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("Value must be positive.")
        return value

    @field_validator("storage_summary_days")
    @classmethod
    def _validate_summary_days(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("storage_summary_days must be positive.")
        return value

    @field_validator("api_rate_limit_profiles", mode="before")
    @classmethod
    def _normalize_rate_profiles(cls, value):
        if value in (None, ""):
            return {}
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError as exc:  # pragma: no cover - invalid config
                raise ValueError("api_rate_limit_profiles must be valid JSON.") from exc
        if not isinstance(value, dict):
            raise ValueError("api_rate_limit_profiles must be a mapping.")
        normalized: Dict[str, Tuple[int, int]] = {}
        for key, profile in value.items():
            if isinstance(profile, (list, tuple)) and len(profile) == 2:
                requests, window = profile
            elif isinstance(profile, dict):
                requests = profile.get("requests")
                window = profile.get("window")
            else:
                raise ValueError(
                    f"Invalid rate profile for {key}; expected [requests, window]."
                )
            requests_int = int(requests)
            window_int = int(window)
            if requests_int <= 0 or window_int <= 0:
                raise ValueError(
                    f"Rate profile values must be positive for key '{key}'."
                )
            normalized[key] = (requests_int, window_int)
        return normalized

    @field_validator("allowed_url_schemes")
    @classmethod
    def _validate_url_schemes(cls, value: Tuple[str, ...]) -> Tuple[str, ...]:
        if not value:
            raise ValueError("allowed_url_schemes must contain at least one scheme.")
        normalized = tuple(scheme.lower().strip() for scheme in value if scheme.strip())
        if not normalized:
            raise ValueError("allowed_url_schemes must contain valid entries.")
        return normalized

    @field_validator(
        "feature_recency_halflife_hours",
        "feature_engagement_normalizer",
        "transformer_learning_rate",
        "transformer_weight_decay",
        "transformer_feedback_weight",
    )
    @classmethod
    def _validate_positive_float(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("Feature parameters must be positive.")
        return value

    def lowercased_included_topics(self) -> Tuple[str, ...]:
        return tuple(topic.lower() for topic in self.included_topics)

    def lowercased_excluded_keywords(self) -> Tuple[str, ...]:
        return tuple(keyword.lower() for keyword in self.excluded_keywords)

    def lowercased_ted_topics(self) -> Tuple[str, ...]:
        return tuple(topic.lower() for topic in self.ted_topics)

    @model_validator(mode="after")
    def _validate_storage_backend(self) -> "AggregatorConfig":
        if self.storage_backend == "postgres" and not self.storage_pg_dsn:
            raise ValueError(
                "storage_pg_dsn must be set when storage_backend='postgres'."
            )
        return self

    @field_validator("email_recipients", mode="before")
    @classmethod
    def _parse_recipients(cls, value):
        if isinstance(value, str):
            parts = [item.strip() for item in value.split(",") if item.strip()]
            return tuple(parts)
        return value
