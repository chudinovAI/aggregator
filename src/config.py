"""
Centralized configuration management powered by pydantic-settings.
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field, PositiveInt, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    """Supported runtime environments."""

    LOCAL = "local"
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class AppSettings(BaseModel):
    """Application metadata and runtime toggles."""

    name: str = Field(default="news-aggregator", description="Human-readable service name.")
    version: str = Field(default="0.1.0", description="Deployed application version.")
    environment: Environment = Field(
        default=Environment.LOCAL, description="Deployment environment identifier."
    )
    debug: bool = Field(default=False, description="Enable debug features and verbose logs.")


class DatabaseSettings(BaseModel):
    """Database connection settings."""

    url: str = Field(
        default="postgresql+psycopg://postgres:postgres@localhost:5432/news_aggregator",
        description="SQLAlchemy-compatible database URL.",
    )
    pool_size: PositiveInt = Field(default=10, description="Database connection pool size.")
    echo: bool = Field(default=False, description="Enable SQL echo for debugging.")


class RedisSettings(BaseModel):
    """Redis connection and caching behavior."""

    url: str = Field(default="redis://localhost:6379/0", description="Redis connection URL.")
    max_connections: PositiveInt = Field(
        default=20, description="Maximum concurrent Redis connections."
    )
    cache_ttl_seconds: PositiveInt = Field(
        default=900, description="Default TTL for cached payloads."
    )
    lock_ttl_seconds: PositiveInt = Field(default=60, description="TTL for distributed lock keys.")


class TelegramBotSettings(BaseModel):
    """Telegram bot integration parameters."""

    token: str = Field(default="", description="Bot API token issued by BotFather.")
    admin_id: int | str | None = Field(
        default=None, description="Telegram admin user ID (number) or username (string)."
    )
    web_app_url: str = Field(
        default="", description="URL for Telegram Web App (must be HTTPS)."
    )

    @field_validator("admin_id", mode="before")
    @classmethod
    def parse_admin_id(cls, v: object) -> object:
        """Parse admin_id: empty string to None, numeric string to int, keep username as str."""
        if v == "" or v is None:
            return None
        if isinstance(v, str) and v.isdigit():
            return int(v)
        return v


class LoggingSettings(BaseModel):
    """Logging configuration shared across the project."""

    level: str = Field(default="INFO", description="Root logging level.")
    format: str = Field(
        default="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        description="Standard logging format string.",
    )
    directory: Path = Field(default=Path("logs"), description="Directory for log files.")
    file_name: str = Field(default="app.log", description="Primary log file name.")
    max_bytes: PositiveInt = Field(
        default=5 * 1024 * 1024, description="Maximum file size before rotating."
    )
    backup_count: PositiveInt = Field(default=5, description="Number of rotated log files to keep.")


class MLSettings(BaseModel):
    """Machine learning classifier settings."""

    model_path: Path = Field(
        default=Path("models/news_classifier/model.pt"),
        description="Path to the serialized classifier artifact.",
    )
    threshold: float = Field(
        default=0.65,
        ge=0.0,
        le=1.0,
        description="Minimum probability required to mark an article as relevant.",
    )


class ParsingSettings(BaseModel):
    """Content parsing cadence and limits."""

    interval_seconds: PositiveInt = Field(
        default=300, description="Interval between parser runs in seconds."
    )
    max_articles_per_source: PositiveInt = Field(
        default=100, description="Maximum articles processed per source per run."
    )
    retention_days: PositiveInt = Field(
        default=30, description="Number of days to retain parsed articles."
    )
    request_delay_seconds: float = Field(
        default=0.2, description="Delay between API requests to avoid rate limiting."
    )
    request_timeout_seconds: float = Field(
        default=15.0, description="HTTP request timeout in seconds."
    )


class RedditSourceSettings(BaseModel):
    """Reddit source configuration."""

    enabled: bool = Field(default=True, description="Enable Reddit source.")
    subreddits: list[str] = Field(
        default=[
            # General programming
            "programming",
            "coding",
            "softwaredevelopment",
            "ExperiencedDevs",
            "AskProgramming",
            # Python
            "Python",
            "django",
            "flask",
            "FastAPI",
            "learnpython",
            # JavaScript/Web
            "javascript",
            "typescript",
            "reactjs",
            "vuejs",
            "angular",
            "sveltejs",
            "node",
            "nextjs",
            "webdev",
            "Frontend",
            # Machine Learning / AI
            "MachineLearning",
            "deeplearning",
            "artificial",
            "LocalLLaMA",
            "ChatGPT",
            "OpenAI",
            "LanguageTechnology",
            "MLOps",
            # DevOps / Cloud
            "devops",
            "aws",
            "googlecloud",
            "azure",
            "docker",
            "kubernetes",
            "terraform",
            "CICD",
            # Data
            "datascience",
            "dataengineering",
            "BigData",
            "datasets",
            # Databases
            "PostgreSQL",
            "mysql",
            "mongodb",
            "redis",
            "SQL",
            # Systems / Languages
            "rust",
            "golang",
            "cpp",
            "java",
            "csharp",
            "swift",
            "kotlin",
            # Linux / Sysadmin
            "linux",
            "linuxadmin",
            "sysadmin",
            "selfhosted",
            "homelab",
            # Security
            "netsec",
            "cybersecurity",
            "hacking",
            "ReverseEngineering",
            "privacy",
            # Career
            "cscareerquestions",
            "cscareerquestionsEU",
            "remotework",
            "learnprogramming",
            # Startups / Business
            "startups",
            "SaaS",
            "Entrepreneur",
            "indiehackers",
            # Open Source
            "opensource",
            "github",
            "commandline",
        ],
        description="Subreddits to collect from.",
    )
    user_agent: str = Field(default="news-aggregator/0.1", description="User agent for Reddit API.")


class HackerNewsSourceSettings(BaseModel):
    """HackerNews source configuration."""

    enabled: bool = Field(default=True, description="Enable HackerNews source.")
    max_concurrent_requests: PositiveInt = Field(
        default=10, description="Max concurrent API requests."
    )


class SourcesSettings(BaseModel):
    """Aggregated source configurations."""

    reddit: RedditSourceSettings = RedditSourceSettings()
    hackernews: HackerNewsSourceSettings = HackerNewsSourceSettings()


class Settings(BaseSettings):
    """Top-level application settings loaded from the environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    app: AppSettings = AppSettings()
    database: DatabaseSettings = DatabaseSettings()
    redis: RedisSettings = RedisSettings()
    telegram: TelegramBotSettings = TelegramBotSettings()
    logging: LoggingSettings = LoggingSettings()
    ml: MLSettings = MLSettings()
    parsing: ParsingSettings = ParsingSettings()
    sources: SourcesSettings = SourcesSettings()


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance loaded from the current environment."""

    return Settings()


__all__ = [
    "AppSettings",
    "DatabaseSettings",
    "Environment",
    "HackerNewsSourceSettings",
    "LoggingSettings",
    "MLSettings",
    "ParsingSettings",
    "RedditSourceSettings",
    "RedisSettings",
    "Settings",
    "SourcesSettings",
    "TelegramBotSettings",
    "get_settings",
]
