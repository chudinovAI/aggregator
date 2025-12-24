import asyncpraw

from .config import AggregatorConfig


def build_reddit_client(config: AggregatorConfig) -> asyncpraw.Reddit:
    return asyncpraw.Reddit(
        client_id=config.reddit_client_id,
        client_secret=config.reddit_client_secret,
        user_agent="AdvancedNewsAggregator/2.0",
    )


def build_youtube_client(config: AggregatorConfig) -> str:
    return config.youtube_api_key
