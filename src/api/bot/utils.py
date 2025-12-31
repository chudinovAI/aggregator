"""
Helper functions for the Telegram bot.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

from aiogram.exceptions import TelegramBadRequest
from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from ...db.models import Post
from ...db.repository import UserRepository
from .constants import AVAILABLE_SOURCES

if TYPE_CHECKING:
    from aiogram.types import InlineKeyboardMarkup

LOGGER = logging.getLogger(__name__)

# Load topic keywords from config
_TOPICS_CONFIG: dict[str, Any] = {}


def _load_topics_config() -> dict[str, Any]:
    """Load topics configuration with keywords."""
    global _TOPICS_CONFIG
    if _TOPICS_CONFIG:
        return _TOPICS_CONFIG

    config_path = Path(__file__).parents[3] / "config" / "topics.json"
    if config_path.exists():
        try:
            with open(config_path) as f:
                data = json.load(f)
                _TOPICS_CONFIG = data.get("topics", {})
        except Exception as e:
            LOGGER.warning("Failed to load topics config: %s", e)
            _TOPICS_CONFIG = {}
    return _TOPICS_CONFIG


def _get_keywords_for_topic(topic: str) -> list[str]:
    """Get all keywords for a topic from config, or return topic itself."""
    config = _load_topics_config()

    # Normalize topic name (e.g., "machine learning" -> "machine_learning")
    topic_key = topic.lower().replace(" ", "_")

    if topic_key in config:
        keywords = config[topic_key].get("keywords", [])
        if keywords:
            return keywords

    # Fallback: return the topic itself as a keyword
    return [topic]


async def get_user_data(session: AsyncSession, telegram_id: int) -> dict[str, Any]:
    """Get user data with defaults."""
    repo = UserRepository(session)
    user = await repo.get_or_create(telegram_id)
    return {
        "user": user,
        "topics": user.topics or [],
        "sources": user.sources or AVAILABLE_SOURCES.copy(),
        "period": user.period or "7d",
    }


async def get_top_posts(
    session: AsyncSession,
    sources: list[str],
    topics: list[str],
    period_days: int,
    limit: int = 10,
    offset: int = 0,
) -> tuple[list[Post], int]:
    """Fetch top posts based on user preferences.

    If topics are specified but no posts match, falls back to showing
    all posts (without topic filter) to ensure users always see content.

    Sources are matched using LIKE since source_name contains prefixes
    like "reddit/r/python" or "hackernews".

    Topics are expanded using keywords from config/topics.json for
    better matching (e.g., "machine learning" also searches for "ML", "LLM", etc.)
    """
    cutoff = datetime.now(UTC) - timedelta(days=period_days)

    base_query = (
        select(Post)
        .where(Post.published_at >= cutoff)
        .order_by(Post.classifier_score.desc(), Post.published_at.desc())
    )

    # Filter by sources using LIKE (source_name is "reddit/r/xxx" or "hackernews")
    if sources:
        source_conditions = []
        for source in sources:
            # Match source at the beginning of source_name
            source_conditions.append(Post.source_name.ilike(f"{source}%"))
        base_query = base_query.where(or_(*source_conditions))

    # Try filtering by topics first
    if topics:
        topic_conditions = []
        for topic in topics:
            # Get all keywords for this topic from config
            keywords = _get_keywords_for_topic(topic)
            for keyword in keywords:
                # Case-insensitive search in title and content
                pattern = f"%{keyword}%"
                topic_conditions.append(Post.title.ilike(pattern))
                topic_conditions.append(Post.content.ilike(pattern))

        if topic_conditions:
            topic_query = base_query.where(or_(*topic_conditions))

            # Check if topic filter returns any results
            count_query = select(func.count()).select_from(topic_query.subquery())
            count_result = await session.execute(count_query)
            total = count_result.scalar_one()

            if total > 0:
                # Use topic-filtered query
                query = topic_query.offset(offset).limit(limit)
                result = await session.scalars(query)
                return list(result), total

    # Fallback: no topics or no matches - return all posts
    count_query = select(func.count()).select_from(base_query.subquery())
    count_result = await session.execute(count_query)
    total = count_result.scalar_one()

    query = base_query.offset(offset).limit(limit)
    result = await session.scalars(query)
    return list(result), total


def format_post_message(posts: list[Post], page: int = 0) -> str:
    """Format posts for display."""
    if not posts:
        return (
            "No posts found matching your criteria\\.\n\n"
            "Try adjusting your topics, sources, or search period\\."
        )

    lines = ["*Top Posts*\n"]
    for i, post in enumerate(posts, start=1):
        score_bar = (
            "🟢"
            if post.classifier_score >= 0.8
            else ("🟡" if post.classifier_score >= 0.5 else "🔴")
        )
        source = escape_markdown(post.source_name.replace("_", " ").title())
        age = escape_markdown(format_age(post.published_at))

        lines.append(
            f"{i}\\. {score_bar} *{escape_markdown(post.title[:60])}*\n   _{source}_ • {age}\n"
        )

    return "\n".join(lines)


def format_age(dt: datetime) -> str:
    """Format datetime as relative age string."""
    now = datetime.now(UTC)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    delta = now - dt

    if delta.days > 0:
        return f"{delta.days}d ago"
    hours = delta.seconds // 3600
    if hours > 0:
        return f"{hours}h ago"
    minutes = delta.seconds // 60
    return f"{minutes}m ago"


def escape_markdown(text: str) -> str:
    """Escape markdown special characters."""
    special_chars = [
        "_",
        "*",
        "[",
        "]",
        "(",
        ")",
        "~",
        "`",
        ">",
        "#",
        "+",
        "-",
        "=",
        "|",
        "{",
        "}",
        ".",
        "!",
    ]
    for char in special_chars:
        text = text.replace(char, f"\\{char}")
    return text


__all__ = [
    "LOGGER",
    "get_user_data",
    "get_top_posts",
    "format_post_message",
    "format_age",
    "escape_markdown",
    "safe_edit_text",
]


async def safe_edit_text(
    message: Any,
    text: str,
    reply_markup: InlineKeyboardMarkup | None = None,
    parse_mode: str | None = "MarkdownV2",
) -> None:
    """
    Edit message text, suppressing 'message is not modified' errors.

    This error occurs when trying to edit a message with identical content,
    which is common when users click the same button twice.

    Args:
        message: The message to edit (Message or MaybeInaccessibleMessage)
        text: The new text content
        reply_markup: Optional inline keyboard markup
        parse_mode: Parse mode for the text (default: MarkdownV2)
    """
    try:
        await message.edit_text(text, reply_markup=reply_markup, parse_mode=parse_mode)
    except TelegramBadRequest as e:
        if "message is not modified" not in str(e):
            raise
