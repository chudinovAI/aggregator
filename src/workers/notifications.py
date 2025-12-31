"""
Notification tasks for background workers.

This module provides tasks for sending user notifications:
- send_daily_digest: Send personalized daily digest to users
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from ..config import Settings
from ..db.models import Post, User
from .types import TaskResult

LOGGER = logging.getLogger(__name__)


async def send_daily_digest(
    session_factory: async_sessionmaker[AsyncSession],
    settings: Settings,
    *,
    telegram_bot: Any | None = None,
    top_n: int = 10,
) -> TaskResult:
    """
    Send daily digest notifications to users.

    This task:
    1. Fetches users with configured preferences
    2. Gets top posts matching each user's preferences
    3. Sends personalized digest via Telegram

    Args:
        session_factory: SQLAlchemy async session factory
        settings: Application settings
        telegram_bot: Optional Telegram bot instance for sending messages
        top_n: Number of top posts to include in digest

    Returns:
        TaskResult with notification statistics
    """
    started_at = datetime.now(UTC)
    LOGGER.info("Starting send_daily_digest task")

    if not telegram_bot:
        LOGGER.warning("Telegram bot not configured, skipping digest")
        return TaskResult(
            task_name="send_daily_digest",
            success=False,
            message="Telegram bot not configured",
            details={},
            started_at=started_at,
            finished_at=datetime.now(UTC),
        )

    users_notified = 0
    errors: list[str] = []

    async with session_factory() as session:
        # Get all users with preferences
        users_query = select(User).where(func.array_length(User.topics, 1) > 0)
        users_result = await session.scalars(users_query)
        users = list(users_result)

        LOGGER.info("Found %d users with preferences", len(users))

        if not users:
            return TaskResult(
                task_name="send_daily_digest",
                success=True,
                message="No users with preferences",
                details={"users_count": 0},
                started_at=started_at,
                finished_at=datetime.now(UTC),
            )

        # Get top posts from last 24 hours
        yesterday = datetime.now(UTC) - timedelta(days=1)
        posts_query = (
            select(Post)
            .where(Post.published_at >= yesterday)
            .order_by(Post.classifier_score.desc())
            .limit(top_n * 2)  # Get more to filter per user
        )
        posts_result = await session.scalars(posts_query)
        all_posts = list(posts_result)

        if not all_posts:
            LOGGER.info("No new posts for digest")
            return TaskResult(
                task_name="send_daily_digest",
                success=True,
                message="No new posts",
                details={"posts_count": 0},
                started_at=started_at,
                finished_at=datetime.now(UTC),
            )

        for user in users:
            try:
                # Filter posts by user's topics (simple keyword matching)
                user_topics = set(t.lower() for t in (user.topics or []))
                matched_posts: list[Post] = []

                for post in all_posts:
                    post_text = f"{post.title} {post.content}".lower()
                    if any(topic in post_text for topic in user_topics):
                        matched_posts.append(post)
                        if len(matched_posts) >= top_n:
                            break

                if not matched_posts:
                    continue

                # Format digest message
                message_lines = ["*Daily News Digest*\n"]
                for i, post in enumerate(matched_posts, 1):
                    title = post.title[:60]
                    if len(post.title) > 60:
                        title += "..."
                    message_lines.append(f"{i}. [{title}]({post.source_url})")

                message = "\n".join(message_lines)

                # Send via Telegram
                try:
                    await telegram_bot.send_message(
                        user.telegram_id,
                        message,
                        parse_mode="Markdown",
                        disable_web_page_preview=True,
                    )
                    users_notified += 1
                except Exception as exc:
                    LOGGER.warning(
                        "Failed to send digest to user %d: %s",
                        user.telegram_id,
                        exc,
                    )
                    errors.append(f"User {user.telegram_id}: {exc}")

            except Exception as exc:
                LOGGER.exception("Error processing digest for user %d", user.id)
                errors.append(f"User {user.id}: {exc}")

    finished_at = datetime.now(UTC)
    success = len(errors) == 0

    LOGGER.info(
        "send_daily_digest completed: notified=%d, errors=%d",
        users_notified,
        len(errors),
    )

    return TaskResult(
        task_name="send_daily_digest",
        success=success,
        message=f"Notified {users_notified} users",
        details={
            "users_notified": users_notified,
            "total_users": len(users),
            "errors": errors,
        },
        started_at=started_at,
        finished_at=finished_at,
    )


__all__ = [
    "send_daily_digest",
]
