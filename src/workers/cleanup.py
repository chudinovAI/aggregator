"""
Cleanup tasks for background workers.

This module provides tasks for cleaning up old data:
- cleanup_old_posts: Remove posts older than retention period
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

from sqlalchemy import delete, func, select
from sqlalchemy.engine import CursorResult
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from ..config import Settings
from ..db.models import Post
from .types import TaskResult

LOGGER = logging.getLogger(__name__)


async def cleanup_old_posts(
    session_factory: async_sessionmaker[AsyncSession],
    settings: Settings,
    *,
    retention_days: int | None = None,
) -> TaskResult:
    """
    Remove posts older than the retention period.

    This task:
    1. Calculates the cutoff date based on retention settings
    2. Deletes posts older than the cutoff
    3. Cleans up orphaned read records

    Args:
        session_factory: SQLAlchemy async session factory
        settings: Application settings
        retention_days: Override retention period (defaults to settings)

    Returns:
        TaskResult with cleanup statistics
    """
    started_at = datetime.now(UTC)
    LOGGER.info("Starting cleanup_old_posts task")

    days = retention_days or settings.parsing.retention_days
    cutoff = datetime.now(UTC) - timedelta(days=days)

    LOGGER.info("Cleaning posts older than %s (%d days)", cutoff, days)

    async with session_factory() as session:
        # Count posts to delete
        count_query = select(func.count()).select_from(Post).where(Post.published_at < cutoff)
        count_result = await session.execute(count_query)
        posts_to_delete = count_result.scalar_one()

        if posts_to_delete == 0:
            LOGGER.info("No posts to clean up")
            return TaskResult(
                task_name="cleanup_old_posts",
                success=True,
                message="No posts to clean up",
                details={
                    "cutoff_date": cutoff.isoformat(),
                    "retention_days": days,
                    "posts_deleted": 0,
                },
                started_at=started_at,
                finished_at=datetime.now(UTC),
            )

        # Delete old posts (cascade will handle UserPostRead)
        delete_stmt = delete(Post).where(Post.published_at < cutoff)
        cursor_result: CursorResult[tuple[()]] = await session.execute(delete_stmt)  # type: ignore[assignment]
        deleted_count = cursor_result.rowcount or 0

        await session.commit()

    finished_at = datetime.now(UTC)
    LOGGER.info("Cleaned up %d old posts", deleted_count)

    return TaskResult(
        task_name="cleanup_old_posts",
        success=True,
        message=f"Deleted {deleted_count} posts",
        details={
            "cutoff_date": cutoff.isoformat(),
            "retention_days": days,
            "posts_deleted": deleted_count,
        },
        started_at=started_at,
        finished_at=finished_at,
    )


__all__ = [
    "cleanup_old_posts",
]
