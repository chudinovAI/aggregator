"""
User repository for data access operations on User entities.
"""

from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy import func, select

from ..models import User, UserPostRead
from .base import BaseRepository


class UserRepository(BaseRepository):
    """Data access helpers for User entities."""

    async def get_or_create(self, telegram_id: int) -> User:
        """
        Fetch a user by Telegram ID or create a new record.

        Uses unique index on telegram_id for O(log n) lookup.
        """
        user = await self._session.scalar(select(User).where(User.telegram_id == telegram_id))
        if user:
            return user

        user = User(telegram_id=telegram_id)
        self._session.add(user)
        await self._session.flush()
        return user

    async def get_by_telegram_id(self, telegram_id: int) -> User | None:
        """
        Fetch a user by Telegram ID without creating.

        Returns None if not found.
        """
        return await self._session.scalar(select(User).where(User.telegram_id == telegram_id))

    async def update_preferences(
        self,
        telegram_id: int,
        *,
        topics: Sequence[str] | None = None,
        sources: Sequence[str] | None = None,
        period: str | None = None,
    ) -> User:
        """
        Update stored topics/sources/period for a user, creating the user if necessary.

        Uses JSONB for efficient topic/source storage.
        """
        user = await self.get_or_create(telegram_id)
        if topics is not None:
            user.topics = list(dict.fromkeys(t.strip() for t in topics if t.strip()))
        if sources is not None:
            user.sources = list(dict.fromkeys(s.strip() for s in sources if s.strip()))
        if period is not None:
            user.period = period
        await self._session.flush()
        return user

    async def get_users_with_topics(self, limit: int = 1000) -> list[User]:
        """
        Get all users who have configured topics.

        Useful for batch digest sending.
        """
        stmt = select(User).where(func.jsonb_array_length(User.topics) > 0).limit(limit)
        result = await self._session.scalars(stmt)
        return list(result)

    async def get_user_read_post_ids(
        self,
        user_id: int,
        post_ids: list[int],
    ) -> set[int]:
        """
        Check which posts from a list have been read by the user.

        Efficient for checking read status of multiple posts.

        Returns:
            Set of post IDs that have been read
        """
        if not post_ids:
            return set()

        stmt = (
            select(UserPostRead.post_id)
            .where(UserPostRead.user_id == user_id)
            .where(UserPostRead.post_id.in_(post_ids))
        )
        result = await self._session.scalars(stmt)
        return set(result)


__all__ = ["UserRepository"]
