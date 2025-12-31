"""
Base repository class with shared utilities.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession


class BaseRepository:
    """
    Base class for all repositories.

    Provides common session handling and can be extended
    with shared query utilities.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    @property
    def session(self) -> AsyncSession:
        """Expose the underlying session for transaction management."""
        return self._session


__all__ = ["BaseRepository"]
