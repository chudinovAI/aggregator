"""
Shared pytest fixtures for database, Redis, settings, and classifier mocks.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from src.config import Settings
from src.ml.classifier import (
    ClassifierConfig,
    PredictionResult,
)
from tests.test_models import MockBase


class MockRedisClient:
    """Minimal in-memory Redis replacement for async tests."""

    def __init__(self) -> None:
        self._store: dict[str, str] = {}

    async def get(self, key: str) -> str | None:
        return self._store.get(key)

    async def set(self, key: str, value: str, ex: int | None = None) -> None:  # noqa: ARG002
        self._store[key] = value

    async def delete(self, key: str) -> None:
        self._store.pop(key, None)

    async def flushdb(self) -> None:
        self._store.clear()


@pytest.fixture(scope="session")
async def db_engine() -> AsyncIterator[AsyncEngine]:
    """Provide an async in-memory SQLite engine for tests.

    Uses MockBase which has SQLite-compatible models (no TSVECTOR, JSONB).
    """
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    async with engine.begin() as connection:
        await connection.run_sync(MockBase.metadata.create_all)
    try:
        yield engine
    finally:
        await engine.dispose()


@pytest.fixture
async def db_session(db_engine: AsyncEngine) -> AsyncIterator[AsyncSession]:
    """Create a transactional AsyncSession per test."""

    session_maker = async_sessionmaker(db_engine, expire_on_commit=False)
    async with session_maker() as session:
        yield session
        await session.rollback()
        for table in reversed(MockBase.metadata.sorted_tables):
            await session.execute(table.delete())
        await session.commit()


@pytest.fixture
async def mock_redis() -> AsyncIterator[MockRedisClient]:
    """Yield a mock Redis client with in-memory storage."""

    client = MockRedisClient()
    try:
        yield client
    finally:
        await client.flushdb()


@pytest.fixture
def settings_override(monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Override global settings with test-friendly configuration."""

    from src import config as config_module

    base_settings = config_module.Settings()
    database = base_settings.database.model_copy(
        update={"url": "sqlite+aiosqlite:///:memory:", "echo": False}
    )
    redis_conf = base_settings.redis.model_copy(
        update={"url": "redis://localhost:6379/0", "max_connections": 5}
    )
    overrides = base_settings.model_copy(update={"database": database, "redis": redis_conf})
    monkeypatch.setattr(config_module, "get_settings", lambda: overrides)
    return overrides


@pytest.fixture
def classifier_mock() -> Any:
    """Mock TextClassifier with predictable outputs."""

    config = ClassifierConfig(
        model_path=":memory:",
        confidence_threshold=0.5,
        vectorizer_type="tfidf",
        embedding_model=None,
    )
    explanation = {"interesting": 0.9, "boring": 0.1}

    class _ClassifierDouble:
        def __init__(self) -> None:
            self.config = config
            self._is_loaded = True
            self.predict = AsyncMock(
                return_value=[
                    PredictionResult(label="interesting", confidence=0.9, explanation=explanation)
                ]
            )
            self.batch_predict = MagicMock(
                return_value=[
                    [PredictionResult(label="interesting", confidence=0.9, explanation=explanation)]
                ]
            )
            self.retrain = AsyncMock()

        @property
        def is_loaded(self) -> bool:
            return self._is_loaded

    return _ClassifierDouble()
