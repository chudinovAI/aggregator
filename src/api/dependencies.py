"""
FastAPI dependency injection providers.
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, Header, HTTPException, Request, status
from redis.asyncio import Redis
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from ..config import Settings, get_settings
from ..db.cache import PostsCache
from ..db.repository import PostRepository, UserRepository
from ..ml.classifier import ClassifierConfig, TextClassifier

LOGGER = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Type Aliases for Dependency Injection
# -----------------------------------------------------------------------------

SettingsDep = Annotated[Settings, Depends(get_settings)]


# -----------------------------------------------------------------------------
# Database Session Management
# -----------------------------------------------------------------------------

_engine = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def _get_engine(settings: Settings) -> AsyncEngine:
    """Create or return cached async engine."""
    global _engine
    if _engine is None:
        _engine = create_async_engine(
            str(settings.database.url),
            pool_size=settings.database.pool_size,
            echo=settings.database.echo,
        )
    return _engine


def _get_session_factory(settings: Settings) -> async_sessionmaker[AsyncSession]:
    """Create or return cached session factory."""
    global _session_factory
    if _session_factory is None:
        engine = _get_engine(settings)
        _session_factory = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )
    return _session_factory


async def get_db(
    settings: SettingsDep,
) -> AsyncGenerator[AsyncSession, None]:
    """
    Provide a transactional database session.

    The session is committed on success and rolled back on exception.
    """
    factory = _get_session_factory(settings)
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


DbSessionDep = Annotated[AsyncSession, Depends(get_db)]


# -----------------------------------------------------------------------------
# Repository Dependencies
# -----------------------------------------------------------------------------


async def get_post_repository(session: DbSessionDep) -> PostRepository:
    """Provide PostRepository with current session."""
    return PostRepository(session)


async def get_user_repository(session: DbSessionDep) -> UserRepository:
    """Provide UserRepository with current session."""
    return UserRepository(session)


PostRepoDep = Annotated[PostRepository, Depends(get_post_repository)]
UserRepoDep = Annotated[UserRepository, Depends(get_user_repository)]


# -----------------------------------------------------------------------------
# Redis / Cache
# -----------------------------------------------------------------------------

_redis_client: Redis | None = None


async def get_redis(settings: SettingsDep) -> Redis:
    """
    Provide a shared Redis client.

    The client is lazily initialized and reused across requests.
    """
    global _redis_client
    if _redis_client is None:
        _redis_client = Redis.from_url(
            str(settings.redis.url),
            max_connections=settings.redis.max_connections,
            decode_responses=True,
        )
    return _redis_client


async def get_cache(
    redis: Annotated[Redis, Depends(get_redis)],
    settings: SettingsDep,
) -> PostsCache:
    """Provide PostsCache wrapper around Redis client."""
    return PostsCache(redis, namespace="posts")


RedisDep = Annotated[Redis, Depends(get_redis)]
CacheDep = Annotated[PostsCache, Depends(get_cache)]


# -----------------------------------------------------------------------------
# ML Classifier
# -----------------------------------------------------------------------------

_classifier: TextClassifier | None = None


def get_classifier(settings: SettingsDep) -> TextClassifier:
    """
    Provide a singleton TextClassifier instance.

    The classifier is lazily loaded from the configured model path.
    """
    global _classifier
    if _classifier is None:
        config = ClassifierConfig(
            model_path=settings.ml.model_path,
            confidence_threshold=settings.ml.threshold,
        )
        _classifier = TextClassifier(config)
    return _classifier


ClassifierDep = Annotated[TextClassifier, Depends(get_classifier)]


# -----------------------------------------------------------------------------
# Authentication / User Context (Placeholder)
# -----------------------------------------------------------------------------


async def get_current_user_id(
    x_telegram_id: Annotated[int | None, Header(alias="X-Telegram-ID")] = None,
) -> int | None:
    """
    Extract the current user's Telegram ID from request headers.

    Returns None for anonymous requests.
    """
    return x_telegram_id


async def require_user_id(
    user_id: Annotated[int | None, Depends(get_current_user_id)],
) -> int:
    """
    Require a valid user ID or raise 401.

    Use this dependency for endpoints that require authentication.
    """
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="X-Telegram-ID header is required.",
        )
    return user_id


CurrentUserIdDep = Annotated[int | None, Depends(get_current_user_id)]
RequiredUserIdDep = Annotated[int, Depends(require_user_id)]


# -----------------------------------------------------------------------------
# Request Context
# -----------------------------------------------------------------------------


async def get_request_id(request: Request) -> str:
    """Extract or generate a request ID for tracing."""
    return request.headers.get("X-Request-ID", f"req_{id(request)}")


RequestIdDep = Annotated[str, Depends(get_request_id)]


# -----------------------------------------------------------------------------
# Lifecycle Helpers
# -----------------------------------------------------------------------------


@asynccontextmanager
async def lifespan_dependencies(settings: Settings) -> AsyncIterator[None]:
    """
    Context manager for application lifespan.

    Initializes and cleans up shared resources.
    """
    global _engine, _session_factory, _redis_client, _classifier

    LOGGER.info("Initializing application dependencies...")

    # Initialize engine
    _get_engine(settings)
    _get_session_factory(settings)

    # Initialize Redis
    _redis_client = Redis.from_url(
        str(settings.redis.url),
        max_connections=settings.redis.max_connections,
        decode_responses=True,
    )

    # Pre-warm classifier (optional)
    if settings.ml.model_path.exists():
        try:
            config = ClassifierConfig(
                model_path=settings.ml.model_path,
                confidence_threshold=settings.ml.threshold,
            )
            _classifier = TextClassifier(config)
            LOGGER.info("ML classifier pre-loaded successfully.")
        except Exception as exc:
            LOGGER.warning("Failed to pre-load classifier: %s", exc)

    try:
        yield
    finally:
        LOGGER.info("Shutting down application dependencies...")

        if _redis_client:
            await _redis_client.aclose()  # type: ignore[attr-defined]
            _redis_client = None

        if _engine:
            await _engine.dispose()
            _engine = None
            _session_factory = None

        _classifier = None


# -----------------------------------------------------------------------------
# Health Check Helpers
# -----------------------------------------------------------------------------


async def check_database_health(settings: Settings) -> tuple[bool, float]:
    """Check database connectivity and return (healthy, latency_ms)."""
    factory = _get_session_factory(settings)
    start = time.perf_counter()
    try:
        async with factory() as session:
            await session.execute(text("SELECT 1"))
        latency = (time.perf_counter() - start) * 1000
        return True, latency
    except Exception as exc:
        LOGGER.error("Database health check failed: %s", exc)
        latency = (time.perf_counter() - start) * 1000
        return False, latency


async def check_redis_health(redis: Redis) -> tuple[bool, float]:
    """Check Redis connectivity and return (healthy, latency_ms)."""
    start = time.perf_counter()
    try:
        await redis.ping()
        latency = (time.perf_counter() - start) * 1000
        return True, latency
    except Exception as exc:
        LOGGER.error("Redis health check failed: %s", exc)
        latency = (time.perf_counter() - start) * 1000
        return False, latency


__all__ = [
    "CacheDep",
    "ClassifierDep",
    "CurrentUserIdDep",
    "DbSessionDep",
    "PostRepoDep",
    "RedisDep",
    "RequestIdDep",
    "RequiredUserIdDep",
    "SettingsDep",
    "UserRepoDep",
    "check_database_health",
    "check_redis_health",
    "get_cache",
    "get_classifier",
    "get_current_user_id",
    "get_db",
    "get_redis",
    "get_request_id",
    "get_settings",
    "lifespan_dependencies",
    "require_user_id",
]
