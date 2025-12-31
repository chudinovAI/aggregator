"""
API route definitions for the news aggregator.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, status
from sqlalchemy import func, select

from ..db.models import Post, UserPostRead
from ..db.repositories.post import PostRepository
from .dependencies import (
    CurrentUserIdDep,
    DbSessionDep,
    RedisDep,
    RequiredUserIdDep,
    SettingsDep,
    UserRepoDep,
    check_database_health,
    check_redis_health,
)
from .schemas import (
    ComponentHealth,
    ErrorResponse,
    HealthResponse,
    HealthStatus,
    MarkReadRequest,
    MarkReadResponse,
    PostListResponse,
    PostResponse,
    PostSortField,
    SortOrder,
    UserPreferences,
    UserPreferencesUpdate,
    UserSettingsResponse,
)

LOGGER = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Router Definitions
# -----------------------------------------------------------------------------

health_router = APIRouter(tags=["Health"])
posts_router = APIRouter(prefix="/api/posts", tags=["Posts"])
settings_router = APIRouter(prefix="/api/settings", tags=["Settings"])


# -----------------------------------------------------------------------------
# Health Endpoints
# -----------------------------------------------------------------------------


@health_router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Returns the health status of the service and its dependencies.",
)
async def health_check(
    settings: SettingsDep,
    redis: RedisDep,
) -> HealthResponse:
    """Perform health checks on all service dependencies."""
    components: list[ComponentHealth] = []

    # Check database
    db_healthy, db_latency = await check_database_health(settings)
    components.append(
        ComponentHealth(
            name="database",
            status=HealthStatus.HEALTHY if db_healthy else HealthStatus.UNHEALTHY,
            latency_ms=db_latency,
            message=None if db_healthy else "Database connection failed",
        )
    )

    # Check Redis
    redis_healthy, redis_latency = await check_redis_health(redis)
    components.append(
        ComponentHealth(
            name="redis",
            status=HealthStatus.HEALTHY if redis_healthy else HealthStatus.UNHEALTHY,
            latency_ms=redis_latency,
            message=None if redis_healthy else "Redis connection failed",
        )
    )

    # Determine overall status
    all_healthy = all(c.status == HealthStatus.HEALTHY for c in components)
    any_unhealthy = any(c.status == HealthStatus.UNHEALTHY for c in components)

    if all_healthy:
        overall_status = HealthStatus.HEALTHY
    elif any_unhealthy:
        overall_status = HealthStatus.UNHEALTHY
    else:
        overall_status = HealthStatus.DEGRADED

    return HealthResponse(
        status=overall_status,
        version=settings.app.version,
        environment=settings.app.environment.value,
        components=components,
    )


# -----------------------------------------------------------------------------
# Posts Endpoints
# -----------------------------------------------------------------------------


@posts_router.get(
    "",
    response_model=PostListResponse,
    summary="List posts",
    description="Retrieve a paginated list of posts with optional filtering.",
    responses={
        200: {"description": "Successfully retrieved posts"},
        400: {"model": ErrorResponse, "description": "Invalid query parameters"},
    },
)
async def list_posts(
    session: DbSessionDep,
    settings: SettingsDep,
    current_user_id: CurrentUserIdDep,
    page: Annotated[int, Query(ge=1, description="Page number")] = 1,
    page_size: Annotated[int, Query(ge=1, le=100, description="Items per page")] = 20,
    source: Annotated[str | None, Query(description="Filter by source name")] = None,
    topic: Annotated[str | None, Query(description="Full-text search by topic")] = None,
    min_score: Annotated[
        float, Query(ge=0.0, le=1.0, description="Minimum classifier score")
    ] = 0.0,
    is_read: Annotated[bool | None, Query(description="Filter by read status")] = None,
    sort_by: Annotated[PostSortField, Query(description="Sort field")] = PostSortField.PUBLISHED_AT,
    sort_order: Annotated[SortOrder, Query(description="Sort direction")] = SortOrder.DESC,
) -> PostListResponse:
    """List posts with filtering and pagination."""
    # Build base query
    query = select(Post).where(Post.classifier_score >= min_score)

    # Apply source filter (sanitized for SQL injection prevention)
    if source:
        # Use concat to safely build the pattern - SQLAlchemy handles escaping
        source_pattern = f"%{source}%"
        query = query.where(Post.source_name.ilike(source_pattern))

    # Apply topic full-text search
    if topic and topic.strip():
        ts_query = func.plainto_tsquery("english", topic)
        query = query.where(Post.search_vector.op("@@")(ts_query))

    # Apply read status filter (requires user context)
    if is_read is not None and current_user_id is not None:
        read_subquery = (
            select(UserPostRead.post_id)
            .where(UserPostRead.user_id == current_user_id)
            .scalar_subquery()
        )
        if is_read:
            query = query.where(Post.id.in_(read_subquery))
        else:
            query = query.where(Post.id.notin_(read_subquery))

    # Get total count
    count_query = select(func.count()).select_from(query.subquery())
    total_result = await session.execute(count_query)
    total = total_result.scalar_one()

    # Apply sorting
    sort_column = {
        PostSortField.PUBLISHED_AT: Post.published_at,
        PostSortField.CLASSIFIER_SCORE: Post.classifier_score,
        PostSortField.CREATED_AT: Post.created_at,
    }[sort_by]

    if sort_order == SortOrder.DESC:
        query = query.order_by(sort_column.desc())
    else:
        query = query.order_by(sort_column.asc())

    # Apply pagination
    offset = (page - 1) * page_size
    query = query.offset(offset).limit(page_size)

    # Execute query
    result = await session.scalars(query)
    posts = list(result)

    # Build response
    has_next = (page * page_size) < total

    return PostListResponse(
        items=[PostResponse.model_validate(post) for post in posts],
        total=total,
        page=page,
        page_size=page_size,
        has_next=has_next,
    )


@posts_router.get(
    "/feed",
    response_model=PostListResponse,
    summary="Get personalized feed",
    description="Retrieve top posts. Personalized if user is authenticated, otherwise returns top by score.",
    responses={
        200: {"description": "Successfully retrieved feed"},
    },
)
async def get_feed(
    session: DbSessionDep,
    user_repo: UserRepoDep,
    user_id: CurrentUserIdDep,  # Optional - allows anonymous access
    limit: Annotated[int, Query(ge=1, le=20, description="Max posts to return")] = 10,
    min_score: Annotated[
        float, Query(ge=0.0, le=1.0, description="Minimum classifier score")
    ] = 0.0,
    sort: Annotated[
        str, Query(description="Sort by: published_at or classifier_score")
    ] = "classifier_score",
    order: Annotated[
        str, Query(description="Sort order: asc or desc")
    ] = "desc",
) -> PostListResponse:
    """Get feed. Personalized for authenticated users, top by score for anonymous."""
    topics: list[str] = []
    sources: list[str] = []
    hours = 7 * 24  # Default 7 days

    if user_id:
        user = await user_repo.get_or_create(user_id)
        topics = user.topics or []
        sources = user.sources or []

        # Parse period string (e.g., "7d" -> 7 days -> 168 hours)
        period_str = user.period or "7d"
        try:
            period_days = int(period_str.rstrip("d"))
        except (ValueError, AttributeError):
            period_days = 7
        hours = period_days * 24

    repo = PostRepository(session)

    if not topics:
        # If no topics configured, return recent posts sorted by score
        query = select(Post).where(Post.classifier_score >= min_score)

        # Filter by sources if configured
        if sources:
            source_conditions = []
            for source in sources:
                source_conditions.append(Post.source_name.ilike(f"{source}%"))
            if len(source_conditions) == 1:
                query = query.where(source_conditions[0])
            else:
                from sqlalchemy import or_
                query = query.where(or_(*source_conditions))

        # Apply sort order based on parameters
        if sort == "published_at":
            sort_column = Post.published_at
        else:
            sort_column = Post.classifier_score
        
        if order == "asc":
            query = query.order_by(sort_column.asc(), Post.published_at.desc())
        else:
            query = query.order_by(sort_column.desc(), Post.published_at.desc())
        query = query.limit(limit)

        result = await session.scalars(query)
        posts = list(result)

        return PostListResponse(
            items=[PostResponse.model_validate(post) for post in posts],
            total=len(posts),
            page=1,
            page_size=limit,
            has_next=False,  # No pagination for feed
        )

    # Use PostRepository for topic-based filtering - return top N only
    posts = await repo.get_posts_for_topics(
        topics=topics,
        limit=limit,
        sources=sources if sources else None,
        hours=hours,
        min_score=min_score,
        sort=sort,
        order=order,
    )

    return PostListResponse(
        items=[PostResponse.model_validate(post) for post in posts],
        total=len(posts),
        page=1,
        page_size=limit,
        has_next=False,  # No pagination for feed
    )


@posts_router.get(
    "/{post_id}",
    response_model=PostResponse,
    summary="Get post by ID",
    description="Retrieve a single post by its unique identifier.",
    responses={
        200: {"description": "Successfully retrieved post"},
        404: {"model": ErrorResponse, "description": "Post not found"},
    },
)
async def get_post(
    post_id: int,
    session: DbSessionDep,
) -> PostResponse:
    """Retrieve a single post by ID."""
    post = await session.get(Post, post_id)
    if post is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Post with ID {post_id} not found.",
        )
    return PostResponse.model_validate(post)


@posts_router.post(
    "/{post_id}/read",
    response_model=MarkReadResponse,
    summary="Mark post as read",
    description="Mark a post as read or unread for the current user.",
    responses={
        200: {"description": "Successfully updated read status"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        404: {"model": ErrorResponse, "description": "Post not found"},
    },
)
async def mark_post_read(
    post_id: int,
    body: MarkReadRequest,
    session: DbSessionDep,
    user_repo: UserRepoDep,
    user_id: RequiredUserIdDep,
) -> MarkReadResponse:
    """Mark a post as read or unread."""
    # Verify post exists
    post = await session.get(Post, post_id)
    if post is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Post with ID {post_id} not found.",
        )

    # Get or create user
    user = await user_repo.get_or_create(user_id)

    # Check existing read state
    existing = await session.scalar(
        select(UserPostRead).where(
            UserPostRead.user_id == user.id,
            UserPostRead.post_id == post_id,
        )
    )

    read_at: datetime | None = None

    if body.is_read:
        if existing is None:
            # Create read record
            read_record = UserPostRead(
                user_id=user.id,
                post_id=post_id,
            )
            session.add(read_record)
            await session.flush()
            read_at = read_record.read_at
        else:
            read_at = existing.read_at
    else:
        if existing is not None:
            # Remove read record
            await session.delete(existing)

    return MarkReadResponse(
        post_id=post_id,
        is_read=body.is_read,
        read_at=read_at,
    )


# -----------------------------------------------------------------------------
# Settings Endpoints
# -----------------------------------------------------------------------------


@settings_router.get(
    "",
    response_model=UserSettingsResponse,
    summary="Get user settings",
    description="Retrieve the current user's preferences and settings.",
    responses={
        200: {"description": "Successfully retrieved settings"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
    },
)
async def get_settings_endpoint(
    user_repo: UserRepoDep,
    user_id: RequiredUserIdDep,
) -> UserSettingsResponse:
    """Get current user's settings."""
    user = await user_repo.get_or_create(user_id)

    return UserSettingsResponse(
        user_id=user.id,
        telegram_id=user.telegram_id,
        preferences=UserPreferences(
            topics=user.topics or [],
            sources=user.sources or [],
            period=user.period or "7d",
        ),
        created_at=user.created_at,
    )


@settings_router.post(
    "",
    response_model=UserSettingsResponse,
    summary="Update user settings",
    description="Update the current user's preferences.",
    responses={
        200: {"description": "Successfully updated settings"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
    },
)
async def update_settings(
    body: UserPreferencesUpdate,
    user_repo: UserRepoDep,
    user_id: RequiredUserIdDep,
) -> UserSettingsResponse:
    """Update current user's settings."""
    user = await user_repo.update_preferences(
        user_id,
        topics=body.topics,
        sources=body.sources,
        period=body.period,
    )

    return UserSettingsResponse(
        user_id=user.id,
        telegram_id=user.telegram_id,
        preferences=UserPreferences(
            topics=user.topics or [],
            sources=user.sources or [],
            period=user.period or "7d",
        ),
        created_at=user.created_at,
    )


__all__ = [
    "health_router",
    "posts_router",
    "settings_router",
]
