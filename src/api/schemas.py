"""
Pydantic schemas for API request/response serialization.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator


class SortOrder(str, Enum):
    """Supported sort directions for list queries."""

    ASC = "asc"
    DESC = "desc"


class PostSortField(str, Enum):
    """Fields available for sorting posts."""

    PUBLISHED_AT = "published_at"
    CLASSIFIER_SCORE = "classifier_score"
    CREATED_AT = "created_at"


# -----------------------------------------------------------------------------
# Post Schemas
# -----------------------------------------------------------------------------


class PostBase(BaseModel):
    """Common post attributes shared across schemas."""

    title: str = Field(..., min_length=1, max_length=512, description="Post title.")
    content: str = Field(..., description="Full post content or summary.")
    source_url: str = Field(..., description="Original source URL.")
    source_name: str = Field(..., min_length=1, max_length=128, description="Name of the source.")
    published_at: datetime = Field(..., description="Original publication timestamp.")

    @field_validator("source_url", mode="before")
    @classmethod
    def ensure_absolute_url(cls, v: str) -> str:
        """Convert relative Reddit URLs to absolute."""
        if isinstance(v, str) and v.startswith("/r/"):
            return f"https://www.reddit.com{v}"
        return str(v) if v else ""


class PostResponse(PostBase):
    """Schema returned when fetching a single post or list item."""

    model_config = ConfigDict(from_attributes=True)

    id: int = Field(..., description="Unique post identifier.")
    classifier_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="ML classifier relevance score.",
    )
    is_read: bool = Field(default=False, description="Whether the current user has read this post.")
    created_at: datetime = Field(..., description="Record creation timestamp.")
    updated_at: datetime = Field(..., description="Record last update timestamp.")


class PostListResponse(BaseModel):
    """Paginated list of posts with metadata."""

    items: list[PostResponse] = Field(default_factory=list, description="List of posts.")
    total: int = Field(..., ge=0, description="Total number of matching posts.")
    page: int = Field(..., ge=1, description="Current page number.")
    page_size: int = Field(..., ge=1, le=100, description="Items per page.")
    has_next: bool = Field(..., description="Whether more pages are available.")


class PostFilterParams(BaseModel):
    """Query parameters for filtering posts."""

    source: str | None = Field(default=None, description="Filter by source name.")
    topic: str | None = Field(default=None, description="Full-text search by topic.")
    min_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum classifier score threshold.",
    )
    is_read: bool | None = Field(default=None, description="Filter by read status.")
    sort_by: PostSortField = Field(
        default=PostSortField.PUBLISHED_AT,
        description="Field to sort by.",
    )
    sort_order: SortOrder = Field(default=SortOrder.DESC, description="Sort direction.")


class MarkReadRequest(BaseModel):
    """Request body for marking a post as read."""

    is_read: bool = Field(default=True, description="Mark as read (true) or unread (false).")


class MarkReadResponse(BaseModel):
    """Response after marking a post as read."""

    post_id: int = Field(..., description="Post ID that was updated.")
    is_read: bool = Field(..., description="New read status.")
    read_at: datetime | None = Field(default=None, description="Timestamp when marked as read.")


# -----------------------------------------------------------------------------
# User Settings / Preferences Schemas
# -----------------------------------------------------------------------------


class TopicItem(BaseModel):
    """Single topic preference."""

    name: str = Field(..., min_length=1, max_length=100, description="Topic name.")
    weight: float = Field(
        default=1.0,
        ge=0.0,
        le=10.0,
        description="Relevance weight for this topic.",
    )


class SourceItem(BaseModel):
    """Single source preference."""

    name: str = Field(..., min_length=1, max_length=128, description="Source identifier.")
    enabled: bool = Field(default=True, description="Whether this source is active.")


class TopicList(BaseModel):
    """Collection of topic preferences."""

    topics: list[TopicItem] = Field(default_factory=list, description="User topic preferences.")


class SourceList(BaseModel):
    """Collection of source preferences."""

    sources: list[SourceItem] = Field(default_factory=list, description="User source preferences.")


class UserPreferences(BaseModel):
    """Complete user preferences payload."""

    model_config = ConfigDict(from_attributes=True)

    topics: list[str] = Field(default_factory=list, description="Preferred topic keywords.")
    sources: list[str] = Field(default_factory=list, description="Preferred source names.")
    period: str = Field(default="7d", description="Search period (e.g., '1d', '7d', '30d').")


class UserPreferencesUpdate(BaseModel):
    """Request body for updating user preferences."""

    topics: list[str] | None = Field(
        default=None,
        description="New list of topic keywords (replaces existing).",
    )
    sources: list[str] | None = Field(
        default=None,
        description="New list of source names (replaces existing).",
    )
    period: str | None = Field(
        default=None,
        description="Search period (e.g., '1d', '7d', '30d').",
    )


class UserSettingsResponse(BaseModel):
    """Full user settings response."""

    model_config = ConfigDict(from_attributes=True)

    user_id: int = Field(..., description="Internal user ID.")
    telegram_id: int = Field(..., description="Telegram user ID.")
    preferences: UserPreferences = Field(..., description="User preferences.")
    created_at: datetime = Field(..., description="Account creation timestamp.")


# -----------------------------------------------------------------------------
# Health / Status Schemas
# -----------------------------------------------------------------------------


class HealthStatus(str, Enum):
    """Service health status values."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class ComponentHealth(BaseModel):
    """Health status for a single component."""

    name: str = Field(..., description="Component name.")
    status: HealthStatus = Field(..., description="Component status.")
    latency_ms: float | None = Field(default=None, description="Response latency in milliseconds.")
    message: str | None = Field(default=None, description="Optional status message.")


class HealthResponse(BaseModel):
    """Aggregated health check response."""

    status: HealthStatus = Field(..., description="Overall service status.")
    version: str = Field(..., description="Application version.")
    environment: str = Field(..., description="Deployment environment.")
    components: list[ComponentHealth] = Field(
        default_factory=list,
        description="Individual component health statuses.",
    )


# -----------------------------------------------------------------------------
# Error Schemas
# -----------------------------------------------------------------------------


class ErrorDetail(BaseModel):
    """Structured error detail."""

    field: str | None = Field(default=None, description="Field that caused the error.")
    message: str = Field(..., description="Human-readable error message.")
    code: str | None = Field(default=None, description="Machine-readable error code.")


class ErrorResponse(BaseModel):
    """Standard error response format."""

    error: str = Field(..., description="Error type or category.")
    message: str = Field(..., description="Human-readable error description.")
    details: list[ErrorDetail] = Field(
        default_factory=list,
        description="Detailed error information.",
    )
    request_id: str | None = Field(default=None, description="Request trace ID for debugging.")


__all__ = [
    "ComponentHealth",
    "ErrorDetail",
    "ErrorResponse",
    "HealthResponse",
    "HealthStatus",
    "MarkReadRequest",
    "MarkReadResponse",
    "PostBase",
    "PostFilterParams",
    "PostListResponse",
    "PostResponse",
    "PostSortField",
    "SortOrder",
    "SourceItem",
    "SourceList",
    "TopicItem",
    "TopicList",
    "UserPreferences",
    "UserPreferencesUpdate",
    "UserSettingsResponse",
]
