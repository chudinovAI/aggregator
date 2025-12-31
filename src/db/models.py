"""
SQLAlchemy ORM models for posts, users, and read-state associations.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    Computed,
    DateTime,
    Float,
    ForeignKey,
    Index,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR
from sqlalchemy.ext.asyncio import AsyncAttrs
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _utc_now() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(UTC)


class Base(AsyncAttrs, DeclarativeBase):
    """Declarative base that enables async-friendly ORM operations."""

    pass


class TimestampMixin:
    """Reusable timestamp columns for auditing."""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utc_now,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utc_now,
        onupdate=_utc_now,
        server_default=func.now(),
    )


class Post(TimestampMixin, Base):
    """News post ingested by the aggregator."""

    __tablename__ = "posts"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    source_url: Mapped[str] = mapped_column(String(1024), nullable=False, unique=True)
    source_name: Mapped[str] = mapped_column(String(128), nullable=False)
    published_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    scraped_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utc_now,
        server_default=func.now(),
    )
    classifier_score: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.0,
        server_default=text("0"),
    )
    is_read: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default=text("false"),
    )
    search_vector: Mapped[str] = mapped_column(
        TSVECTOR,
        Computed(
            "to_tsvector('english', coalesce(title, '') || ' ' || coalesce(content, ''))",
            persisted=True,
        ),
        nullable=False,
    )

    read_states: Mapped[list[UserPostRead]] = relationship(
        "UserPostRead",
        back_populates="post",
        cascade="all, delete-orphan",
    )
    read_by_users: Mapped[list[User]] = relationship(
        "User",
        secondary="user_post_reads",
        primaryjoin="Post.id == UserPostRead.post_id",
        secondaryjoin="User.id == UserPostRead.user_id",
        viewonly=True,
        overlaps="read_states,read_posts",
    )

    __table_args__ = (
        UniqueConstraint("source_url", name="uq_posts_source_url"),
        # Full-text search index (GIN)
        Index("ix_posts_search_vector", "search_vector", postgresql_using="gin"),
        # Source filtering index
        Index("ix_posts_source_name", "source_name"),
        # Time-based filtering index
        Index("ix_posts_published_at", "published_at"),
        # Composite index for get_best_posts query (score + recency ordering)
        # Note: Actual DESC ordering is defined in migration
        Index("ix_posts_score_published_desc", "classifier_score", "published_at"),
    )


class User(Base):
    """Represents an end-user consuming the news feed."""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    telegram_id: Mapped[int] = mapped_column(BigInteger, nullable=False, unique=True, index=True)
    topics: Mapped[list[str]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
        server_default=text("'[]'::jsonb"),
    )
    sources: Mapped[list[str]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
        server_default=text("'[]'::jsonb"),
    )
    period: Mapped[str] = mapped_column(
        String(10),
        nullable=False,
        default="7d",
        server_default=text("'7d'"),
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utc_now,
        server_default=func.now(),
    )

    read_states: Mapped[list[UserPostRead]] = relationship(
        "UserPostRead",
        back_populates="user",
        cascade="all, delete-orphan",
    )
    read_posts: Mapped[list[Post]] = relationship(
        "Post",
        secondary="user_post_reads",
        primaryjoin="User.id == UserPostRead.user_id",
        secondaryjoin="Post.id == UserPostRead.post_id",
        viewonly=True,
        overlaps="read_states,read_by_users",
    )


class UserPostRead(Base):
    """Association table capturing which posts have been read by which users."""

    __tablename__ = "user_post_reads"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    post_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("posts.id", ondelete="CASCADE"),
        nullable=False,
    )
    read_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utc_now,
        server_default=func.now(),
    )

    user: Mapped[User] = relationship("User", back_populates="read_states")
    post: Mapped[Post] = relationship("Post", back_populates="read_states")

    __table_args__ = (
        UniqueConstraint("user_id", "post_id", name="uq_user_post"),
        # Composite index for efficient user+post lookups
        Index("ix_user_post_reads_user_post", "user_id", "post_id"),
    )


__all__ = ["Base", "Post", "User", "UserPostRead"]
