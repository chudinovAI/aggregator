"""
Test-specific SQLAlchemy ORM models for SQLite compatibility.

These models mirror the production models but without PostgreSQL-specific
features (TSVECTOR, Computed columns, JSONB) that SQLite doesn't support.

Note: DateTime is used without timezone for SQLite compatibility.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.ext.asyncio import AsyncAttrs
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class MockBase(AsyncAttrs, DeclarativeBase):
    """Declarative base for test models."""

    pass


class MockPost(MockBase):
    """Mock version of Post model without PostgreSQL-specific features.

    Note: Uses DateTime without timezone for SQLite compatibility.
    """

    __tablename__ = "posts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    source_url: Mapped[str] = mapped_column(String(1024), nullable=False, unique=True)
    source_name: Mapped[str] = mapped_column(String(128), nullable=False)
    published_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    scraped_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    classifier_score: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.0,
    )
    is_read: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )
    created_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    updated_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    read_states: Mapped[list[MockUserPostRead]] = relationship(
        "MockUserPostRead",
        back_populates="post",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        UniqueConstraint("source_url", name="uq_posts_source_url"),
        Index("ix_posts_source_name", "source_name"),
        Index("ix_posts_published_at", "published_at"),
        Index("ix_posts_score_published_desc", "classifier_score", "published_at"),
    )


class MockUser(MockBase):
    """Mock version of User model without PostgreSQL-specific features."""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    telegram_id: Mapped[int] = mapped_column(Integer, nullable=False, unique=True, index=True)
    # Use Text instead of JSONB for SQLite compatibility
    topics: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        default="[]",
    )
    sources: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        default="[]",
    )
    created_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    read_states: Mapped[list[MockUserPostRead]] = relationship(
        "MockUserPostRead",
        back_populates="user",
        cascade="all, delete-orphan",
    )


class MockUserPostRead(MockBase):
    """Mock version of UserPostRead association table."""

    __tablename__ = "user_post_reads"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    post_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("posts.id", ondelete="CASCADE"),
        nullable=False,
    )
    read_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    user: Mapped[MockUser] = relationship("MockUser", back_populates="read_states")
    post: Mapped[MockPost] = relationship("MockPost", back_populates="read_states")

    __table_args__ = (
        UniqueConstraint("user_id", "post_id", name="uq_user_post"),
        Index("ix_user_post_reads_user_post", "user_id", "post_id"),
    )


__all__ = ["MockBase", "MockPost", "MockUser", "MockUserPostRead"]
