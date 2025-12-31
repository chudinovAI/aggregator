"""
Unit tests for PostRepository and UserRepository.

Uses SQLite in-memory database with test-specific models that don't
have PostgreSQL-specific features (TSVECTOR, JSONB, pg_insert).

Note: Tests for PostgreSQL-specific functionality (UPSERT, full-text search)
are marked with @pytest.mark.postgres and should be run against a real
PostgreSQL instance.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from tests.factories import PostFactory, UserFactory
from tests.test_models import MockPost, MockUser, MockUserPostRead


def _utcnow() -> datetime:
    """Return current UTC time as naive datetime for SQLite compatibility."""
    return datetime.now(UTC).replace(tzinfo=None)


class TestPostRepositoryBasic:
    """
    Basic tests for post retrieval that work with SQLite.

    These tests don't use the full PostRepository as it has PostgreSQL-specific
    code (pg_insert). Instead, they test the query patterns directly.
    """

    @pytest.fixture
    async def sample_posts(self, db_session: AsyncSession) -> list[MockPost]:
        """Create sample posts in the database."""
        posts = [
            PostFactory.build(
                title=f"Post {i}",
                classifier_score=0.1 * i,
                source_name="reddit" if i % 2 == 0 else "hackernews",
                published_at=_utcnow() - timedelta(hours=i),
            )
            for i in range(1, 11)
        ]
        for post in posts:
            db_session.add(post)
        await db_session.flush()
        return posts

    @pytest.mark.asyncio
    async def test_select_posts_by_score(
        self, db_session: AsyncSession, sample_posts: list[MockPost]
    ) -> None:
        """Test selecting posts filtered by minimum score."""
        stmt = (
            select(MockPost)
            .where(MockPost.classifier_score >= 0.5)
            .order_by(MockPost.classifier_score.desc())
        )
        result = await db_session.scalars(stmt)
        posts = list(result)

        assert len(posts) == 6  # Posts 5-10 have scores 0.5-1.0
        assert all(post.classifier_score >= 0.5 for post in posts)

    @pytest.mark.asyncio
    async def test_select_posts_with_limit(
        self, db_session: AsyncSession, sample_posts: list[MockPost]
    ) -> None:
        """Test limiting results."""
        stmt = select(MockPost).order_by(MockPost.classifier_score.desc()).limit(3)
        result = await db_session.scalars(stmt)
        posts = list(result)

        assert len(posts) == 3

    @pytest.mark.asyncio
    async def test_select_posts_orders_by_score_desc(
        self, db_session: AsyncSession, sample_posts: list[MockPost]
    ) -> None:
        """Test that posts are ordered by classifier_score descending."""
        stmt = select(MockPost).order_by(MockPost.classifier_score.desc()).limit(10)
        result = await db_session.scalars(stmt)
        posts = list(result)

        scores = [post.classifier_score for post in posts]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_select_posts_with_pagination(
        self, db_session: AsyncSession, sample_posts: list[MockPost]
    ) -> None:
        """Test pagination with offset."""
        first_page_stmt = (
            select(MockPost).order_by(MockPost.classifier_score.desc()).limit(3).offset(0)
        )
        second_page_stmt = (
            select(MockPost).order_by(MockPost.classifier_score.desc()).limit(3).offset(3)
        )

        first_page = list(await db_session.scalars(first_page_stmt))
        second_page = list(await db_session.scalars(second_page_stmt))

        assert len(first_page) == 3
        assert len(second_page) == 3

        # No overlap
        first_ids = {p.id for p in first_page}
        second_ids = {p.id for p in second_page}
        assert first_ids.isdisjoint(second_ids)

    @pytest.mark.asyncio
    async def test_select_posts_by_source(
        self, db_session: AsyncSession, sample_posts: list[MockPost]
    ) -> None:
        """Test filtering by source_name."""
        stmt = select(MockPost).where(MockPost.source_name == "reddit")
        result = await db_session.scalars(stmt)
        posts = list(result)

        assert len(posts) == 5
        assert all(post.source_name == "reddit" for post in posts)

    @pytest.mark.asyncio
    async def test_select_posts_by_date(
        self, db_session: AsyncSession, sample_posts: list[MockPost]
    ) -> None:
        """Test filtering by published_after date."""
        cutoff = _utcnow() - timedelta(hours=5)
        stmt = select(MockPost).where(MockPost.published_at >= cutoff)
        result = await db_session.scalars(stmt)
        posts = list(result)

        # Posts 1-5 are within 5 hours, but post 5 is exactly at cutoff
        # Due to timing, we may get 4-5 posts
        assert len(posts) >= 4
        assert all(post.published_at >= cutoff for post in posts)

    @pytest.mark.asyncio
    async def test_select_posts_by_ids(
        self, db_session: AsyncSession, sample_posts: list[MockPost]
    ) -> None:
        """Test selecting posts by list of IDs."""
        target_ids = [sample_posts[0].id, sample_posts[2].id, sample_posts[4].id]
        stmt = select(MockPost).where(MockPost.id.in_(target_ids))
        result = await db_session.scalars(stmt)
        posts = list(result)

        assert len(posts) == 3
        result_ids = {post.id for post in posts}
        assert result_ids == set(target_ids)


class TestUnreadPostsQueries:
    """Tests for unread posts query patterns."""

    @pytest.fixture
    async def user_with_posts(self, db_session: AsyncSession) -> tuple[MockUser, list[MockPost]]:
        """Create a user and sample posts."""
        user = UserFactory.build()
        db_session.add(user)

        posts = [
            PostFactory.build(
                title=f"Post {i}",
                classifier_score=0.1 * i,
            )
            for i in range(1, 11)
        ]
        for post in posts:
            db_session.add(post)
        await db_session.flush()
        return user, posts

    @pytest.mark.asyncio
    async def test_count_unread_no_reads(
        self, db_session: AsyncSession, user_with_posts: tuple[MockUser, list[MockPost]]
    ) -> None:
        """Test unread count when user has not read any posts."""
        user, posts = user_with_posts

        # Count posts that don't have a read record for this user
        from sqlalchemy import func

        stmt = (
            select(func.count(MockPost.id))
            .select_from(MockPost)
            .outerjoin(
                MockUserPostRead,
                (MockUserPostRead.post_id == MockPost.id) & (MockUserPostRead.user_id == user.id),
            )
            .where(MockUserPostRead.id.is_(None))
        )
        result = await db_session.execute(stmt)
        count = result.scalar_one()

        assert count == 10

    @pytest.mark.asyncio
    async def test_count_unread_some_reads(
        self, db_session: AsyncSession, user_with_posts: tuple[MockUser, list[MockPost]]
    ) -> None:
        """Test unread count when user has read some posts."""
        user, posts = user_with_posts

        # Mark 3 posts as read
        for post in posts[:3]:
            read = MockUserPostRead(user_id=user.id, post_id=post.id)
            db_session.add(read)
        await db_session.flush()

        from sqlalchemy import func

        stmt = (
            select(func.count(MockPost.id))
            .select_from(MockPost)
            .outerjoin(
                MockUserPostRead,
                (MockUserPostRead.post_id == MockPost.id) & (MockUserPostRead.user_id == user.id),
            )
            .where(MockUserPostRead.id.is_(None))
        )
        result = await db_session.execute(stmt)
        count = result.scalar_one()

        assert count == 7

    @pytest.mark.asyncio
    async def test_get_unread_posts(
        self, db_session: AsyncSession, user_with_posts: tuple[MockUser, list[MockPost]]
    ) -> None:
        """Test fetching unread posts."""
        user, posts = user_with_posts

        read_post_ids = {posts[0].id, posts[1].id, posts[2].id}
        for post in posts[:3]:
            read = MockUserPostRead(user_id=user.id, post_id=post.id)
            db_session.add(read)
        await db_session.flush()

        stmt = (
            select(MockPost)
            .outerjoin(
                MockUserPostRead,
                (MockUserPostRead.post_id == MockPost.id) & (MockUserPostRead.user_id == user.id),
            )
            .where(MockUserPostRead.id.is_(None))
        )
        result = await db_session.scalars(stmt)
        unread_posts = list(result)

        assert len(unread_posts) == 7
        assert all(post.id not in read_post_ids for post in unread_posts)


class TestUserQueries:
    """Tests for user-related query patterns."""

    @pytest.mark.asyncio
    async def test_create_user(self, db_session: AsyncSession) -> None:
        """Test creating a new user."""
        user = MockUser(telegram_id=123456789, topics="[]", sources="[]")
        db_session.add(user)
        await db_session.flush()

        assert user.id is not None
        assert user.telegram_id == 123456789

    @pytest.mark.asyncio
    async def test_get_user_by_telegram_id(self, db_session: AsyncSession) -> None:
        """Test fetching user by Telegram ID."""
        user = UserFactory.build(telegram_id=987654321)
        db_session.add(user)
        await db_session.flush()

        stmt = select(MockUser).where(MockUser.telegram_id == 987654321)
        fetched = await db_session.scalar(stmt)

        assert fetched is not None
        assert fetched.id == user.id

    @pytest.mark.asyncio
    async def test_get_nonexistent_user(self, db_session: AsyncSession) -> None:
        """Test that fetching nonexistent user returns None."""
        stmt = select(MockUser).where(MockUser.telegram_id == 999888777)
        result = await db_session.scalar(stmt)

        assert result is None

    @pytest.mark.asyncio
    async def test_update_user_topics(self, db_session: AsyncSession) -> None:
        """Test updating user topics (stored as JSON string)."""
        import json

        user = UserFactory.build(telegram_id=111222333)
        db_session.add(user)
        await db_session.flush()

        # Update topics
        user.topics = json.dumps(["python", "rust", "go"])
        await db_session.flush()

        # Refresh and verify
        await db_session.refresh(user)
        topics = json.loads(user.topics)
        assert topics == ["python", "rust", "go"]


class TestUserPostReadAssociation:
    """Tests for the user-post read association."""

    @pytest.mark.asyncio
    async def test_mark_post_as_read(self, db_session: AsyncSession) -> None:
        """Test creating a read association."""
        user = UserFactory.build()
        post = PostFactory.build()
        db_session.add(user)
        db_session.add(post)
        await db_session.flush()

        read = MockUserPostRead(user_id=user.id, post_id=post.id)
        db_session.add(read)
        await db_session.flush()

        assert read.id is not None
        assert read.user_id == user.id
        assert read.post_id == post.id

    @pytest.mark.asyncio
    async def test_get_read_post_ids_for_user(self, db_session: AsyncSession) -> None:
        """Test fetching which posts a user has read."""
        user = UserFactory.build()
        db_session.add(user)

        posts = [PostFactory.build() for _ in range(5)]
        for post in posts:
            db_session.add(post)
        await db_session.flush()

        # Mark posts 0, 2 as read
        for i in [0, 2]:
            read = MockUserPostRead(user_id=user.id, post_id=posts[i].id)
            db_session.add(read)
        await db_session.flush()

        # Query read post IDs
        all_post_ids = [p.id for p in posts]
        stmt = (
            select(MockUserPostRead.post_id)
            .where(MockUserPostRead.user_id == user.id)
            .where(MockUserPostRead.post_id.in_(all_post_ids))
        )
        result = await db_session.scalars(stmt)
        read_ids = set(result)

        assert read_ids == {posts[0].id, posts[2].id}

    @pytest.mark.asyncio
    async def test_read_status_per_user(self, db_session: AsyncSession) -> None:
        """Test that read status is per-user."""
        user1 = UserFactory.build(telegram_id=100001)
        user2 = UserFactory.build(telegram_id=100002)
        db_session.add(user1)
        db_session.add(user2)

        post = PostFactory.build()
        db_session.add(post)
        await db_session.flush()

        # Only user2 reads the post
        read = MockUserPostRead(user_id=user2.id, post_id=post.id)
        db_session.add(read)
        await db_session.flush()

        # Check user1's read status
        stmt = (
            select(MockUserPostRead.post_id)
            .where(MockUserPostRead.user_id == user1.id)
            .where(MockUserPostRead.post_id == post.id)
        )
        result = await db_session.scalars(stmt)
        user1_read_ids = set(result)

        assert user1_read_ids == set()  # user1 hasn't read anything
