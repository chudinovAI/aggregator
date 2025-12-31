"""
Factory Boy fixtures for building domain objects in tests.

Note: Uses naive datetimes for SQLite compatibility.
"""

from __future__ import annotations

from datetime import UTC, datetime

import factory

from src.aggregator.parsers.base import ParsedPost
from tests.test_models import MockPost, MockUser, MockUserPostRead


def _utcnow() -> datetime:
    """Return current UTC time as naive datetime for SQLite compatibility."""
    return datetime.now(UTC).replace(tzinfo=None)


class PostFactory(factory.Factory):
    """Create MockPost ORM instances without persisting them."""

    class Meta:
        model = MockPost

    title = factory.Faker("sentence", nb_words=8)
    content = factory.Faker("paragraph", nb_sentences=3)
    source_url = factory.Sequence(lambda index: f"https://example.com/articles/{index}")
    source_name = factory.Iterator(["reddit", "hackernews"])
    published_at = factory.LazyFunction(_utcnow)
    scraped_at = factory.LazyFunction(_utcnow)
    classifier_score = factory.Faker("pyfloat", min_value=0.0, max_value=1.0)
    is_read = False


class UserFactory(factory.Factory):
    """Create MockUser ORM instances suitable for tests."""

    class Meta:
        model = MockUser

    telegram_id = factory.Sequence(lambda index: 10_000 + index)
    # Store as JSON string for SQLite compatibility
    topics = factory.LazyFunction(lambda: "[]")
    sources = factory.LazyFunction(lambda: "[]")
    created_at = factory.LazyFunction(_utcnow)


class UserPostReadFactory(factory.Factory):
    """Associate a user with a read post."""

    class Meta:
        model = MockUserPostRead

    user = factory.SubFactory(UserFactory)
    post = factory.SubFactory(PostFactory)
    read_at = factory.LazyFunction(_utcnow)


class ParsedPostFactory(factory.Factory):
    """Factory for ParsedPost dataclass used in parsers.

    Note: ParsedPost uses timezone-aware datetimes as it's not stored in SQLite.
    """

    class Meta:
        model = ParsedPost

    id = factory.Sequence(lambda index: f"parsed-{index}")
    title = factory.Faker("sentence", nb_words=10)
    content = factory.Faker("paragraph")
    source_url = factory.Sequence(lambda index: f"https://example.com/parsed/{index}")
    source_name = factory.Iterator(["reddit", "hackernews"])
    published_at = factory.LazyFunction(lambda: datetime.utcnow())
    fetched_at = factory.LazyFunction(lambda: datetime.utcnow())
    raw_data = factory.LazyFunction(dict)


__all__ = [
    "ParsedPostFactory",
    "PostFactory",
    "UserFactory",
    "UserPostReadFactory",
]
