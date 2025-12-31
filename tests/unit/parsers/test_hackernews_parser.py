"""
Unit tests for HackerNewsParser.

Uses respx to mock HTTP requests to the HackerNews API.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import httpx
import pytest
import respx

from src.aggregator.exceptions import FetchError, ParseError
from src.aggregator.parsers.hackernews import (
    ITEM_URL_TEMPLATE,
    TOPSTORIES_URL,
    HackerNewsParser,
)


class TestHackerNewsParser:
    """Tests for HackerNewsParser."""

    @pytest.fixture
    def parser(self) -> HackerNewsParser:
        """Create a HackerNewsParser with minimal delay."""
        return HackerNewsParser(
            request_delay_seconds=0,
            max_concurrent_requests=5,
        )

    @pytest.fixture
    def sample_story_ids(self) -> list[int]:
        """Sample story IDs response."""
        return [12345, 12346, 12347]

    @pytest.fixture
    def sample_stories(self) -> dict[int, dict]:
        """Sample story data keyed by ID."""
        return {
            12345: {
                "id": 12345,
                "title": "Python ML Tutorial",
                "url": "https://example.com/python-ml",
                "by": "user1",
                "score": 150,
                "time": 1700000000,
                "type": "story",
                "descendants": 42,
            },
            12346: {
                "id": 12346,
                "title": "Rust Systems Programming",
                "text": "A deep dive into Rust for systems programming.",
                "by": "user2",
                "score": 200,
                "time": 1700001000,
                "type": "story",
                "descendants": 75,
            },
            12347: {
                "id": 12347,
                "title": "Job: Senior Developer at TechCo",
                "url": "https://techco.com/jobs",
                "by": "techco",
                "score": 10,
                "time": 1700002000,
                "type": "job",
                "descendants": 0,
            },
        }

    def test_build_topic_url_returns_topstories(self, parser: HackerNewsParser) -> None:
        """Test that build_topic_url always returns top stories URL."""
        assert parser.build_topic_url("python", 50) == TOPSTORIES_URL
        assert parser.build_topic_url("anything", 100) == TOPSTORIES_URL

    @respx.mock
    @pytest.mark.asyncio
    async def test_fetch_content_success(self, parser: HackerNewsParser) -> None:
        """Test successful content fetch."""
        respx.get(TOPSTORIES_URL).mock(return_value=httpx.Response(200, json=[1, 2, 3]))

        content = await parser.fetch_content(TOPSTORIES_URL)

        # JSON may or may not have spaces depending on serialization
        assert content in ("[1, 2, 3]", "[1,2,3]")

    @respx.mock
    @pytest.mark.asyncio
    async def test_fetch_content_http_error_raises_fetch_error(
        self, parser: HackerNewsParser
    ) -> None:
        """Test that HTTP errors raise FetchError."""
        respx.get(TOPSTORIES_URL).mock(
            return_value=httpx.Response(500, text="Internal Server Error")
        )

        with pytest.raises(FetchError, match="HackerNews request failed"):
            await parser.fetch_content(TOPSTORIES_URL)

    @respx.mock
    @pytest.mark.asyncio
    async def test_fetch_content_connection_error_raises_fetch_error(
        self, parser: HackerNewsParser
    ) -> None:
        """Test that connection errors raise FetchError."""
        respx.get(TOPSTORIES_URL).mock(side_effect=httpx.ConnectError("Connection refused"))

        with pytest.raises(FetchError):
            await parser.fetch_content(TOPSTORIES_URL)

    @pytest.mark.asyncio
    async def test_extract_posts_invalid_json_raises_parse_error(
        self, parser: HackerNewsParser
    ) -> None:
        """Test that invalid JSON raises ParseError."""
        with pytest.raises(ParseError, match="not valid JSON"):
            await parser.extract_posts("not valid json", "topic")

    @pytest.mark.asyncio
    async def test_extract_posts_non_list_raises_parse_error(
        self, parser: HackerNewsParser
    ) -> None:
        """Test that non-list response raises ParseError."""
        with pytest.raises(ParseError, match="Expected a list"):
            await parser.extract_posts('{"key": "value"}', "topic")

    @respx.mock
    @pytest.mark.asyncio
    async def test_extract_posts_fetches_stories(
        self,
        parser: HackerNewsParser,
        sample_story_ids: list[int],
        sample_stories: dict[int, dict],
    ) -> None:
        """Test that extract_posts fetches individual stories."""
        # Mock story endpoint for each ID
        for story_id in sample_story_ids:
            url = ITEM_URL_TEMPLATE.format(item_id=story_id)
            respx.get(url).mock(return_value=httpx.Response(200, json=sample_stories[story_id]))

        content = json.dumps(sample_story_ids)
        posts = await parser.extract_posts(content, "")

        assert len(posts) == 3  # All stories/jobs included

    @respx.mock
    @pytest.mark.asyncio
    async def test_extract_posts_filters_by_topic(
        self,
        parser: HackerNewsParser,
        sample_story_ids: list[int],
        sample_stories: dict[int, dict],
    ) -> None:
        """Test that posts are filtered by topic keyword."""
        for story_id in sample_story_ids:
            url = ITEM_URL_TEMPLATE.format(item_id=story_id)
            respx.get(url).mock(return_value=httpx.Response(200, json=sample_stories[story_id]))

        content = json.dumps(sample_story_ids)
        posts = await parser.extract_posts(content, "python")

        # Only post with "Python" in title should match
        assert len(posts) == 1
        assert "Python" in posts[0].title

    @respx.mock
    @pytest.mark.asyncio
    async def test_extract_posts_special_topics_no_filter(
        self,
        parser: HackerNewsParser,
        sample_story_ids: list[int],
        sample_stories: dict[int, dict],
    ) -> None:
        """Test that special topics (top, all, hackernews) don't filter."""
        for story_id in sample_story_ids:
            url = ITEM_URL_TEMPLATE.format(item_id=story_id)
            respx.get(url).mock(return_value=httpx.Response(200, json=sample_stories[story_id]))

        content = json.dumps(sample_story_ids)

        for topic in ["top", "all", "hackernews"]:
            posts = await parser.extract_posts(content, topic)
            assert len(posts) == 3  # All posts included

    @respx.mock
    @pytest.mark.asyncio
    async def test_extract_posts_handles_failed_story_fetch(
        self,
        parser: HackerNewsParser,
        sample_story_ids: list[int],
        sample_stories: dict[int, dict],
    ) -> None:
        """Test that failed story fetches are skipped gracefully."""
        # First story returns 500, others succeed
        respx.get(ITEM_URL_TEMPLATE.format(item_id=12345)).mock(
            return_value=httpx.Response(500, text="Error")
        )
        respx.get(ITEM_URL_TEMPLATE.format(item_id=12346)).mock(
            return_value=httpx.Response(200, json=sample_stories[12346])
        )
        respx.get(ITEM_URL_TEMPLATE.format(item_id=12347)).mock(
            return_value=httpx.Response(200, json=sample_stories[12347])
        )

        content = json.dumps(sample_story_ids)
        posts = await parser.extract_posts(content, "")

        # Only 2 posts fetched successfully
        assert len(posts) == 2

    @respx.mock
    @pytest.mark.asyncio
    async def test_extract_posts_skips_non_story_items(
        self,
        parser: HackerNewsParser,
    ) -> None:
        """Test that non-story/job items are skipped."""
        comment_item = {
            "id": 99999,
            "text": "This is a comment",
            "by": "commenter",
            "type": "comment",
            "time": 1700000000,
        }

        respx.get(ITEM_URL_TEMPLATE.format(item_id=99999)).mock(
            return_value=httpx.Response(200, json=comment_item)
        )

        content = json.dumps([99999])
        posts = await parser.extract_posts(content, "")

        assert len(posts) == 0  # Comment should be skipped

    @respx.mock
    @pytest.mark.asyncio
    async def test_story_to_post_conversion(
        self,
        parser: HackerNewsParser,
        sample_stories: dict[int, dict],
    ) -> None:
        """Test that story dict is correctly converted to ParsedPost."""
        respx.get(ITEM_URL_TEMPLATE.format(item_id=12345)).mock(
            return_value=httpx.Response(200, json=sample_stories[12345])
        )

        content = json.dumps([12345])
        posts = await parser.extract_posts(content, "")

        assert len(posts) == 1
        post = posts[0]

        assert post.id == "12345"
        assert post.title == "Python ML Tutorial"
        assert post.source_url == "https://example.com/python-ml"
        assert post.source_name == "hackernews"
        assert post.raw_data["score"] == 150
        assert post.raw_data["by"] == "user1"
        assert post.raw_data["descendants"] == 42

    @respx.mock
    @pytest.mark.asyncio
    async def test_story_without_url_uses_hn_discussion_link(
        self,
        parser: HackerNewsParser,
        sample_stories: dict[int, dict],
    ) -> None:
        """Test that stories without URL use HN discussion link."""
        respx.get(ITEM_URL_TEMPLATE.format(item_id=12346)).mock(
            return_value=httpx.Response(200, json=sample_stories[12346])
        )

        content = json.dumps([12346])
        posts = await parser.extract_posts(content, "")

        assert len(posts) == 1
        # Story 12346 has no URL, should use HN discussion link
        assert posts[0].source_url == "https://news.ycombinator.com/item?id=12346"

    def test_to_datetime_valid_timestamp(self) -> None:
        """Test timestamp conversion with valid Unix timestamp."""
        timestamp = 1700000000
        result = HackerNewsParser._to_datetime(timestamp)

        assert result.tzinfo is not None
        assert result == datetime.fromtimestamp(1700000000, tz=UTC)

    def test_to_datetime_invalid_timestamp_returns_now(self) -> None:
        """Test that invalid timestamps return current time."""
        before = datetime.now(UTC)
        result = HackerNewsParser._to_datetime(None)
        after = datetime.now(UTC)

        assert before <= result <= after

    def test_to_datetime_zero_returns_now(self) -> None:
        """Test that zero timestamp returns current time."""
        before = datetime.now(UTC)
        result = HackerNewsParser._to_datetime(0)
        after = datetime.now(UTC)

        assert before <= result <= after

    def test_validate_post_valid(self, parser: HackerNewsParser) -> None:
        """Test validation of valid post."""
        from src.aggregator.parsers.base import ParsedPost

        post = ParsedPost(
            id="123",
            title="Test Title",
            content="Test content",
            source_url="https://example.com/article",
            source_name="hackernews",
            published_at=datetime.now(UTC),
            fetched_at=datetime.now(UTC),
            raw_data={},
        )

        assert parser.validate_post(post) is True

    def test_validate_post_missing_title(self, parser: HackerNewsParser) -> None:
        """Test validation fails for missing title."""
        from src.aggregator.parsers.base import ParsedPost

        post = ParsedPost(
            id="123",
            title="",
            content="Test content",
            source_url="https://example.com/article",
            source_name="hackernews",
            published_at=datetime.now(UTC),
            fetched_at=datetime.now(UTC),
            raw_data={},
        )

        assert parser.validate_post(post) is False

    def test_validate_post_missing_url(self, parser: HackerNewsParser) -> None:
        """Test validation fails for missing URL."""
        from src.aggregator.parsers.base import ParsedPost

        post = ParsedPost(
            id="123",
            title="Test Title",
            content="Test content",
            source_url="",
            source_name="hackernews",
            published_at=datetime.now(UTC),
            fetched_at=datetime.now(UTC),
            raw_data={},
        )

        assert parser.validate_post(post) is False

    @pytest.mark.asyncio
    async def test_aclose_closes_owned_client(self) -> None:
        """Test that aclose closes the client when parser owns it."""
        parser = HackerNewsParser(request_delay_seconds=0)

        # Should not raise
        await parser.aclose()

    @pytest.mark.asyncio
    async def test_aclose_does_not_close_injected_client(self) -> None:
        """Test that aclose doesn't close externally injected client."""
        async with httpx.AsyncClient() as client:
            parser = HackerNewsParser(
                client=client,
                request_delay_seconds=0,
            )

            await parser.aclose()

            # Client should still be open
            assert not client.is_closed


class TestHackerNewsParserIntegration:
    """Integration-style tests for full fetch flow."""

    @respx.mock
    @pytest.mark.asyncio
    async def test_full_fetch_flow(self) -> None:
        """Test complete fetch flow from topic to posts."""
        parser = HackerNewsParser(request_delay_seconds=0)

        # Mock top stories endpoint
        story_ids = [100, 101]
        respx.get(TOPSTORIES_URL).mock(return_value=httpx.Response(200, json=story_ids))

        # Mock individual story endpoints
        respx.get(ITEM_URL_TEMPLATE.format(item_id=100)).mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": 100,
                    "title": "Test Story One",
                    "url": "https://example.com/one",
                    "by": "author1",
                    "score": 100,
                    "time": 1700000000,
                    "type": "story",
                    "descendants": 10,
                },
            )
        )
        respx.get(ITEM_URL_TEMPLATE.format(item_id=101)).mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": 101,
                    "title": "Test Story Two",
                    "text": "Story content here",
                    "by": "author2",
                    "score": 50,
                    "time": 1700001000,
                    "type": "story",
                    "descendants": 5,
                },
            )
        )

        # Execute full fetch
        url = parser.build_topic_url("all", 100)
        content = await parser.fetch_content(url)
        posts = await parser.extract_posts(content, "all")

        assert len(posts) == 2
        assert posts[0].id == "100"
        assert posts[1].id == "101"

        await parser.aclose()
