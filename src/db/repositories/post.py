"""
Post repository for data access operations on Post entities.

Query Optimization Notes:
- Uses UPSERT (ON CONFLICT) for efficient save_post
- Leverages composite indexes for ordering queries
- Uses LEFT JOIN instead of correlated subqueries for counts
- Includes ts_rank for full-text search relevance
- Batch operations where possible
"""

from __future__ import annotations

from datetime import datetime, timedelta

from sqlalchemy import Select, case, func, literal_column, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine import CursorResult

from ...aggregator.parsers.base import ParsedPost
from ..models import Post, UserPostRead
from .base import BaseRepository


class PostRepository(BaseRepository):
    """
    Data access helpers for Post entities.

    Optimizations implemented:
    - UPSERT for save_post (avoids SELECT + INSERT/UPDATE)
    - Batch save support for multiple posts
    - Full-text search with ts_rank ordering
    - Efficient unread count using LEFT JOIN
    - Covering index hints via column selection
    """

    async def get_best_posts(
        self,
        min_score: float,
        limit: int,
        offset: int = 0,
        *,
        source_name: str | None = None,
        published_after: datetime | None = None,
    ) -> list[Post]:
        """
        Return posts ordered by classifier score and recency.

        Query uses composite index: ix_posts_classifier_score_published_at
        Time Complexity: O(log n + limit) with index

        Args:
            min_score: Minimum classifier score threshold
            limit: Maximum posts to return
            offset: Pagination offset
            source_name: Optional source filter (uses index ix_posts_source_name)
            published_after: Optional date filter
        """
        stmt: Select[tuple[Post]] = select(Post).where(Post.classifier_score >= min_score)

        if source_name:
            stmt = stmt.where(Post.source_name == source_name)

        if published_after:
            stmt = stmt.where(Post.published_at >= published_after)

        stmt = (
            stmt.order_by(Post.classifier_score.desc(), Post.published_at.desc())
            .limit(limit)
            .offset(offset)
        )

        result = await self._session.scalars(stmt)
        return list(result)

    async def get_by_topic(
        self,
        topic: str,
        limit: int,
        *,
        min_score: float | None = None,
        order_by_relevance: bool = True,
    ) -> list[Post]:
        """
        Fetch posts matched by full-text search against the topic.

        Uses GIN index: ix_posts_search_vector
        Optionally orders by ts_rank for relevance scoring.

        Args:
            topic: Search query (uses plainto_tsquery)
            limit: Maximum posts to return
            min_score: Optional minimum classifier score
            order_by_relevance: If True, order by ts_rank; else by published_at
        """
        if not topic.strip():
            return []

        ts_query = func.plainto_tsquery("english", topic)

        stmt: Select[tuple[Post]] = select(Post).where(Post.search_vector.op("@@")(ts_query))

        if min_score is not None:
            stmt = stmt.where(Post.classifier_score >= min_score)

        if order_by_relevance:
            # Order by full-text relevance score, then recency
            ts_rank = func.ts_rank(Post.search_vector, ts_query)
            stmt = stmt.order_by(ts_rank.desc(), Post.published_at.desc())
        else:
            stmt = stmt.order_by(Post.published_at.desc())

        stmt = stmt.limit(limit)
        result = await self._session.scalars(stmt)
        return list(result)

    async def search_posts(
        self,
        query: str,
        limit: int,
        offset: int = 0,
        *,
        sources: list[str] | None = None,
        min_score: float = 0.0,
        published_after: datetime | None = None,
    ) -> tuple[list[Post], int]:
        """
        Advanced search with multiple filters and total count.

        Returns both results and total count in a single query pattern.
        Uses window function for efficient count.

        Args:
            query: Full-text search query
            limit: Page size
            offset: Pagination offset
            sources: Optional list of source names
            min_score: Minimum classifier score
            published_after: Optional date filter

        Returns:
            Tuple of (posts, total_count)
        """
        ts_query = func.plainto_tsquery("english", query) if query.strip() else None
        ts_rank = func.ts_rank(Post.search_vector, ts_query) if ts_query else literal_column("1")

        # Build base conditions
        conditions = [Post.classifier_score >= min_score]

        if ts_query:
            conditions.append(Post.search_vector.op("@@")(ts_query))

        if sources:
            conditions.append(Post.source_name.in_(sources))

        if published_after:
            conditions.append(Post.published_at >= published_after)

        # Count query (separate for clarity and potential caching)
        count_stmt = select(func.count()).select_from(Post).where(*conditions)
        count_result = await self._session.execute(count_stmt)
        total = count_result.scalar_one()

        if total == 0:
            return [], 0

        # Data query with ordering
        data_stmt: Select[tuple[Post]] = (
            select(Post)
            .where(*conditions)
            .order_by(
                ts_rank.desc() if ts_query else Post.classifier_score.desc(),
                Post.published_at.desc(),
            )
            .offset(offset)
            .limit(limit)
        )

        result = await self._session.scalars(data_stmt)
        return list(result), total

    async def save_post(
        self,
        parsed_post: ParsedPost,
        *,
        classifier_score: float = 0.0,
    ) -> Post | None:
        """
        Insert or update a post using UPSERT (ON CONFLICT).

        Optimization: Uses PostgreSQL INSERT ... ON CONFLICT DO UPDATE
        to avoid separate SELECT + INSERT/UPDATE pattern.

        Time Complexity: O(log n) with unique index on source_url
        """
        if not parsed_post.source_url:
            return None

        # Use PostgreSQL UPSERT
        values = {
            "title": parsed_post.title,
            "content": parsed_post.content,
            "source_url": parsed_post.source_url,
            "source_name": parsed_post.source_name,
            "published_at": parsed_post.published_at,
            "scraped_at": parsed_post.fetched_at,
            "classifier_score": classifier_score,
        }

        stmt = pg_insert(Post).values(**values)

        # On conflict, update everything except source_url and created_at
        # Only update classifier_score if new value is higher (or if current is 0)
        update_dict = {
            "title": stmt.excluded.title,
            "content": stmt.excluded.content,
            "source_name": stmt.excluded.source_name,
            "published_at": stmt.excluded.published_at,
            "scraped_at": stmt.excluded.scraped_at,
            "classifier_score": case(
                (Post.classifier_score == 0, stmt.excluded.classifier_score),
                else_=func.greatest(Post.classifier_score, stmt.excluded.classifier_score),
            ),
            "updated_at": func.now(),
        }

        upsert_stmt = stmt.on_conflict_do_update(
            index_elements=["source_url"],
            set_=update_dict,
        ).returning(Post)

        result = await self._session.execute(upsert_stmt)
        post = result.scalar_one_or_none()
        return post

    async def save_posts_batch(
        self,
        posts_with_scores: list[tuple[ParsedPost, float]],
    ) -> int:
        """
        Batch insert/update multiple posts efficiently.

        Uses single UPSERT statement for all posts.
        Much more efficient than individual save_post calls.

        Args:
            posts_with_scores: List of (ParsedPost, classifier_score) tuples

        Returns:
            Number of posts upserted
        """
        if not posts_with_scores:
            return 0

        values_list = []
        for parsed_post, score in posts_with_scores:
            if not parsed_post.source_url:
                continue
            values_list.append(
                {
                    "title": parsed_post.title,
                    "content": parsed_post.content,
                    "source_url": parsed_post.source_url,
                    "source_name": parsed_post.source_name,
                    "published_at": parsed_post.published_at,
                    "scraped_at": parsed_post.fetched_at,
                    "classifier_score": score,
                }
            )

        if not values_list:
            return 0

        stmt = pg_insert(Post).values(values_list)

        update_dict = {
            "title": stmt.excluded.title,
            "content": stmt.excluded.content,
            "source_name": stmt.excluded.source_name,
            "published_at": stmt.excluded.published_at,
            "scraped_at": stmt.excluded.scraped_at,
            "classifier_score": case(
                (Post.classifier_score == 0, stmt.excluded.classifier_score),
                else_=func.greatest(Post.classifier_score, stmt.excluded.classifier_score),
            ),
            "updated_at": func.now(),
        }

        upsert_stmt = stmt.on_conflict_do_update(
            index_elements=["source_url"],
            set_=update_dict,
        )

        result: CursorResult[tuple[()]] = await self._session.execute(upsert_stmt)  # type: ignore[assignment]
        return result.rowcount or 0

    async def get_unread_count(self, user_id: int) -> int:
        """
        Return the number of posts the given user has not read.

        Optimization: Uses LEFT JOIN + NULL check instead of NOT EXISTS subquery.
        This allows better query planning with proper indexes.

        Uses index: ix_user_post_reads_user_post
        """
        # LEFT JOIN approach - often more efficient than NOT EXISTS
        stmt = (
            select(func.count(Post.id))
            .select_from(Post)
            .outerjoin(
                UserPostRead, (UserPostRead.post_id == Post.id) & (UserPostRead.user_id == user_id)
            )
            .where(UserPostRead.id.is_(None))
        )
        result = await self._session.execute(stmt)
        return int(result.scalar_one())

    async def get_unread_posts(
        self,
        user_id: int,
        limit: int,
        offset: int = 0,
        *,
        min_score: float = 0.0,
    ) -> list[Post]:
        """
        Get unread posts for a user, ordered by score and recency.

        Uses LEFT JOIN for efficiency.
        """
        stmt: Select[tuple[Post]] = (
            select(Post)
            .outerjoin(
                UserPostRead, (UserPostRead.post_id == Post.id) & (UserPostRead.user_id == user_id)
            )
            .where(UserPostRead.id.is_(None))
            .where(Post.classifier_score >= min_score)
            .order_by(Post.classifier_score.desc(), Post.published_at.desc())
            .limit(limit)
            .offset(offset)
        )
        result = await self._session.scalars(stmt)
        return list(result)

    async def mark_posts_read(
        self,
        user_id: int,
        post_ids: list[int],
    ) -> int:
        """
        Batch mark multiple posts as read.

        Uses INSERT ... ON CONFLICT DO NOTHING for efficiency.

        Returns:
            Number of newly marked posts
        """
        if not post_ids:
            return 0

        values = [{"user_id": user_id, "post_id": post_id} for post_id in post_ids]

        stmt = pg_insert(UserPostRead).values(values)
        stmt = stmt.on_conflict_do_nothing(constraint="uq_user_post")

        result: CursorResult[tuple[()]] = await self._session.execute(stmt)  # type: ignore[assignment]
        return result.rowcount or 0

    async def get_posts_by_ids(self, post_ids: list[int]) -> list[Post]:
        """
        Fetch multiple posts by ID in a single query.

        Useful for avoiding N+1 when you have a list of IDs.
        """
        if not post_ids:
            return []

        stmt = select(Post).where(Post.id.in_(post_ids))
        result = await self._session.scalars(stmt)
        return list(result)

    async def get_posts_for_topics(
        self,
        topics: list[str],
        limit: int = 10,
        *,
        sources: list[str] | None = None,
        hours: int = 24,
        min_score: float = 0.0,
        sort: str = "classifier_score",
        order: str = "desc",
    ) -> list[Post]:
        """
        Get posts matching user topics.

        Uses two strategies:
        1. Match source_name against topic-related subreddits/sources
        2. Full-text search in title/content as fallback

        Args:
            topics: List of user interest topics
            limit: Maximum posts to return
            sources: Optional list of source types to filter (reddit, hackernews)
            hours: How far back to search (default 24h)
            min_score: Minimum classifier score

        Returns:
            List of posts ordered by relevance
        """
        if not topics:
            return []

        # Topic to source patterns mapping
        topic_source_patterns = self._get_topic_source_patterns(topics)

        # Time cutoff
        cutoff = func.now() - timedelta(hours=hours)

        # Base conditions
        base_conditions = [
            Post.published_at >= cutoff,
            Post.classifier_score >= min_score,
        ]

        # Filter by source type if specified
        if sources:
            source_conditions = []
            for source in sources:
                source_conditions.append(Post.source_name.ilike(f"{source}%"))
            if source_conditions:
                base_conditions.append(
                    source_conditions[0] if len(source_conditions) == 1
                    else source_conditions[0] | source_conditions[1]
                )

        # Build source_name pattern matching for topics
        # e.g., topic "python" matches "reddit/r/python", "reddit/r/django", etc.
        pattern_conditions = []
        for pattern in topic_source_patterns:
            pattern_conditions.append(Post.source_name.ilike(f"%{pattern}%"))

        # Also add full-text search as additional matching criteria
        topic_queries = [func.plainto_tsquery("english", topic) for topic in topics]
        if len(topic_queries) == 1:
            combined_query = topic_queries[0]
        else:
            combined_query = topic_queries[0]
            for tq in topic_queries[1:]:
                combined_query = combined_query.op("||")(tq)

        fts_condition = Post.search_vector.op("@@")(combined_query)

        # Combine: (source patterns OR full-text search) AND base conditions
        if pattern_conditions:
            from sqlalchemy import or_
            topic_match = or_(*pattern_conditions, fts_condition)
        else:
            topic_match = fts_condition

        all_conditions = base_conditions + [topic_match]

        # Determine sort column
        if sort == "published_at":
            sort_column = Post.published_at
        else:
            sort_column = Post.classifier_score
        
        # Apply ordering
        if order == "asc":
            order_clause = [sort_column.asc(), Post.published_at.desc()]
        else:
            order_clause = [sort_column.desc(), Post.published_at.desc()]

        stmt: Select[tuple[Post]] = (
            select(Post)
            .where(*all_conditions)
            .order_by(*order_clause)
            .limit(limit)
        )

        result = await self._session.scalars(stmt)
        return list(result)

    def _get_topic_source_patterns(self, topics: list[str]) -> list[str]:
        """
        Map user topics to source_name patterns.
        
        Returns patterns that match subreddit/source names.
        """
        # Mapping of topics to related subreddit/source keywords
        # Use specific subreddit names to avoid false positives
        topic_mappings: dict[str, list[str]] = {
            "python": ["r/python", "r/django", "r/flask", "r/fastapi", "r/learnpython"],
            "machine learning": ["r/machinelearning", "r/deeplearning", "r/locallama", "r/mlops", "r/datascience"],
            "ai": ["r/machinelearning", "r/chatgpt", "r/openai", "r/claudeai", "r/ollama", "r/langchain", "r/locallama", "r/artificial"],
            "llm": ["r/locallama", "r/chatgpt", "r/openai", "r/claudeai", "r/ollama", "r/langchain"],
            "web development": ["r/webdev", "r/javascript", "r/reactjs", "r/vuejs", "r/frontend", "r/nextjs", "r/sveltejs"],
            "javascript": ["r/javascript", "r/reactjs", "r/vuejs", "r/node", "r/nextjs", "r/typescript"],
            "typescript": ["r/typescript", "r/angular", "r/nextjs"],
            "react": ["r/reactjs", "r/nextjs"],
            "devops": ["r/devops", "r/docker", "r/kubernetes", "r/aws", "r/terraform", "r/sre", "r/cicd"],
            "cloud": ["r/aws", "r/googlecloud", "r/azure"],
            "kubernetes": ["r/kubernetes", "r/docker"],
            "security": ["r/netsec", "r/cybersecurity", "r/hacking", "r/privacy", "r/reverseengineering"],
            "databases": ["r/postgresql", "r/mysql", "r/mongodb", "r/redis", "r/database", "r/sql"],
            "linux": ["r/linux", "r/linuxadmin", "r/sysadmin", "r/selfhosted", "r/homelab"],
            "rust": ["r/rust", "r/learnrust"],
            "golang": ["r/golang"],  # Only exact match, "go" is too generic
            "data science": ["r/datascience", "r/dataengineering", "r/bigdata", "r/machinelearning"],
            "open source": ["r/opensource", "r/github", "r/selfhosted"],
            "career": ["r/cscareerquestions", "r/experienceddevs", "r/remotework"],
            "startups": ["r/startups", "r/saas", "r/entrepreneur", "r/indiehackers"],
        }

        patterns = set()
        for topic in topics:
            topic_lower = topic.lower().strip()

            # Check if topic has a known mapping
            if topic_lower in topic_mappings:
                patterns.update(topic_mappings[topic_lower])
            else:
                # Use topic itself as pattern (handles custom topics)
                # Add r/ prefix for subreddit matching
                clean = topic_lower.replace(" ", "")
                patterns.add(f"r/{clean}")
                patterns.add(clean)  # Also match in title/content

        return list(patterns)

    async def get_top_posts_by_score(
        self,
        limit: int = 10,
        *,
        sources: list[str] | None = None,
        hours: int = 24,
        min_score: float = 0.0,
    ) -> list[Post]:
        """
        Get top posts by classifier score without topic filtering.

        Useful as fallback when no topic matches found.

        Args:
            limit: Maximum posts to return
            sources: Optional list of source types (reddit, hackernews)
            hours: How far back to search
            min_score: Minimum classifier score

        Returns:
            List of posts ordered by score
        """
        cutoff = func.now() - timedelta(hours=hours)

        conditions = [
            Post.published_at >= cutoff,
            Post.classifier_score >= min_score,
        ]

        # Filter by source type using prefix matching
        if sources:
            from sqlalchemy import or_
            source_conditions = [Post.source_name.ilike(f"{source}%") for source in sources]
            if len(source_conditions) == 1:
                conditions.append(source_conditions[0])
            else:
                conditions.append(or_(*source_conditions))

        stmt: Select[tuple[Post]] = (
            select(Post)
            .where(*conditions)
            .order_by(Post.classifier_score.desc(), Post.published_at.desc())
            .limit(limit)
        )

        result = await self._session.scalars(stmt)
        return list(result)

    async def get_recent_posts_count(
        self,
        hours: int = 24,
        source_name: str | None = None,
    ) -> int:
        """
        Get count of posts from the last N hours.

        Uses index on published_at.
        """
        # Use Python timedelta to avoid SQL injection
        cutoff = func.now() - timedelta(hours=hours)

        stmt = select(func.count()).select_from(Post).where(Post.published_at >= cutoff)

        if source_name:
            stmt = stmt.where(Post.source_name == source_name)

        result = await self._session.execute(stmt)
        return int(result.scalar_one())


__all__ = ["PostRepository"]
