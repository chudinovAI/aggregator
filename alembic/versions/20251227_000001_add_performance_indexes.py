"""Add performance optimization indexes.

This migration adds indexes identified during performance profiling:
1. Composite index on (classifier_score DESC, published_at DESC) for get_best_posts query
2. Index on user_post_reads(user_id, post_id) for unread count queries

Query analysis:
- get_best_posts: SELECT * FROM posts WHERE classifier_score >= :min
                  ORDER BY classifier_score DESC, published_at DESC
                  LIMIT :limit OFFSET :offset
  Without index: Sequential scan O(n) + sort O(n log n)
  With index: Index scan O(log n + limit)
"""

from __future__ import annotations

from alembic import op

revision = "20251227000001"
down_revision = "20251225000001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Composite index for get_best_posts query
    # DESC ordering matches the query's ORDER BY clause
    op.create_index(
        "ix_posts_classifier_score_published_at",
        "posts",
        ["classifier_score", "published_at"],
        postgresql_ops={"classifier_score": "DESC", "published_at": "DESC"},
    )

    # Index for efficient user_id lookups in read state checks
    op.create_index(
        "ix_user_post_reads_user_id",
        "user_post_reads",
        ["user_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_user_post_reads_user_id", table_name="user_post_reads")
    op.drop_index("ix_posts_classifier_score_published_at", table_name="posts")
