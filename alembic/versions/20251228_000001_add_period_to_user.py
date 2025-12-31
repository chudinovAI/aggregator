"""Add period field to users table.

Revision ID: 20251228000001
Revises: 20251227000002
Create Date: 2025-12-28
"""

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision = "20251228000001"
down_revision = "20251227000002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "users",
        sa.Column(
            "period",
            sa.String(length=10),
            server_default=sa.text("'7d'"),
            nullable=False,
        ),
    )


def downgrade() -> None:
    op.drop_column("users", "period")
