"""Add cache token columns to token_usage, llm_interactions, and agent_runs.

Per-call: cache_read_input_tokens / cache_creation_input_tokens (nullable;
None means provider didn't report, 0 means reported zero).

Run-level rollup: total_cache_read_tokens / total_cache_creation_tokens
(default 0 — sums across calls; None values treated as 0 in aggregation).

Revision ID: a8b9c0d1e2f3
Revises: 351153c8797a
Create Date: 2026-04-16
"""

from __future__ import annotations

from typing import Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "a8b9c0d1e2f3"
down_revision: Union[str, None] = "351153c8797a"
branch_labels: Union[str, tuple[str, ...], None] = None
depends_on: Union[str, tuple[str, ...], None] = None


def upgrade() -> None:
    # Per-call cache fields (nullable)
    op.add_column(
        "token_usage",
        sa.Column("cache_read_input_tokens", sa.Integer(), nullable=True),
    )
    op.add_column(
        "token_usage",
        sa.Column("cache_creation_input_tokens", sa.Integer(), nullable=True),
    )
    op.add_column(
        "llm_interactions",
        sa.Column("cache_read_input_tokens", sa.Integer(), nullable=True),
    )
    op.add_column(
        "llm_interactions",
        sa.Column("cache_creation_input_tokens", sa.Integer(), nullable=True),
    )

    # Run-level rollup (default 0 — non-null aggregate)
    op.add_column(
        "agent_runs",
        sa.Column(
            "total_cache_read_tokens", sa.Integer(), nullable=False, server_default="0"
        ),
    )
    op.add_column(
        "agent_runs",
        sa.Column(
            "total_cache_creation_tokens",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
    )


def downgrade() -> None:
    op.drop_column("agent_runs", "total_cache_creation_tokens")
    op.drop_column("agent_runs", "total_cache_read_tokens")
    op.drop_column("llm_interactions", "cache_creation_input_tokens")
    op.drop_column("llm_interactions", "cache_read_input_tokens")
    op.drop_column("token_usage", "cache_creation_input_tokens")
    op.drop_column("token_usage", "cache_read_input_tokens")
