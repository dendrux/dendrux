"""Add reasoning token columns to token_usage, llm_interactions, and agent_runs.

Per-call: reasoning_tokens (nullable; None means provider didn't report).
Run-level rollup: total_reasoning_tokens (default 0).

Reasoning tokens are billed *within* output tokens — this is an informational
breakdown, never added to totals.

Revision ID: a4b5c6d7e8f9
Revises: f3a4b5c6d7e8
Create Date: 2026-06-17
"""

from __future__ import annotations

from typing import Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "a4b5c6d7e8f9"
down_revision: Union[str, None] = "f3a4b5c6d7e8"
branch_labels: Union[str, tuple[str, ...], None] = None
depends_on: Union[str, tuple[str, ...], None] = None


def upgrade() -> None:
    # Per-call reasoning tokens (nullable)
    op.add_column(
        "token_usage",
        sa.Column("reasoning_tokens", sa.Integer(), nullable=True),
    )
    op.add_column(
        "llm_interactions",
        sa.Column("reasoning_tokens", sa.Integer(), nullable=True),
    )

    # Run-level rollup (default 0 — non-null aggregate)
    op.add_column(
        "agent_runs",
        sa.Column(
            "total_reasoning_tokens", sa.Integer(), nullable=False, server_default="0"
        ),
    )


def downgrade() -> None:
    op.drop_column("agent_runs", "total_reasoning_tokens")
    op.drop_column("llm_interactions", "reasoning_tokens")
    op.drop_column("token_usage", "reasoning_tokens")
