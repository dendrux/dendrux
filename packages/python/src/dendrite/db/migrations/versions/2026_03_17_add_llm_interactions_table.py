"""add llm_interactions table for evidence layer

Revision ID: c3d4e5f6a7b8
Revises: b7c8d9e0f1a2
Create Date: 2026-03-17 00:00:00.000000+00:00
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "c3d4e5f6a7b8"
down_revision: Union[str, None] = "b7c8d9e0f1a2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "llm_interactions",
        sa.Column("id", sa.String(length=26), nullable=False),
        sa.Column("agent_run_id", sa.String(length=26), nullable=False),
        sa.Column("iteration_index", sa.Integer(), nullable=False),
        sa.Column("model", sa.String(length=255), nullable=True),
        sa.Column("provider", sa.String(length=100), nullable=True),
        sa.Column("semantic_request", sa.JSON(), nullable=True),
        sa.Column("semantic_response", sa.JSON(), nullable=True),
        sa.Column("provider_request", sa.JSON(), nullable=True),
        sa.Column("provider_response", sa.JSON(), nullable=True),
        sa.Column("input_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("output_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("cost_usd", sa.Numeric(precision=10, scale=6), nullable=True),
        sa.Column("duration_ms", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=True),
        sa.ForeignKeyConstraint(
            ["agent_run_id"],
            ["agent_runs.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_llm_interactions_agent_run_id", "llm_interactions", ["agent_run_id"])


def downgrade() -> None:
    op.drop_index("ix_llm_interactions_agent_run_id", table_name="llm_interactions")
    op.drop_table("llm_interactions")
