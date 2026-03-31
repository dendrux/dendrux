"""add run_events table for durable observability

Revision ID: b7c8d9e0f1a2
Revises: a1b2c3d4e5f6
Create Date: 2026-03-15 22:00:00.000000+00:00
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "b7c8d9e0f1a2"
down_revision: Union[str, None] = "a1b2c3d4e5f6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "run_events",
        sa.Column("id", sa.String(length=26), nullable=False),
        sa.Column("agent_run_id", sa.String(length=26), nullable=False),
        sa.Column("event_type", sa.String(length=50), nullable=False),
        sa.Column("sequence_index", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("iteration_index", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("correlation_id", sa.String(length=26), nullable=True),
        sa.Column("data", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=True),
        sa.ForeignKeyConstraint(
            ["agent_run_id"],
            ["agent_runs.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_run_events_agent_run_id_seq", "run_events", ["agent_run_id", "sequence_index"])
    op.create_index("ix_run_events_event_type", "run_events", ["event_type"])


def downgrade() -> None:
    op.drop_index("ix_run_events_event_type", table_name="run_events")
    op.drop_index("ix_run_events_agent_run_id_seq", table_name="run_events")
    op.drop_table("run_events")
