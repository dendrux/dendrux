"""Add retry_of_run_id column to agent_runs.

Revision ID: f6a7b8c9d0e1
Revises: e5f6a7b8c9d0
Create Date: 2026-04-05
"""

from __future__ import annotations

from typing import Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "f6a7b8c9d0e1"
down_revision: Union[str, None] = "e5f6a7b8c9d0"
branch_labels: Union[str, tuple[str, ...], None] = None
depends_on: Union[str, tuple[str, ...], None] = None


def upgrade() -> None:
    op.add_column(
        "agent_runs",
        sa.Column("retry_of_run_id", sa.String(26), sa.ForeignKey("agent_runs.id", ondelete="SET NULL"), nullable=True),
    )
    op.create_index(
        "ix_agent_runs_retry_of_run_id",
        "agent_runs",
        ["retry_of_run_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_agent_runs_retry_of_run_id", table_name="agent_runs")
    op.drop_column("agent_runs", "retry_of_run_id")
