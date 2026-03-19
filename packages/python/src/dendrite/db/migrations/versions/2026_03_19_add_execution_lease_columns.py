"""add execution lease columns for Sprint 4 crash-safe coordination

Revision ID: d4e5f6a7b8c9
Revises: c3d4e5f6a7b8
Create Date: 2026-03-19 00:00:00.000000+00:00
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "d4e5f6a7b8c9"
down_revision: Union[str, None] = "c3d4e5f6a7b8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("agent_runs", sa.Column("executor_id", sa.String(255), nullable=True))
    op.add_column("agent_runs", sa.Column("lease_nonce", sa.String(26), nullable=True))
    op.add_column("agent_runs", sa.Column("heartbeat_at", sa.DateTime(), nullable=True))
    op.add_column(
        "agent_runs",
        sa.Column("retry_count", sa.Integer(), nullable=False, server_default="0"),
    )
    op.add_column(
        "agent_runs",
        sa.Column("max_retries", sa.Integer(), nullable=False, server_default="3"),
    )


def downgrade() -> None:
    op.drop_column("agent_runs", "max_retries")
    op.drop_column("agent_runs", "retry_count")
    op.drop_column("agent_runs", "heartbeat_at")
    op.drop_column("agent_runs", "lease_nonce")
    op.drop_column("agent_runs", "executor_id")
