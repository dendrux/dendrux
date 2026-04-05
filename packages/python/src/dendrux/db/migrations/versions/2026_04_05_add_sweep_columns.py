"""add last_progress_at and failure_reason to agent_runs for stale-run sweep

Revision ID: d4e5f6a7b8c9
Revises: c3d4e5f6a7b8
Create Date: 2026-04-05 00:00:00.000000+00:00
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
    op.add_column(
        "agent_runs",
        sa.Column("last_progress_at", sa.DateTime(), nullable=True),
    )
    op.add_column(
        "agent_runs",
        sa.Column("failure_reason", sa.String(length=100), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("agent_runs", "failure_reason")
    op.drop_column("agent_runs", "last_progress_at")
