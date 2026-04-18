"""Add cancel_requested flag to agent_runs.

Cooperative cancellation signal. ``cancel_run()`` sets this to True;
the runner checks it at top-of-iteration and pre-pause checkpoints,
exiting cleanly as ``cancelled`` when observed. Cleared on terminal
finalize so a stale flag never lingers on a successful run.

Revision ID: c1d2e3f4a5b6
Revises: a8b9c0d1e2f3
Create Date: 2026-04-18
"""

from __future__ import annotations

from typing import Union

from alembic import op
import sqlalchemy as sa

revision: str = "c1d2e3f4a5b6"
down_revision: Union[str, None] = "a8b9c0d1e2f3"
branch_labels: Union[str, tuple[str, ...], None] = None
depends_on: Union[str, tuple[str, ...], None] = None


def upgrade() -> None:
    op.add_column(
        "agent_runs",
        sa.Column(
            "cancel_requested",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
    )


def downgrade() -> None:
    op.drop_column("agent_runs", "cancel_requested")
