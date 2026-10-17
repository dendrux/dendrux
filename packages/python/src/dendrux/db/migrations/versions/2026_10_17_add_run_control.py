"""Run control: pause_requested flag on agent_runs and the run_inbox table.

``pause_requested`` is the cooperative pause flag set by
``Agent.request_pause``; existing rows default to false. ``run_inbox`` holds
notes submitted with ``Agent.submit_note``; rows go with their run
(``ON DELETE CASCADE``). The ``metadata`` column maps to the ``metadata_``
model attribute (``metadata`` is reserved on declarative models).

Revision ID: b5c6d7e8f9a0
Revises: a4b5c6d7e8f9
Create Date: 2026-10-17
"""

from __future__ import annotations

from typing import Union

from alembic import op
import sqlalchemy as sa

revision: str = "b5c6d7e8f9a0"
down_revision: Union[str, None] = "a4b5c6d7e8f9"
branch_labels: Union[str, tuple[str, ...], None] = None
depends_on: Union[str, tuple[str, ...], None] = None


def upgrade() -> None:
    op.add_column(
        "agent_runs",
        sa.Column(
            "pause_requested",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
    )
    op.create_table(
        "run_inbox",
        sa.Column("id", sa.String(26), primary_key=True),
        sa.Column(
            "run_id",
            sa.String(26),
            sa.ForeignKey("agent_runs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("metadata", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("delivered_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("delivered_iteration", sa.Integer(), nullable=True),
    )
    op.create_index(
        "ix_run_inbox_run_pending",
        "run_inbox",
        ["run_id", "delivered_at", "created_at", "id"],
    )


def downgrade() -> None:
    op.drop_index("ix_run_inbox_run_pending", table_name="run_inbox")
    op.drop_table("run_inbox")
    op.drop_column("agent_runs", "pause_requested")
