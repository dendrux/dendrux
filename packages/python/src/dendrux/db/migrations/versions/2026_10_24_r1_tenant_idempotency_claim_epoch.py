"""R1 (0.5.0): claim fencing epoch and tenant-scoped idempotency.

Adds ``agent_runs.claim_epoch`` (C1): ``INTEGER NOT NULL DEFAULT 0``. Every
claim increments it and status writes check it, so a worker that lost its
claim cannot finalize the run. The constant server default makes the column
add metadata-only on Postgres 11+ and lets 0.4.1 workers keep inserting rows
during a rolling deploy.

TODO(T1, R1): add the tenant-scoped idempotency index work to this same
revision: drop ``ix_agent_runs_idempotency_key``, create
``uq_agent_runs_tenant_idempotency_key`` on
``(COALESCE(tenant_id, ''), idempotency_key) WHERE idempotency_key IS NOT NULL``,
and create the ``agent_runs (tenant_id, created_at DESC)`` index (D-124).
Mirror each step in ``downgrade()``.

Revision ID: d7e8f9a0b1c2
Revises: c6d7e8f9a0b1
Create Date: 2026-10-24
"""

from __future__ import annotations

from typing import Union

from alembic import op
import sqlalchemy as sa

revision: str = "d7e8f9a0b1c2"
down_revision: Union[str, None] = "c6d7e8f9a0b1"
branch_labels: Union[str, tuple[str, ...], None] = None
depends_on: Union[str, tuple[str, ...], None] = None


def upgrade() -> None:
    op.add_column(
        "agent_runs",
        sa.Column(
            "claim_epoch",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("0"),
        ),
    )
    # TODO(T1, R1): tenant-scoped idempotency indexes go here.


def downgrade() -> None:
    # TODO(T1, R1): reverse the tenant-scoped idempotency indexes here first.
    op.drop_column("agent_runs", "claim_epoch")
