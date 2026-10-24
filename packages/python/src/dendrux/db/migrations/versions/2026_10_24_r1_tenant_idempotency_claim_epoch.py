"""R1 (0.5.0): claim fencing epoch and tenant-scoped idempotency.

- Adds ``agent_runs.claim_epoch`` (C1): ``INTEGER NOT NULL DEFAULT 0``. Every
  claim increments it and status writes check it, so a worker that lost its
  claim cannot finalize the run. The constant server default makes the column
  add metadata-only on Postgres 11+ and lets 0.4.1 workers keep inserting rows
  during a rolling deploy.
- Makes the idempotency key unique per tenant instead of globally: creates
  ``uq_agent_runs_tenant_idempotency_key`` on
  ``(COALESCE(tenant_id, ''), idempotency_key) WHERE idempotency_key IS NOT NULL``,
  then drops the global ``ix_agent_runs_idempotency_key`` (and the
  ``agent_runs_idempotency_key_key`` constraint a ``create_all`` schema has).
  The new index is built before the old one is dropped, so uniqueness is
  never unenforced. Existing rows always fit: globally unique keys are unique
  per tenant too.
- Adds ``ix_agent_runs_tenant_id_created_at`` on ``(tenant_id, created_at DESC)``
  for tenant-scoped listings (D-124).

Every index step is idempotent (``IF [NOT] EXISTS``). A database stamped at
this revision by an earlier build that only added ``claim_epoch`` recovers
with ``dendrux db migrate`` after ``alembic downgrade -1``; ``dendrux db check``
and the startup schema check report the missing index.

The index builds take a ``SHARE`` lock on ``agent_runs`` (writes wait, reads
do not) for as long as the build runs. On very large tables, build the two
indexes first with ``CREATE ... INDEX CONCURRENTLY IF NOT EXISTS`` using the
same names and definitions; this migration then skips them.

Downgrade refuses (and changes nothing) when two tenants already share an
idempotency key, because the global unique index of 0.4.1 cannot be rebuilt
over them.

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

_TENANT_KEY_INDEX = (
    "CREATE UNIQUE INDEX IF NOT EXISTS uq_agent_runs_tenant_idempotency_key "
    "ON agent_runs (coalesce(tenant_id, ''), idempotency_key) "
    "WHERE idempotency_key IS NOT NULL"
)
_TENANT_CREATED_INDEX = (
    "CREATE INDEX IF NOT EXISTS ix_agent_runs_tenant_id_created_at "
    "ON agent_runs (tenant_id, created_at DESC)"
)
_RETRY_OF_INDEX = (
    "CREATE INDEX IF NOT EXISTS ix_agent_runs_retry_of_run_id ON agent_runs (retry_of_run_id)"
)
_GLOBAL_KEY_INDEX = (
    "CREATE UNIQUE INDEX IF NOT EXISTS ix_agent_runs_idempotency_key "
    "ON agent_runs (idempotency_key)"
)
_SHARED_KEYS = (
    "SELECT COUNT(*) FROM (SELECT idempotency_key FROM agent_runs "
    "WHERE idempotency_key IS NOT NULL GROUP BY idempotency_key HAVING COUNT(*) > 1) AS d"
)


def _has_column(table: str, column: str) -> bool:
    inspector = sa.inspect(op.get_bind())
    return any(c["name"] == column for c in inspector.get_columns(table))


def upgrade() -> None:
    if not _has_column("agent_runs", "claim_epoch"):
        op.add_column(
            "agent_runs",
            sa.Column(
                "claim_epoch",
                sa.Integer(),
                nullable=False,
                server_default=sa.text("0"),
            ),
        )
    op.execute(_TENANT_KEY_INDEX)
    op.execute("DROP INDEX IF EXISTS ix_agent_runs_idempotency_key")
    if op.get_bind().dialect.name == "postgresql":
        op.execute(
            "ALTER TABLE agent_runs DROP CONSTRAINT IF EXISTS agent_runs_idempotency_key_key"
        )
    op.execute(_TENANT_CREATED_INDEX)
    # Present on every migrated database already; the models now declare it too.
    op.execute(_RETRY_OF_INDEX)


def downgrade() -> None:
    shared = op.get_bind().execute(sa.text(_SHARED_KEYS)).scalar_one()
    if shared:
        raise RuntimeError(
            f"Cannot downgrade R1: {shared} idempotency key(s) are used by more than one "
            "tenant, and the 0.4.1 schema requires globally unique keys. Clear "
            "idempotency_key on the extra runs (UPDATE agent_runs SET idempotency_key = NULL "
            "WHERE ...) and run the downgrade again."
        )
    op.execute(_GLOBAL_KEY_INDEX)
    op.execute("DROP INDEX IF EXISTS uq_agent_runs_tenant_idempotency_key")
    op.execute("DROP INDEX IF EXISTS ix_agent_runs_tenant_id_created_at")
    if _has_column("agent_runs", "claim_epoch"):
        op.drop_column("agent_runs", "claim_epoch")
