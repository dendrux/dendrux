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
  for tenant-scoped listings.

Every index step is idempotent (``IF [NOT] EXISTS``). A database stamped at
this revision by an earlier build that only added ``claim_epoch`` recovers
with ``dendrux db migrate`` after ``alembic downgrade -1``; ``dendrux db check``
and the startup schema check report the missing index.

On Postgres every step runs on the direct migration connection outside a
transaction. The column add and the constraint drop wait at most 5 seconds for
their lock (``lock_timeout``); the indexes are built and dropped
``CONCURRENTLY``, so writers keep running, and their waits for open
transactions share the same 5 second bound. A step that cannot get its lock
refuses with a retry message. Rerunning the migration skips what is already
done and rebuilds an index that an interrupted concurrent build left invalid.
SQLite runs the same steps as ordinary transactional DDL.

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

from dendrux.db._index_check import require_index
from dendrux.db._migration_ddl import (
    bounded_step,
    build_index_concurrently,
    drop_index_concurrently,
    online,
    sqlstate,
)

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


_TABLE = "agent_runs"


def _has_column(table: str, column: str) -> bool:
    inspector = sa.inspect(op.get_bind())
    return any(c["name"] == column for c in inspector.get_columns(table))


def _shared_keys_error(count: int) -> RuntimeError:
    return RuntimeError(
        f"Cannot downgrade R1: {count} idempotency key(s) are used by more than one "
        "tenant, and the 0.4.1 schema requires globally unique keys. Clear "
        "idempotency_key on the extra runs (UPDATE agent_runs SET idempotency_key = NULL "
        "WHERE ...) and run the downgrade again."
    )


def _concurrently(ddl: str) -> str:
    return ddl.replace(" INDEX IF NOT EXISTS ", " INDEX CONCURRENTLY IF NOT EXISTS ", 1)


def _step(statement: str, step: str) -> None:
    bounded_step(op.get_bind(), statement, revision=revision, step=step)


def _pg_build_index(name: str, ddl: str) -> None:
    bind = op.get_bind()
    build_index_concurrently(bind, name, _TABLE, _concurrently(ddl), revision=revision)
    require_index(bind, name, _TABLE, ddl)


def _pg_has_constraint(name: str) -> bool:
    found = op.get_bind().execute(
        sa.text(
            "SELECT 1 FROM pg_constraint WHERE conrelid = 'agent_runs'::regclass "
            "AND conname = :name"
        ),
        {"name": name},
    )
    return found.scalar_one_or_none() is not None


def _upgrade_pg() -> None:
    with online(op.get_context()):
        if not _has_column(_TABLE, "claim_epoch"):
            _step(
                "ALTER TABLE agent_runs ADD COLUMN IF NOT EXISTS claim_epoch "
                "INTEGER NOT NULL DEFAULT 0",
                "add_claim_epoch",
            )
        _pg_build_index("uq_agent_runs_tenant_idempotency_key", _TENANT_KEY_INDEX)
        _pg_build_index("ix_agent_runs_tenant_id_created_at", _TENANT_CREATED_INDEX)
        # The tenant-scoped unique index is valid, so uniqueness is never unenforced.
        drop_index_concurrently(op.get_bind(), "ix_agent_runs_idempotency_key", revision=revision)
        if _pg_has_constraint("agent_runs_idempotency_key_key"):
            _step(
                "ALTER TABLE agent_runs DROP CONSTRAINT IF EXISTS agent_runs_idempotency_key_key",
                "drop_constraint.agent_runs_idempotency_key_key",
            )
        # Present on every migrated database already; the models now declare it too.
        _step(_concurrently(_RETRY_OF_INDEX), "build_index.ix_agent_runs_retry_of_run_id")


def _downgrade_pg() -> None:
    with online(op.get_context()):
        try:
            _pg_build_index("ix_agent_runs_idempotency_key", _GLOBAL_KEY_INDEX)
        except sa.exc.DBAPIError as error:
            if sqlstate(error) != "23505":
                raise
            # A writer shared a key after the check. The failed build left an invalid
            # unique index that would still reject cross-tenant keys: remove it.
            drop_index_concurrently(
                op.get_bind(), "ix_agent_runs_idempotency_key", revision=revision
            )
            shared = op.get_bind().execute(sa.text(_SHARED_KEYS)).scalar_one()
            raise _shared_keys_error(shared) from None
        for name in ("uq_agent_runs_tenant_idempotency_key", "ix_agent_runs_tenant_id_created_at"):
            drop_index_concurrently(op.get_bind(), name, revision=revision)
        if _has_column(_TABLE, "claim_epoch"):
            _step("ALTER TABLE agent_runs DROP COLUMN IF EXISTS claim_epoch", "drop_claim_epoch")


def upgrade() -> None:
    if op.get_bind().dialect.name == "postgresql":
        _upgrade_pg()
        return
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
    require_index(
        op.get_bind(), "uq_agent_runs_tenant_idempotency_key", "agent_runs", _TENANT_KEY_INDEX
    )
    op.execute(_TENANT_CREATED_INDEX)
    require_index(
        op.get_bind(), "ix_agent_runs_tenant_id_created_at", "agent_runs", _TENANT_CREATED_INDEX
    )
    op.execute("DROP INDEX IF EXISTS ix_agent_runs_idempotency_key")
    op.execute(_RETRY_OF_INDEX)


def downgrade() -> None:
    # Before any commit: a refusal also rolls back a later revision's downgrade.
    shared = op.get_bind().execute(sa.text(_SHARED_KEYS)).scalar_one()
    if shared:
        raise _shared_keys_error(shared)
    if op.get_bind().dialect.name == "postgresql":
        _downgrade_pg()
        return
    op.execute(_GLOBAL_KEY_INDEX)
    require_index(op.get_bind(), "ix_agent_runs_idempotency_key", "agent_runs", _GLOBAL_KEY_INDEX)
    op.execute("DROP INDEX IF EXISTS uq_agent_runs_tenant_idempotency_key")
    op.execute("DROP INDEX IF EXISTS ix_agent_runs_tenant_id_created_at")
    if _has_column("agent_runs", "claim_epoch"):
        op.drop_column("agent_runs", "claim_epoch")
