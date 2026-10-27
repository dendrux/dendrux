"""Bound maintenance pages without indexing heartbeat columns.

Postgres builds/drops concurrently on the direct migration connection and
repairs invalid interrupted builds. SQLite uses ordinary transactional DDL.

Revision ID: e8f9a0b1c2d3
Revises: d7e8f9a0b1c2
"""

from contextlib import nullcontext

import sqlalchemy as sa
from alembic import op

from dendrux.db._index_check import index_matches
from dendrux.db._migration_ddl import (
    bounded_step,
    build_index_concurrently,
    drop_index_concurrently,
    online,
)

revision = "e8f9a0b1c2d3"
down_revision = "d7e8f9a0b1c2"
branch_labels = None
depends_on = None

_INDEXES = (
    ("ix_agent_runs_sweep_running", "agent_runs", "status, id", ""),
    ("ix_agent_runs_sweep_waiting", "agent_runs", "status, status_changed_at, id", ""),
    ("ix_run_events_started", "run_events", "agent_run_id", " WHERE event_type = 'run.started'"),
)


def upgrade() -> None:
    bind = op.get_bind()
    pg = bind.dialect.name == "postgresql"
    if pg:
        _online_pg_clock()
    else:
        columns = {c["name"] for c in sa.inspect(bind).get_columns("agent_runs")}
        if "status_changed_at" not in columns:
            op.add_column("agent_runs", sa.Column("status_changed_at", sa.DateTime(timezone=True)))
        op.execute(
            "UPDATE agent_runs SET status_changed_at = updated_at WHERE status_changed_at IS NULL"
        )
        _alter_clock(drop=False)
    with online(op.get_context()) if pg else nullcontext():
        for name, table, keys, predicate in _INDEXES:
            concurrent = " CONCURRENTLY" if pg else ""
            statement = f"CREATE INDEX{concurrent} IF NOT EXISTS {name} ON {table} ({keys}){predicate}"
            if pg:
                build_index_concurrently(bind, name, table, statement, revision=revision)
            else:
                op.execute(statement)
            expected_predicate = predicate
            if pg and predicate:
                column, value = (
                    ("event_type", "run.started")
                    if table == "run_events"
                    else ("status", "running")
                )
                expected_predicate = f" WHERE (({column})::text = '{value}'::text)"
            if not index_matches(
                bind, name, table, f"CREATE INDEX {name} ON {table} ({keys}){expected_predicate}"
            ):
                raise RuntimeError(
                    f"Index {name} has an unexpected definition or is not ready. "
                    "Rebuild it with the migration's exact definition before retrying."
                )


def _exclusive_pg_step(statement: str, step: str, *, transactional: bool = False) -> None:
    """Bound DDL lock waits, including concurrent-index virtualxid waits.

    ``transactional`` keeps the statement in the enclosing transaction: a later
    revision may refuse a multi-revision downgrade, and the column drop and revision
    stamp must roll back with it.
    """
    bounded_step(
        op.get_bind(), statement, revision=revision, step=step, transactional=transactional
    )


def _online_pg_clock() -> None:
    """Commit short DDL and 5k-row batches on the direct migration connection.

    New inserts get a current clock from the default, installed atomically
    with the nullable column. SDK transitions explicitly set their clock;
    backfill never overwrites one. A validated CHECK avoids SET NOT NULL's
    table scan under ACCESS EXCLUSIVE (Postgres 12+).
    """
    with online(op.get_context()):
        _exclusive_pg_step(
            "ALTER TABLE agent_runs ADD COLUMN IF NOT EXISTS status_changed_at TIMESTAMPTZ, "
            "ALTER COLUMN status_changed_at SET DEFAULT now()",
            "add_status_changed_at",
        )
        # Freeze the initial high-water mark so continual inserts cannot
        # extend the migration indefinitely. Later inserts have the default.
        end = op.get_bind().execute(sa.text("SELECT max(id) FROM agent_runs")).scalar_one()
        after = None
        while end is not None:
            lower = "id > :after AND " if after is not None else ""
            last_id = op.get_bind().execute(
                sa.text(
                    "SELECT max(id) FROM (SELECT id FROM agent_runs WHERE "
                    + lower + "id <= :end ORDER BY id LIMIT 5000) AS batch"
                ),
                {"after": after, "end": end},
            ).scalar_one()
            if last_id is None:
                break
            # Explicit, bound range endpoints let the planner restrict the
            # UPDATE itself to this page. A CTE join can hash/scan all runs
            # for every batch even when its input has only 5k IDs.
            op.get_bind().execute(
                sa.text(
                    "UPDATE agent_runs SET status_changed_at = "
                    "COALESCE(updated_at, created_at, now()) WHERE "
                    + lower + "id <= :last AND status_changed_at IS NULL"
                ),
                {"after": after, "last": last_id},
            )
            after = last_id
        constraint = "ck_agent_runs_status_clock_not_null"
        # An interrupted run may already have installed/validated the check.
        exists = (
            op.get_bind()
            .execute(
                sa.text(
                    "SELECT 1 FROM pg_constraint WHERE conrelid='agent_runs'::regclass "
                    "AND conname=:name"
                ),
                {"name": constraint},
            )
            .scalar_one_or_none()
        )
        if exists is None:
            _exclusive_pg_step(
                f"ALTER TABLE agent_runs ADD CONSTRAINT {constraint} "
                "CHECK (status_changed_at IS NOT NULL) NOT VALID",
                "add_status_clock_check",
            )
        op.execute(f"ALTER TABLE agent_runs VALIDATE CONSTRAINT {constraint}")
        _exclusive_pg_step(
            "ALTER TABLE agent_runs ALTER COLUMN status_changed_at SET NOT NULL",
            "set_status_changed_at_not_null",
        )
        _exclusive_pg_step(
            f"ALTER TABLE agent_runs DROP CONSTRAINT {constraint}", "drop_status_clock_check"
        )


def downgrade() -> None:
    pg = op.get_bind().dialect.name == "postgresql"
    with online(op.get_context()) if pg else nullcontext():
        for name, _, _, _ in reversed(_INDEXES):
            if pg:
                drop_index_concurrently(op.get_bind(), name, revision=revision)
            else:
                op.execute(f"DROP INDEX IF EXISTS {name}")
    if pg:
        _exclusive_pg_step(
            "ALTER TABLE agent_runs DROP COLUMN status_changed_at",
            "drop_status_changed_at",
            transactional=True,
        )
    else:
        _alter_clock(drop=True)


def _alter_clock(*, drop: bool) -> None:
    bind = op.get_bind()
    # Alembic's SQLite reflection omits expression indexes. Preserve those
    # definitions across the table rebuild required for nullability changes.
    indexes = {}
    if bind.dialect.name == "sqlite":
        indexes = dict(
            bind.execute(
                sa.text(
                    "SELECT name, sql FROM sqlite_master WHERE type='index' "
                    "AND tbl_name='agent_runs' AND sql IS NOT NULL"
                )
            ).all()
        )
    with op.batch_alter_table("agent_runs") as batch:
        if drop:
            batch.drop_column("status_changed_at")
        else:
            batch.alter_column(
                "status_changed_at",
                existing_type=sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.func.now(),
            )
    if indexes:
        retained = set(
            bind.execute(
                sa.text(
                    "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='agent_runs'"
                )
            ).scalars()
        )
        for name, ddl in indexes.items():
            if name not in retained:
                op.execute(ddl)
