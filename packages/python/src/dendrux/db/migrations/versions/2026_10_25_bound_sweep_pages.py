"""Bound maintenance pages by status, age and id; index event classification.

Postgres builds/drops concurrently on the direct migration connection. If an
interrupted build leaves an invalid index, verification refuses to continue;
repair that index before retrying. SQLite uses ordinary transactional DDL.

Revision ID: e8f9a0b1c2d3
Revises: d7e8f9a0b1c2
"""

from contextlib import nullcontext

from alembic import op

from dendrux.db._index_check import require_index

revision = "e8f9a0b1c2d3"
down_revision = "d7e8f9a0b1c2"
branch_labels = None
depends_on = None

_INDEXES = (
    ("ix_agent_runs_sweep_progress", "agent_runs", "status, coalesce(last_progress_at, created_at), id"),
    ("ix_agent_runs_sweep_waiting", "agent_runs", "status, updated_at, id"),
    ("ix_run_events_run_type", "run_events", "agent_run_id, event_type"),
)


def upgrade() -> None:
    pg = op.get_bind().dialect.name == "postgresql"
    with op.get_context().autocommit_block() if pg else nullcontext():
        for name, table, keys in _INDEXES:
            concurrent = " CONCURRENTLY" if pg else ""
            op.execute(f"CREATE INDEX{concurrent} IF NOT EXISTS {name} ON {table} ({keys})")
            require_index(op.get_bind(), name, table, f"CREATE INDEX {name} ON {table} ({keys})")


def downgrade() -> None:
    pg = op.get_bind().dialect.name == "postgresql"
    with op.get_context().autocommit_block() if pg else nullcontext():
        for name, _, _ in reversed(_INDEXES):
            concurrent = " CONCURRENTLY" if pg else ""
            op.execute(f"DROP INDEX{concurrent} IF EXISTS {name}")
