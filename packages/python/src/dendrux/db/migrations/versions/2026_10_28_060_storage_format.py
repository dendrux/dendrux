"""Add recovery storage without activating uniqueness or format-2 writes.

Constant defaults are metadata-only on PostgreSQL 11+. Each statement is
bounded and independently restartable, including settings creation/seeding.
Downgrade is permitted only before activation has ever started.

Revision ID: f9a0b1c2d3e4
Revises: e8f9a0b1c2d3
"""

from contextlib import nullcontext

from alembic import op
import sqlalchemy as sa

from dendrux.db._migration_ddl import bounded_step, online
from dendrux.db._storage_schema import ADDITIONS

revision = "f9a0b1c2d3e4"
down_revision = "e8f9a0b1c2d3"
branch_labels = None
depends_on = None


def _step(statement, step):
    bounded_step(op.get_bind(), statement, revision=revision, step=step)


def upgrade():
    bind = op.get_bind()
    pg = bind.dialect.name == "postgresql"
    with online(op.get_context()) if pg else nullcontext():
        for table, columns in ADDITIONS.items():
            present = set() if pg else {c["name"] for c in sa.inspect(bind).get_columns(table)}
            for name, definition in columns.items():
                if name in present:
                    continue
                if not pg:
                    definition = definition.replace("JSONB", "JSON").replace("TIMESTAMPTZ", "DATETIME")
                guard = " IF NOT EXISTS" if pg else ""
                _step(f"ALTER TABLE {table} ADD COLUMN{guard} {name} {definition}", f"add.{table}.{name}")
        timestamp = "TIMESTAMPTZ" if pg else "DATETIME"
        _step(
            "CREATE TABLE IF NOT EXISTS dendrux_settings ("
            "key VARCHAR(64) PRIMARY KEY, value VARCHAR(255) NOT NULL, "
            f"updated_at {timestamp} NOT NULL)",
            "create.dendrux_settings",
        )
        _step(
            "INSERT INTO dendrux_settings (key, value, updated_at) "
            "VALUES ('storage_format', '1', CURRENT_TIMESTAMP) ON CONFLICT (key) DO NOTHING",
            "seed.storage_format",
        )


def downgrade():
    bind = op.get_bind()
    if sa.inspect(bind).has_table("dendrux_settings"):
        started = bind.execute(sa.text(
            "SELECT 1 FROM dendrux_settings WHERE key = 'activate_step'"
        )).scalar_one_or_none()
        if started is not None:
            raise RuntimeError(
                "Cannot downgrade after activation has started. "
                "Restore the pre-activation backup to roll back."
            )
    pg = bind.dialect.name == "postgresql"
    with online(op.get_context()) if pg else nullcontext():
        for table, columns in reversed(tuple(ADDITIONS.items())):
            present = set() if pg else {c["name"] for c in sa.inspect(bind).get_columns(table)}
            for name in reversed(columns):
                if pg or name in present:
                    guard = " IF EXISTS" if pg else ""
                    _step(f"ALTER TABLE {table} DROP COLUMN{guard} {name}", f"drop.{table}.{name}")
        _step("DROP TABLE IF EXISTS dendrux_settings", "drop.dendrux_settings")
