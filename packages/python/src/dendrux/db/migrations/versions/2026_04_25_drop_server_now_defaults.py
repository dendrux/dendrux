"""Drop server-side ``now()`` defaults; rely on Python ``_utc_now``.

Phase 3 of the aware-UTC sprint. The model now declares
``default=_utc_now`` / ``onupdate=_utc_now`` on every timestamp column
(see ``db/models.py``), so the database's own ``now()`` is never
consulted by application writes. Dropping the residual server defaults
keeps the schema honest:

- The dialect-level ambiguity around ``now()`` (PG returns
  ``transaction_timestamp`` in the session timezone; SQLite returns a
  naive UTC string) no longer matters — the only writer is Python.
- A row inserted via raw SQL on Postgres without a timestamp would
  previously inherit ``now()``; after this migration it raises a
  NOT NULL violation, surfacing the bypass instead of masking it.

Postgres only — SQLite stores defaults inline in CREATE TABLE and would
require a full table rebuild to remove them. Since the Python-side
default always wins on application writes, the dormant SQLite defaults
are harmless.

Revision ID: f3a4b5c6d7e8
Revises: e2f3a4b5c6d7
Create Date: 2026-04-25
"""

from __future__ import annotations

from typing import Union

from alembic import op

revision: str = "f3a4b5c6d7e8"
down_revision: Union[str, None] = "e2f3a4b5c6d7"
branch_labels: Union[str, tuple[str, ...], None] = None
depends_on: Union[str, tuple[str, ...], None] = None


# 9 columns, not 10 — ``agent_runs.last_progress_at`` was always nullable
# with no ``server_default=func.now()`` (it's set explicitly in
# ``create_run`` and ``touch_progress``), so there is no default to drop
# on that column. The companion migration ``e2f3a4b5c6d7`` has 10 entries
# because every datetime column needed the TIMESTAMPTZ type change.
_PG_COLUMNS: list[tuple[str, str]] = [
    ("agent_runs", "created_at"),
    ("agent_runs", "updated_at"),
    ("react_traces", "created_at"),
    ("react_traces", "updated_at"),
    ("tool_calls", "created_at"),
    ("tool_calls", "updated_at"),
    ("token_usage", "created_at"),
    ("llm_interactions", "created_at"),
    ("run_events", "created_at"),
]


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return
    for table, column in _PG_COLUMNS:
        op.execute(f"ALTER TABLE {table} ALTER COLUMN {column} DROP DEFAULT")


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return
    for table, column in _PG_COLUMNS:
        op.execute(f"ALTER TABLE {table} ALTER COLUMN {column} SET DEFAULT now()")
