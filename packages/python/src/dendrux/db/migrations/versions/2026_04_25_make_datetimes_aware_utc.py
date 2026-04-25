"""Make all datetime columns timezone-aware (TIMESTAMPTZ on PG).

Dendrux's invariant is "all timestamps are aware-UTC." Until this migration,
the schema declared plain ``DateTime`` (TIMESTAMP WITHOUT TIME ZONE on
Postgres), but writes used ``datetime.now(UTC)`` (aware). Postgres rejected
those writes outright; SQLite tolerated them but stripped tzinfo silently.

This migration aligns the schema with the write contract:

- Postgres: convert each ``timestamp`` column to ``timestamptz`` and
  re-interpret existing values as UTC. Equivalent to
  ``ALTER COLUMN ... TYPE timestamptz USING ... AT TIME ZONE 'UTC'``.
  Existing stored values are assumed to be UTC (consistent with how the
  app intended to write them — even if the column type couldn't enforce it).
- SQLite: no DDL change required. SQLite has no native TIMESTAMPTZ; the
  ``timezone=True`` flag on SQLAlchemy is purely declarative on this
  backend. The store-boundary normalizer in ``runtime/state.py`` is what
  actually guarantees aware-UTC reads on SQLite.

Revision ID: e2f3a4b5c6d7
Revises: c1d2e3f4a5b6
Create Date: 2026-04-25
"""

from __future__ import annotations

from typing import Union

from alembic import op

revision: str = "e2f3a4b5c6d7"
down_revision: Union[str, None] = "c1d2e3f4a5b6"
branch_labels: Union[str, tuple[str, ...], None] = None
depends_on: Union[str, tuple[str, ...], None] = None


_PG_COLUMNS: list[tuple[str, str]] = [
    ("agent_runs", "last_progress_at"),
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
        op.execute(
            f'ALTER TABLE {table} '
            f'ALTER COLUMN {column} TYPE TIMESTAMPTZ '
            f'USING {column} AT TIME ZONE \'UTC\''
        )


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return
    for table, column in _PG_COLUMNS:
        op.execute(
            f'ALTER TABLE {table} '
            f'ALTER COLUMN {column} TYPE TIMESTAMP '
            f'USING {column} AT TIME ZONE \'UTC\''
        )
