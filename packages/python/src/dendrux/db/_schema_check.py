"""Read-only check that a non-SQLite database has the tables and columns we need.

Postgres users run Alembic after upgrading Dendrux. If they forget, the first
query fails deep inside a run with a raw ``UndefinedColumnError``. This check
runs once per engine, before any run is created, and turns that into a single
actionable :class:`~dendrux.errors.SchemaOutOfDateError`.

It inspects the live tables and columns and compares them with
``Base.metadata``. It deliberately ignores Alembic's version table (apps may
create the schema themselves), extra tables and columns, and indexes. It never
writes or issues DDL. SQLite is skipped: it upgrades itself in place
(:mod:`dendrux.db._sqlite_sync`). ``DENDRUX_SKIP_SCHEMA_CHECK=1`` disables it.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from sqlalchemy import inspect

from dendrux.db.models import Base
from dendrux.errors import SchemaNotInitializedError, SchemaOutOfDateError

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping

    from sqlalchemy import Connection
    from sqlalchemy.ext.asyncio import AsyncEngine

__all__ = [
    "SKIP_ENV",
    "compare_schema",
    "ensure_schema_current",
    "read_actual_schema",
]

SKIP_ENV = "DENDRUX_SKIP_SCHEMA_CHECK"
_TRUTHY = {"1", "true", "yes", "on"}


def _skipped() -> bool:
    return os.environ.get(SKIP_ENV, "").strip().lower() in _TRUTHY


def compare_schema(actual: Mapping[str, Collection[str]]) -> tuple[list[str], list[str]]:
    """Compare the models with ``actual`` (table name -> column names).

    Returns ``(missing_tables, missing_columns)``; columns are reported as
    ``"table.column"`` and only for tables that exist. Names compare
    case-insensitively. Anything extra in ``actual`` is ignored.
    """
    present = {name.lower(): {c.lower() for c in cols} for name, cols in actual.items()}
    missing_tables: list[str] = []
    missing_columns: list[str] = []
    for table in Base.metadata.sorted_tables:
        columns = present.get(table.name.lower())
        if columns is None:
            missing_tables.append(table.name)
            continue
        missing_columns.extend(
            f"{table.name}.{column.name}"
            for column in table.columns
            if column.name.lower() not in columns
        )
    return missing_tables, missing_columns


def read_actual_schema(sync_conn: Connection) -> dict[str, list[str]]:
    """Table name -> column names for every model table present in the database."""
    inspector = inspect(sync_conn)
    wanted = {t.name.lower() for t in Base.metadata.tables.values()}
    present = [name for name in inspector.get_table_names() if name.lower() in wanted]
    if not present:
        return {}
    columns = inspector.get_multi_columns(filter_names=present)
    return {table: [c["name"] for c in cols] for (_schema, table), cols in columns.items()}


async def ensure_schema_current(engine: AsyncEngine) -> None:
    """Raise if the database lacks tables or columns the models define.

    A no-op on SQLite and when ``DENDRUX_SKIP_SCHEMA_CHECK=1``. Read-only: one
    connection, a table listing and one batched column listing.

    Raises:
        SchemaNotInitializedError: The database has none of the Dendrux tables.
        SchemaOutOfDateError: Some tables or columns are missing; run
            ``dendrux db migrate``.
    """
    if engine.dialect.name == "sqlite" or _skipped():
        return
    async with engine.connect() as conn:
        actual = await conn.run_sync(read_actual_schema)
    if not actual:
        raise SchemaNotInitializedError()
    missing_tables, missing_columns = compare_schema(actual)
    if missing_tables or missing_columns:
        raise SchemaOutOfDateError(missing_tables, missing_columns)
