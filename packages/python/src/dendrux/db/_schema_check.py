"""Read-only check that a non-SQLite database has the tables and columns we need.

Postgres users run Alembic after upgrading Dendrux. If they forget, the first
query fails deep inside a run with a raw ``UndefinedColumnError``. This check
runs once per engine, before any run is created, and turns that into a single
actionable :class:`~dendrux.errors.SchemaOutOfDateError`.

It inspects the live tables and columns and verifies that every table and
column in ``Base.metadata`` is present, by exact name. It does not compare
column types, nullability or defaults. It deliberately ignores Alembic's
version table (apps may create the schema themselves), extra tables and
columns, and indexes. It never writes or issues DDL. SQLite is skipped: it
upgrades itself in place (:mod:`dendrux.db._sqlite_sync`).
``DENDRUX_SKIP_SCHEMA_CHECK=1`` disables it. If the database refuses the
catalog read (for example for lack of privileges) the check logs a warning and
lets startup continue; connection failures propagate.
"""

from __future__ import annotations

import logging
import os
import weakref
from typing import TYPE_CHECKING

from sqlalchemy import inspect
from sqlalchemy.exc import DBAPIError

from dendrux.db.models import Base
from dendrux.errors import SchemaNotInitializedError, SchemaOutOfDateError

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping

    from sqlalchemy import Connection, Engine
    from sqlalchemy.ext.asyncio import AsyncEngine

__all__ = [
    "SKIP_ENV",
    "compare_schema",
    "ensure_schema_current",
    "read_actual_schema",
    "was_verified",
]

logger = logging.getLogger(__name__)
_VERIFIED: weakref.WeakSet[Engine] = weakref.WeakSet()

SKIP_ENV = "DENDRUX_SKIP_SCHEMA_CHECK"
_TRUTHY = {"1", "true", "yes", "on"}


def _skipped() -> bool:
    return os.environ.get(SKIP_ENV, "").strip().lower() in _TRUTHY


def compare_schema(actual: Mapping[str, Collection[str]]) -> tuple[list[str], list[str]]:
    """Compare the models with ``actual`` (table name -> column names).

    Returns ``(missing_tables, missing_columns)``; columns are reported as
    ``"table.column"`` and only for tables that exist. Names must match
    exactly, as the database stores them (an uppercase or quoted name is a
    different object from the lowercase one Dendrux queries). Anything extra
    in ``actual`` is ignored.
    """
    present = {name: set(cols) for name, cols in actual.items()}
    missing_tables: list[str] = []
    missing_columns: list[str] = []
    for table in Base.metadata.sorted_tables:
        columns = present.get(table.name)
        if columns is None:
            missing_tables.append(table.name)
            continue
        missing_columns.extend(
            f"{table.name}.{column.name}" for column in table.columns if column.name not in columns
        )
    return missing_tables, missing_columns


def read_actual_schema(sync_conn: Connection) -> dict[str, list[str]]:
    """Table name -> column names for every model table present in the database."""
    inspector = inspect(sync_conn)
    wanted = {t.name for t in Base.metadata.tables.values()}
    present = [name for name in inspector.get_table_names() if name in wanted]
    if not present:
        return {}
    columns = inspector.get_multi_columns(filter_names=present)
    return {table: [c["name"] for c in cols] for (_schema, table), cols in columns.items()}


def was_verified(engine: AsyncEngine) -> bool:
    """True if :func:`ensure_schema_current` fully checked and passed this engine."""
    return engine.sync_engine in _VERIFIED


async def ensure_schema_current(engine: AsyncEngine) -> None:
    """Raise if the database lacks tables or columns the models define.

    A no-op on SQLite and when ``DENDRUX_SKIP_SCHEMA_CHECK=1``. Read-only: one
    connection, a table listing and one batched column listing. It verifies
    presence by name only; column types and defaults are not compared. If the
    catalog read fails with a database error (such as insufficient
    privileges) a warning is logged and the check is skipped; failures to
    connect propagate.

    Raises:
        SchemaNotInitializedError: The database has none of the Dendrux tables.
        SchemaOutOfDateError: Some tables or columns are missing; run
            ``dendrux db migrate``.
    """
    if engine.dialect.name == "sqlite" or _skipped():
        return
    async with engine.connect() as conn:
        try:
            actual = await conn.run_sync(read_actual_schema)
        except DBAPIError as exc:
            if exc.connection_invalidated:
                raise
            logger.warning("schema check skipped: %s", type(exc.orig or exc).__name__)
            return
    if not actual:
        raise SchemaNotInitializedError()
    missing_tables, missing_columns = compare_schema(actual)
    if missing_tables or missing_columns:
        raise SchemaOutOfDateError(missing_tables, missing_columns)
    _VERIFIED.add(engine.sync_engine)
