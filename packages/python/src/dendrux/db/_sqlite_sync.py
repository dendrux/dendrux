"""Bring an existing SQLite file up to the current model schema.

``Base.metadata.create_all`` creates missing tables but never adds columns to
tables that already exist, so a SQLite file written by an older Dendrux
version breaks after an upgrade (``no such column``). Postgres users run
Alembic; SQLite is zero-config, so it needs an in-place sync instead.

The sync is additive only: it adds missing columns and indexes. It never
drops, renames or alters anything, and never touches tables outside
``Base.metadata``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sqlalchemy import UniqueConstraint
from sqlalchemy.exc import OperationalError
from sqlalchemy.schema import CreateColumn, CreateIndex

from dendrux.db.models import Base

if TYPE_CHECKING:
    from sqlalchemy import Column, Connection, Table

logger = logging.getLogger(__name__)


class SqliteSchemaSyncError(RuntimeError):
    """A SQLite database cannot be upgraded in place to the current schema."""


def create_sqlite_schema(sync_conn: Connection) -> None:
    """Create missing tables, then sync columns and indexes of existing ones.

    Pass to ``AsyncConnection.run_sync`` at every SQLite schema-init site.
    """
    try:
        Base.metadata.create_all(sync_conn)
    except OperationalError:
        # A concurrent startup created a table between create_all's existence
        # check and its CREATE TABLE; the retry sees it and skips.
        Base.metadata.create_all(sync_conn)
    sync_sqlite_schema(sync_conn)


def sync_sqlite_schema(sync_conn: Connection) -> list[str]:
    """Add model columns and indexes missing from existing SQLite tables.

    Returns a description of every change made (empty when already current).
    Idempotent and safe under concurrent startup. A no-op on non-SQLite
    connections.

    Raises:
        SqliteSchemaSyncError: A missing column cannot be added safely
            (NOT NULL without a server default, or a primary key), or SQLite
            rejected the ``ALTER TABLE``. Raised before any change is made
            for the unsafe-column case.
    """
    if sync_conn.dialect.name != "sqlite":
        return []

    tables = [t for t in Base.metadata.sorted_tables if _existing_columns(sync_conn, t)]

    plan = {
        t.name: [c for c in t.columns if c.name not in _existing_columns(sync_conn, t)]
        for t in tables
    }
    _raise_if_unsafe(tables, plan)

    changes: list[str] = []
    for table in tables:
        added = [c for c in plan[table.name] if _add_column(sync_conn, table, c)]
        changes.extend(f"added column {table.name}.{c.name}" for c in added)
        changes.extend(_ensure_unique_indexes(sync_conn, table, {c.name for c in added}))
        changes.extend(_ensure_indexes(sync_conn, table))

    for change in changes:
        logger.info("SQLite schema sync: %s", change)
    return changes


def _existing_columns(conn: Connection, table: Table) -> set[str]:
    quoted = conn.dialect.identifier_preparer.quote(table.name)
    rows = conn.exec_driver_sql(f"PRAGMA table_info({quoted})").fetchall()
    return {row[1] for row in rows}


def _existing_indexes(conn: Connection, table: Table) -> set[str]:
    quoted = conn.dialect.identifier_preparer.quote(table.name)
    rows = conn.exec_driver_sql(f"PRAGMA index_list({quoted})").fetchall()
    return {row[1] for row in rows}


def _is_safe(column: Column[object]) -> bool:
    if column.primary_key:
        return False
    return bool(column.nullable) or column.server_default is not None


def _raise_if_unsafe(tables: list[Table], plan: dict[str, list[Column[object]]]) -> None:
    unsafe = [f"{t.name}.{c.name}" for t in tables for c in plan[t.name] if not _is_safe(c)]
    if unsafe:
        raise SqliteSchemaSyncError(
            f"This SQLite database was created by an older Dendrux version and is "
            f"missing column(s) {', '.join(unsafe)}, which are NOT NULL without a "
            f"server default and cannot be added to an existing SQLite table. "
            f"Run `dendrux db migrate` against this database, or recreate the "
            f"database file."
        )


def _add_column(conn: Connection, table: Table, column: Column[object]) -> bool:
    """Add one column; return False if a concurrent process added it first."""
    preparer = conn.dialect.identifier_preparer
    spec = str(CreateColumn(column).compile(dialect=conn.dialect))
    for fk in column.foreign_keys:
        target = fk.column
        spec += f" REFERENCES {preparer.quote(target.table.name)} ({preparer.quote(target.name)})"
        if fk.ondelete:
            spec += f" ON DELETE {fk.ondelete}"
    statement = f"ALTER TABLE {preparer.quote(table.name)} ADD COLUMN {spec}"
    try:
        conn.exec_driver_sql(statement)
    except OperationalError as exc:
        if column.name in _existing_columns(conn, table):
            return False
        raise SqliteSchemaSyncError(
            f"Could not add column {table.name}.{column.name} to the existing SQLite "
            f"database ({exc.orig}). Run `dendrux db migrate` against this database, "
            f"or recreate the database file."
        ) from exc
    return True


def _ensure_unique_indexes(conn: Connection, table: Table, added: set[str]) -> list[str]:
    """SQLite cannot ADD COLUMN ... UNIQUE; enforce it with a unique index."""
    preparer = conn.dialect.identifier_preparer
    created: list[str] = []
    for constraint in table.constraints:
        if not isinstance(constraint, UniqueConstraint):
            continue
        names = [c.name for c in constraint.columns]
        if not added.intersection(names):
            continue
        index_name = constraint.name or f"uq_{table.name}_{'_'.join(names)}"
        cols = ", ".join(preparer.quote(n) for n in names)
        conn.exec_driver_sql(
            f"CREATE UNIQUE INDEX IF NOT EXISTS {preparer.quote(str(index_name))} "
            f"ON {preparer.quote(table.name)} ({cols})"
        )
        created.append(f"created unique index {index_name}")
    return created


def _ensure_indexes(conn: Connection, table: Table) -> list[str]:
    existing = _existing_indexes(conn, table)
    created: list[str] = []
    for index in sorted(table.indexes, key=lambda i: str(i.name)):
        if index.name in existing:
            continue
        conn.execute(CreateIndex(index, if_not_exists=True))
        created.append(f"created index {index.name}")
    return created
