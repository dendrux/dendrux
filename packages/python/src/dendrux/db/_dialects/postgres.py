"""PostgreSQL catalog reads, maintenance locks and bounded DDL (moved, unchanged)."""

from __future__ import annotations

import re
from contextlib import contextmanager
from typing import TYPE_CHECKING

from sqlalchemy import exc, text

from dendrux.db._dialects import LockWaitTimeout

if TYPE_CHECKING:
    from collections.abc import Iterator

    from sqlalchemy import Connection

MIGRATION_LOCK = 0x44454E44525558
"""Stable, database-wide advisory lock shared by migration and activation."""

_LOCK_NOT_AVAILABLE = "55P03"


def sqlstate(error: exc.DBAPIError) -> str | None:
    """The Postgres SQLSTATE of a driver error (asyncpg or psycopg)."""
    return getattr(error.orig, "sqlstate", None) or getattr(error.orig, "pgcode", None)


def read_indexes(conn: Connection, *, tables: list[str]) -> dict[str, tuple[str, str, bool]]:
    """One catalog query: names mapped to table, definition, valid-and-ready."""
    rows = conn.execute(
        text(
            "SELECT i.relname, t.relname, pg_get_indexdef(i.oid), "
            "x.indisvalid AND x.indisready, n.nspname "
            "FROM pg_class i JOIN pg_namespace n ON n.oid = i.relnamespace "
            "JOIN pg_index x ON x.indexrelid = i.oid "
            "JOIN pg_class t ON t.oid = x.indrelid "
            "WHERE n.nspname = current_schema() AND t.relname = ANY(:tables)"
        ),
        {"tables": tables},
    )
    return {
        row[0]: (
            row[1],
            re.sub(rf'(?i)(\bON\s+)(?:"{re.escape(row[4])}"|{re.escape(row[4])})\.', r"\1", row[2]),
            bool(row[3]),
        )
        for row in rows
    }


def index_exists(conn: Connection, name: str) -> bool:
    statement = (
        "SELECT 1 FROM pg_class i JOIN pg_namespace n ON n.oid=i.relnamespace "
        "WHERE n.nspname=current_schema() AND i.relkind IN ('i', 'I') AND i.relname=:name"
    )
    return conn.execute(text(statement), {"name": name}).scalar_one_or_none() is not None


def index_definition(conn: Connection, name: str, table: str, *, require_valid: bool) -> str | None:
    """The stored definition of ``name`` on ``table``; None if absent or not valid-and-ready."""
    entry = read_indexes(conn, tables=[table]).get(name)
    if entry is None or entry[0] != table or (require_valid and not entry[2]):
        return None
    return entry[1]


def read_columns(conn: Connection, tables: list[str]) -> dict[str, list[str]]:
    """Table name -> column names, for the listed tables present in the current schema."""
    rows = conn.execute(
        text(
            "SELECT t.relname, a.attname FROM pg_class t "
            "JOIN pg_namespace n ON n.oid=t.relnamespace "
            "JOIN pg_attribute a ON a.attrelid=t.oid "
            "WHERE n.nspname=current_schema() AND t.relkind IN ('r', 'p') "
            "AND a.attnum > 0 AND NOT a.attisdropped AND t.relname=ANY(:tables)"
        ),
        {"tables": tables},
    )
    actual: dict[str, list[str]] = {}
    for table_name, column_name in rows:
        actual.setdefault(table_name, []).append(column_name)
    return actual


@contextmanager
def maintenance_lock(conn: Connection) -> Iterator[None]:
    """Activation: the session advisory lock, held across committed batches."""
    conn.execution_options(isolation_level="AUTOCOMMIT")
    acquired = bool(
        conn.execute(text(f"SELECT pg_try_advisory_lock({MIGRATION_LOCK})")).scalar_one()
    )
    conn.commit()
    if not acquired:
        raise RuntimeError("Another Dendrux database migration or activation is running; retry.")
    conn.execution_options(isolation_level="READ COMMITTED")
    try:
        yield
    finally:
        conn.rollback()
        conn.execution_options(isolation_level="AUTOCOMMIT")
        conn.execute(text(f"SELECT pg_advisory_unlock({MIGRATION_LOCK})"))
        conn.commit()


@contextmanager
def migration_lock(connection: Connection) -> Iterator[None]:
    """Alembic runner: lock before reading the revision; keep it through all commits."""
    isolation = connection.get_isolation_level()
    connection.execution_options(isolation_level="AUTOCOMMIT")
    acquired = False
    try:
        # A blocking SELECT retains a query snapshot while waiting and can
        # deadlock the holder's CREATE INDEX CONCURRENTLY. Refuse promptly;
        # callers can retry after the active migration runner finishes.
        acquired = connection.execute(
            text(f"SELECT pg_try_advisory_lock({MIGRATION_LOCK})")
        ).scalar_one()
        connection.commit()
        if not acquired:
            raise RuntimeError(
                "Another Dendrux database migration is running; retry when it finishes."
            )
        connection.execution_options(isolation_level=isolation)
        yield
    finally:
        connection.rollback()
        connection.execution_options(isolation_level="AUTOCOMMIT")
        if acquired:
            connection.execute(text(f"SELECT pg_advisory_unlock({MIGRATION_LOCK})"))
            connection.commit()
        connection.execution_options(isolation_level=isolation)


def bounded_execute(
    conn: Connection, statement: str, *, timeout_seconds: int, transactional: bool
) -> None:
    """Run ``statement`` waiting at most ``timeout_seconds`` for locks.

    Autocommit by default, restoring the session's ``lock_timeout``; with
    ``transactional=True`` a savepoint scopes ``SET LOCAL``. A lock timeout raises
    :class:`LockWaitTimeout`.
    """
    previous: str = str(conn.execute(text("SHOW lock_timeout")).scalar_one())
    try:
        if transactional:
            with conn.begin_nested():
                conn.execute(text(f"SET LOCAL lock_timeout = '{timeout_seconds}s'"))
                conn.execute(text(statement))
                conn.execute(
                    text("SELECT set_config('lock_timeout', :previous, true)"),
                    {"previous": previous},
                )
        else:
            conn.execute(text(f"SET lock_timeout = '{timeout_seconds}s'"))
            try:
                conn.execute(text(statement))
            finally:
                conn.execute(
                    text("SELECT set_config('lock_timeout', :previous, false)"),
                    {"previous": previous},
                )
    except exc.DBAPIError as error:
        if sqlstate(error) != _LOCK_NOT_AVAILABLE:
            raise
        raise LockWaitTimeout from None
