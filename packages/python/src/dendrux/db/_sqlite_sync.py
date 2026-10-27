"""Bring an existing SQLite file up to the current model schema.

``Base.metadata.create_all`` creates missing tables but never adds columns to
tables that already exist, so a SQLite file written by an older Dendrux
version breaks after an upgrade (``no such column``). Postgres users run
Alembic; SQLite is zero-config, so it needs an in-place sync instead.

The sync is additive only: it adds missing columns and indexes. It never
drops, renames or alters anything, and never touches tables outside
``Base.metadata``. Plain column indexes are matched by their columns;
expression, partial and descending indexes are matched by name and by the
``CREATE INDEX`` text SQLite stored for them.

A file created before 0.5.0 keeps its inline ``UNIQUE (idempotency_key)``:
removing it would need a table rebuild. The tenant-scoped idempotency lookup
still closes the cross-tenant leak there; a key reused by another tenant fails
with ``IdempotencyConflictError`` (no run id) instead of creating a second run.
One warning is logged per file. It runs inside one ``BEGIN IMMEDIATE`` transaction, so a
failure rolls every change back and concurrent startups serialize on the
write lock. Every run reconciles all model uniqueness and index invariants
against what the database actually has, so an interrupted older upgrade heals.
"""

from __future__ import annotations

import datetime as dt
import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from sqlalchemy import Column as _Column
from sqlalchemy import UniqueConstraint, literal
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.schema import CreateColumn, CreateIndex, DefaultClause
from sqlalchemy.sql.elements import False_, Null, TextClause, True_

from dendrux.db._pool_guard import close_fully, serialized, shares_one_connection
from dendrux.db.models import Base
from dendrux.errors import SqliteSchemaSyncError

if TYPE_CHECKING:
    from sqlalchemy import Column, Connection, Index, Table
    from sqlalchemy.ext.asyncio import AsyncEngine

__all__ = [
    "SqliteSchemaSyncError",
    "create_sqlite_schema",
    "create_sqlite_schema_sync",
    "sync_sqlite_schema",
]

logger = logging.getLogger(__name__)

_MIGRATE_HINT = "Run `dendrux db migrate` against this database, or recreate the database file."
_CONSTANT_DEFAULT = re.compile(
    r"^\s*(?:NULL|TRUE|FALSE|[+-]?\d+(?:\.\d+)?|'(?:[^']|'')*')\s*$", re.IGNORECASE
)


async def create_sqlite_schema(engine: AsyncEngine) -> None:
    """Create missing tables, then sync columns and indexes of existing ones.

    Owns its connection and transaction: pass the engine, never call it from
    inside a caller's transaction. A no-op on non-SQLite engines.
    """
    if engine.dialect.name != "sqlite":
        return
    if not shares_one_connection(engine):
        async with engine.connect() as conn:
            await conn.run_sync(create_sqlite_schema_sync)
        return
    async with serialized(engine):
        conn = await engine.connect()
        try:
            await conn.run_sync(create_sqlite_schema_sync)
        finally:
            await close_fully(conn.close)


def create_sqlite_schema_sync(sync_conn: Connection) -> list[str]:
    """Synchronous form of :func:`create_sqlite_schema` on a dedicated connection.

    The connection must not be inside an open transaction: this function
    starts and commits (or rolls back) its own.
    """
    if sync_conn.dialect.name != "sqlite":
        return []
    return _run(sync_conn, create_tables=True)


def sync_sqlite_schema(sync_conn: Connection) -> list[str]:
    """Add model columns and indexes missing from existing SQLite tables.

    Returns a description of every change made (empty when already current).
    Takes the write lock only when there is work; owns its transaction, so
    the connection must not be inside an open one. A no-op on non-SQLite
    connections.

    Raises:
        SqliteSchemaSyncError: A missing column cannot be added safely, an
            index the schema needs collides with an unrelated object, the
            write lock stayed busy, or SQLite rejected a statement. Nothing is
            applied for an unsafe column; any other failure rolls back what
            was applied.
    """
    if sync_conn.dialect.name != "sqlite":
        return []
    return _run(sync_conn, create_tables=False)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

_LOCK_ATTEMPTS = 3


def _run(conn: Connection, *, create_tables: bool) -> list[str]:
    _warn_legacy_idempotency_unique(conn)
    if not _has_work(conn, create_tables):
        return []
    _begin_immediate(conn)
    changes: list[str] = []
    try:
        # Another process may have finished the job while we waited for the lock.
        if _has_work(conn, create_tables):
            if create_tables:
                Base.metadata.create_all(conn)
            _apply(conn, changes)
    except BaseException as exc:
        _rollback(conn)
        if isinstance(exc, SqliteSchemaSyncError) and changes:
            raise SqliteSchemaSyncError(
                f"{exc} Applied before the failure: {'; '.join(changes)}. "
                f"All of these were rolled back."
            ) from exc
        raise
    conn.exec_driver_sql("COMMIT")
    for change in changes:
        logger.info("SQLite schema sync: %s", change)
    return changes


def _has_work(conn: Connection, create_tables: bool) -> bool:
    """Read-only plan: does the database still differ from the models?"""
    _raise_if_unsafe(conn)
    tables = _existing_model_tables(conn)
    if create_tables and len(tables) != len(Base.metadata.tables):
        return True
    return bool(_unmet(conn, tables))


def _begin_immediate(conn: Connection) -> None:
    for attempt in range(_LOCK_ATTEMPTS):
        try:
            conn.exec_driver_sql("BEGIN IMMEDIATE")
            return
        except OperationalError as exc:
            if attempt == _LOCK_ATTEMPTS - 1:
                raise SqliteSchemaSyncError(
                    f"Could not take the SQLite write lock to upgrade this database's "
                    f"schema after {_LOCK_ATTEMPTS} attempts: another connection is "
                    f"holding it ({exc.orig}). Each attempt waits for the connection's "
                    f"busy timeout (5 seconds by default; raise it with "
                    f"connect_args={{'timeout': N}}). Retry once the other writer finishes."
                ) from exc


def _rollback(conn: Connection) -> None:
    try:
        conn.exec_driver_sql("ROLLBACK")
    except OperationalError:
        logger.warning("SQLite schema sync: ROLLBACK failed", exc_info=True)


def _apply(conn: Connection, changes: list[str]) -> None:
    tables = _existing_model_tables(conn)
    for table in tables:
        db_columns = _column_names(conn, table)
        for column in table.columns:
            if column.name.lower() not in db_columns:
                _add_column(conn, table, column)
                changes.append(f"added column {table.name}.{column.name}")
    for table in tables:
        _reconcile_indexes(conn, table, changes)
    problems = _unmet(conn, tables)
    if problems:
        raise SqliteSchemaSyncError(
            f"The SQLite schema is still not current after the upgrade: {'; '.join(problems)}. "
            f"{_MIGRATE_HINT}"
        )


# ---------------------------------------------------------------------------
# Introspection
# ---------------------------------------------------------------------------


def _quote(conn: Connection, name: str) -> str:
    return conn.dialect.identifier_preparer.quote(name)


def _table_names(conn: Connection) -> set[str]:
    rows = conn.exec_driver_sql("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    return {row[0].lower() for row in rows}


def _existing_model_tables(conn: Connection) -> list[Table]:
    present = _table_names(conn)
    return [t for t in Base.metadata.sorted_tables if t.name.lower() in present]


def _column_names(conn: Connection, table: Table) -> set[str]:
    rows = conn.exec_driver_sql(f"PRAGMA table_info({_quote(conn, table.name)})").fetchall()
    return {row[1].lower() for row in rows}


@dataclass(frozen=True)
class _DbIndex:
    unique: bool
    partial: bool
    columns: tuple[str | None, ...]


def _table_indexes(conn: Connection, table: Table) -> dict[str, _DbIndex]:
    found: dict[str, _DbIndex] = {}
    listing = conn.exec_driver_sql(f"PRAGMA index_list({_quote(conn, table.name)})").fetchall()
    for row in listing:
        name, unique, partial = row[1], bool(row[2]), bool(row[4]) if len(row) > 4 else False
        info = conn.exec_driver_sql(f"PRAGMA index_info({_quote(conn, name)})").fetchall()
        columns = tuple(
            c[2].lower() if c[2] is not None else None for c in sorted(info, key=lambda r: r[0])
        )
        found[name.lower()] = _DbIndex(unique=unique, partial=partial, columns=columns)
    return found


def _index_owners(conn: Connection) -> dict[str, str]:
    rows = conn.exec_driver_sql("SELECT name, tbl_name FROM sqlite_master WHERE type = 'index'")
    return {row[0].lower(): row[1].lower() for row in rows.fetchall()}


def _index_sql(conn: Connection) -> dict[str, str]:
    """Index name -> its normalized ``CREATE INDEX`` text (explicit indexes only)."""
    rows = conn.exec_driver_sql(
        "SELECT name, sql FROM sqlite_master WHERE type = 'index' AND sql IS NOT NULL"
    )
    return {row[0].lower(): _normalize_sql(row[1]) for row in rows.fetchall()}


def _normalize_sql(sql: str) -> str:
    """Comparable form of a CREATE INDEX statement: case, quoting and spacing removed."""
    text = re.sub(r'["`\[\]]', "", sql.lower())
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s*([(),=])\s*", r"\1", text)
    return text.replace(" if not exists ", " ")


_LEGACY_WARNED: set[str] = set()


def _warn_legacy_idempotency_unique(conn: Connection) -> None:
    """Log once per file when ``agent_runs`` still has the pre-0.5 global key UNIQUE."""
    if "agent_runs" not in _table_names(conn):
        return
    table = Base.metadata.tables["agent_runs"]
    legacy = any(
        index.unique and not index.partial and index.columns == ("idempotency_key",)
        for index in _table_indexes(conn, table).values()
    )
    key = str(conn.engine.url.database or "")
    if not legacy or key in _LEGACY_WARNED:
        return
    _LEGACY_WARNED.add(key)
    logger.warning(
        "SQLite schema sync: this database was created before 0.5.0 and keeps a global "
        "UNIQUE on agent_runs.idempotency_key (removing it needs a table rebuild). "
        "Idempotency lookups are tenant-scoped, so no run leaks across tenants, but a key "
        "already used by another tenant fails with IdempotencyConflictError. Recreate the "
        "database file to make keys unique per tenant only."
    )


# ---------------------------------------------------------------------------
# Columns
# ---------------------------------------------------------------------------


def _unsafe_reason(column: Column[Any]) -> str | None:
    """Why ``ALTER TABLE ADD COLUMN`` cannot add this column, or None if it can."""
    if column.primary_key:
        return "is a primary key"
    default = column.server_default
    if default is None:
        return None if column.nullable else "is NOT NULL without a server default"
    if not isinstance(default, DefaultClause):
        return "has a computed or database-generated default"
    arg = default.arg
    if isinstance(arg, (str, True_, False_, Null)):
        return None
    if isinstance(arg, TextClause) and _CONSTANT_DEFAULT.match(arg.text):
        return None
    return "has a non-constant server default"


def _raise_if_unsafe(conn: Connection) -> None:
    unsafe: list[str] = []
    for table in _existing_model_tables(conn):
        db_columns = _column_names(conn, table)
        for column in table.columns:
            reason = (
                _unsafe_reason(column)
                if column.name.lower() not in db_columns and not _is_status_clock(table, column)
                else None
            )
            if reason is not None:
                unsafe.append(f"{table.name}.{column.name} ({reason})")
    if unsafe:
        raise SqliteSchemaSyncError(
            f"This SQLite database was created by an older Dendrux version and is missing "
            f"column(s) that SQLite cannot add to an existing table: {', '.join(unsafe)}. "
            f"{_MIGRATE_HINT}"
        )


def _is_status_clock(table: Table, column: Column[Any]) -> bool:
    return table.name == "agent_runs" and column.name == "status_changed_at"


def _add_column(conn: Connection, table: Table, column: Column[Any]) -> None:
    status_clock = _is_status_clock(table, column)
    ddl_column = column
    if status_clock:
        # SQLite permits only a constant default on ADD COLUMN. New SDK inserts
        # always set this clock explicitly; existing rows inherit their last
        # recorded update, under the same write lock and transaction as the DDL.
        clock_default = str(
            literal(dt.datetime.now(dt.UTC), type_=column.type).compile(
                dialect=conn.dialect, compile_kwargs={"literal_binds": True}
            )
        )
        ddl_column = _Column(
            column.name, column.type, nullable=False, server_default=TextClause(clock_default)
        )
    spec = str(CreateColumn(ddl_column).compile(dialect=conn.dialect))
    for fk in column.foreign_keys:
        target = fk.column
        spec += f" REFERENCES {_quote(conn, target.table.name)} ({_quote(conn, target.name)})"
        if fk.ondelete:
            spec += f" ON DELETE {fk.ondelete}"
    try:
        conn.exec_driver_sql(f"ALTER TABLE {_quote(conn, table.name)} ADD COLUMN {spec}")
        if status_clock:
            conn.exec_driver_sql(
                "UPDATE agent_runs SET status_changed_at = "
                f"COALESCE(updated_at, created_at, {clock_default})"
            )
    except OperationalError as exc:
        raise SqliteSchemaSyncError(
            f"Could not add column {table.name}.{column.name} to the existing SQLite "
            f"database ({exc.orig}). {_MIGRATE_HINT}"
        ) from exc


# ---------------------------------------------------------------------------
# Indexes and uniqueness
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Want:
    name: str
    columns: tuple[str, ...]
    unique: bool


def _is_plain(index: Index) -> bool:
    """A named index on bare columns with no WHERE clause: matched by its columns."""
    where = index.dialect_options["sqlite"].get("where")
    return where is None and all(isinstance(e, _Column) for e in index.expressions)


def _wanted_indexes(table: Table) -> list[_Want]:
    wants = [
        _Want(str(i.name), tuple(c.name.lower() for c in i.columns), bool(i.unique))
        for i in table.indexes
        if i.name and _is_plain(i)
    ]
    for constraint in table.constraints:
        if not isinstance(constraint, UniqueConstraint):
            continue
        names = tuple(c.name for c in constraint.columns)
        lowered = tuple(n.lower() for n in names)
        if any(w.unique and w.columns == lowered for w in wants):
            continue
        name = str(constraint.name) if constraint.name else f"uq_{table.name}_{'_'.join(names)}"
        wants.append(_Want(name, lowered, True))
    return wants


def _enforces(index: _DbIndex, want: _Want) -> bool:
    if index.columns != want.columns or index.partial:
        return False
    return index.unique or not want.unique


def _state(
    table: Table, want: _Want, indexes: dict[str, _DbIndex], owners: dict[str, str]
) -> str | None:
    """None when satisfied, "missing" when creatable, else a conflict description."""
    existing = indexes.get(want.name.lower())
    if existing is not None:
        if _enforces(existing, want):
            return None
        kind = "unique " if want.unique else ""
        return (
            f"index '{want.name}' already exists on {table.name} but does not match the "
            f"required {kind}index on ({', '.join(want.columns)})"
        )
    owner = owners.get(want.name.lower())
    if owner is not None:
        return f"index name '{want.name}' is already used by an index on table '{owner}'"
    if want.unique and any(_enforces(i, want) for i in indexes.values()):
        return None
    return "missing"


@dataclass(frozen=True)
class _Exact:
    name: str
    ddl: str


def _exact_indexes(conn: Connection, table: Table) -> list[_Exact]:
    """Expression, partial or ordered indexes, compared by name plus stored SQL."""
    return [
        _Exact(str(i.name), str(CreateIndex(i).compile(dialect=conn.dialect)))
        for i in table.indexes
        if i.name and not _is_plain(i)
    ]


def _exact_state(
    table: Table, want: _Exact, stored: dict[str, str], owners: dict[str, str]
) -> str | None:
    """None when the stored index matches, "missing" when creatable, else a conflict."""
    owner = owners.get(want.name.lower())
    if owner is not None and owner != table.name.lower():
        return f"index name '{want.name}' is already used by an index on table '{owner}'"
    sql = stored.get(want.name.lower())
    if sql is None:
        return (
            "missing"
            if owner is None
            else (f"index '{want.name}' on {table.name} has no stored definition to compare")
        )
    if sql != _normalize_sql(want.ddl):
        return (
            f"index '{want.name}' already exists on {table.name} but its definition "
            f"differs from the required one ({want.ddl})"
        )
    return None


def _reconcile_exact(conn: Connection, table: Table, changes: list[str]) -> None:
    for want in _exact_indexes(conn, table):
        state = _exact_state(table, want, _index_sql(conn), _index_owners(conn))
        if state is None:
            continue
        if state != "missing":
            raise SqliteSchemaSyncError(f"Cannot reconcile {table.name}: {state}. {_MIGRATE_HINT}")
        try:
            conn.exec_driver_sql(want.ddl)
        except (OperationalError, IntegrityError) as exc:
            raise SqliteSchemaSyncError(
                f"Could not create index '{want.name}' on {table.name} ({exc.orig}). If "
                f"existing rows violate uniqueness, fix them first. {_MIGRATE_HINT}"
            ) from exc
        changes.append(f"created index {want.name}")


def _reconcile_indexes(conn: Connection, table: Table, changes: list[str]) -> None:
    _reconcile_exact(conn, table, changes)
    indexes = _table_indexes(conn, table)
    owners = _index_owners(conn)
    for want in _wanted_indexes(table):
        state = _state(table, want, indexes, owners)
        if state is None:
            continue
        if state != "missing":
            raise SqliteSchemaSyncError(f"Cannot reconcile {table.name}: {state}. {_MIGRATE_HINT}")
        cols = ", ".join(_quote(conn, c) for c in want.columns)
        unique = "UNIQUE " if want.unique else ""
        try:
            conn.exec_driver_sql(
                f"CREATE {unique}INDEX {_quote(conn, want.name)} "
                f"ON {_quote(conn, table.name)} ({cols})"
            )
        except (OperationalError, IntegrityError) as exc:
            raise SqliteSchemaSyncError(
                f"Could not create {unique.lower()}index '{want.name}' on "
                f"{table.name}({', '.join(want.columns)}) ({exc.orig}). If existing rows "
                f"violate uniqueness, fix them first. {_MIGRATE_HINT}"
            ) from exc
        changes.append(f"created {unique.lower()}index {want.name}")
        indexes = _table_indexes(conn, table)
        owners = _index_owners(conn)


def _unmet(conn: Connection, tables: list[Table]) -> list[str]:
    """Every model column and index invariant the database still does not meet."""
    owners = _index_owners(conn)
    problems: list[str] = []
    for table in tables:
        db_columns = _column_names(conn, table)
        problems.extend(
            f"column {table.name}.{c.name} is missing"
            for c in table.columns
            if c.name.lower() not in db_columns
        )
        indexes = _table_indexes(conn, table)
        for want in _wanted_indexes(table):
            state = _state(table, want, indexes, owners)
            if state is not None:
                problems.append(
                    f"{table.name}: index '{want.name}' is missing" if state == "missing" else state
                )
        stored = _index_sql(conn)
        for exact in _exact_indexes(conn, table):
            state = _exact_state(table, exact, stored, owners)
            if state == "missing":
                problems.append(f"{table.name}: index '{exact.name}' is missing")
            elif state is not None:
                problems.append(state)
    return problems
