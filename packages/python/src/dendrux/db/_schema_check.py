"""Read-only, state-aware schema checks for PostgreSQL, SQLite and MySQL/MariaDB.

Postgres users run Alembic after upgrading Dendrux. If they forget, the first
query fails deep inside a run with a raw ``UndefinedColumnError``. This check
runs once per engine, before any run is created, and turns that into a single
actionable :class:`~dendrux.errors.SchemaOutOfDateError`.

It inspects the live tables and columns and verifies that every table and
column in ``Base.metadata`` is present, by exact name. On Postgres it also
verifies named index definitions and valid/ready catalog flags, plus unique
constraint names
the models declare (for example the tenant-scoped idempotency index added in
0.5.0), so a database that was not migrated fails at startup instead of
running without the constraint. It does not compare column types,
nullability or defaults. It deliberately ignores Alembic's
version table (apps may create the schema themselves) and extra tables,
columns and indexes. It never writes or issues DDL. SQLite startup first
upgrades itself in place (:mod:`dendrux.db._sqlite_sync`), then checks currency.
``DENDRUX_SKIP_SCHEMA_CHECK=1`` disables it. If the database refuses the
catalog read (for example for lack of privileges) the check logs a warning and
lets startup continue; connection failures propagate.
"""

from __future__ import annotations

import logging
import os
import weakref
from typing import TYPE_CHECKING

from sqlalchemy import UniqueConstraint, inspect, select
from sqlalchemy.exc import DBAPIError

from dendrux.db._dialects import family, mysql, postgres, sqlite
from dendrux.db._storage_schema import ACTIVATION_INDEXES
from dendrux.db.models import Base, DendruxSetting
from dendrux.errors import SchemaNotInitializedError, SchemaOutOfDateError

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping

    from sqlalchemy import Connection, Engine
    from sqlalchemy.ext.asyncio import AsyncEngine

__all__ = [
    "SKIP_ENV",
    "compare_indexes",
    "compare_schema",
    "ensure_schema_current",
    "read_actual_indexes",
    "read_actual_schema",
    "required_indexes",
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
    tables = sorted(Base.metadata.tables)
    found = family(sync_conn.dialect)
    if found == "postgresql":
        return postgres.read_columns(sync_conn, tables)
    if found == "mysql":
        return mysql.read_columns(sync_conn, tables)
    return sqlite.read_columns(sync_conn, tables)


def required_indexes(state: str = "pre-activation") -> dict[str, str]:
    """Named index or unique constraint -> its table, for every model table."""
    wanted: dict[str, str] = {}
    for table in Base.metadata.sorted_tables:
        for index in table.indexes:
            if index.name:
                wanted[str(index.name)] = table.name
        for constraint in table.constraints:
            if isinstance(constraint, UniqueConstraint) and constraint.name:
                wanted[str(constraint.name)] = table.name
    if state != "active":
        for name, (owner, _, old) in ACTIVATION_INDEXES.items():
            wanted.pop(name, None)
            if old:
                wanted[old] = owner
    return wanted


def read_schema_phase(conn: Connection, *, known_tables: Collection[str] | None = None) -> str:
    """Read the frozen state table, refusing inconsistent settings."""
    has_settings = (
        "dendrux_settings" in known_tables
        if known_tables is not None
        else inspect(conn).has_table("dendrux_settings")
    )
    if not has_settings:
        return "pre-activation"
    settings = dict(
        conn.execute(
            select(DendruxSetting.key, DendruxSetting.value).where(
                DendruxSetting.key.in_(("storage_format", "activate_step"))
            )
        )
        .tuples()
        .all()
    )
    storage_format = settings.get("storage_format", "1")
    step = settings.get("activate_step")
    if storage_format not in {"1", "2"}:
        raise RuntimeError("Invalid dendrux_settings storage format; repair the settings first.")
    if step is not None and step != "done":
        return "activating"
    if storage_format == "2" and step == "done":
        return "active"
    if storage_format == "1" and step is None:
        return "pre-activation"
    raise RuntimeError("Inconsistent dendrux_settings activation state; repair the settings first.")


def compare_indexes(
    actual: Collection[str], tables: Collection[str], *, state: str = "pre-activation"
) -> list[str]:
    """State-derived missing indexes; active schemas must also lack replaced names."""
    present = set(actual)
    wanted = required_indexes(state)
    if state == "activating":
        for name, (_, _, old) in ACTIVATION_INDEXES.items():
            if old and name in present:
                wanted.pop(old, None)
    missing = [
        f"{table}.{name} (index)"
        for name, table in sorted(wanted.items())
        if table in tables and name not in present
    ]
    if state == "active":
        existing = actual.existing if isinstance(actual, _IndexNames) else present
        missing.extend(
            f"{table}.{old} (replaced index must be absent)"
            for table, _, old in ACTIVATION_INDEXES.values()
            if old and old in existing
        )
    return missing


class _IndexNames(set[str]):
    def __init__(self, valid: set[str], existing: set[str]) -> None:
        super().__init__(valid)
        self.existing = existing


def read_actual_indexes(sync_conn: Connection) -> set[str]:
    """Valid, ready indexes whose definitions match, on PostgreSQL, SQLite or MySQL."""
    from sqlalchemy.schema import CreateIndex

    from dendrux.db._index_check import _canonical

    found = family(sync_conn.dialect)
    if found == "mysql":
        valid, existing = mysql.index_names(sync_conn)
        return _IndexNames(valid, existing)
    if found == "postgresql":
        entries = postgres.read_indexes(sync_conn, tables=sorted(Base.metadata.tables))
    else:
        entries = sqlite.read_index_entries(sync_conn)
    names = {name for name, entry in entries.items() if entry[2]}
    if found == "sqlite":
        names |= sqlite.matching_unique_constraints(sync_conn, Base.metadata.sorted_tables)
    expected = {
        str(index.name): (table.name, str(CreateIndex(index).compile(dialect=sync_conn.dialect)))
        for table in Base.metadata.sorted_tables
        for index in table.indexes
    }
    for _, (owner, column, old) in ACTIVATION_INDEXES.items():
        if old:
            expected[old] = (owner, f"CREATE INDEX {old} ON {owner} (agent_run_id, {column})")
    for name, (owner, ddl) in expected.items():
        entry = entries.get(name)
        if entry is not None and (entry[0] != owner or _canonical(entry[1]) != _canonical(ddl)):
            names.discard(name)
    return _IndexNames(names, set(entries))


def schema_problems(conn: Connection) -> list[str]:
    """Read-only structural check shared by the CLI, activation and runtime startup.

    On MySQL/MariaDB it compares the whole head definition (types, NO PAD collations,
    keys and indexes) and the settings (active format, no unfinished bootstrap).
    """
    if family(conn.dialect) == "mysql":
        return mysql.check(conn).problems()
    actual = read_actual_schema(conn)
    tables, columns = compare_schema(actual)
    missing = columns + [f"table {name}" for name in tables]
    if actual and conn.dialect.name in {"postgresql", "sqlite"}:
        phase = read_schema_phase(conn, known_tables=actual)
        names = read_actual_indexes(conn)
        missing.extend(compare_indexes(names, actual.keys(), state=phase))

    return missing


def was_verified(engine: AsyncEngine) -> bool:
    """True if :func:`ensure_schema_current` fully checked and passed this engine."""
    return engine.sync_engine in _VERIFIED


async def ensure_schema_current(engine: AsyncEngine) -> None:
    """Raise if the database lacks tables, columns or (Postgres) indexes the models define.

    A no-op when ``DENDRUX_SKIP_SCHEMA_CHECK=1``. Read-only: one
    connection, a table listing, one batched column listing and, on Postgres,
    index catalog reads. Named model indexes must match their definitions and
    be valid and ready; column types and defaults are not compared. Missing indexes are
    listed in ``SchemaOutOfDateError.missing_indexes`` as
    ``"table.index (index)"``. If the
    catalog read fails with a database error (such as insufficient
    privileges) a warning is logged and the check is skipped; failures to
    connect propagate.

    Raises:
        SchemaNotInitializedError: The database has none of the Dendrux tables.
        SchemaOutOfDateError: Some tables or columns are missing; run
            ``dendrux db migrate``.
    """
    if _skipped():
        return
    if family(engine.dialect) == "mysql":
        await _ensure_mysql_current(engine)
        return
    indexes: set[str] | None = None
    async with engine.connect() as conn:
        try:
            actual = await conn.run_sync(read_actual_schema)
            phase = "pre-activation"
            if actual and engine.dialect.name in {"postgresql", "sqlite"}:
                phase = await conn.run_sync(lambda c: read_schema_phase(c, known_tables=actual))
                indexes = await conn.run_sync(read_actual_indexes)
        except DBAPIError as exc:
            if exc.connection_invalidated:
                raise
            logger.warning("schema check skipped: %s", type(exc.orig or exc).__name__)
            return
    if not actual:
        raise SchemaNotInitializedError()
    missing_tables, missing_columns = compare_schema(actual)
    missing_indexes = (
        compare_indexes(indexes, actual.keys(), state=phase) if indexes is not None else []
    )
    if missing_tables or missing_columns or missing_indexes:
        raise SchemaOutOfDateError(missing_tables, missing_columns, missing_indexes)
    _VERIFIED.add(engine.sync_engine)


async def _ensure_mysql_current(engine: AsyncEngine) -> None:
    async with engine.connect() as conn:
        try:
            report = await conn.run_sync(mysql.check)
        except DBAPIError as exc:
            if exc.connection_invalidated:
                raise
            logger.warning("schema check skipped: %s", type(exc.orig or exc).__name__)
            return
    if not report.present:
        raise SchemaNotInitializedError()
    if report.missing_tables or report.columns or report.indexes:
        raise SchemaOutOfDateError(report.missing_tables, report.columns, report.indexes)
    _VERIFIED.add(engine.sync_engine)
