"""Resumable, operator-invoked uniqueness activation on a dedicated connection."""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sqlalchemy import delete, func, inspect, select, text
from sqlalchemy.exc import IntegrityError

from dendrux import _codes
from dendrux.db._dialects import is_mysql_family, mysql, postgres, require_family, sqlite
from dendrux.db._index_check import index_exists, require_index
from dendrux.db._migration_ddl import build_index_concurrently, drop_index_concurrently, sqlstate
from dendrux.db._schema_check import read_schema_phase, schema_problems
from dendrux.db._settings import put_setting
from dendrux.db._storage_schema import ACTIVATION_INDEXES, index_ddl
from dendrux.db.models import Base, DendruxSetting

if TYPE_CHECKING:
    from collections.abc import Iterator

    from sqlalchemy import Connection, Table
    from sqlalchemy.ext.asyncio import AsyncEngine

logger = logging.getLogger(__name__)
_REVISION = "f9a0b1c2d3e4"
_FIELDS = {
    "react_traces": ("role", "content", "order_index", "trace_kind", "meta"),
    # The released schema stores result_payload in the JSON `result` column.
    "tool_calls": (
        "tool_call_id",
        "provider_tool_call_id",
        "tool_name",
        "target",
        "params",
        "result",
        "success",
        "iteration_index",
        "error_message",
        "meta",
        "outcome",
        "idempotent",
    ),
    "run_events": ("event_type", "iteration_index", "correlation_id", "data"),
}


class ActivationConflictError(RuntimeError):
    """Bounded duplicate diagnostic containing only table, count and run identifiers."""

    def __init__(self, table: str, count: int, run_ids: list[str]) -> None:
        self.table = table
        self.count = count
        self.run_ids = tuple(run_ids[:20])
        super().__init__(
            f"Conflicting duplicates: table={table} count={count} "
            f"run_ids={','.join(self.run_ids)}. Resolve these duplicates and retry."
        )


class ActivationIndexConflictError(RuntimeError):
    """A writer introduced a duplicate after the dedupe pass finished."""


async def activate_database(
    engine: AsyncEngine, *, batch_size: int = 5000, dedupe_only: bool = False
) -> None:
    """Check currency, dedupe and optionally activate; never auto-upgrade a schema.

    Owns a dedicated direct connection. SQLite retains its exclusive file lock
    across committed cursor batches using locking_mode=EXCLUSIVE; other processes
    must be stopped by the operator. PostgreSQL uses the migration runner's
    session advisory lock across batches and concurrent index builds. MySQL/MariaDB
    hold the maintenance ``GET_LOCK`` (their schemas are bootstrapped active, so
    there is nothing to activate); on any error that connection is invalidated.
    """
    if isinstance(batch_size, bool) or not 1 <= batch_size <= 5000:
        raise ValueError("batch_size must be between 1 and 5000")
    async with engine.connect() as conn:
        await conn.run_sync(lambda c: _activate(c, batch_size, dedupe_only))


@contextmanager
def _maintenance_connection(conn: Connection) -> Iterator[None]:
    found = require_family(conn.dialect, "Activation")
    if found == "postgresql":
        lock = postgres.maintenance_lock(conn)
    elif found == "sqlite":
        lock = sqlite.maintenance_lock(conn)
    else:
        lock = mysql.maintenance_lock(conn)
    with lock:
        yield


def _mysql_active(conn: Connection) -> bool:
    """An absent settings table is left to the currency check (it reports what is missing)."""
    if not inspect(conn).has_table("dendrux_settings"):
        return True
    values = mysql.read_settings(conn)
    return values.get("storage_format") == "2" and values.get("activate_step") == "done"


def _require_current(conn: Connection) -> None:
    from alembic.script import ScriptDirectory

    head = ScriptDirectory(str(Path(__file__).parent / "migrations")).get_current_head()
    if inspect(conn).has_table("alembic_version"):
        revision = conn.execute(
            text("SELECT version_num FROM alembic_version")
        ).scalar_one_or_none()
        if revision != head:
            raise RuntimeError("Database check is not current. Run `dendrux db migrate` first.")
    problems = schema_problems(conn)
    if problems:
        raise RuntimeError("Database check is not current: " + ", ".join(problems))


def _checkpoint(conn: Connection, table: str, cursor: str) -> None:
    put_setting(conn, f"dedupe_cursor_{table}", cursor)
    conn.commit()


def _page(
    conn: Connection, table: Table, column: str, after: str, end: str, size: int
) -> list[Any]:
    candidate = table.alias("candidate")
    keeper = table.alias("keeper")
    first_id = (
        select(func.min(keeper.c.id))
        .where(
            keeper.c.agent_run_id == candidate.c.agent_run_id,
            keeper.c[column] == candidate.c[column],
        )
        .correlate(candidate)
        .scalar_subquery()
    )
    return list(
        conn.execute(
            select(candidate.c.id, first_id.label("keeper_id"), candidate.c.agent_run_id)
            .where(candidate.c.id > after, candidate.c.id <= end)
            .order_by(candidate.c.id)
            .limit(size)
        )
    )


def _duplicates(conn: Connection, table: Table, page: list[Any]) -> tuple[list[str], list[str]]:
    pairs = [(row.id, row.keeper_id, row.agent_run_id) for row in page if row.id != row.keeper_id]
    if not pairs:
        return [], []
    columns = [table.c.id, table.c.content_digest, *(table.c[name] for name in _FIELDS[table.name])]
    identical = []
    conflicting = []
    # Payload memory is bounded independently of the 5k ID cursor batch.
    for start in range(0, len(pairs), 100):
        group = pairs[start : start + 100]
        ids = sorted({id for row in group for id in row[:2]})
        rows = {row.id: row for row in conn.execute(select(*columns).where(table.c.id.in_(ids)))}
        for id, keeper_id, run_id in group:
            current, keeper = rows[id], rows[keeper_id]
            if current.content_digest == keeper.content_digest and (
                current.content_digest is not None
                or json.dumps(
                    tuple(current)[2:], sort_keys=True, separators=(",", ":"), ensure_ascii=False
                )
                == json.dumps(
                    tuple(keeper)[2:], sort_keys=True, separators=(",", ":"), ensure_ascii=False
                )
            ):
                identical.append(id)
            else:
                conflicting.append(run_id)
    return identical, conflicting


def _preflight(conn: Connection, batch_size: int) -> None:
    # Scan every table before deleting anything, including after interruption.
    for table_name, column, _ in ACTIVATION_INDEXES.values():
        table = Base.metadata.tables[table_name]
        end = conn.execute(select(func.max(table.c.id))).scalar_one()
        after = ""
        count = 0
        first_runs: list[str] = []
        while end is not None:
            page = _page(conn, table, column, after, end, batch_size)
            if not page:
                break
            _, conflicts = _duplicates(conn, table, page)
            count += len(conflicts)
            first_runs = sorted(set(first_runs).union(conflicts))[:20]
            after = page[-1].id
        if count:
            _codes.log_code(
                logger, _codes.WRITE_CONFLICT, run_id=first_runs[0], table=table_name, count=count
            )
            raise ActivationConflictError(table_name, count, first_runs)


def _dedupe(conn: Connection, batch_size: int) -> None:
    for table_name, column, _ in ACTIVATION_INDEXES.values():
        table = Base.metadata.tables[table_name]
        after = conn.execute(
            select(DendruxSetting.value).where(DendruxSetting.key == f"dedupe_cursor_{table_name}")
        ).scalar_one_or_none()
        if after == "":
            continue
        put_setting(conn, "activate_step", f"dedupe:{table_name}")
        after = after or ""
        end = conn.execute(select(func.max(table.c.id))).scalar_one()
        conn.commit()
        while end is not None:
            page = _page(conn, table, column, after, end, batch_size)
            if not page:
                break
            identical, conflicts = _duplicates(conn, table, page)
            if conflicts:
                raise ActivationConflictError(
                    table_name, len(conflicts), sorted(set(conflicts))[:20]
                )
            for start in range(0, len(identical), 200):
                ids = identical[start : start + 200]
                conn.execute(delete(table).where(table.c.id.in_(ids)))
            after = page[-1].id
            _checkpoint(conn, table_name, after)
        _checkpoint(conn, table_name, "")


def _activate(conn: Connection, batch_size: int, dedupe_only: bool) -> None:
    with _maintenance_connection(conn):
        if is_mysql_family(conn.dialect) and not _mysql_active(conn):
            raise RuntimeError(
                "MySQL/MariaDB schemas are created active by `dendrux db migrate`; activation "
                "does not apply to this one. Run `dendrux db check` and repair its settings."
            )
        _require_current(conn)
        if read_schema_phase(conn) == "active":
            conn.commit()
            return
        put_setting(conn, "activate_step", "dedupe:react_traces", overwrite=False)
        conn.commit()  # The downgrade barrier exists before any activation work.
        _preflight(conn, batch_size)
        conn.commit()
        step = conn.execute(
            select(DendruxSetting.value).where(DendruxSetting.key == "activate_step")
        ).scalar_one()
        conn.commit()
        if dedupe_only or step.startswith("dedupe:"):
            _dedupe(conn, batch_size)
        if dedupe_only:
            return
        pg = conn.dialect.name == "postgresql"
        for name, (table, _, _) in ACTIVATION_INDEXES.items():
            put_setting(conn, "activate_step", f"index:{name}")
            conn.commit()
            try:
                if pg:
                    conn.execution_options(isolation_level="AUTOCOMMIT")
                    build_index_concurrently(
                        conn, name, table, index_ddl(name, concurrently=True), revision=_REVISION
                    )
                else:
                    conn.exec_driver_sql("BEGIN EXCLUSIVE")
                    conn.execute(text(index_ddl(name)))
                require_index(conn, name, table, index_ddl(name))
                conn.commit()
            except IntegrityError as error:
                if (
                    sqlstate(error) != "23505"
                    and getattr(error.orig, "sqlite_errorname", None) != "SQLITE_CONSTRAINT_UNIQUE"
                ):
                    raise
                conn.rollback()
                if pg:
                    conn.execution_options(isolation_level="READ COMMITTED")
                conn.execute(
                    delete(DendruxSetting).where(DendruxSetting.key == f"dedupe_cursor_{table}")
                )
                put_setting(conn, "activate_step", f"dedupe:{table}")
                conn.commit()
                _codes.log_code(logger, _codes.ACTIVATION_CONFLICT, table=table, index=name)
                raise ActivationIndexConflictError(
                    "duplicates appeared after dedupe; rerun dendrux db dedupe"
                ) from None
            if pg:
                conn.execution_options(isolation_level="READ COMMITTED")
        for table, column, old in ACTIVATION_INDEXES.values():
            if old and index_exists(conn, old):
                require_index(
                    conn,
                    old,
                    table,
                    f"CREATE INDEX {old} ON {table} (agent_run_id, {column})",
                    require_valid=False,
                )
        conn.commit()
        for _, (_, _, old) in ACTIVATION_INDEXES.items():
            if not old:
                continue
            put_setting(conn, "activate_step", f"drop:{old}")
            conn.commit()
            if pg:
                conn.execution_options(isolation_level="AUTOCOMMIT")
                drop_index_concurrently(conn, old, revision=_REVISION)
            else:
                conn.exec_driver_sql("BEGIN EXCLUSIVE")
                conn.execute(text(f"DROP INDEX IF EXISTS {old}"))
            conn.commit()
            if pg:
                conn.execution_options(isolation_level="READ COMMITTED")
        put_setting(conn, "activate_step", "format")
        conn.commit()
        # Publish the format and completion marker atomically.
        put_setting(conn, "storage_format", "2")
        put_setting(conn, "activate_step", "done")
        conn.commit()
