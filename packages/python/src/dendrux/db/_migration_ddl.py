"""Bounded Postgres DDL steps shared by the Alembic revisions.

Every step waits at most ``LOCK_TIMEOUT_SECONDS`` for a lock, including the waits of
concurrent index builds and drops for open transactions. A step that cannot get its
lock refuses with a retry message and logs one ``migration_lock_timeout`` line
(revision, step name, timeout; never SQL text).
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from sqlalchemy import exc, text

from dendrux import _codes
from dendrux.db._index_check import read_pg_indexes

if TYPE_CHECKING:
    from collections.abc import Iterator

    from sqlalchemy import Connection

logger = logging.getLogger(__name__)

LOCK_TIMEOUT_SECONDS = 5
LOCK_TIMEOUT_CODE = "migration_lock_timeout"
_LOCK_NOT_AVAILABLE = "55P03"


def sqlstate(error: exc.DBAPIError) -> str | None:
    """The Postgres SQLSTATE of a driver error (asyncpg or psycopg)."""
    return getattr(error.orig, "sqlstate", None) or getattr(error.orig, "pgcode", None)


@contextmanager
def online(context: Any) -> Iterator[None]:
    """Run the enclosed steps outside the migration transaction (Alembic's autocommit block).

    ``context`` is the Alembic migration context (``op.get_context()``).
    """
    with context.autocommit_block():
        yield


def bounded_step(
    conn: Connection,
    statement: str,
    *,
    revision: str,
    step: str,
    transactional: bool = False,
) -> None:
    """Run one DDL statement with a bounded lock wait.

    By default ``conn`` is in autocommit mode (inside :func:`online`) and the session's
    previous ``lock_timeout`` is restored afterwards. With ``transactional=True`` the
    statement stays in the enclosing transaction: a savepoint scopes ``SET LOCAL`` and
    restores it on timeout.
    """
    if conn.dialect.name != "postgresql":
        conn.execute(text(statement))
        return
    previous: str = str(conn.execute(text("SHOW lock_timeout")).scalar_one())
    try:
        if transactional:
            with conn.begin_nested():
                conn.execute(text(f"SET LOCAL lock_timeout = '{LOCK_TIMEOUT_SECONDS}s'"))
                conn.execute(text(statement))
                conn.execute(
                    text("SELECT set_config('lock_timeout', :previous, true)"),
                    {"previous": previous},
                )
        else:
            conn.execute(text(f"SET lock_timeout = '{LOCK_TIMEOUT_SECONDS}s'"))
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
        fields: dict[str, Any] = {"step": step, "timeout_seconds": LOCK_TIMEOUT_SECONDS}
        fields["revision"] = revision
        _codes.log_code(logger, _codes.MIGRATION_LOCK_TIMEOUT, **fields)
        raise RuntimeError(
            f"Timed out acquiring the Dendrux migration lock after {LOCK_TIMEOUT_SECONDS} "
            "seconds. Retry the migration operation after the blocking transaction finishes."
        ) from None


def build_index_concurrently(
    conn: Connection, name: str, table: str, statement: str, *, revision: str
) -> None:
    """Run ``statement`` (a ``CREATE ... INDEX CONCURRENTLY IF NOT EXISTS`` of ``name``).

    An invalid index of the same name, left by an interrupted concurrent build, is
    dropped first so the rerun rebuilds it. A valid index is kept; callers verify its
    definition. Must run inside :func:`online`.
    """
    entry = read_pg_indexes(conn, tables=[table]).get(name)
    if entry is not None and not entry[2]:
        drop_index_concurrently(conn, name, revision=revision, step=f"repair_index.{name}")
    bounded_step(conn, statement, revision=revision, step=f"build_index.{name}")


def drop_index_concurrently(
    conn: Connection, name: str, *, revision: str, step: str | None = None
) -> None:
    """``DROP INDEX CONCURRENTLY IF EXISTS``, bounded. Must run inside :func:`online`."""
    bounded_step(
        conn,
        f"DROP INDEX CONCURRENTLY IF EXISTS {name}",
        revision=revision,
        step=step or f"drop_index.{name}",
    )
