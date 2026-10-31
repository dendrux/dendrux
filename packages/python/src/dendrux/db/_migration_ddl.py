"""Bounded DDL steps shared by the Alembic revisions and the MySQL-family bootstrap.

Every step waits at most ``LOCK_TIMEOUT_SECONDS`` for a lock, including (on Postgres) the
waits of concurrent index builds and drops for open transactions. A step that cannot get
its lock refuses with a retry message and logs one ``migration_lock_timeout`` line
(revision, step name, timeout; never SQL text).
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from sqlalchemy import text

from dendrux import _codes
from dendrux.db._dialects import LockWaitTimeout, family, mysql, postgres
from dendrux.db._dialects.postgres import sqlstate

if TYPE_CHECKING:
    from collections.abc import Iterator

    from sqlalchemy import Connection

__all__ = [
    "LOCK_TIMEOUT_SECONDS",
    "bounded_step",
    "build_index_concurrently",
    "drop_index_concurrently",
    "online",
    "sqlstate",
]

logger = logging.getLogger(__name__)

LOCK_TIMEOUT_SECONDS = 5
LOCK_TIMEOUT_CODE = "migration_lock_timeout"


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

    Postgres: by default ``conn`` is in autocommit mode (inside :func:`online`) and the
    session's previous ``lock_timeout`` is restored afterwards; with ``transactional=True``
    the statement stays in the enclosing transaction and a savepoint scopes ``SET LOCAL``.
    MySQL/MariaDB: ``lock_wait_timeout`` and ``innodb_lock_wait_timeout`` are bounded for
    the statement and restored (their DDL commits implicitly). SQLite runs it directly.
    """
    found = family(conn.dialect)
    try:
        if found == "postgresql":
            postgres.bounded_execute(
                conn, statement, timeout_seconds=LOCK_TIMEOUT_SECONDS, transactional=transactional
            )
        elif found == "mysql":
            mysql.bounded_execute(conn, statement, timeout_seconds=LOCK_TIMEOUT_SECONDS)
        else:
            conn.execute(text(statement))
    except LockWaitTimeout:
        fields: dict[str, Any] = {"timeout_seconds": LOCK_TIMEOUT_SECONDS}
        # Log fields are a closed set: values the registry does not know are left out.
        for name, value in (("step", step), ("revision", revision)):
            if _codes._valid_field(name, value):
                fields[name] = value
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
    definition. Must run inside :func:`online`. Postgres only.
    """
    entry = postgres.read_indexes(conn, tables=[table]).get(name)
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
