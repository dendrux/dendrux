"""SQLite catalog reads and the activation lock (moved, unchanged).

Schema upgrades in place live in :mod:`dendrux.db._sqlite_sync`.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

from sqlalchemy import UniqueConstraint, inspect, text
from sqlalchemy.exc import OperationalError

if TYPE_CHECKING:
    from collections.abc import Iterator

    from sqlalchemy import Connection, Table


def index_exists(conn: Connection, name: str) -> bool:
    statement = "SELECT 1 FROM sqlite_master WHERE type='index' AND name=:name"
    return conn.execute(text(statement), {"name": name}).scalar_one_or_none() is not None


def index_definition(conn: Connection, name: str, table: str) -> str | None:
    """The stored ``CREATE INDEX`` text of ``name`` on ``table``, or None."""
    row = conn.execute(
        text("SELECT tbl_name, sql FROM sqlite_master WHERE type = 'index' AND name = :name"),
        {"name": name},
    ).first()
    if row is None or row[0] != table or row[1] is None:
        return None
    return str(row[1])


def read_columns(conn: Connection, tables: list[str]) -> dict[str, list[str]]:
    """Table name -> column names via the inspector (also the fallback for other dialects)."""
    inspector = inspect(conn)
    wanted = set(tables)
    present = [name for name in inspector.get_table_names() if name in wanted]
    if not present:
        return {}
    columns = inspector.get_multi_columns(filter_names=present)
    return {table: [c["name"] for c in cols] for (_schema, table), cols in columns.items()}


def read_index_entries(conn: Connection) -> dict[str, tuple[str, str, bool]]:
    """Explicit indexes: name -> (table, stored SQL, True)."""
    return {
        row[0]: (row[1], row[2], True)
        for row in conn.execute(
            text(
                "SELECT name, tbl_name, sql FROM sqlite_master "
                "WHERE type='index' AND sql IS NOT NULL"
            )
        )
    }


def matching_unique_constraints(conn: Connection, model_tables: list[Table]) -> set[str]:
    """Declared unique constraints whose live columns match (inline, so not in the index list)."""
    names: set[str] = set()
    inspector = inspect(conn)
    for model_table in model_tables:
        declared = [
            c for c in model_table.constraints if isinstance(c, UniqueConstraint) and c.name
        ]
        if not declared or not inspector.has_table(model_table.name):
            continue
        constraints = {
            item["name"]: item["column_names"]
            for item in inspector.get_unique_constraints(model_table.name)
        }
        for constraint in declared:
            if constraints.get(str(constraint.name)) == [c.name for c in constraint.columns]:
                names.add(str(constraint.name))
    return names


@contextmanager
def maintenance_lock(conn: Connection) -> Iterator[None]:
    """Activation: an exclusive file lock retained across committed batches."""
    timeout = conn.exec_driver_sql("PRAGMA busy_timeout").scalar_one()
    conn.exec_driver_sql("PRAGMA busy_timeout=0")
    conn.exec_driver_sql("PRAGMA locking_mode=EXCLUSIVE")
    try:
        try:
            conn.exec_driver_sql("BEGIN EXCLUSIVE")
        except OperationalError:
            raise RuntimeError(
                "Cannot activate SQLite while another connection holds a live transaction. "
                "Stop other processes, finish the transaction and retry."
            ) from None
        yield
    finally:
        conn.rollback()
        conn.exec_driver_sql("PRAGMA locking_mode=NORMAL")
        # A transaction releases the retained exclusive lock when switching back.
        conn.exec_driver_sql("BEGIN")
        conn.exec_driver_sql("COMMIT")
        conn.exec_driver_sql(f"PRAGMA busy_timeout={int(timeout)}")
        conn.commit()
