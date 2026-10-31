"""MySQL 8.4 / MariaDB 10.11 schema: replacement indexes, checks, locks and bootstrap.

The models' expression index (tenant idempotency) and partial index (started events)
do not exist on these servers. In their place, under the same names:

- ``agent_runs.tenant_scope``: a stored generated ``COALESCE(tenant_id, '')`` column
  with the NO PAD collation, and the unique index ``(tenant_scope, idempotency_key)``.
  Lookups by tenant and key must compare ``tenant_scope`` (:func:`tenant_scope_column`);
  MariaDB does not match the expression to the index.
- ``ix_run_events_started`` on ``(agent_run_id, event_type)``.
"""

from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING, Any

from sqlalchemy import String, insert, literal_column, text
from sqlalchemy.dialects.mysql import insert as mysql_insert
from sqlalchemy.exc import DBAPIError, IntegrityError

from dendrux.db._dialects._mysql_types import nopad_collation
from dendrux.db.models import DendruxSetting

if TYPE_CHECKING:
    from sqlalchemy import Connection
    from sqlalchemy.sql.elements import ColumnElement
    from sqlalchemy.sql.selectable import FromClause

TENANT_SCOPE_COLUMN = "tenant_scope"
TENANT_SCOPE_EXPRESSION = "COALESCE(tenant_id, '')"

EXTRA_INDEXES: dict[str, tuple[str, bool, tuple[str, ...]]] = {
    "uq_agent_runs_tenant_idempotency_key": (
        "agent_runs",
        True,
        (TENANT_SCOPE_COLUMN, "idempotency_key"),
    ),
    "ix_run_events_started": ("run_events", False, ("agent_run_id", "event_type")),
}
"""MySQL-only index name -> (table, unique, columns). Replaces the model definitions."""


def tenant_scope_ddl(dialect: Any) -> str:
    """``ALTER TABLE`` adding the generated tenant scope with the server's collation."""
    return (
        f"ALTER TABLE agent_runs ADD COLUMN {TENANT_SCOPE_COLUMN} VARCHAR(255) "
        f"CHARACTER SET utf8mb4 COLLATE {nopad_collation(dialect)} "
        f"GENERATED ALWAYS AS ({TENANT_SCOPE_EXPRESSION}) STORED"
    )


def extra_index_ddl(name: str) -> str:
    table, unique, columns = EXTRA_INDEXES[name]
    kind = "UNIQUE INDEX" if unique else "INDEX"
    return f"CREATE {kind} {name} ON {table} ({', '.join(columns)})"


def tenant_scope_column(table: FromClause) -> ColumnElement[str]:
    """``<table>.tenant_scope`` for lookups by tenant and idempotency key on MySQL/MariaDB.

    Compare it with ``tenant_id or ''``. The column is not mapped on the model (it does
    not exist on Postgres or SQLite).
    """
    name = getattr(table, "name", "agent_runs")
    return literal_column(f"{name}.{TENANT_SCOPE_COLUMN}", type_=String(255))


def create_extras(connection: Any, table: str) -> None:
    """Create the MySQL-only objects of ``table`` (``create_all``'s after-create hook)."""
    if table == "agent_runs":
        connection.execute(text(tenant_scope_ddl(connection.dialect)))
    for name, (owner, _, _) in EXTRA_INDEXES.items():
        if owner == table:
            connection.execute(text(extra_index_ddl(name)))


DUPLICATE_KEY = 1062
LOCK_WAIT_TIMEOUT = 1205


def errno(error: BaseException) -> int | None:
    """The MySQL/MariaDB error number of a driver error (asyncmy, aiomysql, PyMySQL)."""
    orig = error.orig if isinstance(error, DBAPIError) else error
    args = getattr(orig, "args", ())
    return args[0] if args and isinstance(args[0], int) else None


def put_setting(conn: Connection, key: str, value: str, *, overwrite: bool) -> None:
    """Write one setting in the caller's transaction (``key`` is the only unique key).

    Overwrite is an upsert; otherwise a plain INSERT whose duplicate-key error (the
    primary key, the table's only unique key) means the setting already exists. MySQL
    rolls back only the failed statement, so the transaction stays usable.
    """
    row = {"key": key, "value": value, "updated_at": dt.datetime.now(dt.UTC)}
    if overwrite:
        upsert = mysql_insert(DendruxSetting).values(**row)
        conn.execute(
            upsert.on_duplicate_key_update(
                value=upsert.inserted.value, updated_at=upsert.inserted.updated_at
            )
        )
        return
    try:
        conn.execute(insert(DendruxSetting).values(**row))
    except IntegrityError as error:
        if errno(error) != DUPLICATE_KEY:
            raise
