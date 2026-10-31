"""MySQL 8.4 / MariaDB 10.11 schema: replacement indexes, checks, locks and bootstrap.

The models' expression index (tenant idempotency) and partial index (started events)
do not exist on these servers. In their place, under the same names:

- ``agent_runs.tenant_scope``: a stored generated ``COALESCE(tenant_id, '')`` column
  with the NO PAD collation, and the unique index ``(tenant_scope, idempotency_key)``.
  Lookups by tenant and key must compare ``tenant_scope`` (:func:`tenant_scope_column`);
  MariaDB does not match the expression to the index.
- ``ix_run_events_started`` on ``(agent_run_id, event_type)``.

Schema history (owner decision D2) starts from a fresh baseline: ``dendrux db migrate``
bootstraps an empty database straight to the current head (:func:`migrate`), resumes an
interrupted bootstrap, upgrades a stamped Dendrux schema with later revisions, and
refuses anything else. DDL is not transactional here, so every step is restartable and
an existing object must match the head definition exactly. Maintenance holds
``GET_LOCK`` on one dedicated connection, always released, and the connection is
invalidated on any error.
"""

from __future__ import annotations

import datetime as dt
import re
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from sqlalchemy import (
    String,
    UniqueConstraint,
    bindparam,
    delete,
    insert,
    literal_column,
    select,
    text,
)
from sqlalchemy.dialects.mysql import insert as mysql_insert
from sqlalchemy.exc import DBAPIError, IntegrityError
from sqlalchemy.schema import CreateIndex, CreateTable
from sqlalchemy.sql.elements import UnaryExpression
from sqlalchemy.sql.operators import desc_op

from dendrux.db._dialects import LockWaitTimeout
from dendrux.db._dialects._mysql_types import nopad_collation
from dendrux.db._storage_schema import ACTIVATION_INDEXES
from dendrux.db.models import Base, DendruxSetting

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Iterator

    from sqlalchemy import Column, Connection, Index
    from sqlalchemy.engine import Dialect
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

MYSQL_BASELINE_REVISION = "f9a0b1c2d3e4"
"""The 0.6.0 revision MySQL/MariaDB history starts at (fresh baseline, D2)."""

BOOTSTRAP_MARKER = "bootstrap_in_progress"
SETTINGS_TABLE = "dendrux_settings"
VERSION_TABLE = "alembic_version"
DUPLICATE_KEY = 1062
LOCK_WAIT_TIMEOUT = 1205
_LOCK_PREFIX = "dendrux_maint_"

_before_step: Callable[[str], None] | None = None
"""Test seam: called with each bootstrap step's name before the step runs."""


# ---------------------------------------------------------------------------
# MySQL-only objects
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Errors and settings
# ---------------------------------------------------------------------------


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


def read_settings(conn: Connection, *, limit: int = 100) -> dict[str, str]:
    """Up to ``limit`` settings rows (the table holds a handful)."""
    rows = conn.execute(select(DendruxSetting.key, DendruxSetting.value).limit(limit))
    return {str(key): str(value) for key, value in rows}


# ---------------------------------------------------------------------------
# Head definition and catalog
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ColumnShape:
    data_type: str
    length: int | None
    precision: tuple[int, ...] | None
    nullable: bool
    default: str | None
    collation: str | None
    generated: str | None = None


@dataclass(frozen=True)
class IndexShape:
    table: str
    unique: bool
    columns: tuple[tuple[str, str], ...]


@dataclass
class TableShape:
    engine: str
    columns: dict[str, ColumnShape]
    primary_key: tuple[str, ...]
    foreign_keys: frozenset[tuple[tuple[str, ...], str, tuple[str, ...], str]]
    uniques: dict[str, IndexShape] = field(default_factory=dict)


@dataclass
class Catalog:
    tables: dict[str, TableShape]
    indexes: dict[tuple[str, str], IndexShape]
    """Keyed by (table, index name): MySQL index names are only unique per table."""


_TYPES = {
    "VARCHAR": "varchar",
    "CHAR": "char",
    "LONGTEXT": "longtext",
    "LONGBLOB": "longblob",
    "DATETIME": "datetime",
    "INTEGER": "int",
    "SMALLINT": "smallint",
    "BOOL": "tinyint",
    "NUMERIC": "decimal",
}


def _is_mariadb(dialect: Any) -> bool:
    return dialect.name == "mariadb" or bool(getattr(dialect, "is_mariadb", False))


def _normal_default(value: Any) -> str | None:
    if value is None:
        return None
    text_value = str(value).strip()
    if text_value.upper() == "NULL":
        return None
    while text_value.startswith("(") and text_value.endswith(")"):
        text_value = text_value[1:-1].strip()
    if len(text_value) >= 2 and text_value[0] == text_value[-1] == "'":
        text_value = text_value[1:-1]
    lowered = text_value.lower()
    return {"false": "0", "true": "1"}.get(lowered, lowered)


def _normal_expression(value: Any) -> str | None:
    if not value:
        return None
    return re.sub(r"\s+|`|\\|_utf8mb4", "", str(value)).lower()


def _expected_column(column: Column[Any], dialect: Dialect, ddl: Any) -> ColumnShape:
    spec = str(dialect.type_compiler_instance.process(column.type, type_expression=column))
    match = re.match(r"(\w+)(?:\((\d+)(?:,\s*(\d+))?\))?", spec)
    if match is None or (match[1].upper() not in _TYPES and match[1].upper() != "JSON"):
        raise RuntimeError(f"No MySQL catalog mapping for {column.table.name}.{column.name}")
    kind = match[1].upper()
    args = tuple(int(a) for a in match.groups()[1:] if a is not None)
    # MariaDB's JSON is a LONGTEXT alias with a JSON_VALID check.
    mariadb_json = "longtext" if _is_mariadb(dialect) else "json"
    data_type = mariadb_json if kind == "JSON" else _TYPES[kind]
    collation = re.search(r"COLLATE (\w+)", spec)
    return ColumnShape(
        data_type=data_type,
        length=args[0] if data_type in {"varchar", "char"} else None,
        precision=args if data_type in {"decimal", "datetime"} else None,
        nullable=bool(column.nullable),
        default=_normal_default(ddl.get_column_default_string(column)),
        collation=collation[1] if collation and data_type in {"varchar", "char"} else None,
    )


def tenant_scope_shape(dialect: Dialect) -> ColumnShape:
    return ColumnShape(
        data_type="varchar",
        length=255,
        precision=None,
        nullable=True,
        default=None,
        collation=nopad_collation(dialect),
        generated="stored:" + str(_normal_expression(TENANT_SCOPE_EXPRESSION)),
    )


def _index_columns(index: Index) -> tuple[tuple[str, str], ...]:
    columns: list[tuple[str, str]] = []
    for expression in index.expressions:
        if isinstance(expression, UnaryExpression) and expression.modifier is desc_op:
            columns.append((str(getattr(expression.element, "name", "")), "D"))
        elif hasattr(expression, "name") and hasattr(expression, "table"):
            columns.append((str(expression.name), "A"))
        else:
            raise RuntimeError(f"Index {index.name} has no MySQL-family form")
    return tuple(columns)


def expected_tables(dialect: Dialect) -> dict[str, TableShape]:
    """Every model table as this server must hold it (``agent_runs`` with ``tenant_scope``)."""
    ddl = dialect.ddl_compiler(dialect, None)  # type: ignore[arg-type]
    tables: dict[str, TableShape] = {}
    for table in Base.metadata.sorted_tables:
        columns = {c.name: _expected_column(c, dialect, ddl) for c in table.columns}
        if table.name == "agent_runs":
            columns[TENANT_SCOPE_COLUMN] = tenant_scope_shape(dialect)
        tables[table.name] = TableShape(
            engine="InnoDB",
            columns=columns,
            primary_key=tuple(c.name for c in table.primary_key.columns),
            foreign_keys=frozenset(
                (
                    tuple(c.name for c in fk.columns),
                    fk.referred_table.name,
                    tuple(e.column.name for e in fk.elements),
                    (fk.ondelete or "NO ACTION").upper(),
                )
                for fk in table.foreign_key_constraints
            ),
            uniques={
                str(c.name): IndexShape(
                    table.name, True, tuple((col.name, "A") for col in c.columns)
                )
                for c in table.constraints
                if isinstance(c, UniqueConstraint) and c.name
            },
        )
    return tables


def _model_indexes() -> list[Index]:
    return [
        index
        for table in Base.metadata.sorted_tables
        for index in sorted(table.indexes, key=lambda i: str(i.name))
        if str(index.name) not in EXTRA_INDEXES
    ]


def expected_indexes() -> dict[str, IndexShape]:
    """Named indexes in creation order: the model's, then the MySQL-only replacements."""
    shapes = {
        str(index.name): IndexShape(
            str(getattr(index.table, "name", "")), bool(index.unique), _index_columns(index)
        )
        for index in _model_indexes()
    }
    for name, (table, unique, columns) in EXTRA_INDEXES.items():
        shapes[name] = IndexShape(table, unique, tuple((c, "A") for c in columns))
    return shapes


def read_catalog(conn: Connection, tables: Collection[str]) -> Catalog:
    """Four catalog queries for ``tables`` in the current database."""
    names = sorted(tables)
    params = {"names": names}
    catalog = Catalog(tables={}, indexes={})
    if not names:
        return catalog
    for name, engine in conn.execute(
        text(
            "SELECT TABLE_NAME, ENGINE FROM information_schema.tables "
            "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME IN :names"
        ).bindparams(_expanding("names")),
        params,
    ):
        catalog.tables[name] = TableShape(str(engine), {}, (), frozenset())
    for row in conn.execute(
        text(
            "SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE, CHARACTER_MAXIMUM_LENGTH, "
            "NUMERIC_PRECISION, NUMERIC_SCALE, DATETIME_PRECISION, IS_NULLABLE, "
            "COLUMN_DEFAULT, COLLATION_NAME, EXTRA, GENERATION_EXPRESSION "
            "FROM information_schema.columns "
            "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME IN :names"
        ).bindparams(_expanding("names")),
        params,
    ):
        shape = catalog.tables.get(row[0])
        if shape is None:
            continue
        data_type = str(row[2]).lower()
        precision: tuple[int, ...] | None = None
        if data_type == "decimal":
            precision = (int(row[4]), int(row[5]))
        elif data_type == "datetime":
            precision = (int(row[6] or 0),)
        expression = _normal_expression(row[11])
        stored = "stored" if "STORED" in str(row[10] or "").upper() else "virtual"
        shape.columns[row[1]] = ColumnShape(
            data_type=data_type,
            length=int(row[3]) if data_type in {"varchar", "char"} else None,
            precision=precision,
            nullable=row[7] == "YES",
            default=None if expression else _normal_default(row[8]),
            collation=row[9] if data_type in {"varchar", "char"} else None,
            generated=f"{stored}:{expression}" if expression else None,
        )
    grouped: dict[tuple[str, str], tuple[bool, list[tuple[str, str]]]] = {}
    for table, index, non_unique, column, collation, sub_part in conn.execute(
        text(
            "SELECT TABLE_NAME, INDEX_NAME, NON_UNIQUE, COLUMN_NAME, COLLATION, SUB_PART "
            "FROM information_schema.statistics "
            "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME IN :names "
            "ORDER BY TABLE_NAME, INDEX_NAME, SEQ_IN_INDEX"
        ).bindparams(_expanding("names")),
        params,
    ):
        entry = grouped.setdefault((table, index), (not int(non_unique), []))
        name = str(column) if column is not None else "<expression>"
        if sub_part is not None:
            name = f"{name}({int(sub_part)})"
        entry[1].append((name, str(collation or "A")))
    for (table, index), (unique, columns) in grouped.items():
        if table not in catalog.tables:
            continue
        if index == "PRIMARY":
            catalog.tables[table].primary_key = tuple(c for c, _ in columns)
        else:
            catalog.indexes[(table, index)] = IndexShape(table, unique, tuple(columns))
    foreign: dict[tuple[str, str], tuple[list[str], str, list[str], str]] = {}
    for table, name, column, referenced, referenced_column, rule in conn.execute(
        text(
            "SELECT k.TABLE_NAME, k.CONSTRAINT_NAME, k.COLUMN_NAME, k.REFERENCED_TABLE_NAME, "
            "k.REFERENCED_COLUMN_NAME, r.DELETE_RULE "
            "FROM information_schema.key_column_usage k "
            "JOIN information_schema.referential_constraints r "
            "ON r.CONSTRAINT_SCHEMA = k.CONSTRAINT_SCHEMA "
            "AND r.CONSTRAINT_NAME = k.CONSTRAINT_NAME AND r.TABLE_NAME = k.TABLE_NAME "
            "WHERE k.TABLE_SCHEMA = DATABASE() AND k.TABLE_NAME IN :names "
            "AND k.REFERENCED_TABLE_NAME IS NOT NULL "
            "ORDER BY k.TABLE_NAME, k.CONSTRAINT_NAME, k.ORDINAL_POSITION"
        ).bindparams(_expanding("names")),
        params,
    ):
        entry_fk = foreign.setdefault((table, name), ([], str(referenced), [], str(rule)))
        entry_fk[0].append(str(column))
        entry_fk[2].append(str(referenced_column))
    for (table, _), (columns_fk, referenced, ref_columns, rule) in foreign.items():
        if table in catalog.tables:
            shape = catalog.tables[table]
            shape.foreign_keys = shape.foreign_keys | {
                (tuple(columns_fk), referenced, tuple(ref_columns), rule.upper())
            }
    return catalog


def _expanding(name: str) -> Any:
    return bindparam(name, expanding=True)


def owned_tables() -> set[str]:
    """The table names Dendrux owns in a database it may share with the app."""
    return set(Base.metadata.tables) | {VERSION_TABLE}


def base_tables(conn: Connection, names: Collection[str]) -> set[str]:
    """Which of ``names`` exist as base tables (never lists the app's own tables)."""
    rows = conn.execute(
        text(
            "SELECT TABLE_NAME FROM information_schema.tables "
            "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_TYPE = 'BASE TABLE' "
            "AND TABLE_NAME IN :names"
        ).bindparams(_expanding("names")),
        {"names": sorted(names)},
    )
    return {str(row[0]) for row in rows}


def table_problems(
    name: str, expected: TableShape, actual: TableShape, *, exact: bool, scope_optional: bool
) -> list[str]:
    """Differences between a live table and its head definition (names only, no data)."""
    problems: list[str] = []
    if actual.engine != expected.engine:
        problems.append(f"{name} (engine {actual.engine}, expected {expected.engine})")
    for column, want in expected.columns.items():
        have = actual.columns.get(column)
        if have is None:
            if not (scope_optional and column == TENANT_SCOPE_COLUMN):
                problems.append(f"{name}.{column}")
            continue
        for attribute in ("data_type", "length", "precision", "nullable", "default"):
            if getattr(have, attribute) != getattr(want, attribute):
                problems.append(
                    f"{name}.{column} ({attribute} {getattr(have, attribute)}, "
                    f"expected {getattr(want, attribute)})"
                )
        if have.collation != want.collation:
            problems.append(
                f"{name}.{column} (collation {have.collation}, expected {want.collation})"
            )
        if have.generated != want.generated:
            problems.append(f"{name}.{column} (generated column differs)")
    if exact:
        problems.extend(
            f"{name}.{column} (unexpected column)"
            for column in actual.columns
            if column not in expected.columns
        )
    if actual.primary_key != expected.primary_key:
        problems.append(
            f"{name} (primary key {actual.primary_key}, expected {expected.primary_key})"
        )
    for key in sorted(expected.foreign_keys - actual.foreign_keys):
        problems.append(
            f"{name} (foreign key {', '.join(key[0])} -> {key[1]} missing or different)"
        )
    if exact:
        for key in sorted(actual.foreign_keys - expected.foreign_keys):
            problems.append(f"{name} (unexpected foreign key {', '.join(key[0])} -> {key[1]})")
    return problems


@dataclass
class SchemaReport:
    present: set[str]
    missing_tables: list[str]
    columns: list[str]
    indexes: list[str]

    def problems(self) -> list[str]:
        return [*self.columns, *self.indexes, *(f"table {t}" for t in self.missing_tables)]


def check(conn: Connection, *, exact: bool = False, settings: bool = True) -> SchemaReport:
    """Compare the live schema with the head definition (read-only, five queries).

    ``exact`` also reports extra columns and foreign keys (the bootstrap's rule); the
    runtime check ignores extras, as on Postgres. Extra unique indexes are always
    reported: they can reject writes the store expects to succeed.
    """
    dialect = conn.dialect
    wanted = expected_tables(dialect)
    catalog = read_catalog(conn, wanted)
    report = SchemaReport(set(catalog.tables), [], [], [])
    for name, shape in wanted.items():
        actual = catalog.tables.get(name)
        if actual is None:
            report.missing_tables.append(name)
            continue
        report.columns.extend(
            table_problems(name, shape, actual, exact=exact, scope_optional=False)
        )
    expected = _all_index_shapes(wanted)
    for name, want in expected.items():
        if want.table not in catalog.tables:
            continue
        have = catalog.indexes.get((want.table, name))
        if have is None:
            report.indexes.append(f"{want.table}.{name} (index)")
        elif have != want:
            report.indexes.append(f"{want.table}.{name} (index definition differs)")
    for (table, name), found in sorted(catalog.indexes.items()):
        owner_shape = expected.get(name)
        if found.unique and (owner_shape is None or owner_shape.table != table):
            report.indexes.append(f"{table}.{name} (unexpected unique index)")
    for owner, _, old in ACTIVATION_INDEXES.values():
        if old and (owner, old) in catalog.indexes:
            report.indexes.append(f"{owner}.{old} (replaced index must be absent)")
    if settings and SETTINGS_TABLE in catalog.tables and not report.missing_tables:
        values = read_settings(conn)
        if BOOTSTRAP_MARKER in values:
            report.columns.append(
                f"{SETTINGS_TABLE}.{BOOTSTRAP_MARKER} (bootstrap unfinished; "
                "run `dendrux db migrate`)"
            )
        if values.get("storage_format") != "2" or values.get("activate_step") != "done":
            report.columns.append(
                f"{SETTINGS_TABLE}.storage_format (expected an active format 2 schema)"
            )
    return report


def _all_index_shapes(tables: dict[str, TableShape]) -> dict[str, IndexShape]:
    """Named indexes plus the tables' unique constraints (both live in the index catalog)."""
    shapes = expected_indexes()
    for table in tables.values():
        shapes.update(table.uniques)
    return shapes


def index_names(conn: Connection) -> tuple[set[str], set[str]]:
    """(names whose definition matches the head, every existing index name)."""
    wanted = expected_tables(conn.dialect)
    catalog = read_catalog(conn, wanted)
    expected = _all_index_shapes(wanted)
    valid = {
        name for name, want in expected.items() if catalog.indexes.get((want.table, name)) == want
    }
    return valid, {name for _, name in catalog.indexes}


def read_columns(conn: Connection, tables: list[str]) -> dict[str, list[str]]:
    """Table name -> column names for the listed tables (one catalog query)."""
    if not tables:
        return {}
    rows = conn.execute(
        text(
            "SELECT TABLE_NAME, COLUMN_NAME FROM information_schema.columns "
            "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME IN :names "
            "ORDER BY TABLE_NAME, ORDINAL_POSITION"
        ).bindparams(_expanding("names")),
        {"names": tables},
    )
    actual: dict[str, list[str]] = {}
    for table, column in rows:
        actual.setdefault(str(table), []).append(str(column))
    return actual


_DDL = re.compile(
    r"^\s*CREATE\s+(UNIQUE\s+)?INDEX\s+(?:CONCURRENTLY\s+)?(?:IF NOT EXISTS\s+)?"
    r"`?(\w+)`?\s+ON\s+`?(\w+)`?\s*\((.*)\)\s*$",
    re.I | re.S,
)


def index_exists(conn: Connection, name: str) -> bool:
    row = conn.execute(
        text(
            "SELECT 1 FROM information_schema.statistics "
            "WHERE TABLE_SCHEMA = DATABASE() AND INDEX_NAME = :name LIMIT 1"
        ),
        {"name": name},
    ).first()
    return row is not None


def index_matches(conn: Connection, name: str, table: str, expected: str) -> bool:
    """Whether index ``name`` on ``table`` has the plain-column shape ``expected`` states."""
    match = _DDL.match(expected)
    if match is None or match[2] != name or match[3] != table:
        return False
    columns = []
    for part in match[4].split(","):
        words = part.replace("`", "").split()
        direction = "D" if len(words) > 1 and words[1].upper() == "DESC" else "A"
        columns.append((words[0], direction))
    have = read_catalog(conn, [table]).indexes.get((table, name))
    return have == IndexShape(table, bool(match[1]), tuple(columns))


# ---------------------------------------------------------------------------
# Maintenance lock and bounded DDL
# ---------------------------------------------------------------------------


def lock_name(conn: Connection) -> str:
    """The maintenance lock of the current database (``GET_LOCK`` names are server-wide)."""
    name = conn.execute(text(f"SELECT CONCAT('{_LOCK_PREFIX}', SHA1(DATABASE()))")).scalar()
    if name is None:
        raise RuntimeError("Connect to a database: the URL names none.")
    return str(name)


@contextmanager
def maintenance_lock(conn: Connection) -> Iterator[None]:
    """Hold the database's maintenance ``GET_LOCK`` on this connection, made dedicated.

    The connection is detached from its pool first, so it is never handed to another
    caller, and closed when the caller closes it. The lock is session-scoped (commit and
    rollback do not release it). Acquisition is inside the protected region: on any
    error, including one right after ``GET_LOCK`` succeeded, the connection is
    invalidated, so the server drops the session and any lock it holds. On success it is
    released with ``RELEASE_LOCK``; a failed release invalidates as well.
    """
    conn.detach()
    error: BaseException | None = None
    name = ""
    try:
        name = lock_name(conn)
        acquired = conn.execute(text("SELECT GET_LOCK(:name, 0)"), {"name": name}).scalar()
        conn.commit()
        if acquired != 1:
            raise RuntimeError(
                "Another Dendrux database migration or activation is running; retry when it "
                "finishes."
            )
        yield
    except BaseException as exc:
        error = exc
        raise
    finally:
        if error is not None:
            _drop_session(conn)
        else:
            try:
                conn.rollback()
                conn.execute(text("SELECT RELEASE_LOCK(:name)"), {"name": name})
                conn.commit()
            except BaseException:
                _drop_session(conn)
                raise


def _drop_session(conn: Connection) -> None:
    """Close the server session (and with it any lock it holds), then invalidate.

    A detached connection has no pool record, so ``invalidate()`` alone would only
    forget the driver connection, leaving the session (and its lock) open.
    """
    try:
        raw = conn.connection.dbapi_connection
        if raw is not None:
            raw.close()
    except Exception:  # already closed or broken: the invalidation below still runs
        pass
    conn.invalidate()


def bounded_execute(
    conn: Connection, statement: str, *, timeout_seconds: int, transactional: bool = False
) -> None:
    """Run one DDL statement waiting at most ``timeout_seconds`` for metadata/row locks.

    DDL commits implicitly here, so ``transactional`` has no effect. The session's
    previous waits are restored. A lock timeout (1205) raises :class:`LockWaitTimeout`.
    """
    seconds = int(timeout_seconds)
    previous = conn.execute(
        text("SELECT @@SESSION.lock_wait_timeout, @@SESSION.innodb_lock_wait_timeout")
    ).one()
    conn.execute(
        text(
            f"SET SESSION lock_wait_timeout = {seconds}, "
            f"SESSION innodb_lock_wait_timeout = {seconds}"
        )
    )
    try:
        conn.execute(text(statement))
    except DBAPIError as error:
        if errno(error) == LOCK_WAIT_TIMEOUT:
            raise LockWaitTimeout from None
        raise
    finally:
        conn.execute(
            text(
                f"SET SESSION lock_wait_timeout = {int(previous[0])}, "
                f"SESSION innodb_lock_wait_timeout = {int(previous[1])}"
            )
        )


# ---------------------------------------------------------------------------
# Migrate: bootstrap, resume, upgrade or refuse
# ---------------------------------------------------------------------------


def bootstrap_steps(dialect: Dialect) -> list[str]:
    """The fixed bootstrap order (each step is restartable)."""
    steps = ["create:" + SETTINGS_TABLE, "marker"]
    steps += [f"create:{t.name}" for t in Base.metadata.sorted_tables if t.name != SETTINGS_TABLE]
    steps.append(f"column:agent_runs.{TENANT_SCOPE_COLUMN}")
    steps += [f"index:{name}" for name in expected_indexes()]
    steps += ["seed", "stamp", "clear_marker"]
    return steps


def log_step(step: str) -> str | None:
    """The registered ``migration_lock_timeout`` step of a bootstrap step (None: no DDL)."""
    kind, _, target = step.partition(":")
    if kind == "create":
        return f"create.{target}"
    if kind == "column":
        return f"add.{target}"
    if kind == "index":
        return f"build_index.{target}"
    return "stamp" if step == "stamp" else None


def upgradeable_revisions(script: Any, head: str) -> set[str]:
    """Revisions a stamped MySQL database may upgrade from: the baseline and its
    descendants, except head (``iterate_revisions`` leaves the lower bound out)."""
    between = {rev.revision for rev in script.iterate_revisions(head, MYSQL_BASELINE_REVISION)}
    return (between | {MYSQL_BASELINE_REVISION}) - {head}


def _refuse(reason: str) -> RuntimeError:
    return RuntimeError(f"Cannot migrate this MySQL/MariaDB database: {reason}")


def classify(conn: Connection, *, head: str, upgradeable: Collection[str]) -> str:
    """``bootstrap`` | ``current`` | ``upgrade``, or a refusal.

    Only Dendrux-owned objects decide (its table names, the Alembic version table and the
    settings marker): the database may also hold the app's tables, which are never read
    or touched.
    """
    present = base_tables(conn, owned_tables())
    if not present:
        return "bootstrap"
    if SETTINGS_TABLE in present:
        try:
            values = read_settings(conn)
        except DBAPIError:
            raise _refuse(f"{SETTINGS_TABLE} is not a Dendrux settings table.") from None
        marker = values.get(BOOTSTRAP_MARKER)
        if marker is not None:
            if marker != head:
                raise _refuse(
                    f"a bootstrap by another Dendrux version (revision {marker}) is "
                    f"unfinished; this version bootstraps {head}. Finish it with that "
                    "version, or drop the partial schema."
                )
            return "bootstrap"
        if present == {SETTINGS_TABLE} and not values:
            return "bootstrap"  # crashed between creating the table and writing the marker
    if VERSION_TABLE in present:
        revisions = [
            str(r) for r in conn.execute(text("SELECT version_num FROM alembic_version")).scalars()
        ]
        if revisions == [head]:
            return "current"
        if len(revisions) == 1 and revisions[0] in upgradeable:
            return "upgrade"
        shown = ", ".join(revisions) or "none"
        raise _refuse(
            f"it is stamped with revision {shown}, which is not a Dendrux MySQL revision "
            f"(MySQL history starts at {MYSQL_BASELINE_REVISION})."
        )
    raise _refuse(
        f"it holds Dendrux tables in an unrecognised state ({', '.join(sorted(present))}) "
        "with no bootstrap marker and no Dendrux revision. Drop them, or point Dendrux at "
        "another database."
    )


def migrate(
    conn: Connection,
    *,
    head: str,
    upgradeable: Collection[str],
    upgrade: Callable[[], None],
) -> str:
    """Bring the database to ``head`` under the maintenance lock; return what was done."""
    with maintenance_lock(conn):
        state = classify(conn, head=head, upgradeable=upgradeable)
        conn.commit()
        if state == "bootstrap":
            _bootstrap(conn, head)
        elif state == "current":
            problems = check(conn, exact=True).problems()
            if problems:
                raise _refuse(
                    f"it is stamped {head} but differs from it: {'; '.join(problems[:10])}"
                )
        else:
            upgrade()
        conn.commit()
        return state


def _step(name: str) -> None:
    hook = _before_step
    if hook is not None:
        hook(name)


def _bootstrap(conn: Connection, head: str) -> None:
    from dendrux.db._migration_ddl import bounded_step

    dialect = conn.dialect
    wanted = expected_tables(dialect)
    catalog = read_catalog(conn, [*wanted, VERSION_TABLE])
    conn.commit()

    def ddl(statement: str, step: str) -> None:
        bounded_step(conn, statement, revision=head, step=log_step(step) or step)

    def table(name: str) -> None:
        _step(f"create:{name}")
        actual = catalog.tables.get(name)
        if actual is None:
            model = Base.metadata.tables[name]
            ddl(str(CreateTable(model).compile(dialect=dialect)).strip(), f"create:{name}")
            return
        problems = table_problems(name, wanted[name], actual, exact=True, scope_optional=True)
        problems += [
            f"{name}.{unique} (unique constraint)"
            for unique, shape in wanted[name].uniques.items()
            if catalog.indexes.get((name, unique)) != shape
        ]
        if problems:
            raise _refuse(
                f"the existing table {name} differs from this version's definition "
                f"({'; '.join(problems[:10])}). Drop it, or use an empty database."
            )

    table(SETTINGS_TABLE)
    _step("marker")
    if read_settings(conn).get(BOOTSTRAP_MARKER) is None:
        conn.execute(
            insert(DendruxSetting).values(
                key=BOOTSTRAP_MARKER, value=head, updated_at=dt.datetime.now(dt.UTC)
            )
        )
    conn.commit()
    for model in Base.metadata.sorted_tables:
        if model.name != SETTINGS_TABLE:
            table(model.name)

    _step(f"column:agent_runs.{TENANT_SCOPE_COLUMN}")
    scope = catalog.tables.get("agent_runs", TableShape("", {}, (), frozenset())).columns.get(
        TENANT_SCOPE_COLUMN
    )
    if scope is None:
        ddl(tenant_scope_ddl(dialect), f"column:agent_runs.{TENANT_SCOPE_COLUMN}")
    elif scope != tenant_scope_shape(dialect):
        raise _refuse("the existing agent_runs.tenant_scope column differs from its definition.")

    models = {str(index.name): index for index in _model_indexes()}
    for name, want in expected_indexes().items():
        _step(f"index:{name}")
        have = catalog.indexes.get((want.table, name))
        if have is None:
            statement = (
                str(CreateIndex(models[name]).compile(dialect=dialect))
                if name in models
                else extra_index_ddl(name)
            )
            ddl(statement, f"index:{name}")
        elif have != want:
            raise _refuse(
                f"the existing index {want.table}.{name} differs from this version's "
                "definition. Drop it, or use an empty database."
            )

    _step("seed")
    put_setting(conn, "storage_format", "2", overwrite=False)
    put_setting(conn, "activate_step", "done", overwrite=False)
    conn.commit()
    values = read_settings(conn)
    if values.get("storage_format") != "2" or values.get("activate_step") != "done":
        raise _refuse("the settings table holds a different storage format.")

    _step("stamp")
    conn.commit()
    problems = check(conn, exact=True, settings=False).problems()
    if problems:
        raise _refuse(f"the schema differs after bootstrap: {'; '.join(problems[:10])}")
    if not base_tables(conn, [VERSION_TABLE]):
        ddl(
            f"CREATE TABLE {VERSION_TABLE} (version_num VARCHAR(32) NOT NULL, "
            "CONSTRAINT alembic_version_pkc PRIMARY KEY (version_num)) ENGINE=InnoDB",
            "stamp",
        )
    revisions = [
        str(r) for r in conn.execute(text("SELECT version_num FROM alembic_version")).scalars()
    ]
    if not revisions:
        conn.execute(
            text("INSERT INTO alembic_version (version_num) VALUES (:head)"), {"head": head}
        )
    elif revisions != [head]:
        raise _refuse(f"it is stamped with {', '.join(revisions)} during bootstrap.")
    conn.commit()

    _step("clear_marker")
    conn.execute(delete(DendruxSetting).where(DendruxSetting.key == BOOTSTRAP_MARKER))
    conn.commit()
