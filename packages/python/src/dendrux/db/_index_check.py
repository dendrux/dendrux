"""Conservative index-definition checks for migration replacements and startup."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from sqlalchemy import text

if TYPE_CHECKING:
    from sqlalchemy import Connection


def _canonical(ddl: str) -> str:
    # Protect SQL string literals before normalizing catalog decoration.
    literals: list[str] = []

    def protect(match: re.Match[str]) -> str:
        literals.append(match.group())
        return f"@literal{len(literals) - 1}@"

    ddl = re.sub(r"'(?:''|[^'])*'", protect, ddl)
    ddl = re.sub(r"\bIF NOT EXISTS\b", "", ddl, flags=re.I)
    ddl = re.sub(r"\bUSING btree\b", "", ddl, flags=re.I)
    for i, value in enumerate(literals):
        if value == "''":
            ddl = re.sub(
                rf"@literal{i}@::(?:character varying|text)\b", f"@literal{i}@", ddl, flags=re.I
            )
    ddl = re.sub(r'"([a-z_][a-z_0-9]*)"', r"\1", ddl)
    ddl = re.sub(r"\bWHERE\s+\(\s*([a-z_]+\s+IS NOT NULL)\s*\)", r"WHERE \1", ddl, flags=re.I)
    tokens = re.split(r'("(?:""|[^"])*")', ddl)
    ddl = "".join(
        token if i % 2 else re.sub(r"\s+", "", token).lower() for i, token in enumerate(tokens)
    )
    return re.sub(r"@literal(\d+)@", lambda m: literals[int(m[1])], ddl)


def read_pg_indexes(conn: Connection, *, tables: list[str]) -> dict[str, tuple[str, str, bool]]:
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


def index_matches(conn: Connection, name: str, table: str, expected: str) -> bool:
    """Require a valid, ready index of the exact intended definition and table."""
    if conn.dialect.name == "postgresql":
        entry = read_pg_indexes(conn, tables=[table]).get(name)
        if entry is None or entry[0] != table or not entry[2]:
            return False
        actual = entry[1]
    elif conn.dialect.name == "sqlite":
        row = conn.execute(
            text("SELECT tbl_name, sql FROM sqlite_master WHERE type = 'index' AND name = :name"),
            {"name": name},
        ).first()
        if row is None or row[0] != table or row[1] is None:
            return False
        actual = row[1]
    else:
        raise RuntimeError("Index verification supports PostgreSQL and SQLite only.")
    return _canonical(actual) == _canonical(expected)


def require_index(conn: Connection, name: str, table: str, expected: str) -> None:
    if not index_matches(conn, name, table, expected):
        raise RuntimeError(
            f"Index {name} has an unexpected definition or is invalid/not ready. "
            "Repair or rebuild it with the migration's exact definition before retrying; "
            "the previous uniqueness constraint has been retained."
        )
