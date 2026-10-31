"""Conservative index-definition checks for migration replacements and startup."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from dendrux.db._dialects import family, postgres, require_family, sqlite

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


read_pg_indexes = postgres.read_indexes
"""One catalog query: names mapped to table, definition, valid-and-ready (Postgres)."""


def index_exists(conn: Connection, name: str) -> bool:
    """Whether an index name exists in the current schema, regardless of its owner."""
    found = family(conn.dialect)
    if found == "postgresql":
        return postgres.index_exists(conn, name)
    if found == "mysql":
        from dendrux.db._dialects import mysql

        return mysql.index_exists(conn, name)
    return sqlite.index_exists(conn, name)


def index_matches(
    conn: Connection, name: str, table: str, expected: str, *, require_valid: bool = True
) -> bool:
    """Check exact definition and table, normally requiring a valid, ready index."""
    found = require_family(conn.dialect, "Index verification")
    if found == "mysql":
        from dendrux.db._dialects import mysql

        return mysql.index_matches(conn, name, table, expected)
    if found == "postgresql":
        actual = postgres.index_definition(conn, name, table, require_valid=require_valid)
    else:
        actual = sqlite.index_definition(conn, name, table)
    if actual is None:
        return False
    return _canonical(actual) == _canonical(expected)


def require_index(
    conn: Connection, name: str, table: str, expected: str, *, require_valid: bool = True
) -> None:
    if not index_matches(conn, name, table, expected, require_valid=require_valid):
        raise RuntimeError(
            f"Index {name} has an unexpected definition or is invalid/not ready. "
            "Repair or rebuild it with the migration's exact definition before retrying; "
            "the previous uniqueness constraint has been retained."
        )
