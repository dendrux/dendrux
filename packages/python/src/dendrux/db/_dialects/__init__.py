"""Per-dialect schema maintenance: catalog reads, index checks, locks and bounded DDL.

``postgres``, ``sqlite`` and ``mysql`` hold everything that differs by database for
schema checks, activation and migrations. The shared modules (``_schema_check``,
``_index_check``, ``_activate``, ``_migration_ddl``) dispatch here by dialect. One
SQLAlchemy dialect name, ``mysql``, reaches both MySQL and MariaDB; the connected server
decides the differences (``dialect.is_mariadb``). URLs with the ``mariadb`` dialect name
are treated as the same family.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from sqlalchemy.engine import Dialect

Family = Literal["postgresql", "sqlite", "mysql"]
MYSQL_FAMILY = frozenset({"mysql", "mariadb"})


def family(dialect: Dialect) -> Family | None:
    """The supported database family of ``dialect``, or None for any other database."""
    name = dialect.name
    if name in MYSQL_FAMILY:
        return "mysql"
    if name == "postgresql":
        return "postgresql"
    if name == "sqlite":
        return "sqlite"
    return None


def is_mysql_family(dialect: Dialect) -> bool:
    """True for MySQL and MariaDB, whichever SQLAlchemy dialect name reached them."""
    return dialect.name in MYSQL_FAMILY


def require_family(dialect: Dialect, what: str) -> Family:
    """The family, or a clear refusal naming the supported databases."""
    found = family(dialect)
    if found is None:
        raise RuntimeError(f"{what} supports PostgreSQL, SQLite and MySQL/MariaDB only.")
    return found
