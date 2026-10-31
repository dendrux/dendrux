"""MySQL-family column types used by the models as dialect variants.

Each helper returns the generic type the models always had, with a MySQL/MariaDB variant,
so Postgres and SQLite DDL never changes. The NO PAD collation is chosen from the
connected server at DDL compile time (one SQLAlchemy dialect name reaches both servers):
``utf8mb4_0900_bin`` on MySQL 8.4, ``utf8mb4_nopad_bin`` on MariaDB 10.11. ``utf8mb4_bin``
is never used: it is PAD SPACE, so ``'t' = 't '`` would hold for tenants and keys.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

from sqlalchemy import CHAR, DateTime, LargeBinary, String, Text, func
from sqlalchemy.dialects.mysql import DATETIME, LONGBLOB, LONGTEXT
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.sql.functions import FunctionElement
from sqlalchemy.types import TypeDecorator, TypeEngine

MYSQL_NOPAD_COLLATION = "utf8mb4_0900_bin"
MARIADB_NOPAD_COLLATION = "utf8mb4_nopad_bin"
PAD_SPACE_BINARY_COLLATION = "utf8mb4_bin"
"""Forbidden on identifier columns: it ignores trailing spaces when comparing."""

_FAMILY = ("mysql", "mariadb")


def is_mariadb(dialect: Any) -> bool:
    """True when the dialect (connected, or named ``mariadb``) talks to MariaDB."""
    return dialect.name == "mariadb" or bool(getattr(dialect, "is_mariadb", False))


def nopad_collation(dialect: Any) -> str:
    """The NO PAD binary utf8mb4 collation of the server behind ``dialect``."""
    return MARIADB_NOPAD_COLLATION if is_mariadb(dialect) else MYSQL_NOPAD_COLLATION


class NoPadString(String):
    """``VARCHAR(n)`` in utf8mb4 with the server's NO PAD binary collation."""


class NoPadChar(CHAR):
    """``CHAR(n)`` in utf8mb4 with the server's NO PAD binary collation."""


@compiles(NoPadString)
def _compile_string(type_: NoPadString, compiler: Any, **kw: Any) -> str:
    return str(compiler.process(String(type_.length), **kw))


@compiles(NoPadChar)
def _compile_char(type_: NoPadChar, compiler: Any, **kw: Any) -> str:
    return str(compiler.process(CHAR(type_.length), **kw))


@compiles(NoPadString, "mysql")
@compiles(NoPadString, "mariadb")
def _compile_string_mysql(type_: NoPadString, compiler: Any, **kw: Any) -> str:
    collation = nopad_collation(compiler.dialect)
    return f"VARCHAR({type_.length}) CHARACTER SET utf8mb4 COLLATE {collation}"


@compiles(NoPadChar, "mysql")
@compiles(NoPadChar, "mariadb")
def _compile_char_mysql(type_: NoPadChar, compiler: Any, **kw: Any) -> str:
    collation = nopad_collation(compiler.dialect)
    return f"CHAR({type_.length}) CHARACTER SET utf8mb4 COLLATE {collation}"


class UTCDateTime(TypeDecorator[dt.datetime]):
    """``DATETIME(6)`` holding naive UTC; binds aware values only, reads back aware UTC.

    MySQL-family DATETIME has no zone and the driver drops offsets, so every value is
    converted to UTC explicitly. A naive datetime is refused rather than guessed.
    """

    impl = DATETIME(fsp=6)
    cache_ok = True

    def process_bind_param(self, value: Any, dialect: Any) -> Any:
        if not isinstance(value, dt.datetime):
            return value
        if value.tzinfo is None:
            raise ValueError("MySQL-family timestamps need an aware datetime; naive refused")
        return value.astimezone(dt.UTC).replace(tzinfo=None)

    def process_result_value(self, value: Any, dialect: Any) -> Any:
        if not isinstance(value, dt.datetime):
            return value
        if value.tzinfo is None:
            return value.replace(tzinfo=dt.UTC)
        return value.astimezone(dt.UTC)


class utc_now_default(FunctionElement[dt.datetime]):  # noqa: N801 (SQL function style)
    """Server default: the database clock. ``now()`` on Postgres and SQLite (unchanged),
    ``(UTC_TIMESTAMP(6))`` on MySQL/MariaDB, which is UTC whatever the session zone."""

    type = DateTime()
    inherit_cache = True
    name = "utc_now_default"


@compiles(utc_now_default)
def _compile_now(element: utc_now_default, compiler: Any, **kw: Any) -> str:
    return str(compiler.process(func.now(), **kw))


@compiles(utc_now_default, "mysql")
@compiles(utc_now_default, "mariadb")
def _compile_now_mysql(element: utc_now_default, compiler: Any, **kw: Any) -> str:
    return "(UTC_TIMESTAMP(6))"


def nopad_string(length: int) -> TypeEngine[str]:
    """``String(length)``; NO PAD binary VARCHAR on MySQL/MariaDB."""
    return String(length).with_variant(NoPadString(length), *_FAMILY)


def nopad_char(length: int) -> TypeEngine[str]:
    """``CHAR(length)``; NO PAD binary CHAR on MySQL/MariaDB."""
    return CHAR(length).with_variant(NoPadChar(length), *_FAMILY)


def long_text() -> TypeEngine[str]:
    """``Text``; ``LONGTEXT`` on MySQL/MariaDB (plain TEXT stops at 64 KiB)."""
    return Text().with_variant(LONGTEXT(), *_FAMILY)


def long_binary() -> TypeEngine[bytes]:
    """``LargeBinary``; ``LONGBLOB`` on MySQL/MariaDB (plain BLOB stops at 64 KiB)."""
    return LargeBinary().with_variant(LONGBLOB(), *_FAMILY)


def utc_datetime() -> TypeEngine[dt.datetime]:
    """``DateTime(timezone=True)``; UTC ``DATETIME(6)`` on MySQL/MariaDB."""
    return DateTime(timezone=True).with_variant(UTCDateTime(), *_FAMILY)


MYSQL_TABLE_OPTIONS: dict[str, Any] = {
    f"{prefix}_{option}": value
    for prefix in _FAMILY
    for option, value in (("engine", "InnoDB"), ("default_charset", "utf8mb4"))
}
"""Table keyword arguments: InnoDB, utf8mb4 (ignored by every other dialect)."""
