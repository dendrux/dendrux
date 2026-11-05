"""Classify driver metadata without exposing driver messages."""

from __future__ import annotations

import errno
import re
import sqlite3
from dataclasses import dataclass
from typing import Any, Literal

DbErrorKind = Literal[
    "unique_violation",
    "deadlock_or_serialization",
    "lock_timeout",
    "connection_lost",
    "statement_too_large",
    "missing_table",
    "missing_column",
    "other",
]

#: MySQL/MariaDB: ER_NET_PACKET_TOO_LARGE and ER_WARN_ALLOWED_PACKET_OVERFLOWED.
_PACKET_TOO_LARGE = frozenset({1153, 1301})
#: Packages whose exception classes carry MySQL server diagnostics as their first argument.
_MYSQL_DRIVERS = frozenset({"asyncmy", "aiomysql", "pymysql", "MySQLdb", "mysql"})


def _is_mysql_driver_error(error: BaseException) -> bool:
    """True for a MySQL driver's own exception, not a socket error or any other class."""
    if isinstance(error, OSError):
        return False
    return any(cls.__module__.split(".", 1)[0] in _MYSQL_DRIVERS for cls in type(error).__mro__)


@dataclass(frozen=True)
class DbError:
    """Safe database failure metadata, suitable for structured diagnostics."""

    kind: DbErrorKind
    constraint: str | None = None
    code: int | str | None = None


def error_chain(exc: BaseException) -> list[BaseException]:
    """Inspect nested driver exceptions once, without stringifying wrappers."""
    chain = []
    seen = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(current)
        current = _nested_error(current)
    return chain


def _nested_error(error: BaseException) -> BaseException | None:
    nested = getattr(error, "orig", None)
    if isinstance(nested, BaseException):
        return nested
    if error.__cause__ is not None:
        return error.__cause__
    return None if error.__suppress_context__ else error.__context__


def _is_connection_loss_code(code: str) -> bool:
    return code.startswith("08") or code in ("57P01", "57P02", "57P03")


#: PostgreSQL SQLSTATE codes mapped to a kind with no extra detail (23505 is handled apart).
_SQLSTATE_KINDS: dict[str, DbErrorKind] = {
    "42P01": "missing_table",
    "42703": "missing_column",
    "40P01": "deadlock_or_serialization",
    "40001": "deadlock_or_serialization",
    "55P03": "lock_timeout",
}
#: MySQL/MariaDB server and client error numbers mapped to a kind (1062 is handled apart).
_MYSQL_ERRNO_KINDS: dict[int, DbErrorKind] = {
    1146: "missing_table",
    1054: "missing_column",
    1213: "deadlock_or_serialization",
    1205: "lock_timeout",
    **dict.fromkeys((1927, 2006, 2013, 2014, 2045, 2055, 4031), "connection_lost"),
}
_SQLITE_UNIQUE_COLUMNS = re.compile(r"UNIQUE constraint failed: ([\w.]+(?:, [\w.]+)*)")
_SQLITE_UNIQUE_INDEX = re.compile(r"UNIQUE constraint failed: index '([\w]+)'")
_MYSQL_DUPLICATE_KEY = re.compile(r" for key ['`]((?:[A-Za-z0-9_]+\.)?[A-Za-z0-9_]+)['`]$")


def _first_int_arg(error: BaseException) -> int | None:
    return error.args[0] if error.args and isinstance(error.args[0], int) else None


def _stage_packet_too_large(chain: list[BaseException]) -> DbError | None:
    """A MySQL driver's own packet-too-large number anywhere in the chain."""
    for error in chain:
        code = _first_int_arg(error)
        if code in _PACKET_TOO_LARGE and _is_mysql_driver_error(error):
            return DbError("statement_too_large", code=code)
    return None


def _constraint_name(chain: list[BaseException]) -> Any:
    """First string constraint name in the chain, else the last value seen."""
    constraint = None
    for nested in chain:
        constraint = getattr(nested, "constraint_name", None) or getattr(
            getattr(nested, "diag", None), "constraint_name", None
        )
        if isinstance(constraint, str):
            break
    return constraint


def _sqlstate_error(code: str, chain: list[BaseException]) -> DbError:
    # Read the metadata for every SQLSTATE, as the single pass always did: the
    # getters can raise, and that must not depend on the code.
    constraint = _constraint_name(chain)
    if code == "23505":
        return DbError("unique_violation", constraint, code)
    kind = _SQLSTATE_KINDS.get(code)
    if kind is not None:
        return DbError(kind, code=code)
    if _is_connection_loss_code(code):
        return DbError("connection_lost", code=code)
    return DbError("other", code=code)


def _stage_sqlstate(chain: list[BaseException]) -> DbError | None:
    """The first element with a string SQLSTATE and no numeric first argument decides."""
    for error in chain:
        if _first_int_arg(error) is not None:
            continue
        code = getattr(error, "sqlstate", None) or getattr(error, "pgcode", None)
        if isinstance(code, str):
            return _sqlstate_error(code, chain)
    return None


def _sqlite_unique_constraint(error: BaseException) -> str | None:
    # SQLite exposes unique column lists only in its message. Keep only
    # identifiers; never retain arbitrary message text or bound values.
    match = _SQLITE_UNIQUE_COLUMNS.fullmatch(str(error))
    if match is None:
        match = _SQLITE_UNIQUE_INDEX.fullmatch(str(error))
    return match[1] if match else None


def _sqlite_missing(message: str, code: int) -> DbError | None:
    if message.startswith("no such table:"):
        return DbError("missing_table", code=code)
    if message.startswith("no such column:") or " has no column named " in message:
        return DbError("missing_column", code=code)
    return None


def _sqlite_error(error: BaseException, code: int) -> DbError:
    if isinstance(error, sqlite3.Error):
        missing = _sqlite_missing(str(error), code)
        if missing is not None:
            return missing
    if code & 0xFF in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED):
        return DbError("lock_timeout", code=code)
    if code in (sqlite3.SQLITE_CONSTRAINT_UNIQUE, sqlite3.SQLITE_CONSTRAINT_PRIMARYKEY):
        return DbError("unique_violation", _sqlite_unique_constraint(error), code)
    return DbError("other", code=code)


def _stage_sqlite(chain: list[BaseException]) -> DbError | None:
    """The first element with an integer ``sqlite_errorcode`` decides."""
    for error in chain:
        code = getattr(error, "sqlite_errorcode", None)
        if isinstance(code, int):
            return _sqlite_error(error, code)
    return None


def _is_connection_reset(error: BaseException, code: int | None) -> bool:
    return _is_connection_type(error) or _is_connection_errno(error, code)


def _is_connection_type(error: BaseException) -> bool:
    return isinstance(error, (ConnectionError, TimeoutError)) and error.errno is None


def _is_connection_errno(error: BaseException, code: int | None) -> bool:
    # Read the errno constants at call time: they are platform values, and tests
    # substitute them.
    return isinstance(error, OSError) and (getattr(error, "errno", None) or code) in (
        errno.ECONNREFUSED,
        errno.ECONNRESET,
        errno.ECONNABORTED,
        errno.EPIPE,
        errno.ENOTCONN,
        errno.ETIMEDOUT,
        errno.EHOSTUNREACH,
        errno.ENETUNREACH,
        errno.ENETDOWN,
    )


def _duplicate_key_error(error: BaseException, code: int) -> DbError:
    # asyncmy only supplies the key in the message. Accept a terminal,
    # identifier-only key; never retain the duplicate value or message.
    args = error.args
    text = args[1] if len(args) > 1 else None
    match = _MYSQL_DUPLICATE_KEY.search(text) if isinstance(text, str) else None
    return DbError("unique_violation", match[1].rsplit(".", 1)[-1] if match else None, code)


def _driver_error(error: BaseException, code: int | None) -> DbError | None:
    """One element's OS or MySQL errno result, or None when it decides nothing."""
    if _is_connection_reset(error, code):
        return DbError("connection_lost", code=code)
    if code is None:
        return None
    if code == 0:
        if type(error).__name__ == "InterfaceError":
            return DbError("connection_lost", code=code)
        return None
    if code == 1062:
        return _duplicate_key_error(error, code)
    kind = _MYSQL_ERRNO_KINDS.get(code)
    return DbError(kind, code=code) if kind is not None else None


def _stage_driver(chain: list[BaseException]) -> DbError:
    """OS and MySQL errnos in chain order, then the invalidated flag, then ``other``."""
    unknown_code: int | None = None
    for error in chain:
        code = _first_int_arg(error)
        found = _driver_error(error, code)
        if found is not None:
            return found
        if code is not None and unknown_code is None:
            unknown_code = code
    if any(getattr(error, "connection_invalidated", False) for error in chain):
        return DbError("connection_lost", code=unknown_code)
    return DbError("other", code=unknown_code)


def classify_error(exc: BaseException) -> DbError:
    """Classify server/driver codes, retaining only safe identifiers.

    Four passes over the whole chain, in this order; the first pass that
    decides wins for every element.
    """
    chain = error_chain(exc)
    for stage in (_stage_packet_too_large, _stage_sqlstate, _stage_sqlite):
        found = stage(chain)
        if found is not None:
            return found
    return _stage_driver(chain)
