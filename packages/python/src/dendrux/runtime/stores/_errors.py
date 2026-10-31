"""Classify driver metadata without exposing driver messages."""

from __future__ import annotations

import errno
import re
import sqlite3
from dataclasses import dataclass
from typing import Literal

DbErrorKind = Literal[
    "unique_violation",
    "deadlock_or_serialization",
    "lock_timeout",
    "connection_lost",
    "statement_too_large",
    "other",
]

#: MySQL/MariaDB: ER_NET_PACKET_TOO_LARGE and ER_WARN_ALLOWED_PACKET_OVERFLOWED.
_PACKET_TOO_LARGE = frozenset({1153, 1301})


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
        nested = getattr(current, "orig", None)
        current = nested if isinstance(nested, BaseException) else current.__cause__
    return chain


def classify_error(exc: BaseException) -> DbError:
    """Classify server/driver codes, retaining only safe identifiers."""
    chain = error_chain(exc)
    for error in chain:
        code = error.args[0] if error.args and isinstance(error.args[0], int) else None
        if code in _PACKET_TOO_LARGE:
            return DbError("statement_too_large", code=code)
    for error in chain:
        if error.args and isinstance(error.args[0], int):
            continue
        code = getattr(error, "sqlstate", None) or getattr(error, "pgcode", None)
        if not isinstance(code, str):
            continue
        constraint = None
        for nested in chain:
            constraint = getattr(nested, "constraint_name", None) or getattr(
                getattr(nested, "diag", None), "constraint_name", None
            )
            if isinstance(constraint, str):
                break
        if code == "23505":
            return DbError("unique_violation", constraint, code)
        if code in ("40P01", "40001"):
            return DbError("deadlock_or_serialization", code=code)
        if code == "55P03":
            return DbError("lock_timeout", code=code)
        if code.startswith("08") or code in ("57P01", "57P02", "57P03"):
            return DbError("connection_lost", code=code)
        return DbError("other", code=code)
    for error in chain:
        code = getattr(error, "sqlite_errorcode", None)
        if not isinstance(code, int):
            continue
        if code & 0xFF in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED):
            return DbError("lock_timeout", code=code)
        if code in (sqlite3.SQLITE_CONSTRAINT_UNIQUE, sqlite3.SQLITE_CONSTRAINT_PRIMARYKEY):
            # SQLite exposes unique column lists only in its message. Keep only
            # identifiers; never retain arbitrary message text or bound values.
            match = re.fullmatch(r"UNIQUE constraint failed: ([\w.]+(?:, [\w.]+)*)", str(error))
            constraint = match[1] if match else None
            if constraint is None:
                match = re.fullmatch(r"UNIQUE constraint failed: index '([\w]+)'", str(error))
                constraint = match[1] if match else None
            return DbError("unique_violation", constraint, code)
        return DbError("other", code=code)
    unknown_code: int | None = None
    for error in chain:
        args = error.args
        code = args[0] if args and isinstance(args[0], int) else None
        if isinstance(error, OSError) and (getattr(error, "errno", None) or code) in (
            errno.ECONNRESET,
            errno.ECONNABORTED,
            errno.EPIPE,
            errno.ENOTCONN,
            errno.ETIMEDOUT,
        ):
            return DbError("connection_lost", code=code)
        if code is None:
            continue
        if code == 0 and type(error).__name__ == "InterfaceError":
            return DbError("connection_lost", code=code)
        if code == 1062:
            # asyncmy only supplies the key in the message. Accept a terminal,
            # identifier-only key; never retain the duplicate value or message.
            match = (
                re.search(r" for key ['`]((?:[A-Za-z0-9_]+\.)?[A-Za-z0-9_]+)['`]$", args[1])
                if len(args) > 1 and isinstance(args[1], str)
                else None
            )
            return DbError("unique_violation", match[1].rsplit(".", 1)[-1] if match else None, code)
        if code == 1213:
            return DbError("deadlock_or_serialization", code=code)
        if code == 1205:
            return DbError("lock_timeout", code=code)
        if code in (1927, 2006, 2013, 2014, 2045, 2055, 4031):
            return DbError("connection_lost", code=code)
        if unknown_code is None:
            unknown_code = code
    if any(getattr(error, "connection_invalidated", False) for error in chain):
        return DbError("connection_lost", code=unknown_code)
    return DbError("other", code=unknown_code)
