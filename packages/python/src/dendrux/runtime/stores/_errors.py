"""Classify driver metadata without exposing driver messages."""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from typing import Literal

DbErrorKind = Literal[
    "unique_violation", "deadlock_or_serialization", "lock_timeout", "connection_lost", "other"
]


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
    """Classify PostgreSQL and SQLite using structured server/driver codes."""
    chain = error_chain(exc)
    for error in chain:
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
    if any(getattr(error, "connection_invalidated", False) for error in chain):
        return DbError("connection_lost")
    return DbError("other")
