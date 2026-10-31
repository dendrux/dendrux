"""Refuse a MySQL/MariaDB statement the server would drop the connection over.

The server closes the connection on a statement larger than ``max_allowed_packet``, so an
oversized write surfaces as a lost connection mid-write. This listener measures the statement
text plus every bound value, as the driver will escape them, and raises
:class:`StatementTooLargeError` before anything is sent. It reads a number already in memory
and issues no SQL.
"""

from __future__ import annotations

import logging
import weakref
from typing import TYPE_CHECKING, Any

from sqlalchemy import event

from dendrux import _codes
from dendrux.errors import StatementTooLargeError

if TYPE_CHECKING:
    from sqlalchemy.engine import Connection
    from sqlalchemy.engine.interfaces import DBAPICursor, ExecutionContext
    from sqlalchemy.ext.asyncio import AsyncEngine

logger = logging.getLogger(__name__)

# Each engine's ``max_allowed_packet``, recorded by the store's first-use checks.
_capacity: weakref.WeakKeyDictionary[Any, int] = weakref.WeakKeyDictionary()


def set_capacity(engine: Any, packet: int) -> None:
    """Record the server's ``max_allowed_packet`` for ``engine`` (a sync SQLAlchemy engine)."""
    _capacity[engine] = packet


# The driver escapes these characters (one extra byte each) and adds quotes around a string.
_SPECIAL = ("\0", "\n", "\r", "\x1a", '"', "'", "\\")
_BYTES_SPECIAL = tuple(char.encode() for char in _SPECIAL)
_OTHER = 64  # numbers, dates, NULL, booleans
# Cheap upper bound per string character: four UTF-8 bytes, each possibly escaped.
_QUICK_PER_CHAR = 8


def param_bytes(value: object) -> int:
    """Bytes the driver puts on the wire for one bound value (quotes and escapes included)."""
    if isinstance(value, str):
        size = len(value) if value.isascii() else len(value.encode("utf-8", "surrogatepass"))
        return size + 2 + sum(value.count(char) for char in _SPECIAL)
    if isinstance(value, (bytes, bytearray, memoryview)):
        raw = bytes(value)
        return len(raw) + 9 + sum(raw.count(char) for char in _BYTES_SPECIAL)
    return _OTHER


def _values(row: Any) -> list[Any]:
    if row is None:
        return []
    if isinstance(row, dict):
        return list(row.values())
    if isinstance(row, (list, tuple)):
        return list(row)
    return [row]


def _rows(parameters: Any, executemany: bool) -> list[Any]:
    # asyncmy sends a bulk insert in batches of about 1 MB, but a row is never split:
    # the largest row bounds the largest statement.
    return list(parameters) if executemany else [parameters]


def _text_bytes(statement: str) -> int:
    return len(statement) if statement.isascii() else len(statement.encode("utf-8"))


def _quick_bound(value: object) -> int:
    if isinstance(value, (str, bytes, bytearray)):
        return len(value) * _QUICK_PER_CHAR + 11
    if isinstance(value, memoryview):
        return value.nbytes * 2 + 9
    return _OTHER


def statement_bytes(statement: str, parameters: Any, executemany: bool) -> int:
    """The statement text plus its largest set of bound values, as the driver encodes them."""
    largest = max(
        (
            sum(param_bytes(value) for value in _values(row))
            for row in _rows(parameters, executemany)
        ),
        default=0,
    )
    return _text_bytes(statement) + largest


def before_cursor_execute(
    conn: Connection,
    cursor: DBAPICursor | None,
    statement: str,
    parameters: Any,
    context: ExecutionContext | None,
    executemany: bool,
) -> None:
    capacity = _capacity.get(conn.engine)
    if capacity is None:
        return
    quick = _text_bytes(statement) + max(
        (
            sum(_quick_bound(value) for value in _values(row))
            for row in _rows(parameters, executemany)
        ),
        default=0,
    )
    if quick <= capacity:
        return
    size = statement_bytes(statement, parameters, executemany)
    if size > capacity:
        _codes.log_code(
            logger, _codes.STORE_STATEMENT_LIMIT, run_id=None, bytes=size, limit=capacity
        )
        raise StatementTooLargeError(size, capacity)


def install(engine: AsyncEngine) -> None:
    """Attach the preflight to ``engine`` once, however many stores share it."""
    target = engine.sync_engine
    if not event.contains(target, "before_cursor_execute", before_cursor_execute):
        event.listen(target, "before_cursor_execute", before_cursor_execute)
