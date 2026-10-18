"""Transient-failure retry helpers for durable DB writes.

Shared by both the PersistenceRecorder (evidence writes) and the
SQLAlchemyStateStore (lifecycle writes). Retries only transient DB
failures — connection drops, lock timeouts, SQLite busy, and Postgres
deadlock / serialization failures (SQLSTATE 40P01 / 40001; the database
already rolled the transaction back). Logical failures (IntegrityError, CAS
misses, bad input) propagate immediately.

Retry budget: 3 retries with exponential backoff.
  attempt 0: immediate
  attempt 1: 100ms
  attempt 2: 200ms
  attempt 3: 400ms
  Total max wait: ~700ms
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, TypeVar

from sqlalchemy.exc import DBAPIError, OperationalError
from sqlalchemy.exc import TimeoutError as SATimeoutError

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_BASE_BACKOFF_S = 0.1

T = TypeVar("T")

# Postgres: deadlock_detected, serialization_failure. The whole transaction was
# rolled back, so re-running the attempt (which owns its transaction) is safe.
_RETRYABLE_SQLSTATES = frozenset({"40P01", "40001"})


def _sqlstate(exc: BaseException) -> str | None:
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        for attr in ("sqlstate", "pgcode"):
            value = getattr(current, attr, None)
            if isinstance(value, str):
                return value
        current = getattr(current, "orig", None) or current.__cause__
    return None


def is_transient_db_error(exc: BaseException) -> bool:
    """True for failures a fresh attempt of the same transaction may not hit."""
    if isinstance(exc, (OperationalError, SATimeoutError)):
        return True
    return isinstance(exc, DBAPIError) and _sqlstate(exc) in _RETRYABLE_SQLSTATES


async def retry_transient_db(
    attempt: Callable[[], Coroutine[Any, Any, T]],
    *,
    label: str,
    run_id: str,
) -> T:
    """Retry a DB transaction on transient failures.

    Wraps the entire transaction attempt — session open, execute, commit.
    Retries OperationalError, SQLAlchemy TimeoutError and Postgres
    deadlock / serialization failures (see :func:`is_transient_db_error`).
    Everything else (IntegrityError, ProgrammingError, ValueError, etc.)
    propagates immediately.

    On retry exhaustion, the last transient exception propagates.
    """
    last_exc: Exception | None = None
    for attempt_num in range(_MAX_RETRIES + 1):
        try:
            return await attempt()
        except DBAPIError as exc:
            if not is_transient_db_error(exc):
                raise
            last_exc = exc
            if attempt_num < _MAX_RETRIES:
                await _backoff(attempt_num, label, run_id, exc)
        except SATimeoutError as exc:
            last_exc = exc
            if attempt_num < _MAX_RETRIES:
                await _backoff(attempt_num, label, run_id, exc)
    raise last_exc  # type: ignore[misc]


async def _backoff(attempt_num: int, label: str, run_id: str, exc: Exception) -> None:
    wait = _BASE_BACKOFF_S * (2**attempt_num)
    logger.warning(
        "Transient DB failure %d/%d for %s (run %s): %s — retrying in %.1fs",
        attempt_num + 1,
        _MAX_RETRIES,
        label,
        run_id,
        exc,
        wait,
    )
    await asyncio.sleep(wait)


async def retry_critical(
    coro_fn: Callable[[], Coroutine[Any, Any, Any]],
    *,
    label: str,
    run_id: str,
) -> None:
    """Backward-compatible alias for evidence writes (return None).

    PersistenceRecorder's fail-closed writes all return None. This wrapper
    calls retry_transient_db and discards the return value so the call
    sites don't need to change.
    """
    await retry_transient_db(coro_fn, label=label, run_id=run_id)
