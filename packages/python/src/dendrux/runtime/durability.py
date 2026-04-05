"""Transient-failure retry helpers for durable DB writes.

Shared by both the PersistenceRecorder (evidence writes) and the
SQLAlchemyStateStore (lifecycle writes). Retries only transient DB
failures — connection drops, lock timeouts, SQLite busy. Logical
failures (IntegrityError, CAS misses, bad input) propagate immediately.

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

from sqlalchemy.exc import OperationalError
from sqlalchemy.exc import TimeoutError as SATimeoutError

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_BASE_BACKOFF_S = 0.1

T = TypeVar("T")


async def retry_transient_db(
    attempt: Callable[[], Coroutine[Any, Any, T]],
    *,
    label: str,
    run_id: str,
) -> T:
    """Retry a DB transaction on transient failures.

    Wraps the entire transaction attempt — session open, execute, commit.
    Retries only OperationalError and SQLAlchemy TimeoutError. Everything
    else (IntegrityError, ProgrammingError, ValueError, etc.) propagates
    immediately.

    On retry exhaustion, the last transient exception propagates.
    """
    last_exc: Exception | None = None
    for attempt_num in range(_MAX_RETRIES + 1):
        try:
            return await attempt()
        except (OperationalError, SATimeoutError) as exc:
            last_exc = exc
            if attempt_num < _MAX_RETRIES:
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
    raise last_exc  # type: ignore[misc]


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
