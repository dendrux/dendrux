"""Transient-failure retry helpers for durable DB writes.

Shared by both the PersistenceRecorder (evidence writes) and the
SQL stores (lifecycle writes). Retries transient pre-commit failures after
the operation has rolled back its whole transaction. SQL stores resolve
ambiguous COMMIT outcomes before they reach this general retry boundary.
Logical failures (IntegrityError, CAS misses, bad input) propagate immediately.

Retry budget: 3 retries with exponential backoff.
  attempt 0: immediate
  attempt 1: 100ms
  attempt 2: 200ms
  attempt 3: 400ms
  Total max wait: ~700ms
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from contextvars import ContextVar
from functools import wraps
from typing import TYPE_CHECKING, Any, TypeVar

from sqlalchemy.exc import DBAPIError, OperationalError, SQLAlchemyError, TimeoutError

from dendrux.errors import PersistenceError
from dendrux.errors import StoreOutcomeUnknown as StoreOutcomeUnknown
from dendrux.runtime.stores._errors import classify_error

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_BASE_BACKOFF_S = 0.1

T = TypeVar("T")

_database_operation: ContextVar[tuple[object, str, dict[str, Any]] | None] = ContextVar(
    "dendrux_database_operation", default=None
)


class _CommitNotAppliedError(PersistenceError):
    """Readback proved that the transition can safely be attempted again."""

    def __init__(self) -> None:
        super().__init__(retryable=True)


def is_transient_db_error(exc: BaseException, *, classify: Any = classify_error) -> bool:
    """True for failures a fresh attempt of the same transaction may not hit."""
    if isinstance(exc, PersistenceError):
        return exc.retryable
    classified = classify(exc)
    return (
        isinstance(exc, (OperationalError, TimeoutError))
        or classified.kind in ("deadlock_or_serialization", "lock_timeout", "connection_lost")
        or (
            isinstance(exc, DBAPIError)
            and isinstance(classified.code, str)
            and (
                classified.code.startswith(("08", "57P0")) or classified.code in ("53300", "57014")
            )
        )
    )


async def retry_transient_db(
    attempt: Callable[[], Coroutine[Any, Any, T]],
    *,
    label: str,
    run_id: str,
    classify: Any = classify_error,
) -> T:
    """Retry a DB transaction on transient failures.

    Wraps the entire transaction attempt — session open, execute, commit.
    Retries transient pre-commit connection, availability and transaction failures
    (see :func:`is_transient_db_error`).
    Everything else (IntegrityError, ProgrammingError, ValueError, etc.)
    propagates immediately.

    On retry exhaustion, the last transient exception propagates.
    """
    for attempt_num in range(_MAX_RETRIES + 1):
        try:
            return await attempt()
        except (SQLAlchemyError, PersistenceError) as exc:
            if not is_transient_db_error(exc, classify=classify):
                if isinstance(exc, PersistenceError):
                    raise
                raise PersistenceError() from None
            if attempt_num < _MAX_RETRIES:
                await _backoff(attempt_num, label, run_id, exc)
    raise PersistenceError(retryable=True) from None


async def _backoff(attempt_num: int, label: str, run_id: str, exc: Exception) -> None:
    wait = _BASE_BACKOFF_S * (2**attempt_num)
    logger.warning(
        "Transient DB failure %d/%d for %s (run %s): %s — retrying in %.1fs",
        attempt_num + 1,
        _MAX_RETRIES,
        label,
        run_id,
        type(exc).__name__,
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


def _safe_database_call(method: Any) -> Any:
    signature = inspect.signature(method)

    @wraps(method)
    async def call(*args: Any, **kwargs: Any) -> Any:
        token = None
        if args and not method.__name__.startswith("_"):
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            token = _database_operation.set((args[0], method.__name__, dict(bound.arguments)))
        try:
            return await method(*args, **kwargs)
        except SQLAlchemyError as error:
            classifier = (
                getattr(args[0], "classify_db_error", classify_error) if args else classify_error
            )
            raise PersistenceError(
                retryable=is_transient_db_error(error, classify=classifier)
            ) from None
        finally:
            if token is not None:
                _database_operation.reset(token)

    return call


def safe_database_operations(cls: type[T]) -> type[T]:
    """Normalize database errors after each operation's transaction handling."""
    for name, method in inspect.getmembers(cls, inspect.iscoroutinefunction):
        if not name.startswith("_"):
            setattr(cls, name, _safe_database_call(method))
    return cls
