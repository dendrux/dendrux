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
from contextlib import asynccontextmanager
from contextvars import ContextVar
from functools import wraps
from time import monotonic
from typing import TYPE_CHECKING, Any, TypeVar

from sqlalchemy.exc import (
    DisconnectionError,
    OperationalError,
    SQLAlchemyError,
    TimeoutError,
)

from dendrux import _codes
from dendrux.db._live_updates import LiveUpdates
from dendrux.errors import (
    PersistenceError,
    SchemaNotInitializedError,
    SchemaOutOfDateError,
    StatementTooLargeError,
)
from dendrux.errors import StoreOutcomeUnknown as StoreOutcomeUnknown
from dendrux.runtime.stores._errors import classify_error, error_chain

_EXCEPTION_WALK_LIMIT = 1024

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable, Coroutine

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
BASE_BACKOFF_S = 0.1

T = TypeVar("T")

_database_operation: ContextVar[tuple[object, str, dict[str, Any]] | None] = ContextVar(
    "dendrux_database_operation", default=None
)


_live_update_scope: ContextVar[tuple[object, frozenset[int]] | None] = ContextVar(
    "dendrux_live_update_scope", default=None
)
_sweep_deadlines: ContextVar[tuple[object, dict[int, float]] | None] = ContextVar(
    "dendrux_sweep_deadlines", default=None
)
_failed_page_marks: ContextVar[tuple[object, dict[int, set[str]]] | None] = ContextVar(
    "dendrux_failed_page_marks", default=None
)
_SWEEP_BOUNDARIES = frozenset(
    {
        "_scan_expired_runs",
        "_sweep_runs",
        "mark_lease_expired",
        "sweep_stale_runs",
        "sweep_abandoned_runs",
    }
)


def _task_values(scope: Any, default: Any) -> Any:
    if scope is None or scope[0] is not asyncio.current_task():
        return default
    return scope[1]


async def _flush_live_updates(store: Any) -> None:
    updates = getattr(store, "live_updates", None)
    if not isinstance(updates, LiveUpdates):
        return
    deadlines = _task_values(_sweep_deadlines.get(), {})
    if id(store) in deadlines:
        await updates._flush_before(deadlines[id(store)])
    else:
        await updates.flush()


@asynccontextmanager
async def _store_update_scope(store: Any) -> AsyncIterator[bool]:
    stores = _task_values(_live_update_scope.get(), frozenset())
    if id(store) in stores:
        yield False
        return
    token = _live_update_scope.set((asyncio.current_task(), stores | {id(store)}))
    try:
        yield True
    finally:
        _live_update_scope.reset(token)


# Allow a cancelled request to retain a normally slow COMMIT and fenced cleanup.
# Expiry still joins cancellation acknowledgement; stores must honour cancellation.
_OWNERSHIP_DEADLINE_S = 5.0
_handoff_commit: ContextVar[dict[str, Any] | None] = ContextVar(
    "dendrux_handoff_commit", default=None
)


def _mark_commit_issued() -> None:
    marker = _handoff_commit.get()
    if marker is not None:
        marker["issued"] = True


async def _without_live_flush(
    store: Any, operation: Awaitable[T], marker: dict[str, Any] | None = None
) -> T:
    token = _handoff_commit.set(marker)
    try:
        async with _store_update_scope(store):
            return await operation
    finally:
        _handoff_commit.reset(token)


def _log_handoff_failure(marker: dict[str, Any], run_id: str) -> None:
    code = _codes.CANCEL_PERSIST_FAILED if marker.get("issued") else _codes.PERSISTENCE_ERROR
    _codes.log_code(logger, code, run_id=run_id)


async def _cancel_handoff(
    store: Any,
    task: asyncio.Task[T],
    on_cancel: Callable[[T], Coroutine[Any, Any, Any]],
    marker: dict[str, Any],
    run_id: str,
) -> None:
    try:
        async with asyncio.timeout(_OWNERSHIP_DEADLINE_S):
            result = await task
            await _without_live_flush(store, on_cancel(result))
    except Exception:
        _log_handoff_failure(marker, run_id)
    finally:
        # LiveUpdates supplies its own flush deadline. Do this once even when
        # cleanup times out; hints never decide whether cleanup runs.
        if marker.get("issued") or not task.cancelled():
            await _flush_live_updates(store)


async def _ownership_handoff(
    store: Any,
    operation_factory: Callable[[], Awaitable[T]],
    on_cancel: Callable[[T], Coroutine[Any, Any, Any]],
    *,
    run_id: str,
) -> T:
    from dendrux.runtime.state import unclaimed_store
    from dendrux.runtime.tasks import _drain_cleanup

    native = unclaimed_store(store)
    marker: dict[str, Any] = {}
    token = _handoff_commit.set(marker)
    try:
        # Store-created Tasks copy this context at invocation, before their first step.
        operation = operation_factory()
    finally:
        _handoff_commit.reset(token)
    task = asyncio.create_task(_without_live_flush(native, operation, marker))
    try:
        result = await asyncio.shield(task)
        await _flush_live_updates(native)
        return result
    except asyncio.CancelledError as cancellation:
        # No await between checking COMMIT and interrupting the child.
        if not marker.get("issued"):
            task.cancel()
        try:
            await _drain_cleanup(
                _cancel_handoff(native, task, on_cancel, marker, run_id), run_id=run_id
            )
        except Exception:
            _log_handoff_failure(marker, run_id)
        finally:
            try:
                if inspect.iscoroutine(operation):
                    operation.close()  # The child may have been cancelled before its first step.
            finally:
                raise cancellation


@asynccontextmanager
async def _live_updates_boundary(store: Any, *, sweep: bool = False) -> AsyncIterator[None]:
    budget_token = _sweep_budget(store, sweep)
    try:
        async with _store_update_scope(store) as outermost:
            yield
            if outermost:
                await _flush_live_updates(store)
    finally:
        if budget_token is not None:
            _sweep_deadlines.reset(budget_token)


def _enqueue_live_updates(store: Any, marks: Any, succeeded: bool) -> None:
    deferred = _task_values(_failed_page_marks.get(), {}).get(id(store))
    if not succeeded and deferred is not None:
        deferred.update(marks)
        return
    if deferred is not None:
        deferred.difference_update(marks)
    store.live_updates.enqueue(marks)


@asynccontextmanager
async def _live_updates_page(store: Any) -> AsyncIterator[None]:
    deferred: set[str] = set()
    pages = _task_values(_failed_page_marks.get(), {})
    token = _failed_page_marks.set((asyncio.current_task(), {**pages, id(store): deferred}))
    try:
        yield
    finally:
        try:
            await _flush_live_updates(store)
        finally:
            _failed_page_marks.reset(token)
            # A failed COMMIT's hints wait for the next successful boundary,
            # after earlier successful candidates in this page have flushed.
            if deferred:
                store.live_updates.enqueue(deferred)


def _sweep_budget(store: Any, sweep: bool) -> Any:
    updates = getattr(store, "live_updates", None)
    if not sweep or not isinstance(updates, LiveUpdates):
        return None
    deadlines = _task_values(_sweep_deadlines.get(), {})
    if id(store) in deadlines:
        return None
    return _sweep_deadlines.set(
        (
            asyncio.current_task(),
            {
                **deadlines,
                id(store): monotonic() + updates.flush_deadline_s,
            },
        )
    )


def _with_live_updates(method: Any) -> Any:
    # Hooks borrowing a session or connection stay inside their caller's boundary.
    if {"session", "conn"}.intersection(inspect.signature(method).parameters):
        return method

    @wraps(method)
    async def call(*args: Any, **kwargs: Any) -> Any:
        store = args[0] if args else None
        async with _live_updates_boundary(store, sweep=method.__name__ in _SWEEP_BOUNDARIES):
            return await method(*args, **kwargs)

    return call


class _CommitNotAppliedError(PersistenceError):
    """Readback proved that the transition can safely be attempted again."""

    def __init__(self) -> None:
        super().__init__(retryable=True)


def _log_too_large(error: StatementTooLargeError) -> None:
    if error.limit is None:
        _codes.log_code(logger, _codes.STORE_STATEMENT_LIMIT, run_id=None)
    else:
        _codes.log_code(logger, _codes.STORE_STATEMENT_LIMIT, run_id=None, limit=error.limit)


def _statement_too_large(limit: int | None) -> StatementTooLargeError:
    error = StatementTooLargeError(limit)
    _log_too_large(error)
    return error


def is_transient_db_error(exc: BaseException, *, classify: Any = classify_error) -> bool:
    """True for temporary failures, not proof that COMMIT was unapplied.

    Callers must roll back pre-commit failures and resolve ambiguous commits
    before retrying. StoreOutcomeUnknown remains non-retryable.
    """
    if isinstance(exc, PersistenceError):
        return exc.retryable
    classified = classify(exc)
    if (
        classified.kind in ("statement_too_large", "missing_table", "missing_column")
        or classified.code == "57014"
    ):
        return False
    return (
        _transient_wrapper(exc)
        or classified.kind in ("deadlock_or_serialization", "lock_timeout", "connection_lost")
        or classified.code == "53300"
    )


def _transient_wrapper(exc: BaseException) -> bool:
    return any(
        isinstance(error, (OperationalError, TimeoutError, DisconnectionError))
        for error in error_chain(exc)
    )


async def retry_transient_db(
    attempt: Callable[[], Coroutine[Any, Any, T]],
    *,
    label: str,
    run_id: str,
    classify: Any = classify_error,
    _raw_driver_errors: bool = False,
) -> T:
    """Retry a DB transaction on transient failures.

    Wraps the entire transaction attempt — session open, execute, commit.
    Retries transient pre-commit connection, availability and transaction failures
    (see :func:`is_transient_db_error`).
    SQLAlchemy and persistence errors retain the shared retry policy. Only SQL
    stores with a COMMIT proof boundary opt into raw transient driver failures;
    other raw errors propagate to their caller's normalization boundary.
    Exhaustion raises a retryable PersistenceError.
    """

    def check_retry_error(error: Exception) -> None:
        _require_database_error(error)
        allowed = isinstance(error, (SQLAlchemyError, PersistenceError)) or (
            _raw_driver_errors and is_transient_db_error(error, classify=classify)
        )
        if not allowed:
            raise error

    failure: BaseException | None = None
    last_classified = None
    for attempt_num in range(MAX_RETRIES + 1):
        try:
            return await attempt()
        except Exception as exc:
            check_retry_error(exc)
            last_classified = classify(exc)
            if not is_transient_db_error(exc, classify=classify):
                failure = _database_failure(exc, classify=classify)
                break
            if attempt_num < MAX_RETRIES:
                await _backoff(attempt_num, label, run_id, exc)
    if failure is None:
        failure = PersistenceError(retryable=True)
        _log_database_failure(last_classified, run_id)
    failure.__context__ = None
    failure.__cause__ = None
    raise failure from None


def _require_database_error(error: Exception) -> None:
    if not isinstance(error, (SQLAlchemyError, PersistenceError)) and not _is_driver_failure(error):
        raise error


def _log_database_failure(classified: Any, run_id: Any = None) -> None:
    code = str(getattr(classified, "code", "other"))
    kind = getattr(classified, "kind", "other")
    _codes.log_code(
        logger,
        _codes.PERSISTENCE_ERROR,
        run_id=run_id if _codes._valid_field("run_id", run_id) else None,
    )
    code = code if _codes._valid_field("db_code", code) else "other"
    kind = kind if _codes._valid_field("db_kind", kind) else "other"
    logger.warning(
        "Database failure kind=%s db_code=%s", kind, code, extra={"db_kind": kind, "db_code": code}
    )


async def _backoff(attempt_num: int, label: str, run_id: str, exc: Exception) -> None:
    wait = BASE_BACKOFF_S * (2**attempt_num)
    logger.warning(
        "Transient DB failure %d/%d for %s (run %s): %s — retrying in %.1fs",
        attempt_num + 1,
        MAX_RETRIES,
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
        failure: BaseException | None = None
        if args and not method.__name__.startswith("_"):
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            token = _database_operation.set((args[0], method.__name__, dict(bound.arguments)))
        try:
            return await method(*args, **kwargs)
        except PersistenceError as error:
            facts = getattr(args[0], "_dialect_facts", None) if args else None
            if isinstance(error, StatementTooLargeError):
                error.set_limit(getattr(facts, "max_allowed_packet", None))
            failure = error
        except SQLAlchemyError as error:
            classifier = (
                getattr(args[0], "classify_db_error", classify_error) if args else classify_error
            )
            facts = getattr(args[0], "_dialect_facts", None) if args else None
            failure = _database_failure(
                error, classify=classifier, limit=getattr(facts, "max_allowed_packet", None)
            )
        except Exception as error:
            from dendrux.db._engine_options import mysql_dependency_error

            failure = mysql_dependency_error(error)
            if failure is None:
                if not _is_driver_failure(error):
                    raise
                classifier = (
                    getattr(args[0], "classify_db_error", classify_error)
                    if args
                    else classify_error
                )
                failure = _database_failure(error, classify=classifier)
        finally:
            if token is not None:
                _database_operation.reset(token)
        assert failure is not None
        failure.__context__ = None
        failure.__cause__ = None
        failure.__traceback__ = None
        raise failure from None

    return _with_live_updates(call)


def _is_driver_failure(error: BaseException) -> bool:
    drivers = {
        "asyncmy",
        "aiomysql",
        "pymysql",
        "MySQLdb",
        "mysql",
        "asyncpg",
        "aiosqlite",
        "sqlite3",
        "psycopg",
        "psycopg2",
    }
    if any(cls.__module__.split(".", 1)[0] in drivers for cls in type(error).__mro__):
        return True
    traceback = error.__traceback__
    for _ in range(_EXCEPTION_WALK_LIMIT):
        if traceback is None:
            break
        module = traceback.tb_frame.f_globals.get("__name__", "")
        if module.split(".", 1)[0] in drivers:
            return True
        traceback = traceback.tb_next
    return False


def _database_failure(
    exc: BaseException, *, classify: Any, limit: int | None = None
) -> BaseException:
    if isinstance(exc, PersistenceError):
        return exc
    from dendrux.errors import JSONIntegerOutOfRangeError
    from dendrux.types import InputValidationError

    original = getattr(exc, "orig", None)
    if isinstance(original, (JSONIntegerOutOfRangeError, InputValidationError)):
        return original
    classified = classify(exc)
    operation = _database_operation.get()
    _log_database_failure(classified, None if operation is None else operation[2].get("run_id"))
    if classified.kind == "missing_table":
        return SchemaNotInitializedError()
    if classified.kind == "missing_column":
        return SchemaOutOfDateError()
    if classified.kind == "statement_too_large":
        return _statement_too_large(limit)
    return PersistenceError(retryable=is_transient_db_error(exc, classify=classify))


def safe_database_operations(cls: type[T]) -> type[T]:
    """Normalize database errors after each operation's transaction handling."""
    for name, method in inspect.getmembers(cls, inspect.iscoroutinefunction):
        if not name.startswith("_"):
            setattr(cls, name, _safe_database_call(method))
    return cls
