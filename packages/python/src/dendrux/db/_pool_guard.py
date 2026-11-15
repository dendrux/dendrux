"""Guard event-loop ownership and serialise sessions on single-connection pools.

``StaticPool`` (the default for ``sqlite+aiosqlite:///:memory:``) and
``SingletonThreadPool`` hand every session the same connection, so every
session shares one SQLite transaction. When one session ends, the pool's
reset-on-return ``ROLLBACK`` discards whatever another session has executed
but not yet committed, and that session's later ``commit()`` silently does
nothing. Concurrent writers (parallel tool calls, heartbeats, readers) then
lose rows without any error.

:func:`serialized` holds a lock for the whole life of a session (begin,
commit, close and pool reset) on such pools, so Dendrux's transactions never
overlap. The lock belongs to the *pool*: engines that share a pool (for
example ``engine.execution_options(...)``) share the lock. Engines with any
other pool get no lock and no behaviour change.

Limits, by design:

- Re-entry from the task that already holds the lock does not wait. Such a
  nested session shares the outer session's transaction: its commit also
  publishes the outer session's pending work.
- Other tasks always wait. Never hold a Dendrux session (``get_session`` or a
  store session) across a call that runs an agent or otherwise starts tasks
  that use the same engine: those tasks wait for a lock their parent holds,
  and the call hangs. A waiter that has waited ``WAIT_WARNING_S`` seconds logs
  a warning naming the holder.
- An in-memory engine is single-event-loop. Locks are per event loop; two
  loops in two threads on one in-memory engine are not serialised.

:func:`close_fully` runs a close to completion even when its caller is
cancelled, but once the caller is cancelled it waits at most
``CLOSE_AFTER_CANCEL_S`` seconds. A close still running then is abandoned: it
keeps running in the background, and the single-connection lock (if the caller
holds one) stays held until that close finishes.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import weakref
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from sqlalchemy.pool import SingletonThreadPool, StaticPool

from dendrux import _codes

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable

    from sqlalchemy.ext.asyncio import AsyncEngine
    from sqlalchemy.pool import Pool

logger = logging.getLogger(__name__)

__all__ = [
    "CLOSE_AFTER_CANCEL_S",
    "WAIT_WARNING_S",
    "close_fully",
    "serialized",
    "shares_one_connection",
    "warn_if_shared_connection",
]

WAIT_WARNING_S = 10.0
"""Seconds a task may wait for a single-connection lock before a warning is logged."""

CLOSE_AFTER_CANCEL_S = 5.0
"""Seconds :func:`close_fully` keeps waiting for a close once its caller is cancelled."""

_CONTEXT_DEPTH = 8  # exception __context__ links searched for an unwinding cancellation


class _Guard:
    __slots__ = ("abandoned", "lock", "owner", "users")

    def __init__(self) -> None:
        self.lock = asyncio.Lock()
        self.owner: asyncio.Task[Any] | None = None
        self.users = 0  # holder + waiters; the entry is dropped when this reaches 0
        # Closes the holder abandoned on cancellation; the lock waits for them.
        self.abandoned: list[asyncio.Future[None]] = []


# (pool, loop) -> guard. An entry exists while a task holds or waits for the lock, or
# while a close its holder abandoned is still running. Otherwise an idle pool or a
# closed event loop is never kept alive by this map. Limit: an abandoned close that
# never finishes (aiosqlite 0.22.1 close() after stop()) never runs its callbacks, so
# that pool's lock stays held and its entry (pool and loop) stays here for the life
# of the process; other sessions on that engine wait (with the WAIT_WARNING_S warning).
_GUARDS: dict[tuple[Pool, asyncio.AbstractEventLoop], _Guard] = {}
_WARNED: weakref.WeakSet[Pool] = weakref.WeakSet()


def _bind_to_current_loop(engine: AsyncEngine) -> None:
    """Make every pool checkout on another event loop fail with a clear error.

    asyncpg connections belong to the loop that opened them; sharing the
    global engine across loops (a new loop per request, ``asyncio.run`` per
    call, worker threads) otherwise fails deep inside the driver. SQLite
    connections are not loop-bound, so SQLite engines are left alone.
    """
    if engine.dialect.name == "sqlite":
        return
    from sqlalchemy import event

    owner = asyncio.get_running_loop()

    def _check_loop(*_args: object) -> None:
        try:
            current = asyncio.get_running_loop()
        except RuntimeError:
            return
        if current is not owner:
            raise RuntimeError(
                "The shared dendrux database engine was created "
                "on a different event loop and cannot be used from this one. Create one "
                "state store per process on the loop that serves requests and pass "
                "state_store= to every Agent, or call dendrux.db.session.reset_engine() "
                "before switching loops."
            )

    # Pool checkout events run *after* pre-ping. Reject at the public Engine
    # acquisition boundary, before the pool can ping, open or invalidate an
    # asyncpg connection on the wrong loop. This also covers raw_connection
    # and survives dispose(), which replaces the pool on this same Engine.
    # Guard both public entry points: normal acquisition need not route through
    # raw_connection in Connection.__init__ on any particular SQLAlchemy minor.
    connect = engine.sync_engine.connect
    raw_connection = engine.sync_engine.raw_connection

    def _checked_connect() -> Any:
        _check_loop()
        return connect()

    def _checked_connection() -> Any:
        _check_loop()
        return raw_connection()

    engine.sync_engine.connect = _checked_connect  # type: ignore[method-assign]
    engine.sync_engine.raw_connection = _checked_connection  # type: ignore[method-assign]
    event.listen(engine.sync_engine, "checkout", _check_loop)


def shares_one_connection(engine: AsyncEngine) -> bool:
    """True when every checkout of ``engine`` returns the same DBAPI connection."""
    return isinstance(engine.sync_engine.pool, (StaticPool, SingletonThreadPool))


def warn_if_shared_connection(engine: AsyncEngine) -> None:
    """Log one WARNING per pool when ``engine`` uses a single-connection pool."""
    pool = engine.sync_engine.pool
    if not shares_one_connection(engine) or pool in _WARNED:
        return
    _WARNED.add(pool)
    logger.warning(
        "Dendrux store on a single-connection pool (%s, e.g. in-memory "
        "SQLite): Dendrux's database sessions are serialised and the data is not "
        "durable. Use a SQLite file or Postgres for anything beyond tests and demos.",
        type(pool).__name__,
    )


async def _acquire(guard: _Guard) -> None:
    if not guard.lock.locked():
        await guard.lock.acquire()
        return
    acquire = asyncio.ensure_future(guard.lock.acquire())
    try:
        done, _ = await asyncio.wait({acquire}, timeout=WAIT_WARNING_S)
        if not done:
            holder = guard.owner.get_name() if guard.owner is not None else "unknown"
            logger.warning(
                "Waited %.0fs for the single-connection database lock held by task %r. "
                "A Dendrux session is probably being held across a call that runs an "
                "agent or starts tasks using the same in-memory engine; that call will "
                "not finish. Use a SQLite file or Postgres, or do not hold the session.",
                WAIT_WARNING_S,
                holder,
            )
        await acquire
    except BaseException:
        if acquire.done() and not acquire.cancelled() and acquire.exception() is None:
            guard.lock.release()
        else:
            acquire.cancel()
        raise


@asynccontextmanager
async def serialized(engine: AsyncEngine) -> AsyncIterator[None]:
    """Hold the pool's session lock for the body when the pool shares one connection.

    A no-op for every other pool. Re-entering from the task that already holds
    the lock does not wait (see the module docstring for what that implies).
    """
    if not shares_one_connection(engine):
        yield
        return
    loop = asyncio.get_running_loop()
    key = (engine.sync_engine.pool, loop)
    guard = _GUARDS.get(key)
    task = asyncio.current_task()
    if guard is not None and task is not None and guard.owner is task:
        yield
        return
    if guard is None:
        guard = _GUARDS[key] = _Guard()
    guard.users += 1
    try:
        await _acquire(guard)
    except BaseException:
        _leave(guard, key)
        raise
    guard.owner = task
    try:
        yield
    finally:
        _unlock_when_closed(guard, key)


def _leave(guard: _Guard, key: tuple[Pool, asyncio.AbstractEventLoop]) -> None:
    guard.users -= 1
    if guard.users == 0 and _GUARDS.get(key) is guard:
        del _GUARDS[key]


def _unlock(guard: _Guard, key: tuple[Pool, asyncio.AbstractEventLoop]) -> None:
    guard.owner = None
    guard.lock.release()
    _leave(guard, key)


def _unlock_when_closed(guard: _Guard, key: tuple[Pool, asyncio.AbstractEventLoop]) -> None:
    """Release the lock now, or once every close its holder abandoned has finished.

    An abandoned close may still run its ROLLBACK and pool reset on the shared
    connection; releasing earlier would let another session's transaction overlap it.
    """
    pending = [close for close in guard.abandoned if not close.done()]
    guard.abandoned = []
    if not pending:
        _unlock(guard, key)
        return
    guard.owner = None  # the cancelled holder must not re-enter without waiting
    remaining = len(pending)

    def _closed(_close: asyncio.Future[None]) -> None:
        nonlocal remaining
        remaining -= 1
        if remaining == 0:
            _unlock(guard, key)

    for close in pending:
        close.add_done_callback(_closed)


async def close_fully(close: Callable[[], Awaitable[None]]) -> None:
    """Run ``close()`` to completion even if the caller is cancelled meanwhile.

    A cancellation received while waiting is re-raised once ``close()`` has
    finished, so a lock held around the close is released only after the
    session's ROLLBACK and pool reset have run.

    Once the caller is cancelled, the wait is bounded by ``CLOSE_AFTER_CANCEL_S``.
    The bound starts at the first cancellation received during the wait, or on
    entry when a cancellation is already unwinding (a ``finally`` or ``__aexit__``
    reached by a ``CancelledError``, possibly converted to another exception).
    Later cancellations neither restart nor shorten it. A close still running at
    expiry is abandoned: it is not cancelled and keeps running in the background,
    ``db_close_abandoned`` is logged, and a single-connection lock held by the
    caller is released only when that close finishes.

    The cancellation always wins over a close error: a cancellation received here
    is re-raised, and on entry during an unwinding cancellation this returns so
    the in-flight exception carries on. The close error is then dropped unlogged.
    """
    caller = asyncio.current_task()
    unwinding = _cancellation_unwinding(caller)
    task: asyncio.Future[None] = asyncio.ensure_future(close())
    loop = asyncio.get_running_loop()
    deadline = loop.time() + CLOSE_AFTER_CANCEL_S if unwinding else None
    cancelled = await _await_close(task, caller, deadline)
    if cancelled is None and not unwinding:
        task.result()  # never abandoned without a cancellation, so the close is done
        return
    if task.done():
        _retrieve(task)  # the cancellation wins over a close error
    if cancelled is not None:
        raise cancelled


def _cancellation_unwinding(caller: asyncio.Task[Any] | None) -> bool:
    """True when ``caller`` runs cleanup for a cancellation that is in flight now.

    ``cancelling()`` alone is not proof: it stays raised after a cancellation was
    handled without ``uncancel()`` (a swallowed ``CancelledError``, a TaskGroup on
    Python 3.11 and 3.12). So a ``CancelledError`` must also be the exception being
    handled, or on its ``__context__`` chain when the store converted it.
    """
    if caller is None or not caller.cancelling():
        return False
    handled = sys.exception()
    for _ in range(_CONTEXT_DEPTH):
        if handled is None:
            return False
        if isinstance(handled, asyncio.CancelledError):
            return True
        handled = handled.__context__
    return False


async def _await_close(
    task: asyncio.Future[None], caller: asyncio.Task[Any] | None, deadline: float | None
) -> asyncio.CancelledError | None:
    """Wait for ``task`` without cancelling it; return the caller's first cancellation.

    Unbounded while ``deadline`` is None; the first cancellation received sets it.
    At the deadline the close is abandoned, still running.
    """
    loop = asyncio.get_running_loop()
    cancelled: asyncio.CancelledError | None = None
    while not task.done():
        timeout = None if deadline is None else deadline - loop.time()
        if timeout is not None and timeout <= 0:
            _abandon(task, caller)
            break
        try:
            await asyncio.wait((task,), timeout=timeout)
        except asyncio.CancelledError as exc:
            cancelled = cancelled or exc
            if deadline is None:
                deadline = loop.time() + CLOSE_AFTER_CANCEL_S
    return cancelled


def _abandon(task: asyncio.Future[None], caller: asyncio.Task[Any] | None) -> None:
    # Retrieve a late failure so asyncio never logs its text ("exception was never
    # retrieved"): no exception text in logs.
    task.add_done_callback(_retrieve)
    if caller is not None:
        for guard in _GUARDS.values():
            if guard.owner is caller:
                guard.abandoned.append(task)
    _codes.log_code(logger, _codes.DB_CLOSE_ABANDONED, timeout_s=CLOSE_AFTER_CANCEL_S)


def _retrieve(task: asyncio.Future[None]) -> None:
    if not task.cancelled():
        task.exception()
