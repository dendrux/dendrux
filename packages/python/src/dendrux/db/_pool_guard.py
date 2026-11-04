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
"""

from __future__ import annotations

import asyncio
import logging
import weakref
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from sqlalchemy.pool import SingletonThreadPool, StaticPool

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable

    from sqlalchemy.ext.asyncio import AsyncEngine
    from sqlalchemy.pool import Pool

logger = logging.getLogger(__name__)

__all__ = [
    "WAIT_WARNING_S",
    "close_fully",
    "serialized",
    "shares_one_connection",
    "warn_if_shared_connection",
]

WAIT_WARNING_S = 10.0
"""Seconds a task may wait for a single-connection lock before a warning is logged."""


class _Guard:
    __slots__ = ("lock", "owner", "users")

    def __init__(self) -> None:
        self.lock = asyncio.Lock()
        self.owner: asyncio.Task[Any] | None = None
        self.users = 0  # holder + waiters; the entry is dropped when this reaches 0


# (pool, loop) -> guard. An entry exists only while a task holds or waits for the
# lock, so an idle pool or a closed event loop is never kept alive by this map.
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
        guard.owner = task
        try:
            yield
        finally:
            guard.owner = None
            guard.lock.release()
    finally:
        guard.users -= 1
        if guard.users == 0 and _GUARDS.get(key) is guard:
            del _GUARDS[key]


async def close_fully(close: Callable[[], Awaitable[None]]) -> None:
    """Run ``close()`` to completion even if the caller is cancelled meanwhile.

    A cancellation received while waiting is re-raised once ``close()`` has
    finished, so a lock held around the close is released only after the
    session's ROLLBACK and pool reset have run.
    """
    task = asyncio.ensure_future(close())
    cancelled: BaseException | None = None
    while True:
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as exc:
            if task.done():
                raise
            cancelled = exc
        else:
            break
    if cancelled is not None:
        raise cancelled
