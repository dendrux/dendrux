"""Serialise database work on engines whose pool shares one DBAPI connection.

``StaticPool`` (the default for ``sqlite+aiosqlite:///:memory:``) and
``SingletonThreadPool`` hand every session the same connection, so every
session shares one SQLite transaction. When one session ends, the pool's
reset-on-return ``ROLLBACK`` discards whatever another session has executed
but not yet committed, and that session's later ``commit()`` silently does
nothing. Concurrent writers (parallel tool calls, heartbeats, readers) then
lose rows without any error.

:func:`serialized` holds a per-engine lock for the whole life of a session
(begin, commit, close and pool reset) on such engines, so those transactions
never overlap. It is task-reentrant: a session opened while the same task
already holds the lock runs without waiting. Engines with any other pool get
no lock and no behaviour change.
"""

from __future__ import annotations

import asyncio
import logging
import weakref
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from sqlalchemy.pool import SingletonThreadPool, StaticPool

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from sqlalchemy import Engine
    from sqlalchemy.ext.asyncio import AsyncEngine

logger = logging.getLogger(__name__)

__all__ = ["shares_one_connection", "serialized", "warn_if_shared_connection"]


class _Guard:
    __slots__ = ("lock", "owner")

    def __init__(self) -> None:
        self.lock = asyncio.Lock()
        self.owner: asyncio.Task[object] | None = None


# engine.sync_engine -> event loop -> guard. Keyed per loop because an asyncio
# lock binds to the loop that first waits on it, and an engine can outlive a loop.
_GUARDS: weakref.WeakKeyDictionary[
    Engine, weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, _Guard]
] = weakref.WeakKeyDictionary()
_WARNED: weakref.WeakSet[Engine] = weakref.WeakSet()


def shares_one_connection(engine: AsyncEngine) -> bool:
    """True when every checkout of ``engine`` returns the same DBAPI connection."""
    return isinstance(engine.sync_engine.pool, (StaticPool, SingletonThreadPool))


def warn_if_shared_connection(engine: AsyncEngine) -> None:
    """Log one WARNING per engine when it uses a single-connection pool."""
    sync_engine = engine.sync_engine
    if not shares_one_connection(engine) or sync_engine in _WARNED:
        return
    _WARNED.add(sync_engine)
    logger.warning(
        "Dendrux store on a single-connection pool (%s, e.g. in-memory SQLite, %s): "
        "database sessions are serialised and the data is not durable. "
        "Use a SQLite file or Postgres for anything beyond tests and demos.",
        type(sync_engine.pool).__name__,
        sync_engine.url.render_as_string(hide_password=True),
    )


def _guard_for(engine: AsyncEngine) -> _Guard:
    per_loop = _GUARDS.get(engine.sync_engine)
    if per_loop is None:
        per_loop = _GUARDS[engine.sync_engine] = weakref.WeakKeyDictionary()
    loop = asyncio.get_running_loop()
    guard = per_loop.get(loop)
    if guard is None:
        guard = per_loop[loop] = _Guard()
    return guard


@asynccontextmanager
async def serialized(engine: AsyncEngine) -> AsyncIterator[None]:
    """Hold ``engine``'s session lock for the body when its pool shares one connection.

    A no-op for every other pool. Re-entering from the task that already holds
    the lock does not wait.
    """
    if not shares_one_connection(engine):
        yield
        return
    guard = _guard_for(engine)
    task = asyncio.current_task()
    if task is not None and guard.owner is task:
        yield
        return
    async with guard.lock:
        guard.owner = task
        try:
            yield
        finally:
            guard.owner = None
