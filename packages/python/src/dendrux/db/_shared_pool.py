"""Reference-counted process-local engines for explicit Agent URLs and env stores."""

from __future__ import annotations

import asyncio
import atexit
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from sqlalchemy.ext.asyncio import AsyncEngine


from dendrux.db._engine_options import engine_options, frozen_options, normalized_url
from dendrux.db._pool_guard import close_fully


@dataclass
class _Entry:
    engine: AsyncEngine
    ready: asyncio.Task[None]
    loop: asyncio.AbstractEventLoop
    refs: int = 0

    def can_retire(self) -> bool:
        """A dead owner makes every remaining reference stale."""
        return not self.refs or self.loop.is_closed()


_lock = threading.Lock()
_entries: dict[Any, _Entry] = {}
_by_engine: dict[AsyncEngine, Any] = {}
logger = logging.getLogger(__name__)


async def _initialize(engine: AsyncEngine) -> None:
    from dendrux.db._schema_check import ensure_schema_current
    from dendrux.db._sqlite_sync import create_sqlite_schema
    from dendrux.db.session import _bind_to_current_loop

    if engine.dialect.name == "sqlite":
        database = engine.url.database
        if database and database != ":memory:" and not database.startswith("file:"):
            Path(database).parent.mkdir(parents=True, exist_ok=True)
        await create_sqlite_schema(engine)
    else:
        await ensure_schema_current(engine)
    _bind_to_current_loop(engine)


def _retrieved(task: asyncio.Task[None]) -> None:
    if not task.cancelled():
        task.exception()  # also retrieved when every initializer waiter was cancelled


async def _dispose_entry(entry: _Entry) -> None:
    if entry.loop.is_closed():
        # Public disposal replaces the pool without touching dead-loop drivers.
        # The orphaned pool's connections are released by garbage collection.
        await entry.engine.dispose(close=False)
        logger.warning(
            "Retired a shared engine after its event loop closed; orphaned connections "
            "await garbage collection. Call await agent.close() or use async with "
            "Agent(...) before closing the loop.",
            extra={"code": "shared_engine_loop_closed"},
        )
        return
    if not entry.ready.done():
        entry.ready.cancel()
        await asyncio.gather(entry.ready, return_exceptions=True)
    await entry.engine.dispose()


async def _prepare(
    engine: AsyncEngine,
    initialize: Callable[[AsyncEngine], Awaitable[None]],
    previous: _Entry | None,
) -> None:
    if previous is not None:
        await _dispose_entry(previous)
    await initialize(engine)


async def acquire(
    url: str,
    options: dict[str, Any] | None = None,
    *,
    factory: Callable[..., AsyncEngine] | None = None,
    initialize: Callable[[AsyncEngine], Awaitable[None]] | None = None,
) -> AsyncEngine:
    from sqlalchemy.ext.asyncio import create_async_engine

    parsed = normalized_url(url)
    effective = engine_options(parsed, options)
    key = (parsed, frozen_options(effective))
    loop = asyncio.get_running_loop()
    with _lock:
        entry = _entries.get(key)
        previous = None
        if entry is not None and entry.loop.is_closed():
            previous = _entries.pop(key)
            _by_engine.pop(entry.engine)
            entry = None
        if (
            entry is not None
            and entry.loop is not loop
            and (entry.engine.dialect.name != "sqlite" or not entry.ready.done())
        ):
            raise RuntimeError("The shared database engine belongs to a different event loop")
        if entry is None:
            engine = (factory or create_async_engine)(parsed, **effective)
            ready = asyncio.ensure_future(_prepare(engine, initialize or _initialize, previous))
            ready.add_done_callback(_retrieved)
            entry = _entries[key] = _Entry(engine, ready, loop)
            _by_engine[engine] = key
        entry.refs += 1
    try:
        await asyncio.shield(entry.ready)
    except BaseException:
        await release(entry.engine)
        raise
    return entry.engine


async def release(engine: AsyncEngine) -> None:
    with _lock:
        key = _by_engine.get(engine)
        entry = _entries.get(key)
        if entry is None:
            return
        entry.refs -= 1
        if not entry.can_retire():
            return
        _entries.pop(key)
        _by_engine.pop(engine)

    await close_fully(lambda: _dispose_entry(entry))


def _retire_at_exit() -> None:
    # An event loop may already be closed at interpreter exit. Retire pools
    # synchronously without trying to await foreign-loop asyncpg connections;
    # process exit closes their sockets. Explicit close() is the graceful path.
    with _lock:
        engines = list(_by_engine)
        _entries.clear()
        _by_engine.clear()
    for engine in engines:
        engine.sync_engine.dispose(close=False)


atexit.register(_retire_at_exit)
