"""Reference-counted process-local engines for explicit Agent URLs and env stores."""

from __future__ import annotations

import asyncio
import atexit
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
    reaper: asyncio.Task[None] | None = None


_lock = threading.Lock()
_entries: dict[Any, _Entry] = {}
_by_engine: dict[AsyncEngine, Any] = {}
_reapers: set[asyncio.Task[None]] = set()


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


async def _loop_lifetime(entry: _Entry) -> None:
    """asyncio.run cancels this sentinel before closing its still-live loop.

    Forget ownership first so closing cancelled jobs cannot await this reaper.
    Drain other cancelled tasks before disposal, letting their final writes and
    checkins finish on the owner loop. Other pool reapers are excluded to avoid
    mutual waits. Explicit release has already removed its entry, so is a no-op.
    """
    try:
        await asyncio.Future()
    finally:
        with _lock:
            key = _by_engine.get(entry.engine)
            owned = _entries.get(key) is entry
            if owned:
                _entries.pop(key)
                _by_engine.pop(entry.engine)
        if owned:
            others = [t for t in asyncio.all_tasks() if t.cancelling() and t not in _reapers]
            await asyncio.gather(*others, return_exceptions=True)
            await entry.engine.dispose()


def _reaped(task: asyncio.Task[None]) -> None:
    _reapers.discard(task)
    _retrieved(task)


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
        if entry is not None and entry.loop.is_closed():
            # Defensive fallback for manually closed loops. Normal asyncio.run
            # shutdown has already drained the pool via _loop_lifetime.
            _entries.pop(key)
            _by_engine.pop(entry.engine)
            entry.engine.sync_engine.dispose(close=False)
            entry = None
        if (
            entry is not None
            and entry.loop is not loop
            and (entry.engine.dialect.name != "sqlite" or not entry.ready.done())
        ):
            raise RuntimeError("The shared database engine belongs to a different event loop")
        if entry is None:
            engine = (factory or create_async_engine)(parsed, **effective)
            ready = asyncio.ensure_future((initialize or _initialize)(engine))
            ready.add_done_callback(_retrieved)
            entry = _entries[key] = _Entry(engine, ready, loop)
            _by_engine[engine] = key
            entry.reaper = asyncio.create_task(_loop_lifetime(entry))
            _reapers.add(entry.reaper)
            entry.reaper.add_done_callback(_reaped)
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
        if entry.refs:
            return
        _entries.pop(key)
        _by_engine.pop(engine)

    async def dispose() -> None:
        if entry.reaper is not None:
            entry.reaper.cancel()
            await asyncio.gather(entry.reaper, return_exceptions=True)
        if not entry.ready.done():
            entry.ready.cancel()
            await asyncio.gather(entry.ready, return_exceptions=True)
        await entry.engine.dispose()

    await close_fully(dispose)


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
