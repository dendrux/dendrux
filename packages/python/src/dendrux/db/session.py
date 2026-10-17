"""Session factory — engine creation and async session management.

Zero-config promise: first run() call auto-creates ~/.dendrux/dendrux.db
with all tables. No 'dendrux db migrate' needed for SQLite.

Postgres users set DENDRUX_DATABASE_URL and use Alembic migrations.
"""

from __future__ import annotations

import asyncio
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING
from weakref import WeakKeyDictionary

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

from dendrux.db._sqlite_sync import create_sqlite_schema

_engine: AsyncEngine | None = None
_engine_url: str | None = None
_session_factory: sessionmaker | None = None  # type: ignore[type-arg]
_engine_lock = threading.Lock()

# Default DB location: ~/.dendrux/dendrux.db
# Single well-known path so all commands and examples share the same DB.
_DENDRITE_DIR = Path.home() / ".dendrux"
DEFAULT_SQLITE_URL = f"sqlite+aiosqlite:///{_DENDRITE_DIR / 'dendrux.db'}"


def get_database_url() -> str:
    """Resolve database URL from environment or default to SQLite.

    Default: ~/.dendrux/dendrux.db (created automatically).
    Override: set DENDRUX_DATABASE_URL for Postgres or custom path.
    """
    return os.environ.get("DENDRUX_DATABASE_URL", DEFAULT_SQLITE_URL)


def loop_lock(registry: WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Lock]) -> asyncio.Lock:
    """The asyncio lock for the running loop (asyncio locks are loop-bound)."""
    loop = asyncio.get_running_loop()
    lock = registry.get(loop)
    if lock is None:
        lock = registry[loop] = asyncio.Lock()
    return lock


_init_locks: WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Lock] = WeakKeyDictionary()


def _published(resolved_url: str) -> AsyncEngine | None:
    if _engine is None:
        return None
    if _engine_url != resolved_url:
        raise RuntimeError(
            f"get_engine() already initialized with URL {_engine_url!r}, "
            f"cannot reinitialize with {resolved_url!r}. "
            f"Call reset_engine() first to change databases."
        )
    return _engine


async def get_engine(url: str | None = None) -> AsyncEngine:
    """Get or create the async engine.

    For SQLite, auto-creates all tables on first call and upgrades an older
    file in place. For Postgres, tables must exist via Alembic migrations.

    The engine is published only after initialization succeeds: concurrent
    callers wait for the first one, and a failed initialization disposes its
    candidate engine and leaves nothing cached, so the next call retries.
    """
    global _engine, _engine_url, _session_factory  # noqa: PLW0603
    resolved_url = url or get_database_url()

    ready = _published(resolved_url)
    if ready is not None:
        return ready

    async with loop_lock(_init_locks):
        ready = _published(resolved_url)
        if ready is not None:
            return ready

        # SQLite needs connect_args for async support
        connect_args = {}
        if resolved_url.startswith("sqlite"):
            connect_args["check_same_thread"] = False

        candidate = create_async_engine(
            resolved_url,
            echo=False,
            connect_args=connect_args,
        )
        try:
            if resolved_url.startswith("sqlite"):
                # Ensure the parent directory exists (e.g. ~/.dendrux/)
                db_path = resolved_url.split("///", 1)[-1] if "///" in resolved_url else None
                if db_path:
                    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
                async with candidate.begin() as conn:
                    await conn.run_sync(create_sqlite_schema)
        except BaseException:
            await candidate.dispose()
            raise

        with _engine_lock:
            if _engine is None:
                _engine = candidate
                _engine_url = resolved_url
                _session_factory = sessionmaker(  # type: ignore[call-overload]
                    candidate, class_=AsyncSession, expire_on_commit=False
                )
                return candidate
        # Another thread's loop published first; use theirs.
        await candidate.dispose()
        ready = _published(resolved_url)
        assert ready is not None
        return ready


async def reset_engine() -> None:
    """Dispose the current engine and session factory. Used in tests for cleanup.

    Releases the lock before awaiting dispose() to avoid holding a
    synchronous lock across an async suspension point.
    """
    global _engine, _engine_url, _session_factory  # noqa: PLW0603
    with _engine_lock:
        engine = _engine
        _engine = None
        _engine_url = None
        _session_factory = None
    # Dispose outside the lock — no deadlock risk
    if engine is not None:
        await engine.dispose()


@asynccontextmanager
async def get_session(engine: AsyncEngine | None = None) -> AsyncGenerator[AsyncSession, None]:
    """Async context manager for a database session.

    Usage:
        engine = await get_engine()
        async with get_session(engine) as session:
            session.add(run)
            await session.commit()
    """
    resolved_engine = engine or await get_engine()
    # Use cached factory when using the default engine, create ad-hoc for custom engines
    if engine is None and _session_factory is not None:
        factory = _session_factory
    else:
        factory = sessionmaker(  # type: ignore[call-overload]
            resolved_engine, class_=AsyncSession, expire_on_commit=False
        )
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
