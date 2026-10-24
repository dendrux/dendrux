"""Session factory — engine creation and async session management.

Zero-config promise: first run() call auto-creates ~/.dendrux/dendrux.db
with all tables. No 'dendrux db migrate' needed for SQLite.

Postgres users set DENDRUX_DATABASE_URL and use Alembic migrations.

The shared engine belongs to the event loop that created it. Using it from
another loop on Postgres raises a clear ``RuntimeError`` (instead of
asyncpg's "another operation is in progress"). Applications should create
one state store per process at startup and pass ``state_store=`` to every
Agent. URLs in messages are always masked.
"""

from __future__ import annotations

import asyncio
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

from dendrux._redact import mask_database_url
from dendrux.db._pool_guard import close_fully, serialized, shares_one_connection
from dendrux.db._schema_check import ensure_schema_current
from dendrux.db._sqlite_sync import create_sqlite_schema

_engine: AsyncEngine | None = None
_engine_url: str | None = None
_session_factory: sessionmaker | None = None  # type: ignore[type-arg]
_engine_lock = threading.Lock()
_generation = 0

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


def _published(resolved_url: str) -> AsyncEngine | None:
    if _engine is None:
        return None
    if _engine_url != resolved_url:
        raise RuntimeError(
            f"get_engine() already initialized with URL {mask_database_url(_engine_url or '')}, "
            f"cannot reinitialize with {mask_database_url(resolved_url)}. "
            f"Call reset_engine() first to change databases."
        )
    return _engine


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
                "The shared dendrux database engine (from DENDRUX_DATABASE_URL) was created "
                "on a different event loop and cannot be used from this one. Create one "
                "state store per process on the loop that serves requests and pass "
                "state_store= to every Agent, or call dendrux.db.session.reset_engine() "
                "before switching loops."
            )

    event.listen(engine.sync_engine, "checkout", _check_loop)


async def get_engine(url: str | None = None) -> AsyncEngine:
    """Get or create the async engine.

    For SQLite, auto-creates all tables on first call and upgrades an older
    file in place. For Postgres, tables must exist via Alembic migrations.

    A candidate engine is published only after its initialization succeeds
    and only if ``reset_engine()`` has not run since it started. Concurrent
    first callers each initialize a candidate (idempotent, serialized by
    SQLite's write lock); one is published, the rest are disposed. A failed
    initialization disposes its candidate and caches nothing.
    """
    global _engine, _engine_url, _session_factory  # noqa: PLW0603
    resolved_url = url or get_database_url()

    while True:
        with _engine_lock:
            ready = _published(resolved_url)
            generation = _generation
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
                await create_sqlite_schema(candidate)
            else:
                await ensure_schema_current(candidate)
        except BaseException:
            await candidate.dispose()
            raise

        _bind_to_current_loop(candidate)
        with _engine_lock:
            if _generation == generation and _engine is None:
                _engine = candidate
                _engine_url = resolved_url
                _session_factory = sessionmaker(  # type: ignore[call-overload]
                    candidate, class_=AsyncSession, expire_on_commit=False
                )
                return candidate
        # Reset ran or another caller published first: drop ours and look again.
        await candidate.dispose()


async def reset_engine() -> None:
    """Dispose the current engine and session factory. Used in tests for cleanup.

    Releases the lock before awaiting dispose() to avoid holding a
    synchronous lock across an async suspension point.
    """
    global _engine, _engine_url, _session_factory, _generation  # noqa: PLW0603
    with _engine_lock:
        _generation += 1
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

    On a single-connection pool (in-memory SQLite) the session holds the
    pool's lock until it is closed, so Dendrux's other sessions wait for it.
    Never hold such a session across a call that runs an agent or starts
    tasks using the same engine (for example for a whole web request that
    calls ``agent.run()``): those tasks wait for the lock and the call hangs.
    """
    resolved_engine = engine or await get_engine()
    # Use cached factory when using the default engine, create ad-hoc for custom engines
    if engine is None and _session_factory is not None:
        factory = _session_factory
    else:
        factory = sessionmaker(  # type: ignore[call-overload]
            resolved_engine, class_=AsyncSession, expire_on_commit=False
        )
    if not shares_one_connection(resolved_engine):
        async with factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
        return
    async with serialized(resolved_engine):
        session = factory()
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await close_fully(session.close)
