"""Session factory — engine creation and async session management.

Zero-config promise: first run() call auto-creates ./dendrux.db
with all tables. No 'dendrux db migrate' needed for SQLite.

Postgres users set DENDRUX_DATABASE_URL and use Alembic migrations.

The shared engine belongs to the event loop that created it. Using it from
another loop on Postgres raises a clear ``RuntimeError`` (instead of
asyncpg's "another operation is in progress"). Applications should create
one state store per process at startup and pass ``state_store=`` to every
Agent. URLs in messages are always masked.
"""

from __future__ import annotations

import logging
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

from dendrux import _codes
from dendrux._redact import mask_database_url
from dendrux.db import _shared_pool
from dendrux.db._pool_guard import (
    _bind_to_current_loop,
    close_fully,
    serialized,
    shares_one_connection,
)
from dendrux.db._schema_check import ensure_schema_current
from dendrux.db._sqlite_sync import create_sqlite_schema

_engine: AsyncEngine | None = None
_engine_url: str | None = None
_engine_options_key: Any = None
_session_factory: sessionmaker | None = None  # type: ignore[type-arg]
_engine_lock = threading.Lock()
_generation = 0

logger = logging.getLogger(__name__)
DEFAULT_SQLITE_URL = "sqlite+aiosqlite:///./dendrux.db"


def get_database_url() -> str:
    """Resolve database URL from environment or default to SQLite.

    Default: ./dendrux.db (created automatically).
    Override: set DENDRUX_DATABASE_URL for Postgres or custom path.

    The existing home database is never selected implicitly.
    When only that legacy file exists, resolution logs a warning.
    Explicit environment values take precedence over file checks.
    Applications and commands should use the same working directory.
    """
    return _resolve_database_url(os.environ.get("DENDRUX_DATABASE_URL"))


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


async def _initialize_engine(engine: AsyncEngine) -> None:
    if engine.dialect.name == "sqlite":
        database = engine.url.database
        if database and database != ":memory:" and not database.startswith("file:"):
            Path(database).parent.mkdir(parents=True, exist_ok=True)
        await create_sqlite_schema(engine)
    else:
        await ensure_schema_current(engine)
    _bind_to_current_loop(engine)


async def get_engine(
    url: str | None = None, *, database_options: dict[str, Any] | None = None
) -> AsyncEngine:
    """Resolve the env-store singleton using the shared URL/options pool.

    Defaults for non-SQLite pools are pre_ping=True and recycle=1800. The
    singleton owns one registry reference until reset_engine or process exit.
    Initialization is shared by all first callers and cancellation-safe.
    """
    global _engine, _engine_url, _session_factory, _engine_options_key
    from dendrux.db._engine_options import engine_options, frozen_options, normalized_url

    parsed = normalized_url(url or get_database_url())
    resolved_url = parsed.render_as_string(hide_password=False)
    options_key = frozen_options(engine_options(parsed, database_options))
    while True:
        with _engine_lock:
            ready = _published(resolved_url)
            generation = _generation
            if ready is not None and _engine_options_key != options_key:
                raise RuntimeError(
                    "get_engine() database_options differ; call reset_engine() first"
                )
        if ready is not None:
            return ready
        candidate = await _shared_pool.acquire(
            resolved_url,
            database_options,
            factory=create_async_engine,
            initialize=_initialize_engine,
        )
        with _engine_lock:
            if _generation == generation and _engine is None:
                _engine, _engine_url, _engine_options_key = candidate, resolved_url, options_key
                _session_factory = sessionmaker(  # type: ignore[call-overload]
                    candidate, class_=AsyncSession, expire_on_commit=False
                )
                return candidate
        await _shared_pool.release(candidate)


async def reset_engine() -> None:
    """Dispose the current engine and session factory. Used in tests for cleanup.

    Releases the lock before awaiting dispose() to avoid holding a
    synchronous lock across an async suspension point.
    """
    global _engine, _engine_url, _session_factory, _generation, _engine_options_key  # noqa: PLW0603
    with _engine_lock:
        _generation += 1
        engine = _engine
        _engine = None
        _engine_url = None
        _engine_options_key = None
        _session_factory = None
    # Dispose outside the lock — no deadlock risk
    if engine is not None:
        await _shared_pool.release(engine)


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


def _resolve_database_url(url: str | None) -> str:
    if url is not None:
        return url
    if Path("dendrux.db").exists():
        return DEFAULT_SQLITE_URL
    try:
        legacy_exists = (Path.home() / ".dendrux/dendrux.db").exists()
    except (OSError, RuntimeError):
        legacy_exists = False
    if legacy_exists:
        _codes.log_code(logger, _codes.LOCAL_DATABASE_DEFAULT_CHANGED)
    return DEFAULT_SQLITE_URL
