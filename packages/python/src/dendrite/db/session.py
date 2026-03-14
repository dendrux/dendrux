"""Session factory — engine creation and async session management.

Zero-config promise: first run() call auto-creates ./dendrite.db with all tables.
No 'dendrite db migrate' needed for SQLite.

Postgres users set DENDRITE_DATABASE_URL and use Alembic migrations.
"""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

from dendrite.db.models import Base

_engine: AsyncEngine | None = None
_session_factory: sessionmaker | None = None
_engine_lock = asyncio.Lock()

DEFAULT_SQLITE_URL = "sqlite+aiosqlite:///./dendrite.db"


def get_database_url() -> str:
    """Resolve database URL from environment or default to SQLite."""
    return os.environ.get("DENDRITE_DATABASE_URL", DEFAULT_SQLITE_URL)


async def get_engine(url: str | None = None) -> AsyncEngine:
    """Get or create the async engine.

    For SQLite, auto-creates all tables on first call.
    For Postgres, tables must exist via Alembic migrations.

    Thread-safe: uses asyncio.Lock to prevent duplicate engine creation
    when multiple coroutines call this concurrently.
    """
    global _engine, _session_factory  # noqa: PLW0603
    if _engine is not None:
        return _engine

    async with _engine_lock:
        # Double-check after acquiring lock
        if _engine is not None:
            return _engine

        resolved_url = url or get_database_url()

        # SQLite needs connect_args for async support
        connect_args = {}
        if resolved_url.startswith("sqlite"):
            connect_args["check_same_thread"] = False

        _engine = create_async_engine(
            resolved_url,
            echo=False,
            connect_args=connect_args,
        )

        _session_factory = sessionmaker(_engine, class_=AsyncSession, expire_on_commit=False)

        # Auto-create tables for SQLite (zero-config promise)
        if resolved_url.startswith("sqlite"):
            async with _engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)

        return _engine


async def reset_engine() -> None:
    """Dispose the current engine and session factory. Used in tests for cleanup."""
    global _engine, _session_factory  # noqa: PLW0603
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _session_factory = None


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
        factory = sessionmaker(resolved_engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
