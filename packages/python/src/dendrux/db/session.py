"""Session factory — engine creation and async session management.

Zero-config promise: first run() call auto-creates ~/.dendrux/dendrux.db
with all tables. No 'dendrux db migrate' needed for SQLite.

Postgres users set DENDRUX_DATABASE_URL and use Alembic migrations.
"""

from __future__ import annotations

import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

from dendrux.db.models import Base

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


async def get_engine(url: str | None = None) -> AsyncEngine:
    """Get or create the async engine.

    For SQLite, auto-creates all tables on first call.
    For Postgres, tables must exist via Alembic migrations.

    Thread-safe: uses threading.Lock with double-check pattern to prevent
    duplicate engine creation from concurrent calls.
    """
    global _engine, _engine_url, _session_factory  # noqa: PLW0603
    resolved_url = url or get_database_url()

    if _engine is not None:
        if _engine_url != resolved_url:
            raise RuntimeError(
                f"get_engine() already initialized with URL {_engine_url!r}, "
                f"cannot reinitialize with {resolved_url!r}. "
                f"Call reset_engine() first to change databases."
            )
        return _engine

    with _engine_lock:
        # Double-check after acquiring lock
        if _engine is not None:
            if _engine_url != resolved_url:
                raise RuntimeError(
                    f"get_engine() already initialized with URL {_engine_url!r}, "
                    f"cannot reinitialize with {resolved_url!r}. "
                    f"Call reset_engine() first to change databases."
                )
            return _engine

        # SQLite needs connect_args for async support
        connect_args = {}
        if resolved_url.startswith("sqlite"):
            connect_args["check_same_thread"] = False

        _engine = create_async_engine(
            resolved_url,
            echo=False,
            connect_args=connect_args,
        )

        _engine_url = resolved_url
        _session_factory = sessionmaker(  # type: ignore[call-overload]
            _engine, class_=AsyncSession, expire_on_commit=False
        )

    # Auto-create tables for SQLite (zero-config promise)
    # Outside the lock — idempotent and safe for concurrent calls
    if resolved_url.startswith("sqlite"):
        # Ensure the parent directory exists (e.g. ~/.dendrux/)
        db_path = resolved_url.split("///", 1)[-1] if "///" in resolved_url else None
        if db_path:
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        async with _engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    return _engine


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
