"""Close checked SQL sessions within a bounded wait once their caller is cancelled."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from dendrux.db._pool_guard import close_fully, serialized

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

    from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

    from dendrux.runtime.stores.sql._sql_host import _StoreHost


class _ClosingSessionFactory:
    """Session factory whose sessions close through ``close_fully``.

    SQLAlchemy's ``AsyncSession.__aexit__`` shields its close and waits for it
    without a bound, so a session cancelled twice could wait forever on a close
    that stalls in the driver (aiosqlite 0.22.1 close after stop). ``close_fully``
    runs the close to completion but waits at most ``CLOSE_AFTER_CANCEL_S`` once
    the caller is cancelled.
    """

    def __init__(self, factory: Callable[[], AsyncSession]) -> None:
        self._factory = factory

    @asynccontextmanager
    async def __call__(self) -> AsyncIterator[AsyncSession]:
        session = self._factory()
        try:
            yield session
        finally:
            await close_fully(session.close)


class _SerializedSessionFactory:
    """Session factory that holds the pool's session lock for each session's life.

    The session is closed (ROLLBACK and pool reset included) before the lock
    is released, even when the caller is cancelled during the close.
    """

    def __init__(self, engine: AsyncEngine, factory: Callable[[], AsyncSession]) -> None:
        self._engine = engine
        engine.sync_engine.hide_parameters = True
        self._closing = _ClosingSessionFactory(factory)

    @asynccontextmanager
    async def __call__(self) -> AsyncIterator[AsyncSession]:
        async with serialized(self._engine), self._closing() as session:
            yield session


class _CheckedSessionFactory:
    def __init__(self, store: _StoreHost, factory: Any) -> None:
        self._store = store
        self._factory = factory

    @asynccontextmanager
    async def __call__(self) -> AsyncIterator[AsyncSession]:
        async with self._factory() as session:
            await self._store._check_connection(session)

            yield session


_SerializedSessionFactory.__module__ = "dendrux.runtime.stores.sql.base"
_CheckedSessionFactory.__module__ = "dendrux.runtime.stores.sql.base"
