"""Ready-made kit fixtures for the built-in ``SQLiteStateStore``.

A worked example of what a store author provides, and enough to certify SQLite from an installed
wheel: ``pytest -p dendrux.testing.store_kit -p dendrux.testing.store_kit.sqlite_fixtures``.
``DENDRUX_KIT_SQLITE=file`` uses a database file per test (``single_host``); the default is an
in-memory database (``single_process``).
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import pytest
import pytest_asyncio
from sqlalchemy import event
from sqlalchemy.ext.asyncio import create_async_engine

from dendrux.db._settings import seed_settings
from dendrux.db.models import Base
from dendrux.runtime.stores.sql.sqlite import SQLiteStateStore
from dendrux.testing.store_kit.sql import SqlClock, SqlFaultInjector

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Iterator
    from pathlib import Path

    from sqlalchemy.ext.asyncio import AsyncEngine

MODE_ENV = "DENDRUX_KIT_SQLITE"


class _SqliteWorld:
    def __init__(self, directory: Path) -> None:
        self.file = os.environ.get(MODE_ENV, "memory") == "file"
        self._directory = directory
        self.engines: list[AsyncEngine] = []
        self.url = ""

    def _engine(self, url: str) -> AsyncEngine:
        engine = create_async_engine(url, connect_args={"check_same_thread": False})

        @event.listens_for(engine.sync_engine, "connect")
        def _foreign_keys(dbapi_connection: Any, _record: Any) -> None:
            dbapi_connection.execute("PRAGMA foreign_keys = ON")

        self.engines.append(engine)
        return engine

    async def fresh(self) -> SQLiteStateStore:
        self.url = (
            f"sqlite+aiosqlite:///{self._directory / f'kit{len(self.engines)}.db'}"
            if self.file
            else "sqlite+aiosqlite:///:memory:"
        )
        engine = self._engine(self.url)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            await conn.run_sync(lambda c: seed_settings(c, active=True))
        return SQLiteStateStore(engine)

    async def reopen(self, store: SQLiteStateStore) -> SQLiteStateStore:
        # in memory the data lives in the engine, so the new store shares it
        return await self.second(store) if self.file else SQLiteStateStore(store._engine)

    async def second(self, store: SQLiteStateStore) -> SQLiteStateStore:
        if not self.file:
            raise RuntimeError("an in-memory database has no second connection")
        return SQLiteStateStore(self._engine(self.url))


@pytest_asyncio.fixture
async def dendrux_sqlite_world(tmp_path: Path) -> AsyncIterator[_SqliteWorld]:
    world = _SqliteWorld(tmp_path)
    yield world
    for engine in world.engines:
        await engine.dispose()


@pytest.fixture
def dendrux_store_factory(dendrux_sqlite_world: _SqliteWorld) -> Callable[[], Any]:
    return dendrux_sqlite_world.fresh


@pytest.fixture
def dendrux_store_reopen(dendrux_sqlite_world: _SqliteWorld) -> Callable[[Any], Any]:
    return dendrux_sqlite_world.reopen


@pytest.fixture
def dendrux_store_second_instance(dendrux_sqlite_world: _SqliteWorld) -> Callable[[Any], Any]:
    return dendrux_sqlite_world.second


@pytest.fixture
def dendrux_clock(dendrux_sqlite_world: _SqliteWorld) -> SqlClock:
    return SqlClock(lambda: dendrux_sqlite_world.engines[0])


@pytest.fixture
def dendrux_fault_injector() -> Iterator[SqlFaultInjector]:
    injector = SqlFaultInjector()
    yield injector
    injector.close()
