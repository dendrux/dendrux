"""Store ownership and explicit live-update configuration."""

from collections.abc import AsyncIterator, Sequence
from unittest.mock import AsyncMock

import pytest
from sqlalchemy.ext.asyncio import create_async_engine

from dendrux.db._live_updates import LiveUpdates
from dendrux.runtime.stores._factory import store_for_engine


class RecordingUpdates(LiveUpdates):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.messages = []
        self.aclose_sender = AsyncMock()

    async def send(self, messages: Sequence[str]) -> None:
        self.messages.extend(messages)

    async def receive(self) -> AsyncIterator[str | None]:
        yield None


@pytest.mark.parametrize("dialect", ["sqlite", "postgresql", "mysql"])
async def test_bus_bound_property_close_and_engine_ownership(dialect, monkeypatch):
    urls = {
        "sqlite": "sqlite+aiosqlite://",
        "postgresql": "postgresql+asyncpg://",
        "mysql": "mysql+asyncmy://",
    }
    if dialect == "mysql":
        pytest.importorskip("asyncmy")
    engine = create_async_engine(urls[dialect])
    dispose = AsyncMock()
    monkeypatch.setattr(type(engine), "dispose", dispose)
    bus = RecordingUpdates()
    store = store_for_engine(engine, live_updates=bus)
    assert store.live_updates is bus
    assert bus._store is store
    with pytest.raises(AttributeError):
        store.live_updates = None
    with pytest.raises(ValueError, match="another store"):
        store_for_engine(engine, live_updates=bus)
    await store.close()
    await store.close()
    bus.aclose_sender.assert_awaited_once()
    dispose.assert_not_awaited()


@pytest.mark.parametrize("value", [True, 1, "sentinel"])
async def test_non_native_rejects_invalid_configuration(value):
    engine = create_async_engine("sqlite+aiosqlite://")
    try:
        with pytest.raises(ValueError, match="SQLiteStateStore.*RedisUpdates"):
            store_for_engine(engine, live_updates=value)
    finally:
        await engine.dispose()


async def test_postgres_native_hook_is_lazy_and_typed_until_b():
    from dendrux.errors import LiveUpdatesUnavailableError

    engine = create_async_engine("postgresql+asyncpg://")
    try:
        assert store_for_engine(engine).live_updates is None
        with pytest.raises(LiveUpdatesUnavailableError):
            store_for_engine(engine, live_updates=True)
    finally:
        await engine.dispose()


async def test_mysql_from_url_names_bus_without_forwarding_to_engine():
    pytest.importorskip("asyncmy")
    from dendrux.runtime.stores.sql.mysql import MySQLStateStore

    bus = RecordingUpdates()
    store = MySQLStateStore.from_url("mysql+asyncmy://", live_updates=bus)
    assert store.live_updates is bus
    await store.close()
    await store._engine.dispose()


async def test_agent_closes_only_owned_store(monkeypatch):
    from dendrux import Agent

    agent = Agent(prompt="test", database_url="sqlite+aiosqlite://")
    store = await agent._open_state_store()
    close = AsyncMock(wraps=store.close)
    monkeypatch.setattr(store, "close", close)
    await agent.close()
    await agent.close()
    close.assert_awaited_once()
    external = store_for_engine(create_async_engine("sqlite+aiosqlite://"))
    monkeypatch.setattr(external, "close", AsyncMock())
    await Agent(prompt="test", state_store=external).close()
    external.close.assert_not_awaited()
    await external._engine.dispose()


@pytest.mark.parametrize(
    "dialect,entry",
    [(d, e) for d in ("sqlite", "postgresql", "mysql") for e in ("direct", "factory")]
    + [("mysql", "url")],
)
@pytest.mark.parametrize("off", [None, False])
async def test_disabled_live_updates_is_off_on_every_store_path(dialect, entry, off):
    from dendrux.runtime.stores.sql.mysql import MySQLStateStore
    from dendrux.runtime.stores.sql.postgres import PostgresStateStore
    from dendrux.runtime.stores.sql.sqlite import SQLiteStateStore

    classes = {
        "sqlite": SQLiteStateStore,
        "postgresql": PostgresStateStore,
        "mysql": MySQLStateStore,
    }
    urls = {
        "sqlite": "sqlite+aiosqlite://",
        "postgresql": "postgresql+asyncpg://",
        "mysql": "mysql+asyncmy://",
    }
    if dialect == "mysql":
        pytest.importorskip("asyncmy")
    engine = create_async_engine(urls[dialect])
    try:
        if entry == "url":
            store = MySQLStateStore.from_url(urls[dialect], live_updates=off)
            await engine.dispose()
            engine = store._engine
        elif entry == "factory":
            store = store_for_engine(engine, live_updates=off)
        else:
            store = classes[dialect](engine, live_updates=off)
        assert store.live_updates is None
        await store.close()
    finally:
        await engine.dispose()


@pytest.mark.parametrize("attribute", [None, True, False, object(), "custom"])
async def test_runtime_sweep_ignores_non_bus_live_updates(attribute):
    from datetime import timedelta
    from types import SimpleNamespace

    from dendrux import sweep

    if attribute == "custom":
        attribute = SimpleNamespace(flush_deadline_s=1.0)
    store = SimpleNamespace(
        live_updates=attribute,
        sweep_stale_runs=AsyncMock(return_value=[]),
        sweep_abandoned_runs=AsyncMock(return_value=[]),
    )
    result = await sweep(
        state_store=store,
        stale_running=timedelta(hours=1),
        abandoned_waiting=timedelta(hours=1),
        on_stale="error",
    )
    assert result.stale_running == result.abandoned_waiting == []
    store.sweep_stale_runs.assert_awaited_once()
    store.sweep_abandoned_runs.assert_awaited_once()
