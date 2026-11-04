"""Live-store warnings count only open, reachable owners."""

import gc
import inspect
import warnings

import pytest
from sqlalchemy.ext.asyncio import create_async_engine

from dendrux.runtime.stores._factory import store_for_engine
from tests.unit.test_live_updates_store_lifecycle import RecordingUpdates


async def test_many_stores_warn_once_with_counts_and_hash_only(caplog):
    engine = create_async_engine(
        "postgresql+asyncpg://sentinel:secret@hidden/database_registry_test"
    )
    stores = []
    try:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            for _ in range(160):  # ten times the realistic one-pool-per-worker threshold
                stores.append(store_for_engine(engine, live_updates=RecordingUpdates()))
        matching = [item for item in captured if item.category is RuntimeWarning]
        assert len(matching) == 1
        records = [
            r for r in caplog.records if getattr(r, "code", None) == "live_updates_many_stores"
        ]
        assert len(records) == 1
        assert records[0].count == 17
        assert len(records[0].database_key) == 64
        for value in ("sentinel", "secret", "hidden", "database_registry_test"):
            assert value not in str(matching[0].message)
            assert value not in records[0].getMessage()
        for store in stores:
            await store.close()
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            for _ in range(17):
                stores.append(store_for_engine(engine, live_updates=RecordingUpdates()))
        assert not captured
    finally:
        for store in stores:
            await store.close()
        await engine.dispose()


async def test_closed_and_collected_stores_do_not_count():
    engine = create_async_engine("postgresql+asyncpg:///registry_collection_test")
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        for _ in range(160):
            store = store_for_engine(engine, live_updates=RecordingUpdates())
            await store.close()
        del store
        for _ in range(160):
            store_for_engine(engine, live_updates=RecordingUpdates())
            gc.collect()
    assert not captured
    await engine.dispose()


@pytest.mark.parametrize(
    "dialect,entry",
    [(d, e) for d in ("sqlite", "postgresql", "mysql") for e in ("direct", "factory")]
    + [("mysql", "url")],
)
async def test_many_stores_warning_points_to_developer_call(monkeypatch, dialect, entry):
    from dendrux.runtime.stores.sql import _live_registry
    from dendrux.runtime.stores.sql.mysql import MySQLStateStore
    from dendrux.runtime.stores.sql.postgres import PostgresStateStore
    from dendrux.runtime.stores.sql.sqlite import SQLiteStateStore

    monkeypatch.setattr(_live_registry, "MANY_STORES_THRESHOLD", 0)
    monkeypatch.setattr(_live_registry, "_warned", set())
    monkeypatch.setattr(_live_registry, "_open_stores", {})
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
    bus = RecordingUpdates()
    try:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            if entry == "factory":
                line = inspect.currentframe().f_lineno + 1
                store = store_for_engine(engine, live_updates=bus)
            elif entry == "url":
                line = inspect.currentframe().f_lineno + 1
                store = MySQLStateStore.from_url(urls[dialect], live_updates=bus)
                await engine.dispose()
                engine = store._engine
            else:
                line = inspect.currentframe().f_lineno + 1
                store = classes[dialect](engine, live_updates=bus)
        matching = [item for item in captured if "Open live-update stores" in str(item.message)]
        assert len(matching) == 1
        assert (matching[0].filename, matching[0].lineno) == (__file__, line)
        await store.close()
    finally:
        await engine.dispose()
