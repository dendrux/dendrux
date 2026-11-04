"""Live-store warnings count only open, reachable owners."""

import gc
import warnings

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
