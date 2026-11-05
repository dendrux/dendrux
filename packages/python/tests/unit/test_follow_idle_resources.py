"""What a reader keeps after its last read: nothing without live updates, and a clean loop exit."""

from __future__ import annotations

import asyncio

import pytest
from sqlalchemy.ext.asyncio import create_async_engine

from dendrux.db._sqlite_sync import create_sqlite_schema
from dendrux.errors import RunNotFoundError
from dendrux.store import RunStore, SQLiteStateStore, _follow
from dendrux.types import generate_ulid
from tests.unit.test_follow_coordinator import BatchedState, FakeLive, ticks
from tests.unit.test_follow_coordinator import clock as clock
from tests.unit.test_follow_coordinator import fast_budget as fast_budget


@pytest.fixture
async def sqlite_store(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/reader.db")
    await create_sqlite_schema(engine)
    store = SQLiteStateStore(engine)
    yield store
    await store.close()
    await engine.dispose()


async def test_per_request_readers_without_live_updates_leave_no_task(sqlite_store):
    run_id = generate_ulid()
    await sqlite_store.create_run(run_id, "reader")
    await sqlite_store.save_run_event(run_id, event_type="tick")
    before = asyncio.all_tasks()
    for _ in range(500):
        reader = RunStore(sqlite_store)
        stream = reader.stream_events(run_id, tenant_id=None, after_sequence_index=-1)
        await anext(stream)
        await stream.aclose()
    for _ in range(500):
        with pytest.raises(RunNotFoundError):
            await RunStore(sqlite_store).follow(generate_ulid(), tenant_id=None).start()
    await asyncio.sleep(0)
    assert asyncio.all_tasks() - before == set()


async def test_a_reader_holding_a_listener_still_retires_it_when_idle(clock):
    state = BatchedState()
    state.create("r")
    state.live_updates = FakeLive()
    reader = RunStore(state)
    live = state.live_updates
    try:
        stream = reader.stream_events("r", tenant_id=None, poll_interval_s=0.5)
        reading = asyncio.ensure_future(anext(stream))
        await ticks(20)
        assert reader._lease is not None
        reading.cancel()
        await asyncio.gather(reading, return_exceptions=True)
        await stream.aclose()
        await ticks(20)
        assert reader._lease is not None  # Kept for the next request of a burst.
        clock.advance(_follow.IDLE_EXIT_S + 1)
        await ticks(40)
        assert reader._lease is None and not live._leases
        assert not [t for t in asyncio.all_tasks() if "coordinator" in t.get_name()]
    finally:
        await reader.close()
