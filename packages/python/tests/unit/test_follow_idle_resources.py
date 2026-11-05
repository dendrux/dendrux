"""What a reader keeps after its last read: nothing without live updates, and a clean loop exit."""

from __future__ import annotations

import asyncio
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import create_async_engine

from dendrux.db._sqlite_sync import create_sqlite_schema
from dendrux.errors import RunNotFoundError
from dendrux.store import EventFeed, RunStore, SQLiteStateStore, _follow
from dendrux.types import generate_ulid
from tests.unit.test_follow_coordinator import BatchedState, FakeLive, ticks
from tests.unit.test_follow_coordinator import clock as clock
from tests.unit.test_follow_coordinator import fast_budget as fast_budget

PACKAGE = Path(__file__).resolve().parents[2]


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


async def test_idle_clock_starts_when_the_last_active_run_stops(clock):
    state = BatchedState()
    state.create("r")
    state.live_updates = FakeLive()
    reader = RunStore(state)
    feed = EventFeed(reader, poll_healthy_s=1.0)
    follow = feed.follow("r", tenant_id=None)
    try:
        await follow.start()
        await ticks(20)
        for _ in range(int(_follow.IDLE_EXIT_S) + 5):  # A long run: older than the idle period.
            clock.advance(1.0)
            await ticks(10)
        state.write("r", "run.completed")
        state.statuses["r"] = "success"
        state.live_updates.hint("r")
        for _ in range(40):  # Terminal, with the event still unread by a slow consumer.
            clock.advance(_follow.TERMINAL_DRAIN_GRACE_S)
            await ticks(10)
        assert follow._follower.end_reason == "terminal" and follow._follower.buffer
        assert reader._lease is not None
        clock.advance(_follow.IDLE_EXIT_S + 1)
        await ticks(40)
        assert reader._lease is None
    finally:
        await follow.aclose()
        await reader.close()


EXIT_SCRIPT = """
    import asyncio, sys, tempfile
    sys.path.insert(0, {package!r})
    from sqlalchemy.ext.asyncio import create_async_engine
    from dendrux.db._sqlite_sync import create_sqlite_schema
    from dendrux.store import EventFeed, RunStore, SQLiteStateStore, _follow
    from dendrux.types import generate_ulid
    from tests.unit.test_follow_coordinator import BatchedState, FakeLive

    KEEP = []

    async def sqlite_stream():
        path = tempfile.mkdtemp()
        engine = create_async_engine(f"sqlite+aiosqlite:///{{path}}/x.db")
        await create_sqlite_schema(engine)
        store = SQLiteStateStore(engine)
        run_id = generate_ulid()
        await store.create_run(run_id, "exit")
        await store.save_run_event(run_id, event_type="tick")
        stream = RunStore(store).stream_events(run_id, tenant_id=None, after_sequence_index=-1)
        await anext(stream)
        KEEP.append(stream)

    def live_reader():
        state = BatchedState()
        state.create("r")
        state.write("r")
        state.live_updates = FakeLive()
        return state, RunStore(state)

    async def live_stream():
        _, reader = live_reader()
        stream = reader.stream_events("r", tenant_id=None, after_sequence_index=-1)
        await anext(stream)
        KEEP.append(stream)

    async def live_follow():
        _, reader = live_reader()
        follow = EventFeed(reader).follow("r", tenant_id=None)

        async def read():
            async for _ in follow:
                pass

        KEEP.append(asyncio.create_task(read()))
        await asyncio.sleep(0.05)

    asyncio.run({scenario}())
    KEEP.clear()
    import gc
    gc.collect()
    print("clean exit")
"""


@pytest.mark.parametrize("scenario", ["sqlite_stream", "live_stream", "live_follow"])
def test_reader_left_open_at_loop_exit_is_quiet(scenario):
    script = textwrap.dedent(EXIT_SCRIPT).format(package=str(PACKAGE), scenario=scenario)
    done = subprocess.run(
        [sys.executable, "-X", "dev", "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=PACKAGE,
    )
    assert done.returncode == 0, done.stderr
    assert "clean exit" in done.stdout
    for noise in ("Task was destroyed", "no running event loop", "was never awaited"):
        assert noise not in done.stderr, done.stderr
