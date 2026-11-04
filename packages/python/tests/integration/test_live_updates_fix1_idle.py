"""A transient local send failure must not pin idle readers to fast polling."""

import asyncio

from sqlalchemy import event
from sqlalchemy.ext.asyncio import create_async_engine

from dendrux.db.models import Base
from dendrux.store import EventFeed, RunStore, SQLiteStateStore
from dendrux.types import generate_ulid
from tests.unit.test_live_updates_base import Bus


async def _idle_selects(tmp_path, blip):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/{blip}.db")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    bus = Bus(backoff_initial_s=0.05, backoff_max_s=0.1)
    store = SQLiteStateStore(engine, live_updates=bus)
    runs = [generate_ulid() for _ in range(20)]
    for run in runs:
        await store.create_run(run, "idle")
    reader = RunStore(store)
    feed = EventFeed(reader, poll_interval_s=0.02, poll_max_s=0.05, poll_healthy_s=0.4)
    followers = [feed.follow(run, tenant_id=None) for run in runs]
    count = [0]

    def statement(conn, cursor, sql, *args):
        if sql.lstrip().upper().startswith("SELECT"):
            count[0] += 1

    try:
        for follower in followers:
            await follower.start()
        await asyncio.sleep(0.5)
        if blip:

            async def fail():
                raise TimeoutError

            bus.action = fail
            bus.enqueue(["local-failure"])
            await bus.flush()
            bus.action = None
        await asyncio.sleep(0.5)  # beyond backoff, with no succeeding send
        event.listen(engine.sync_engine, "before_cursor_execute", statement)
        await asyncio.sleep(1.2)
        return count[0], bus.healthy, bus.stats()
    finally:
        for follower in followers:
            await follower.aclose()
        await reader.close()
        await store.close()
        await engine.dispose()


async def test_idle_recovery_restores_follower_safety_poll_rate(tmp_path):
    normal, _, _ = await _idle_selects(tmp_path, False)
    recovered, healthy, stats = await _idle_selects(tmp_path, True)
    assert healthy
    assert stats.last_error_type == "TimeoutError" and stats.dropped_failed == 1
    assert recovered <= normal * 1.5 + 20, (normal, recovered)
