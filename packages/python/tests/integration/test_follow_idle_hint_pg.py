"""An admission during idle sleep remains subscribed to real PostgreSQL hints."""

import asyncio

import pytest

from dendrux.errors import RunNotFoundError
from dendrux.store import EventFeed, PostgresStateStore, PostgresUpdates, RunStore
from dendrux.types import generate_ulid

pytestmark = pytest.mark.parametrize("engine", ["postgres"], indirect=True)


async def test_join_during_idle_sleep_gets_postgres_hint(engine):
    channel = "rf1_idle_" + generate_ulid().lower()
    state = PostgresStateStore(engine, live_updates=PostgresUpdates(channel=channel))
    reader = RunStore(state)
    feed = EventFeed(reader, poll_healthy_s=20)
    rid = generate_ulid()
    await state.create_run(rid, "idle-hint")
    first = feed.follow(rid, tenant_id=None)
    second = feed.follow(rid, tenant_id=None)
    try:
        await first.start()
        async with asyncio.timeout(3):
            while not state.live_updates.healthy:
                await asyncio.sleep(0.01)
        await asyncio.sleep(0.1)
        await first.aclose()
        with pytest.raises(RunNotFoundError):
            await feed.follow(generate_ulid(), tenant_id=None).start()
        await second.start()
        await asyncio.sleep(1.0)
        assert not second._follower.buffer
        await state.save_run_event(rid, event_type="tick")
        assert (await asyncio.wait_for(anext(second), 0.5)).event_type == "tick"
    finally:
        await first.aclose()
        await second.aclose()
        await reader.close()
        await state.close()
