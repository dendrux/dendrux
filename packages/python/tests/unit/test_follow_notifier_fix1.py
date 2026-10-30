"""Review regressions for notifier ownership and follow admission."""

import asyncio
import logging
from typing import TYPE_CHECKING, cast

import pytest

from dendrux.errors import NotifierUnavailableError
from dendrux.store import RunStore, StoredEvent
from dendrux.store._notifier_pg import PostgresEventNotifier
from tests.unit.test_follow_stream_notifier import State as StreamState
from tests.unit.test_notifier_pg import URL

if TYPE_CHECKING:
    from dendrux.runtime.state import StateStore


class State(StreamState):
    async def get_run_events(self, run_id, *, after_sequence_index=None, limit=100, **kwargs):
        return await super().get_run_events(
            run_id, after_sequence_index=after_sequence_index, limit=limit
        )


@pytest.mark.parametrize("kind", ["follow", "stream"])
async def test_p1_3_ten_owned_readers_share_failure_and_poll(monkeypatch, caplog, kind):
    import asyncpg

    caplog.set_level(logging.INFO)
    attempts = 0

    async def blackhole(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        await asyncio.Event().wait()

    monkeypatch.setattr(asyncpg, "connect", blackhole)
    notifier = PostgresEventNotifier(URL, connect_timeout_s=0.02, reconnect_backoff=(1, 1))
    state = State()
    state.write()
    reader = RunStore(cast("StateStore", state))
    reader._owned_notifier = reader._notifier = notifier
    opened = []

    async def read():
        item = (
            reader.follow("run", tenant_id=None, until_terminal=False)
            if kind == "follow"
            else reader.stream_events("run", tenant_id=None)
        )
        opened.append(item)
        return await anext(item)

    started = asyncio.get_running_loop().time()
    try:
        results = await asyncio.gather(*(read() for _ in range(10)), return_exceptions=True)
        elapsed = asyncio.get_running_loop().time() - started
        assert attempts == 1
        assert all(isinstance(item, StoredEvent) for item in results), results
        assert elapsed < 0.55
        assert not notifier.state.healthy
        await reader._start_owned_notifier()
        assert attempts == 1  # The failed attempt is negatively cached for backoff.
        assert [r.code for r in caplog.records].count("notifier_unavailable") == 1
    finally:
        for item in opened:
            await item.aclose()
        await reader.close()


async def test_p1_3_direct_concurrent_start_shares_one_result(monkeypatch):
    import asyncpg

    attempts = 0

    async def blackhole(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        await asyncio.Event().wait()

    monkeypatch.setattr(asyncpg, "connect", blackhole)
    notifier = PostgresEventNotifier(URL, connect_timeout_s=0.01)
    try:
        results = await asyncio.gather(
            *(notifier.start() for _ in range(10)), return_exceptions=True
        )
        assert attempts == 1
        assert all(isinstance(result, NotifierUnavailableError) for result in results)
    finally:
        await notifier.stop()


async def test_p2_4_owned_listener_uses_configured_writer_channel():
    reader = RunStore.from_database_url(
        "sqlite+aiosqlite://", database_options={"notify_channel": "custom_events"}, listen_url=URL
    )
    try:
        assert reader._owned_notifier.channel == "custom_events"
    finally:
        await reader.close()


async def test_owner_default_listener_channel_does_not_enable_writer():
    reader = RunStore.from_database_url("sqlite+aiosqlite://", listen_url=URL)
    try:
        assert reader._owned_notifier.channel == "dendrux_events"
        assert reader._state.notify_channel is None
    finally:
        await reader.close()


async def test_p2_4_owned_listener_rejects_explicit_disabled_writer():
    with pytest.raises(ValueError):
        reader = RunStore.from_database_url(
            "sqlite+aiosqlite://", database_options={"notify_channel": None}, listen_url=URL
        )
        await reader.close()


async def test_p2_6_follow_start_refuses_notifier_cap_before_iteration():
    from dendrux.store import FollowCapacityError

    reader = RunStore(cast("StateStore", State()))
    notifier = reader._notifier = PostgresEventNotifier(URL, max_subscriptions=1)
    first = reader.follow("first", tenant_id=None, until_terminal=False)
    second = reader.follow("second", tenant_id=None, until_terminal=False)
    try:
        await first.start()
        with pytest.raises(FollowCapacityError):
            await second.start()
        assert notifier._count == 1
        assert reader._event_feed.followers == 1
    finally:
        await first.aclose()
        await second.aclose()
        await reader.close()
    assert notifier._count == 0


async def test_p2_6_same_run_followers_share_admitted_subscription():
    reader = RunStore(cast("StateStore", State()))
    notifier = reader._notifier = PostgresEventNotifier(URL, max_subscriptions=1)
    first = reader.follow("same", tenant_id=None, until_terminal=False)
    second = reader.follow("same", tenant_id=None, until_terminal=False)
    try:
        await asyncio.gather(first.start(), second.start())
        assert notifier._count == 1
        await first.aclose()
        assert notifier._count == 1
    finally:
        await second.aclose()
        await reader.close()
    assert notifier._count == 0


async def test_p2_2_health_generation_jitters_the_next_fetch(monkeypatch):
    from tests.unit.test_follow_notifier import setup
    from tests.unit.test_follow_wake import Clock, ticks

    loop = asyncio.get_running_loop()
    clock = Clock(loop.time())
    monkeypatch.setattr(loop, "time", lambda: clock.now)
    # Patch the randomness provider independently of the future implementation.
    import random

    monkeypatch.setattr(random, "uniform", lambda low, high: 0.25)
    store, notifier, owner, run, follower = setup()
    try:
        await ticks(3)
        notifier.health(False)
        await ticks(3)
        assert store.polls == 1
        clock.advance(0.249)
        await ticks(3)
        assert store.polls == 1
        clock.advance(0.002)
        await ticks(4)
        assert store.polls == 2
    finally:
        await owner.aclose()


async def test_p2_7_empty_notification_fetch_gets_one_short_replica_retry(monkeypatch):
    from tests.unit.test_follow_notifier import setup
    from tests.unit.test_follow_wake import Clock, ticks

    loop = asyncio.get_running_loop()
    clock = Clock(loop.time())
    monkeypatch.setattr(loop, "time", lambda: clock.now)
    store, notifier, owner, run, follower = setup()
    try:
        await ticks(3)
        notifier.event.set()
        await ticks(3)
        assert store.polls == 2
        clock.advance(0.501)
        await ticks(4)
        assert store.polls == 3
        clock.advance(29)
        await ticks(4)
        assert store.polls == 3
        clock.advance(1.01)
        await ticks(4)
        assert store.polls == 4
    finally:
        await owner.aclose()


async def test_p2_10_close_disposes_engine_even_when_owned_stop_raises(monkeypatch):
    from unittest.mock import AsyncMock

    from tests.unit.test_follow_notifier import Notifier

    reader = RunStore.from_database_url("sqlite+aiosqlite://")
    notifier = Notifier()
    notifier.stop = AsyncMock(side_effect=RuntimeError("test stop failure"))
    reader._owned_notifier = notifier
    engine = reader._owned_engine
    from types import SimpleNamespace

    dispose = AsyncMock()
    reader._owned_engine = SimpleNamespace(dispose=dispose)
    try:
        with pytest.raises(RuntimeError):
            await reader.close()
        dispose.assert_awaited_once()
        assert reader._owned_engine is None
    finally:
        await engine.dispose()


async def test_p2_6_cancelled_admission_releases_subscription():
    from contextlib import asynccontextmanager

    notifier = PostgresEventNotifier(URL)
    entered = asyncio.Event()
    subscribe = notifier.subscribe

    @asynccontextmanager
    async def blocked(run_id):
        async with subscribe(run_id) as sub:
            entered.set()
            await asyncio.Event().wait()
            yield sub

    notifier.subscribe = blocked
    reader = RunStore(cast("StateStore", State()))
    reader._notifier = notifier
    follow = reader.follow("run", tenant_id=None, until_terminal=False)
    starting = asyncio.create_task(follow.start())
    try:
        await asyncio.wait_for(entered.wait(), 1)
        starting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await starting
        assert notifier._count == 0
        assert reader._event_feed.followers == 0
    finally:
        starting.cancel()
        await asyncio.gather(starting, return_exceptions=True)
        await follow.aclose()
        await reader.close()


async def test_p2_6_abandoned_started_follower_releases_capacity():
    import gc

    from tests.unit.test_follow_wake import ticks

    reader = RunStore(cast("StateStore", State()))
    notifier = reader._notifier = PostgresEventNotifier(URL)
    follow = reader.follow("run", tenant_id=None, until_terminal=False)
    try:
        await follow.start()
        assert notifier._count == 1
        del follow
        gc.collect()
        await ticks(5)
        assert notifier._count == 0
        assert reader._event_feed.followers == 0
    finally:
        await reader.close()
