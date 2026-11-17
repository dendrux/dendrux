"""Regression cases for listener ownership, buffer capacity and polling fairness."""

import asyncio
import logging
from unittest.mock import AsyncMock

import pytest

from dendrux.errors import FollowCapacityError, RunNotFoundError
from dendrux.store import EventFeed, FollowEnd, RunStore, _follow
from tests._helpers.gates import wait_for_gate
from tests.unit.test_follow_coordinator import BatchedState, FakeLive, ticks
from tests.unit.test_follow_coordinator import clock as clock
from tests.unit.test_follow_lease_lifecycle import SlowStopLive


def make_reader(live=None):
    state = BatchedState()
    state.create("r")
    state.live_updates = live or FakeLive()
    return RunStore(state), state


@pytest.mark.parametrize("kind", ["missing", "drained", "stream"])
async def test_short_request_burst_shares_listener_until_idle(monkeypatch, kind):
    monkeypatch.setattr(_follow, "IDLE_EXIT_S", 0.08)
    reader, state = make_reader()
    live = state.live_updates
    acquire = AsyncMock(wraps=live._acquire_listener)
    monkeypatch.setattr(live, "_acquire_listener", acquire)
    state.statuses["r"] = "success"
    state.write("r", "run.completed")
    try:
        for _ in range(5):
            if kind == "missing":
                with pytest.raises(RunNotFoundError):
                    await reader.follow("missing", tenant_id=None).start()
            elif kind == "drained":
                async with reader.follow("r", tenant_id=None, after_sequence_index=0):
                    pass
            else:
                stream = reader.stream_events("r", tenant_id=None)
                await anext(stream)
                await stream.aclose()
        assert acquire.await_count == 1
        assert reader._lease is not None
        await asyncio.sleep(0.12)
        assert reader._lease is None
        assert not live._leases and not live._consumers
    finally:
        await reader.close()
        await live.close()


async def test_join_while_coordinator_idles_receives_real_hub_hint():
    reader, state = make_reader()
    feed = EventFeed(reader, poll_healthy_s=20)
    first = feed.follow("r", tenant_id=None)
    second = feed.follow("r", tenant_id=None)
    try:
        await first.start()
        await ticks(40)
        await first.aclose()
        with pytest.raises(RunNotFoundError):
            await feed.follow("missing", tenant_id=None).start()
        await second.start()
        await ticks(40)
        state.write("r")
        state.live_updates.hint("r")
        item = await asyncio.wait_for(anext(second), 0.5)
        assert item.event_type == "tick"
        assert reader._follow_coordinator()._registration is not None
    finally:
        await second.aclose()
        await reader.close()
        await state.live_updates.close()


async def test_stalled_idle_stop_does_not_block_join_or_polling(monkeypatch):
    monkeypatch.setattr(_follow, "IDLE_EXIT_S", 0.02)
    live = SlowStopLive()
    reader, state = make_reader(live)
    feed = EventFeed(reader, poll_interval_s=0.01, poll_max_s=0.02)
    releasing = asyncio.Event()
    original_release = live._release_listener

    async def release(lease):
        releasing.set()
        await original_release(lease)

    monkeypatch.setattr(live, "_release_listener", release)
    first = feed.follow("r", tenant_id=None)
    second = feed.follow("r", tenant_id=None)
    try:
        await first.start()
        await ticks(40)
        live.stopping.clear()
        await first.aclose()
        await wait_for_gate(releasing)
        assert reader._retiring_lease is not None
        await asyncio.wait_for(second.start(), 0.2)
        await ticks(40)  # Let the join read finish before the event arrives.
        state.write("r")
        assert (await asyncio.wait_for(anext(second), 0.3)).event_type == "tick"
        live.stopping.set()
        async with asyncio.timeout(1):
            while reader._lease is None or not live.healthy:
                await asyncio.sleep(0.001)
        assert len(live._leases) == 1
    finally:
        live.stopping.set()
        await second.aclose()
        await reader.close()
        await live.close()


async def test_request_cancel_does_not_cancel_owned_idle_release(monkeypatch):
    monkeypatch.setattr(_follow, "IDLE_EXIT_S", 0.02)
    reader, state = make_reader()
    live = state.live_updates
    follow = reader.follow("r", tenant_id=None)
    entered, finish = asyncio.Event(), asyncio.Event()
    original = live._release_listener

    async def released_but_not_returned(lease):
        await original(lease)
        entered.set()
        await finish.wait()

    monkeypatch.setattr(live, "_release_listener", released_but_not_returned)
    await follow.start()
    await ticks(40)
    await follow.aclose()
    pending = asyncio.create_task(reader.follow("missing", tenant_id=None).start())
    try:
        await wait_for_gate(entered)
        pending.cancel()
        await asyncio.gather(pending, return_exceptions=True)
        assert reader._lease is None
        assert not live._leases
        finish.set()
        await ticks(40)
    finally:
        finish.set()
        await asyncio.gather(pending, return_exceptions=True)
        await reader.close()
        await live.close()


@pytest.mark.parametrize("ending", ["lagging", "terminal", "error"])
async def test_buffered_end_holds_tenant_capacity_until_read(ending):
    reader, state = make_reader()
    feed = EventFeed(reader, max_followers=2, max_followers_per_tenant=1, buffer=2)
    follow = feed.follow("r", tenant_id=None)
    other = feed.follow("r", tenant_id=None)
    try:
        await follow.start()
        state.write("r")
        events = await reader.get_events("r", limit=10)
        follow._follower.push(events[0])
        follow._follower.end(ending)
        with pytest.raises(FollowCapacityError):
            await other.start()
        assert (await anext(follow)).sequence_index == 0
        assert await anext(follow) == FollowEnd(ending, 0)
        await other.start()
    finally:
        await follow.aclose()
        await other.aclose()
        await reader.close()
        await state.live_updates.close()


async def test_max_duration_discards_buffer_and_releases_slot():
    reader, state = make_reader()
    follow = reader.follow("r", tenant_id=None, max_duration_s=0.02)
    try:
        await follow.start()
        state.write("r")
        follow._follower.push((await reader.get_events("r", limit=10))[0])
        follow._follower.end("lagging")
        await asyncio.sleep(0.05)
        assert not follow._follower.buffer
        assert reader._event_feed.followers == 0
        assert not reader._follow_coordinator().runs
    finally:
        await follow.aclose()
        await reader.close()
        await state.live_updates.close()


async def test_start_with_pending_terminal_event_does_not_sleep(monkeypatch):
    state = BatchedState()
    state.create("r", status="success")
    state.write("r", "run.completed")
    reader = RunStore(state)
    follow = reader.follow("r", tenant_id=None)
    sleep = AsyncMock(wraps=asyncio.sleep)
    monkeypatch.setattr(asyncio, "sleep", sleep)
    try:
        await follow.start()
        assert not follow.drained_terminal
        assert not any(
            call.args == (_follow.TERMINAL_DRAIN_GRACE_S,) for call in sleep.call_args_list
        )
    finally:
        await follow.aclose()
        await reader.close()


async def test_mixed_safety_intervals_keep_independent_windows(clock, monkeypatch):
    reader = RunStore(BatchedState())
    coordinator = reader._follow_coordinator()
    for interval in (10.0, 12.0):
        run = _follow._WatchedRun(coordinator, str(interval), None, 0)
        run.params = _follow._Params(poll_healthy_s=interval)
        coordinator.runs[run.run_id] = run
    monkeypatch.setattr(_follow.random, "uniform", lambda low, high: (low + high) / 2)
    try:
        first = coordinator._safety_at(10, clock.now)
        coordinator._safety_at(12, clock.now)
        assert coordinator._safety_at(10, clock.now + 0.1) == first
    finally:
        await reader.close()


async def test_failed_singleton_keeps_retry_deadline_despite_hints(clock):
    state = BatchedState()
    state.create("bad")
    state.create("good")
    state.fail_runs.add("bad")
    reader = RunStore(state)
    feed = EventFeed(reader)
    follows = [feed.follow(rid, tenant_id=None) for rid in ("bad", "good")]
    try:
        for follow in follows:
            await follow.start()
        coordinator = reader._follow_coordinator()
        for _ in range(12):  # Past the query share's gate, so only the deadline holds it back.
            clock.advance(0.1)
            await ticks(10)
        bad = coordinator.runs["bad"]
        assert bad.suspect
        deadline = bad.events_retry_at
        before = len([read for read in state.event_reads if "bad" in read])
        while clock.now + 0.6 < deadline:
            coordinator.on_hints(("bad",))
            clock.advance(0.6)
            await ticks(20)
        assert len([read for read in state.event_reads if "bad" in read]) == before
        clock.advance(0.7)
        await ticks(20)
        assert len([read for read in state.event_reads if "bad" in read]) == before + 1
    finally:
        for follow in follows:
            await follow.aclose()
        await reader.close()


async def test_missed_hint_diagnostic_is_one_per_generation(caplog):
    state = BatchedState()
    reader = RunStore(state)
    coordinator = reader._follow_coordinator()
    coordinator._healthy = lambda: True
    runs = []
    for i in range(10000):
        rid = str(i)
        state.create(rid)
        state.write(rid)
        run = _follow._WatchedRun(coordinator, rid, None, i)
        run.followers.add(_follow._Follower(False, 2))
        run.reads = 1
        run.reason = "safety"
        coordinator.runs[rid] = run
        runs.append(run)
    try:
        with caplog.at_level(logging.WARNING):
            for offset in range(0, len(runs), 200):
                chunk = runs[offset : offset + 200]
                await coordinator._read_events(chunk, {r: coordinator._snapshot(r) for r in chunk})
        records = [
            r for r in caplog.records if getattr(r, "code", None) == "live_updates_missed_hint"
        ]
        assert len(records) == 1
        assert records[0].count == 200
    finally:
        coordinator.runs.clear()
        await reader.close()


async def test_raw_stream_demand_does_not_spin_idle_coordinator(monkeypatch, clock):
    monkeypatch.setattr(_follow, "IDLE_EXIT_S", 0.01)
    reader, state = make_reader()
    state.write("r")
    stream = reader.stream_events("r", tenant_id=None)
    await anext(stream)
    follow = reader.follow("r", tenant_id=None)
    await follow.start()
    coordinator = reader._follow_coordinator()
    sleeps = []
    original = coordinator._sleep

    async def counted(delay):
        sleeps.append(delay)
        return await original(delay)

    monkeypatch.setattr(coordinator, "_sleep", counted)
    try:
        await follow.aclose()
        await ticks()
        clock.advance(0.07)
        await ticks()
        assert sleeps == pytest.approx([_follow.IDLE_EXIT_S] * 2)
        assert reader._lease is not None
    finally:
        await stream.aclose()
        await reader.close()
        await state.live_updates.close()


async def test_transient_retirement_failure_recovers_with_active_follow(monkeypatch):
    import dendrux.store as store_module

    monkeypatch.setattr(_follow, "IDLE_EXIT_S", 0.01)
    monkeypatch.setattr(store_module, "_LEASE_RETRY_S", 0.02)
    reader, state = make_reader()
    live = state.live_updates
    original = live._release_listener
    failed = asyncio.Event()
    calls = 0

    async def transient(lease):
        nonlocal calls
        calls += 1
        if calls == 1:
            failed.set()
            raise OSError("release temporarily unavailable")
        await original(lease)

    monkeypatch.setattr(live, "_release_listener", transient)
    first = reader.follow("r", tenant_id=None)
    second = reader.follow("r", tenant_id=None)
    try:
        await first.start()
        await ticks()
        await first.aclose()
        await wait_for_gate(failed)
        await second.start()
        async with asyncio.timeout(0.3):
            while reader._lease is None or not live.healthy:
                await asyncio.sleep(0.001)
        assert calls == 2
        assert len(live._leases) == 1
    finally:
        await second.aclose()
        await reader.close()
        await live.close()
