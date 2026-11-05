"""Focused interleavings for previously surviving coordinator guard reverts."""

import asyncio
import logging

import pytest

from dendrux.store import EventFeed, RunStore, _follow
from tests.unit.test_follow_coordinator import BatchedState, ticks
from tests.unit.test_follow_coordinator import clock as clock


def watched(reader, state, rid="r"):
    state.create(rid)
    coordinator = reader._follow_coordinator()
    run = _follow._WatchedRun(coordinator, rid, None, len(coordinator.runs))
    run.followers.add(_follow._Follower(True, 100))
    coordinator.runs[rid] = run
    return coordinator, run


@pytest.mark.parametrize("interval,maximum", [(0.01, 0.02), (1.0, 2.0)])
async def test_first_fallback_poll_honors_configured_interval(clock, interval, maximum):
    state = BatchedState()
    reader = RunStore(state)
    coordinator, run = watched(reader, state)
    run.params = _follow._Params(poll_interval_s=interval, poll_max_s=maximum)
    try:
        coordinator._schedule(run, False, clock.now)
        assert clock.now + interval <= run.events_due <= clock.now + maximum
    finally:
        await reader.close()


async def test_hint_cannot_pull_terminal_drain_before_grace(clock):
    state = BatchedState()
    reader = RunStore(state)
    coordinator, run = watched(reader, state)
    run.status_terminal = True
    run.terminal_drain_at = clock.now + _follow.TERMINAL_DRAIN_GRACE_S
    run.events_due = run.terminal_drain_at
    try:
        coordinator.on_hints((run.run_id,))
        assert run.events_due == run.terminal_drain_at
        await ticks(30)
        assert not state.event_reads
        clock.advance(_follow.TERMINAL_DRAIN_GRACE_S)
        await ticks(30)
        assert len(state.event_reads) == 1
    finally:
        await reader.close()


@pytest.mark.parametrize("changed", ["generation", "hint"])
async def test_inflight_hint_or_generation_prevents_missed_diagnostic(changed, caplog):
    state = BatchedState()
    reader = RunStore(state)
    coordinator, run = watched(reader, state)
    run.reason, run.reads = "safety", 1
    generation = [0]
    coordinator._generation = lambda: generation[0]
    coordinator._healthy = lambda: True
    state.write("r")
    original = state.get_events_for_runs

    async def read(*args, **kwargs):
        rows = await original(*args, **kwargs)
        if changed == "generation":
            generation[0] += 1
        else:
            run.hints += 1
        return rows

    state.get_events_for_runs = read
    try:
        with caplog.at_level(logging.WARNING):
            await coordinator._read_events([run], {run: coordinator._snapshot(run)})
        assert not [
            r for r in caplog.records if getattr(r, "code", None) == "live_updates_missed_hint"
        ]
        state.get_events_for_runs = original
        for _ in range(2):
            generation[0] += 1
            state.write("r")
            run.reason = "safety"
            with caplog.at_level(logging.WARNING):
                await coordinator._read_events([run], {run: coordinator._snapshot(run)})
        records = [
            r for r in caplog.records if getattr(r, "code", None) == "live_updates_missed_hint"
        ]
        assert len(records) == 2 and all(r.count == 1 for r in records)
    finally:
        await reader.close()


async def test_drained_start_releases_capacity_before_iteration():
    state = BatchedState()
    state.create("done", status="success")
    reader = RunStore(state)
    feed = EventFeed(reader, max_followers=1)
    first = feed.follow("done", tenant_id=None)
    second = feed.follow("done", tenant_id=None)
    try:
        await first.start()
        assert first.drained_terminal and feed.followers == 0
        await second.start()
        assert second.drained_terminal and feed.followers == 0
    finally:
        await first.aclose()
        await second.aclose()
        await reader.close()


async def test_sleep_preserves_cancellation_even_when_wait_swallows_it(monkeypatch):
    reader = RunStore(BatchedState())
    coordinator = reader._follow_coordinator()
    entered = asyncio.Event()

    async def swallow():
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            return True

    monkeypatch.setattr(coordinator._wake, "wait", swallow)
    sleeping = asyncio.create_task(coordinator._sleep(100))
    try:
        await entered.wait()
        sleeping.cancel()
        with pytest.raises(asyncio.CancelledError):
            await sleeping
    finally:
        await reader.close()


@pytest.mark.parametrize("replacement", [False, True])
async def test_read_boundary_preserves_cancellation_without_outer_loop(replacement):
    state = BatchedState()
    reader = RunStore(state)
    coordinator, run = watched(reader, state)
    entered = asyncio.Event()

    async def swallow(runs):
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            if replacement:
                raise ValueError("driver replaced cancellation") from None
            return {}

    reading = asyncio.create_task(coordinator._isolated_batch([run], swallow))
    try:
        await entered.wait()
        reading.cancel()
        with pytest.raises(asyncio.CancelledError):
            await reading
    finally:
        await reader.close()


async def test_closed_main_exits_without_needing_task_cancellation(monkeypatch):
    state = BatchedState()
    reader = RunStore(state)
    coordinator, _ = watched(reader, state)
    calls = 0

    async def close_during_sleep(seconds):
        nonlocal calls
        calls += 1
        assert calls == 1, "closed coordinator entered another sleep"
        coordinator._closed = True
        return False

    monkeypatch.setattr(coordinator, "_sleep", close_during_sleep)
    try:
        await coordinator._main()
        assert calls == 1
    finally:
        await reader.close()


async def test_initial_isolation_yields_to_new_healthy_hint():
    state = BatchedState()
    reader = RunStore(state)
    coordinator = reader._follow_coordinator()
    for i in range(100):
        _, run = watched(reader, state, str(i))
        run.events_due = run.status_due = coordinator._now()
        state.fail_runs.add(str(i))
    try:
        await coordinator._cycle(list(coordinator.runs.values()), coordinator._now())
        _, healthy = watched(reader, state, "healthy")
        state.write("healthy")
        coordinator.on_hints(("healthy",))
        await ticks(150)
        assert healthy.head == 0
        assert state.event_reads.index(("healthy",)) < 5
        assert coordinator._isolations
    finally:
        await reader.close()


async def test_suspect_reads_are_singletons_with_reserved_query_share(clock):
    state = BatchedState()
    reader = RunStore(state)
    coordinator = reader._follow_coordinator()
    coordinator._healthy = lambda: True
    for i in range(100):
        _, run = watched(reader, state, str(i))
        run.suspect = True
        run.events_due = run.status_due = clock.now
        state.fail_runs.add(str(i))
    start = clock.now
    try:
        for _ in range(20):
            chunk = coordinator._due_chunk(clock.now)
            if chunk:
                await coordinator._cycle(chunk, clock.now)
            reads = state.event_reads + state.status_reads
            assert all(len(group) == 1 for group in reads)
            assert len(reads) <= 1 + _follow.SUSPECT_READS_PER_S * (clock.now - start)
            clock.advance(0.1)
    finally:
        await reader.close()
