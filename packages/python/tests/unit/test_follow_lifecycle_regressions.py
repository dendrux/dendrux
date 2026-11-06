"""Reader deadlines and shutdown do not depend on another client read."""

import asyncio
import logging

import pytest

from dendrux.store import EventFeed, FollowEnd, RunStore, _follow
from tests.unit.test_follow_coordinator import BatchedState, FakeLive, ticks
from tests.unit.test_follow_coordinator import clock as clock


def reader_with_run(live=False):
    state = BatchedState()
    state.create("r")
    if live:
        state.live_updates = FakeLive()
    return RunStore(state), state


async def test_never_read_follow_releases_capacity_and_watch():
    reader, _ = reader_with_run()
    follow = reader.follow("r", tenant_id=None, max_duration_s=0.02)
    try:
        await follow.start()
        await asyncio.sleep(0.05)
        assert reader._event_feed._open == 0
        assert not reader._follow_coordinator().runs
        assert await anext(follow) == FollowEnd("max_duration", None)
    finally:
        await follow.aclose()
        await reader.close()


async def test_idle_exit_drops_registration_and_lease_and_rejoins(monkeypatch):
    monkeypatch.setattr(_follow, "IDLE_EXIT_S", 0.01)
    reader, state = reader_with_run(True)
    live = state.live_updates
    follow = reader.follow("r", tenant_id=None)
    try:
        await follow.start()
        await follow.aclose()
        await asyncio.sleep(0.04)
        assert reader._lease is None
        assert not live._consumers
        coordinator = reader._follow_coordinator()
        old_task = coordinator._task
        coordinator.on_hints(("r",))
        coordinator.revive()
        assert coordinator._task is old_task
        assert not coordinator.runs
        second = reader.follow("r", tenant_id=None)
        await second.start()
        await ticks()
        assert reader._lease is not None
        assert reader._follow_coordinator()._registration is not None
        await second.aclose()
    finally:
        await reader.close()
        await live.close()


@pytest.mark.parametrize("swallow", ["raise", "return"])
async def test_coordinator_preserves_requested_cancellation(monkeypatch, swallow):
    reader, state = reader_with_run()
    entered = asyncio.Event()

    async def read(*args, **kwargs):
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            if swallow == "raise":
                raise ValueError("driver replaced cancellation") from None
            return {}

    monkeypatch.setattr(state, "get_events_for_runs", read)
    follow = reader.follow("r", tenant_id=None)
    await follow.start()
    await entered.wait()
    task = reader._follow_coordinator()._task
    task.cancel()
    await ticks(40)
    try:
        assert task.done() and task.cancelled()
    finally:
        await follow.aclose()
        await reader.close()


async def test_unexpected_cancel_is_crash_and_hint_revives():
    reader, state = reader_with_run()
    follow = reader.follow("r", tenant_id=None)
    try:
        await follow.start()
        await ticks()
        c = reader._follow_coordinator()
        old = c._task
        old.cancel()
        await asyncio.gather(old, return_exceptions=True)
        await ticks()
        assert c._crashed
        state.write("r")
        c.on_hints(["r"])
        await ticks(40)
        assert c._task is not None and c._task is not old
        assert follow._follower.buffer[0].sequence_index == 0
    finally:
        await follow.aclose()
        await reader.close()


async def test_cross_task_close_ends_pending_iteration():
    reader, _ = reader_with_run()
    follow = reader.follow("r", tenant_id=None)
    task = asyncio.create_task(anext(follow))
    await ticks(40)
    try:
        await follow.aclose()
        with pytest.raises(StopAsyncIteration):
            await asyncio.wait_for(task, 0.1)
        assert reader._event_feed._open == 0
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await reader.close()


async def test_first_unannounced_safety_discovery_logs_immediately(caplog):
    reader, state = reader_with_run()
    c = reader._follow_coordinator()
    run = _follow._WatchedRun(c, "r", None, 0)
    run.reads = 1
    run.reason = "safety"
    c.runs["r"] = run
    state.write("r")
    rows = await reader.get_events("r", limit=10)
    snapshot = _follow._Snapshot(0, 0, "safety", False, False, True, 0)
    with caplog.at_level(logging.WARNING):
        c._diagnose(run, rows, snapshot)
        c._log_missed_hints()
        c._diagnose(run, rows, snapshot)
        c._log_missed_hints()
    assert sum("live_updates_missed_hint" in r.message for r in caplog.records) == 1
    await reader.close()


async def test_second_loop_follow_polls_during_foreign_lease_acquisition():
    reader, state = reader_with_run(True)
    live = state.live_updates
    await live._listener_lock.acquire()
    acquiring = asyncio.create_task(reader._lease_live_updates())
    await ticks()

    async def second_loop():
        follow = reader.follow("r", tenant_id=None, until_terminal=False)
        try:
            await follow.start()
            c = reader._follow_coordinator()
            assert c.live is None
            await ticks()
            assert c._task is not None
        finally:
            await follow.aclose()
            await reader._follow_coordinator().aclose(1)

    try:
        await asyncio.to_thread(lambda: asyncio.run(second_loop()))
    finally:
        live._listener_lock.release()
        await acquiring
        await reader.close()
        await live.close()


async def test_idle_follow_keeps_live_lease_for_raw_stream(monkeypatch, clock):
    monkeypatch.setattr(_follow, "IDLE_EXIT_S", 0.01)
    reader, state = reader_with_run(True)
    state.write("r")
    stream = reader.stream_events("r", tenant_id=None)
    await anext(stream)
    follow = reader.follow("r", tenant_id=None)
    try:
        await follow.start()
        await follow.aclose()
        await ticks()
        clock.advance(0.04)
        await ticks()
        assert reader._lease is not None
        await stream.aclose()
        assert reader._lease is not None
        await ticks()
        clock.advance(0.04)
        await ticks()
        assert reader._lease is None
    finally:
        await reader.close()
        await state.live_updates.close()


async def test_new_admission_during_unwanted_acquisition_release_gets_lease(monkeypatch):
    monkeypatch.setattr(_follow, "IDLE_EXIT_S", 0.01)
    reader, state = reader_with_run(True)
    live = state.live_updates
    await live._listener_lock.acquire()
    first = reader.follow("r", tenant_id=None)
    starting = asyncio.create_task(first.start())
    await ticks()
    starting.cancel()
    await asyncio.gather(starting, return_exceptions=True)
    entered, proceed = asyncio.Event(), asyncio.Event()
    original = live._release_listener

    async def release(lease):
        entered.set()
        await proceed.wait()
        await original(lease)

    monkeypatch.setattr(live, "_release_listener", release)
    await asyncio.sleep(0.03)
    live._listener_lock.release()
    await asyncio.wait_for(entered.wait(), 1)
    second = reader.follow("r", tenant_id=None)
    joining = asyncio.create_task(second.start())
    await ticks()
    proceed.set()
    try:
        await asyncio.wait_for(joining, 1)
        async with asyncio.timeout(1):
            while reader._lease is None:
                await asyncio.sleep(0)
        assert reader._lease is not None
        assert live._leases
    finally:
        await first.aclose()
        await second.aclose()
        await reader.close()
        await live.close()


async def test_cross_task_close_during_timed_read_yields_nothing(monkeypatch):
    reader, state = reader_with_run()
    state.write("r")
    follow = reader.follow("r", tenant_id=None, max_duration_s=0.03)
    await follow.start()
    entered = asyncio.Event()

    async def blocked(*args, **kwargs):
        entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(reader, "get_events", blocked)
    pending = asyncio.create_task(anext(follow))
    await entered.wait()
    await follow.aclose()
    try:
        with pytest.raises(StopAsyncIteration):
            await asyncio.wait_for(pending, 0.2)
    finally:
        pending.cancel()
        await asyncio.gather(pending, return_exceptions=True)
        await reader.close()


async def test_terminal_event_without_terminal_status_keeps_follow_open():
    reader, state = reader_with_run()
    feed = EventFeed(reader, poll_interval_s=0.01)
    follow = feed.follow("r", tenant_id=None)
    try:
        await follow.start()
        state.write("r", "run.completed")
        coordinator = reader._follow_coordinator()
        coordinator.on_hints(("r",))
        assert (await anext(follow)).event_type == "run.completed"
        await asyncio.sleep(0.04)
        assert follow._follower.end_reason is None
        state.write("r", "tick")
        coordinator.on_hints(("r",))
        assert (await anext(follow)).event_type == "tick"
    finally:
        await follow.aclose()
        await reader.close()


async def test_join_during_idle_release_restores_hub(monkeypatch):
    monkeypatch.setattr(_follow, "IDLE_EXIT_S", 0.01)
    reader, state = reader_with_run(True)
    entered, released = asyncio.Event(), asyncio.Event()
    original = reader._release_idle_lease

    async def release():
        if reader._has_reader_demand():
            return await original()
        entered.set()
        await released.wait()
        await original()

    monkeypatch.setattr(reader, "_release_idle_lease", release)
    first = reader.follow("r", tenant_id=None)
    second = reader.follow("r", tenant_id=None)
    try:
        await first.start()
        await first.aclose()
        await asyncio.wait_for(entered.wait(), 1)
        await second.start()
        released.set()
        await ticks(40)
        coordinator = reader._follow_coordinator()
        assert coordinator._registration is not None
        assert reader._lease is not None
        state.write("r")
        coordinator.on_hints(("r",))
        assert (await anext(second)).sequence_index == 0
    finally:
        released.set()
        await second.aclose()
        await reader.close()
        await state.live_updates.close()


@pytest.mark.parametrize("close_during_start", [False, True])
async def test_start_error_propagates_unless_follow_closed(monkeypatch, close_during_start):
    reader, state = reader_with_run()
    entered, released = asyncio.Event(), asyncio.Event()
    follow = reader.follow("r", tenant_id=None)

    async def fail(*args, **kwargs):
        entered.set()
        await released.wait()
        raise RuntimeError("storage unavailable")

    monkeypatch.setattr(state, "get_run_status", fail)
    pending = asyncio.create_task(anext(follow))
    try:
        await entered.wait()
        if close_during_start:
            await follow.aclose()
        released.set()
        if close_during_start:
            with pytest.raises(StopAsyncIteration):
                await pending
        else:
            with pytest.raises(RuntimeError, match="storage unavailable"):
                await pending
        with pytest.raises(StopAsyncIteration):
            await anext(follow)
    finally:
        released.set()
        await follow.aclose()
        await reader.close()
