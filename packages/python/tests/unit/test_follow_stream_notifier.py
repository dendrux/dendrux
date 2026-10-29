"""§12.45 raw streaming keeps its cadence and lifetime, with optional wake hints."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import cast

import pytest

from dendrux.errors import RunNotFoundError
from dendrux.runtime.state import RunEventRecord, StateStore
from dendrux.store import FollowCapacityError, RunStore
from dendrux.store._notifier_pg import PostgresEventNotifier
from tests.unit.test_follow_notifier import Notifier
from tests.unit.test_follow_wake import Clock, ticks

URL = "postgresql+asyncpg://postgres:root@localhost:5432/dendrux_060_p3b_test"


class State:
    def __init__(self):
        self.rows = []
        self.reads = 0
        self.fetching = asyncio.Event()
        self.block = None
        self.fail = False

    async def get_run(self, run_id):
        return SimpleNamespace(tenant_id=None, status="success")

    async def get_run_events(self, run_id, *, after_sequence_index, limit):
        self.reads += 1
        rows = [
            r
            for r in self.rows
            if after_sequence_index is None or r.sequence_index > after_sequence_index
        ][:limit]
        block, self.block = self.block, None
        self.fetching.set()
        if block is not None:
            await block.wait()
        if self.fail:
            raise RuntimeError("test read failure")
        return rows

    def write(self):
        row = RunEventRecord(
            id=str(len(self.rows)),
            event_type="run.completed",
            sequence_index=len(self.rows),
            iteration_index=0,
            data={"raw": "preserve this payload"},
        )
        self.rows.append(row)
        return row


def setup(notifier=None):
    state = State()
    reader = RunStore(cast("StateStore", state))
    reader._notifier = notifier
    return state, reader


@pytest.fixture
async def clock(monkeypatch):
    loop = asyncio.get_running_loop()
    value = Clock(loop.time())
    monkeypatch.setattr(loop, "time", lambda: value.now)
    return value


@pytest.mark.parametrize("mode", ["none", "healthy", "degraded"])
async def test_stream_raw_payload_terminal_lifetime_and_caller_poll_interval(clock, mode):
    notifier = None if mode == "none" else Notifier()
    if notifier is not None:
        notifier.state.healthy = mode == "healthy"
    state, reader = setup(notifier)
    row = state.write()
    stream = reader.stream_events("run", tenant_id=None, poll_interval_s=2)
    pending = None
    try:
        item = await anext(stream)
        assert item.data == row.data and item.event_type == "run.completed"
        assert notifier is None or notifier.subscribers == 1
        pending = asyncio.create_task(anext(stream))
        await ticks(3)
        assert state.reads == 1 and not pending.done()
        clock.advance(1.99)
        await ticks(3)
        assert state.reads == 1
        clock.advance(0.02)
        await ticks(4)
        assert state.reads == 2 and not pending.done()
        if notifier is not None:
            assert notifier.started == notifier.stopped == 0
    finally:
        if pending is not None:
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)
        await stream.aclose()
        await reader.close()
    assert notifier is None or notifier.subscribers == 0


async def test_stream_notification_during_fetch_is_not_lost(clock):
    notifier = Notifier()
    state, reader = setup(notifier)
    release = state.block = asyncio.Event()
    stream = reader.stream_events("run", tenant_id=None, poll_interval_s=100)
    pending = asyncio.create_task(anext(stream))
    try:
        await ticks(3)
        assert state.fetching.is_set()
        row = state.write()
        notifier.event.set()
        release.set()
        await ticks(5)
        assert pending.done(), "notification during the empty fetch must trigger a fresh read"
        assert (await pending).data == row.data
        assert state.reads == 2
    finally:
        release.set()
        pending.cancel()
        await asyncio.gather(pending, return_exceptions=True)
        await stream.aclose()
        await reader.close()
    assert notifier.subscribers == 0


@pytest.mark.parametrize("exit_kind", ["aclose", "cancel", "read_error"])
async def test_stream_subscription_released_on_every_exit(clock, exit_kind):
    notifier = Notifier()
    state, reader = setup(notifier)
    state.write()
    stream = reader.stream_events("run", tenant_id=None, poll_interval_s=1)
    pending = None
    try:
        await anext(stream)
        assert notifier.subscribers == 1
        if exit_kind == "aclose":
            await stream.aclose()
        else:
            pending = asyncio.create_task(anext(stream))
            await ticks(3)
            if exit_kind == "cancel":
                pending.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await pending
            else:
                state.fail = True
                notifier.event.set()
                await ticks(4)
                assert pending.done()
                with pytest.raises(RuntimeError):
                    await pending
        assert notifier.subscribers == 0
    finally:
        if pending is not None:
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)
        await stream.aclose()
        await reader.close()


async def test_stream_capacity_refused_before_first_yield_and_released():
    notifier = PostgresEventNotifier(URL, max_subscriptions=2)
    state, reader = setup(notifier)
    state.write()
    streams = [reader.stream_events("run", tenant_id=None) for _ in range(3)]
    try:
        await anext(streams[0])
        await anext(streams[1])
        with pytest.raises(FollowCapacityError):
            await anext(streams[2])
        await streams[0].aclose()
        replacement = reader.stream_events("run", tenant_id=None)
        try:
            assert (await anext(replacement)).sequence_index == 0
        finally:
            await replacement.aclose()
    finally:
        for stream in streams:
            await stream.aclose()
        await reader.close()
    assert notifier._count == 0


async def test_stream_scope_check_precedes_subscription():
    notifier = Notifier()
    state, reader = setup(notifier)
    stream = reader.stream_events("run", tenant_id="another-tenant")
    try:
        with pytest.raises(RunNotFoundError):
            await anext(stream)
        assert notifier.subscribers == 0
    finally:
        await stream.aclose()
        await reader.close()


async def test_stream_starts_owned_convenience_notifier_once():
    notifier = Notifier()
    state, reader = setup(notifier)
    reader._owned_notifier = notifier
    state.write()
    streams = [reader.stream_events("run", tenant_id=None) for _ in range(2)]
    try:
        for stream in streams:
            await anext(stream)
        assert notifier.started == 1
    finally:
        for stream in streams:
            await stream.aclose()
        await reader.close()
    assert notifier.stopped == 1 and notifier.subscribers == 0


@pytest.mark.parametrize("channel", [None, "custom_events"])
async def test_from_database_url_forwards_store_options_without_mutating_input(channel):
    options = {"notify_channel": channel, "echo": False}
    reader = RunStore.from_database_url("sqlite+aiosqlite://", database_options=options)
    try:
        assert reader._state.notify_channel == channel
        assert options == {"notify_channel": channel, "echo": False}
    finally:
        await reader.close()
