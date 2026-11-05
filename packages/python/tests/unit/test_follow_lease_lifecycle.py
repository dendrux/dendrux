"""A reader's listener lease against close(), cancellation and stream admission.

The gates are the shipped ``LiveUpdates._listener_lock`` and a slow listener
shutdown, so the lease implementation under test is the real one.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

import pytest

from dendrux.store import RunStore
from tests.unit.test_follow_coordinator import BatchedState, FakeLive, ticks


class SlowStopLive(FakeLive):
    """A listener whose shutdown waits on ``stopping`` (a slow socket close)."""

    def __init__(self, **kw: Any) -> None:
        super().__init__(**kw)
        self.stopping = asyncio.Event()
        self.stopping.set()

    async def receive(self) -> Any:
        try:
            yield None
            while True:
                yield await self.inbox.get()
        finally:
            await self.stopping.wait()


def _reader(live: FakeLive | None = None) -> tuple[RunStore, FakeLive]:
    state = BatchedState()
    state.create("r")
    state.live_updates = live = live or FakeLive()
    return RunStore(state), live  # type: ignore[arg-type]


async def _settle(live: FakeLive) -> None:
    async with asyncio.timeout(5):
        while live._listener_task is not None and not live._listener_task.done():
            await asyncio.sleep(0)


async def test_listener_acquired_after_close_is_released_at_once() -> None:
    reader, live = _reader()
    await live._listener_lock.acquire()  # acquisition waits here
    follow = reader.follow("r", tenant_id=None)
    reader._event_feed.stop_timeout_s = 0.05  # close waits this long for the acquisition
    start = asyncio.create_task(follow.start())
    await ticks()
    await reader.close()
    live._listener_lock.release()
    (outcome,) = await asyncio.gather(start, return_exceptions=True)
    await _settle(live)
    assert isinstance(outcome, RuntimeError)
    assert reader._lease is None and not live._leases and not live.healthy
    await follow.aclose()
    await live.close()


@pytest.mark.parametrize("where", ["before_release_lock", "during_listener_shutdown"])
async def test_cancelled_close_keeps_the_handle_and_a_retry_releases_it(where: str) -> None:
    reader, live = _reader(SlowStopLive())
    await reader._lease_live_updates()
    await ticks()
    assert live.healthy and len(live._leases) == 1
    if where == "before_release_lock":
        await live._listener_lock.acquire()
    else:
        live.stopping.clear()
    close = asyncio.create_task(reader.close())
    await ticks()
    close.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await close
    if where == "before_release_lock":
        assert reader._retiring_lease is not None  # owned retirement survives client cancellation
        assert reader._retiring_lease[1] in live._leases
        live._listener_lock.release()
    live.stopping.set()
    await reader.close()
    await _settle(live)
    assert reader._lease is None and not live._leases and not live.healthy
    await live.close()


async def test_concurrent_closes_release_once_and_both_finish() -> None:
    reader, live = _reader()
    await reader._lease_live_updates()
    await ticks()
    await asyncio.wait_for(asyncio.gather(reader.close(), reader.close()), 5)
    await _settle(live)
    assert reader._lease is None and not live._leases and not live.healthy
    await live.close()


async def test_cancelled_stream_admission_never_keeps_a_hub_consumer() -> None:
    reader, live = _reader()
    await live._listener_lock.acquire()  # admission waits in lease acquisition
    stream = reader.stream_events("r", tenant_id=None)
    first = asyncio.create_task(anext(stream))
    await ticks()
    assert live._stream_consumers == 1
    first.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await first
    live._listener_lock.release()
    await stream.aclose()
    assert live._stream_consumers == 0
    await reader.close()
    await live.close()


async def test_failed_stream_admission_never_keeps_a_hub_consumer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader, live = _reader()

    async def broken() -> object:
        raise OSError("listener setup failed")

    monkeypatch.setattr(live, "acquire_listener", broken)
    with pytest.raises(OSError):
        await anext(reader.stream_events("r", tenant_id=None))
    assert live._stream_consumers == 0
    await reader.close()
    await live.close()


@pytest.mark.parametrize("close_first", [True, False], ids=["close_returns_first", "close_waits"])
async def test_cancelled_startup_after_close_never_strands_a_lease(close_first: bool) -> None:
    """Close, a pending acquisition and a peer reader that cancels the startup."""
    reader, live = _reader()
    await live._listener_lock.acquire()  # gate the first acquisition only
    follow = reader.follow("r", tenant_id=None)
    reader._event_feed.stop_timeout_s = 0.05  # close waits this long for the acquisition
    start = asyncio.create_task(follow.start())
    await ticks()

    async def peer() -> None:
        lease = await live.acquire_listener()  # a second reader queued behind startup
        start.cancel()  # an independent cancellation while startup cleans up
        await live.release_listener(lease)

    other = asyncio.create_task(peer())
    await ticks()
    if close_first:
        await reader.close()  # returns before the gate opens (bounded wait)
        live._listener_lock.release()
    else:
        closing = asyncio.create_task(reader.close())
        await ticks()
        live._listener_lock.release()
        await asyncio.wait_for(closing, 5)
    await asyncio.gather(start, return_exceptions=True)
    await asyncio.wait_for(other, 5)
    await follow.aclose()
    await _settle(live)
    assert reader._lease is None and not live._leases and not live.healthy
    await live.close()
