"""One scope is closed on exhaustion, failure, or explicit stream closure."""

from contextlib import asynccontextmanager

import pytest

from dendrux.loops._helpers import leased_stream
from dendrux.types import StreamEvent, StreamEventType


@pytest.mark.parametrize("end", ["close", "exhaust", "fail"])
async def test_stream_owns_exactly_one_scope_and_closes_provider(end):
    calls = []

    class Recorder:
        @asynccontextmanager
        async def lease_scope(self):
            calls.append("enter")
            try:
                yield
            finally:
                calls.append("exit")

    async def provider():
        try:
            yield StreamEvent(type=StreamEventType.TEXT_DELTA, text="x")
            yield StreamEvent(type=StreamEventType.TEXT_DELTA, text="y")
            if end == "fail":
                raise ValueError("provider failure")
        finally:
            calls.append("close")

    stream = leased_stream(provider(), Recorder())
    await anext(stream)
    await anext(stream)
    assert calls == ["enter"]
    if end == "close":
        await stream.aclose()
    else:
        with pytest.raises(ValueError if end == "fail" else StopAsyncIteration):
            await anext(stream)
    await stream.aclose()
    assert calls == ["enter", "close", "exit"]


async def test_legacy_progress_write_finishes_before_tool_cancellation_propagates():
    import asyncio

    from dendrux.runtime.persistence import PersistenceRecorder
    from dendrux.types import ToolCall

    entered, release, completed = asyncio.Event(), asyncio.Event(), asyncio.Event()

    class Store:
        async def touch_progress(self, run_id):
            entered.set()
            await release.wait()
            completed.set()

    recorder = PersistenceRecorder(Store(), "run")
    pending = asyncio.create_task(recorder.on_tool_started("run", ToolCall("work"), 1))
    await entered.wait()
    pending.cancel()
    await asyncio.sleep(0)
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert completed.is_set()
