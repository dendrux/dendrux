"""Hints and gaps coalesce within each raw stream's minimum read interval."""

import asyncio

import pytest

from dendrux.store import RunStore, _follow
from tests.unit.test_follow_coordinator import BatchedState


@pytest.mark.parametrize("signal", ["hint", "gap"])
async def test_stream_wakeups_have_minimum_read_interval(signal):
    state = BatchedState()
    state.create("r")
    reader = RunStore(state)
    waker = _follow._RunWaker("r")
    stream = reader._poll_events("r", None, waker, 0.2)
    task = asyncio.create_task(anext(stream))
    loop = asyncio.get_running_loop()
    started = loop.time()
    try:
        while loop.time() - started < 0.12:
            if signal == "hint":
                waker.on_hints(["r"])
            else:
                waker.on_gap()
            await asyncio.sleep(0.001)
        assert len(state.event_reads) <= 7
        state.write("r")
        waker.on_hints(["r"])
        item = await asyncio.wait_for(task, 0.08)
        assert item.sequence_index == 0
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await stream.aclose()
        await reader.close()
