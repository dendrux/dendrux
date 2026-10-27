"""Large client JSON validation yields to other requests; small inputs stay inline."""

import asyncio
import gc
import threading
import time
from types import SimpleNamespace

import pytest

from dendrux import Agent, tool
from dendrux.runtime.submit import prepare_client_results
from dendrux.types import ToolCall, ToolResult


@tool(target="client")
async def read_range() -> list:
    return []


@pytest.mark.parametrize("shape", ["numbers", "arrays", "objects"])
@pytest.mark.parametrize("from_wire", [False, True])
async def test_near_megabyte_validation_keeps_ticker_responsive(monkeypatch, from_wire, shape):
    import dendrux.types as types

    payload = {
        "numbers": "[" + "0," * 524280 + "0]",
        "arrays": "[" + "[]," * 349520 + "[]]",
        "objects": "[" + "{}," * 349520 + "{}]",
    }[shape]
    call = ToolCall("read_range")
    pause = SimpleNamespace(pending_tool_calls=[call])
    original = types._validate_persistable
    walks = []
    work_intervals = []

    def observed(value, field="input"):
        if not isinstance(value, list) or len(value) <= 100000:
            return original(value, field)
        walks.append(threading.get_ident())
        started = time.monotonic()
        try:
            return original(value, field)
        finally:
            work_intervals.append((started, time.monotonic()))

    monkeypatch.setattr(types, "_validate_persistable", observed)
    ready = asyncio.Event()
    stop = asyncio.Event()
    gaps = []
    gap_intervals = []
    loop = asyncio.get_running_loop()

    async def ticker():
        previous = loop.time()
        previous_cpu = time.process_time()
        ready.set()
        while not stop.is_set():
            await asyncio.sleep(0.001)
            current = loop.time()
            current_cpu = time.process_time()
            gap = current - previous
            gaps.append(gap)
            gap_intervals.append((previous, current, current_cpu - previous_cpu))
            previous, previous_cpu = current, current_cpu

    agent = Agent(prompt="p", tools=[read_range])
    await agent.get_tool_lookups()
    # Measure validation, excluding SDK/executor cold-start initialization.
    await asyncio.to_thread(lambda: None)
    gc.collect()  # exclude garbage left by unrelated earlier test fixtures
    task = asyncio.create_task(ticker())
    await ready.wait()
    try:
        data = dict(name=call.name, call_id=call.id, payload=payload)
        result = (
            await asyncio.to_thread(ToolResult.from_dict, data) if from_wire else ToolResult(**data)
        )
        prepared = await prepare_client_results(agent, "r", pause, [result])
        assert prepared[0].payload == payload
    finally:
        stop.set()
        await task
        await agent.close()
    # Attribute ticker gaps to the structured walk, the work this boundary
    # offloads. Record whole-conversion latency too; decoding/allocation and
    # teardown can trigger unrelated process-wide GC outside this walk.
    stalls = [
        min(
            sum(
                max(0, min(end, work_end) - max(start, work_start))
                for work_start, work_end in work_intervals
            ),
            cpu + 0.001,
        )
        for start, end, cpu in gap_intervals
    ]
    print(
        f"shape={shape} wire={from_wire} max_ticker_ms={max(gaps) * 1000:.3f} "
        f"walk_busy_ms={max(stalls) * 1000:.3f}"
    )
    assert max(stalls) < 0.025, (
        f"structured validation stalled ticker for {max(stalls) * 1000:.1f} ms"
    )
    assert len(walks) == 1, "from_dict and submission must not repeat the structured walk"
    assert all(identity != threading.get_ident() for identity in walks)


async def test_small_validation_stays_inline(monkeypatch):
    import dendrux.runtime.submit as submit

    original = submit._validated_client_results
    threads = []

    def observed(*args):
        threads.append(threading.get_ident())
        return original(*args)

    monkeypatch.setattr(submit, "_validated_client_results", observed)
    call = ToolCall("read_range")
    pause = SimpleNamespace(pending_tool_calls=[call])
    async with Agent(prompt="p", tools=[read_range]) as agent:
        await prepare_client_results(agent, "r", pause, [ToolResult(call.name, call.id, "[1,2]")])
    assert threads == [threading.get_ident()]


@pytest.mark.parametrize("change", ["payload", "metadata", "limit"])
async def test_payload_attestation_does_not_bypass_new_data_or_limits(change):
    from dataclasses import replace

    from dendrux.errors import InvalidToolResultError
    from dendrux.types import Limits

    call = ToolCall("read_range")
    result = ToolResult.from_dict(dict(name=call.name, call_id=call.id, payload='"safe text"'))
    if change == "payload":
        result = replace(result, payload='"bad\\u0000"')
    elif change == "metadata":
        result = replace(result, error="bad\x00")
    pause = SimpleNamespace(pending_tool_calls=[call])
    limits = Limits(max_tool_result_bytes=4) if change == "limit" else Limits()
    async with Agent(prompt="p", tools=[read_range], limits=limits) as agent:
        if change == "limit":
            prepared = await prepare_client_results(agent, "r", pause, [result])
            assert prepared[0].success is False
            assert prepared[0].meta["code"] == "size_limit"
        else:
            with pytest.raises(InvalidToolResultError):
                await prepare_client_results(agent, "r", pause, [result])


def test_client_cannot_supply_a_payload_attestation():
    with pytest.raises(ValueError):
        ToolResult.from_dict(
            dict(
                name="read_range",
                call_id="c",
                payload='"bad\\u0000"',
                _validated_payload='"bad\\u0000"',
            )
        )


async def test_concurrent_large_submissions_apply_backpressure_before_executor(monkeypatch):
    import dendrux.runtime.submit as submit

    original = submit._validated_client_results
    lock = threading.Lock()
    active = peak = 0

    def observed(*args):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        try:
            return original(*args)
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(submit, "_validated_client_results", observed)
    call = ToolCall("read_range")
    pause = SimpleNamespace(pending_tool_calls=[call])
    result = ToolResult(call.name, call.id, "[" + "0," * 262140 + "0]")
    async with Agent(prompt="p", tools=[read_range]) as agent:
        await asyncio.gather(
            *(prepare_client_results(agent, "r", pause, [result]) for _ in range(4))
        )
    assert peak == 1, "large requests must wait before occupying the default executor"


@pytest.mark.parametrize("worker_fails", [False, True])
async def test_cancelled_validation_waits_for_cpu_job_before_releasing_slot(
    monkeypatch, worker_fails
):
    import dendrux.runtime.submit as submit

    entered = asyncio.Event()
    release = threading.Event()
    loop = asyncio.get_running_loop()

    def work(*args):
        loop.call_soon_threadsafe(entered.set)
        # Bounded: if validation ran on the event loop, this blocks the loop, so
        # the test must fail after the timeout instead of hanging.
        if not release.wait(timeout=5):
            raise AssertionError("validation ran on the event loop, not in a worker")
        if worker_fails:
            raise ValueError("invalid payload")
        return []

    monkeypatch.setattr(submit, "_validated_client_results", work)
    result = ToolResult("read_range", "c", "x" * 65536)
    async with Agent(prompt="p", tools=[read_range]) as agent:
        caller = asyncio.create_task(submit._validate_client_results(agent, "r", [result]))
        try:
            await asyncio.wait_for(entered.wait(), timeout=10)
            caller.cancel()
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            assert not caller.done(), "the admission slot must cover the actual CPU job"
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await caller
        finally:
            release.set()
            await asyncio.gather(caller, return_exceptions=True)
