"""Cancellation preserves awaitable compatibility and diagnostic identity."""

import asyncio
import inspect
from functools import wraps

import pytest

from dendrux.loops.react import ReActLoop
from dendrux.runtime import _run_events, runner
from dendrux.runtime.runner import _create_runtime_run
from dendrux.types import ToolResult, generate_ulid
from tests._helpers.gates import wait_for_gate
from tests.integration.test_060_ownership_cancellation import invocation


async def test_task_returning_store_preserves_cancel(store, engine, monkeypatch):
    original = store.create_run
    entered = asyncio.Event()
    children = []

    async def slow(*args, **kwargs):
        entered.set()
        await asyncio.Event().wait()
        return await original(*args, **kwargs)

    @wraps(original)
    def spawned(*args, **kwargs):
        child = asyncio.create_task(slow(*args, **kwargs))
        children.append(child)
        return child

    monkeypatch.setattr(store, "create_run", spawned)
    agent, call, _, _, calls = await invocation("run", store, engine)
    task = asyncio.create_task(call)
    try:
        await wait_for_gate(entered, task=task)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 5)
        assert calls() == 0
        assert children and all(child.cancelled() for child in children)
    finally:
        await asyncio.gather(task, *children, return_exceptions=True)
        await agent.close()


@pytest.mark.parametrize(
    "entry", ["run", "stream", "submit_tool_results", "submit_resume", "reclaim"]
)
async def test_precommit_unwind_failure_has_identity(store, engine, monkeypatch, caplog, entry):
    original = store.create_run
    created_ids = []

    @wraps(original)
    async def create(run_id, *args, **kwargs):
        created_ids.append(run_id)
        return await original(run_id, *args, **kwargs)

    monkeypatch.setattr(store, "create_run", create)
    agent, call, run_id, _, _ = await invocation(entry, store, engine)
    entered = asyncio.Event()

    async def fail_unwind(*args, **kwargs):
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            raise ValueError("private_unwind_sentinel") from None

    monkeypatch.setattr(store, "_write_tx", fail_unwind)
    task = asyncio.create_task(call)
    try:
        await wait_for_gate(entered, task=task)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 5)
        expected = run_id or created_ids[-1]
        records = [r for r in caplog.records if r.name == "dendrux.runtime.durability"]
        assert [(r.code, getattr(r, "run_id", None)) for r in records] == [
            ("persistence_error", expected)
        ]
        assert "private_unwind_sentinel" not in caplog.text
    finally:
        await asyncio.gather(task, return_exceptions=True)
        await agent.close()


async def test_cancel_drain_passes_known_run_id(monkeypatch, caplog):
    run_id = generate_ulid()
    entered, release = asyncio.Event(), asyncio.Event()

    async def failing(*args, **kwargs):
        entered.set()
        await release.wait()
        raise ValueError("private_cleanup_sentinel")

    monkeypatch.setattr(_run_events, "_persist_task_cancellation", failing)
    task = asyncio.create_task(runner._drain_task_cancellation(None, run_id, False))
    await wait_for_gate(entered, task=task)
    task.cancel()
    await asyncio.sleep(0)
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 5)
    assert [(r.code, getattr(r, "run_id", None)) for r in caplog.records] == [
        ("persistence_error", run_id)
    ]
    assert "private_cleanup_sentinel" not in caplog.text


@pytest.mark.parametrize("entry", ["stream", "resume_stream"])
async def test_stream_cleanup_supplies_run_id(store, engine, monkeypatch, entry):
    agent, call, run_id, _, _ = await invocation(entry, store, engine)
    call.close()
    if entry == "stream":
        stream = agent.stream("go", tenant_id="tenant")
    else:
        raw = await store.get_pause_state(run_id)
        stream = agent.resume_stream(
            run_id,
            tenant_id="tenant",
            expected_pause_id=raw["pause_id"],
            tool_results=[
                ToolResult(c["name"], c["id"], '"ok"') for c in raw["pending_tool_calls"]
            ],
        )
    original = runner._shielded_cleanup
    observed = []

    async def capture(cleanup, **kwargs):
        observed.append(kwargs.get("run_id"))
        return await original(cleanup, **kwargs)

    monkeypatch.setattr(_run_events, "_shielded_cleanup", capture)
    try:
        await asyncio.wait_for(stream.__aiter__().__anext__(), 5)
        await asyncio.wait_for(stream.aclose(), 5)
        assert observed == [stream.run_id]
    finally:
        await agent.close()


async def test_unstarted_handoff_operation_is_closed(store, monkeypatch):
    original_create = store.create_run
    original_task = asyncio.create_task
    operations = []

    @wraps(original_create)
    def capture(*args, **kwargs):
        operation = original_create(*args, **kwargs)
        operations.append(operation)
        return operation

    def cancel_before_first_step(coro, **kwargs):
        child = original_task(coro, **kwargs)
        if coro.__qualname__ == "_without_live_flush":
            child.cancel()
        return child

    monkeypatch.setattr(store, "create_run", capture)
    monkeypatch.setattr(asyncio, "create_task", cancel_before_first_step)
    try:
        with pytest.raises(asyncio.CancelledError):
            await _create_runtime_run(store, ReActLoop(), generate_ulid(), "never-started")
        assert len(operations) == 1
        assert inspect.getcoroutinestate(operations[0]) == inspect.CORO_CLOSED
    finally:
        for operation in operations:
            operation.close()


async def test_handoff_second_cancel_failure_carries_run_id(store, monkeypatch, caplog):
    from dendrux.runtime import durability

    run_id = generate_ulid()
    parked, cleanup_flush, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    calls = 0

    async def flush(_store):
        nonlocal calls
        calls += 1
        if calls == 1:
            parked.set()
            await asyncio.Event().wait()
        cleanup_flush.set()
        await release.wait()
        raise ValueError("private_flush_sentinel")

    monkeypatch.setattr(durability, "_flush_live_updates", flush)
    task = asyncio.create_task(_create_runtime_run(store, ReActLoop(), run_id, "second-cancel"))
    try:
        await wait_for_gate(parked, task=task)
        task.cancel()
        await wait_for_gate(cleanup_flush, task=task)
        task.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 5)
        records = [r for r in caplog.records if r.name == "dendrux.runtime.tasks"]
        assert [(r.code, getattr(r, "run_id", None)) for r in records] == [
            ("persistence_error", run_id)
        ]
        assert "private_flush_sentinel" not in caplog.text
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
