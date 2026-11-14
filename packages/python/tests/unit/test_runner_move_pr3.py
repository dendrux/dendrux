"""Characterize fresh lifecycle and stream ownership before the PR3 move."""

import asyncio
from contextvars import ContextVar
from unittest.mock import AsyncMock, Mock

import pytest

from dendrux.loops import _helpers
from dendrux.runtime import runner, state
from dendrux.runtime._managed import DESCEND_KEY as MANAGED_DESCEND_KEY
from dendrux.types import RunEvent, RunEventType, RunResult, RunStatus, generate_ulid
from tests.unit.test_runner_move_contract import store as store


@pytest.mark.parametrize("exit_kind", ["complete", "failure", "early_close", "cancel"])
@pytest.mark.parametrize("close_fails", [False, True])
async def test_stream_close_cancel_matrix_keeps_task_context_and_single_close(
    monkeypatch, exit_kind, close_fails
):
    marker = ContextVar("pr3_stream_marker")
    value = object()
    seen = []
    entered = asyncio.Event()
    release = asyncio.Event()
    failure = RuntimeError("loop iteration failed")
    run_id = generate_ulid()
    record = AsyncMock()
    notify = AsyncMock()
    monkeypatch.setattr(_helpers, "record_run_finished", record)
    monkeypatch.setattr(_helpers, "notify_run_finished", notify)

    async def source():
        token = marker.set(value)
        try:
            seen.append(("iterate", asyncio.current_task(), marker.get()))
            yield RunEvent(type=RunEventType.TEXT_DELTA, text="partial")
            if exit_kind == "failure":
                raise failure
            if exit_kind == "cancel":
                entered.set()
                await release.wait()
            yield RunEvent(
                type=RunEventType.RUN_COMPLETED,
                run_result=RunResult(run_id=run_id, status=RunStatus.SUCCESS, answer="done"),
            )
        finally:
            seen.append(("close", asyncio.current_task(), marker.get()))
            marker.reset(token)
            if close_fails:
                raise RuntimeError("loop close failed")

    class Stream:
        def __init__(self):
            self.inner = source()
            self.aclose = AsyncMock(side_effect=self.close)

        async def close(self):
            await self.inner.aclose()

        def __aiter__(self):
            return self.inner

    stream = Stream()
    shared = {}

    async def consume():
        driver = runner._stream_loop_events(
            stream,
            state_store=None,
            recorder=None,
            notifier=None,
            run_id=run_id,
            provider_kwargs=None,
            shared=shared,
            mcp_meta={},
            new_run=True,
        )
        try:
            if exit_kind == "early_close":
                await anext(driver)
            else:
                return [event async for event in driver]
        finally:
            await driver.aclose()

    task = asyncio.create_task(consume())
    if exit_kind == "cancel":
        await entered.wait()
        task.cancel()
    if exit_kind == "cancel":
        # A failing generator finally replaces cancellation on the current base.
        with pytest.raises(RuntimeError if close_fails else asyncio.CancelledError):
            await task
    elif exit_kind == "failure" or close_fails and exit_kind == "complete":
        with pytest.raises(RuntimeError) as caught:
            await task
        if not close_fails:
            assert caught.value is failure
    else:
        events = await task
        if exit_kind == "complete":
            assert events[-1].run_result.status == RunStatus.SUCCESS
    assert seen == [("iterate", task, value), ("close", task, value)]
    stream.aclose.assert_awaited_once()
    assert shared["partial_text"] == ["partial"]
    expected = int(exit_kind == "complete")
    assert record.await_count == notify.await_count == expected


@pytest.mark.parametrize(
    "status,event_type",
    [
        (RunStatus.SUCCESS, RunEventType.RUN_COMPLETED),
        (RunStatus.ERROR, RunEventType.RUN_ERROR),
        (RunStatus.CANCELLED, RunEventType.RUN_CANCELLED),
        (RunStatus.WAITING_CLIENT_TOOL, RunEventType.RUN_PAUSED),
    ],
)
async def test_terminal_stream_projection_finishes_sinks_in_order(monkeypatch, status, event_type):
    calls = []

    async def record(*args):
        calls.append("record")

    async def notify(*args):
        calls.append("notify")

    monkeypatch.setattr(_helpers, "record_run_finished", AsyncMock(side_effect=record))
    monkeypatch.setattr(_helpers, "notify_run_finished", AsyncMock(side_effect=notify))
    result = RunResult(run_id=generate_ulid(), status=status)
    event = RunEvent(type=event_type, run_result=result)
    projected = await runner._finish_stream_event(
        event,
        result,
        state_store=None,
        recorder=None,
        notifier=None,
        run_id=result.run_id,
        provider_kwargs=None,
    )
    assert projected is event
    assert calls == ["record", "notify"]


async def test_managed_handoff_keeps_lifecycle_sinks_open(monkeypatch):
    record = AsyncMock()
    notify = AsyncMock()
    monkeypatch.setattr(_helpers, "record_run_finished", record)
    monkeypatch.setattr(_helpers, "notify_run_finished", notify)
    lifecycle = runner._RunLifecycle(generate_ulid(), None, None)
    result = RunResult(
        run_id=lifecycle.run_id, status=RunStatus.SUCCESS, meta={MANAGED_DESCEND_KEY: {}}
    )
    await lifecycle._finish(None, None, result)
    await runner._record_finished(None, None, lifecycle.run_id, result)
    assert not lifecycle.recorder_closed and not lifecycle.notifier_closed
    record.assert_not_awaited()
    notify.assert_not_awaited()


async def test_failure_unwind_attempts_each_sink_once(monkeypatch):
    record = AsyncMock()
    notify = AsyncMock()
    monkeypatch.setattr(_helpers, "record_run_failed", record)
    monkeypatch.setattr(_helpers, "notify_run_failed", notify)
    lifecycle = runner._RunLifecycle(generate_ulid(), None, None)
    await lifecycle._fail_sinks(RuntimeError("failure"), None, None)
    await lifecycle._fail_sinks(RuntimeError("failure again"), None, None)
    record.assert_awaited_once()
    notify.assert_awaited_once()
    assert lifecycle.recorder_closed and lifecycle.notifier_closed


@pytest.mark.parametrize("path", ["run", "run_stream", "resume", "resume_stream"])
async def test_claimed_store_fault_reaches_fresh_and_resumed_consumers(store, monkeypatch, path):
    from dendrux.types import LLMResponse
    from tests._helpers.runtime_paths import path_case
    from tests._helpers.scripted_provider import ScriptedLLM

    case = await path_case(store, None, path, ScriptedLLM([LLMResponse(text="done")]))
    failure = RuntimeError("claimed store fault")
    fault = Mock(side_effect=failure)
    monkeypatch.setattr(state, "claimed_store", fault)
    try:
        if path in {"run", "resume"}:
            with pytest.raises(RuntimeError) as caught:
                await case.invoke()
            assert caught.value is failure
        else:
            result = await case.invoke()
            assert result.status == RunStatus.ERROR
            assert case.events[-1].error == "claimed store fault"
        fault.assert_called_once()
        assert case.agent._task_manager._owned == {}
    finally:
        await case.agent.close()
