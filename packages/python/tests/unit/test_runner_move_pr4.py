"""Characterize fresh-stream ownership and retry seams before the PR4 move."""

import asyncio
from contextvars import ContextVar
from unittest.mock import AsyncMock, Mock

import pytest

from dendrux import Agent
from dendrux.llm.mock import MockLLM
from dendrux.loops import ReActLoop, SingleCall
from dendrux.runtime import runner
from dendrux.types import LLMResponse, RunEventType, RunStatus, StreamEvent, StreamEventType
from tests._helpers.runtime_paths import path_case
from tests._helpers.scripted_provider import ScriptedLLM
from tests.unit.test_runner_move_contract import store as store


@pytest.mark.parametrize("loop_type", [ReActLoop, SingleCall])
@pytest.mark.parametrize("ending", ["complete", "failure", "early_close", "cancel"])
async def test_public_fresh_stream_creates_pulls_and_closes_in_consumer_task(
    store, loop_type, ending
):
    marker = ContextVar("pr4_stream_marker", default=None)
    value = object()
    seen = []
    entered = asyncio.Event()
    release = asyncio.Event()

    class Provider(MockLLM):
        def complete_stream(self, *args, **kwargs):
            seen.append(("create", asyncio.current_task(), marker.get()))
            return self.events()

        async def events(self):
            token = marker.set(value)
            try:
                seen.append(("pull", asyncio.current_task(), marker.get()))
                yield StreamEvent(type=StreamEventType.TEXT_DELTA, text="partial")
                if ending == "failure":
                    raise RuntimeError("pr4_stream_failure")
                if ending == "cancel":
                    entered.set()
                    await release.wait()
                yield StreamEvent(type=StreamEventType.DONE, raw=LLMResponse(text="partial"))
            finally:
                seen.append(("close", asyncio.current_task(), marker.get()))
                marker.reset(token)

    provider = Provider([])
    async with Agent(
        prompt="test", provider=provider, loop=loop_type(), state_store=store
    ) as agent:
        stream = agent.stream("go", tenant_id="tenant")
        assert seen == []
        events = []

        async def consume():
            async with stream:
                async for event in stream:
                    events.append(event)
                    if ending == "early_close" and event.type == RunEventType.TEXT_DELTA:
                        break
            assert marker.get() is None

        task = asyncio.create_task(consume())
        if ending == "cancel":
            await asyncio.wait_for(entered.wait(), timeout=5)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            await task
        assert seen == [("create", task, None), ("pull", task, value), ("close", task, value)]
        expected = {
            "complete": RunStatus.SUCCESS,
            "failure": RunStatus.ERROR,
            "early_close": RunStatus.CANCELLED,
            "cancel": RunStatus.CANCELLED,
        }[ending]
        row = await store.get_run(stream.run_id)
        assert row.status == expected.value
        if ending != "failure":
            assert row.answer == "partial"
        if ending == "failure":
            assert events[-1].type == RunEventType.RUN_ERROR
        assert agent._task_manager._owned == {}


async def test_unstarted_stream_close_has_no_store_or_provider_effects(monkeypatch):
    resolver = AsyncMock()
    provider = MockLLM([])
    complete = Mock(side_effect=AssertionError("provider called before iteration"))
    monkeypatch.setattr(provider, "complete_stream", complete)
    stream = runner.run_stream(
        Agent(prompt="test"),
        provider=provider,
        user_input="go",
        tenant_id=None,
        state_store_resolver=resolver,
    )
    await stream.aclose()
    resolver.assert_not_awaited()
    complete.assert_not_called()


@pytest.mark.parametrize("path", ["run", "run_stream", "retry", "resume", "resume_stream"])
async def test_default_strategy_fault_fires_across_fresh_retry_and_resume(store, monkeypatch, path):
    case = await path_case(store, None, path, ScriptedLLM([LLMResponse(text="done")]))
    fault = RuntimeError("pr4_default_strategy_fault")
    constructor = Mock(side_effect=fault)
    monkeypatch.setattr(runner, "NativeToolCalling", constructor)
    try:
        if path == "resume_stream":
            result = await case.invoke()
            assert result.status == RunStatus.WAITING_CLIENT_TOOL
            assert case.events[-1].type == RunEventType.RUN_ERROR
            assert case.events[-1].error == "pr4_default_strategy_fault"
        else:
            with pytest.raises(RuntimeError) as caught:
                await case.invoke()
            assert caught.value is fault
        constructor.assert_called_once_with()
    finally:
        await case.agent.close()


def test_default_strategy_fault_also_reaches_approval_dispatch(monkeypatch):
    fault = RuntimeError("pr4_approval_strategy_fault")
    constructor = Mock(side_effect=fault)
    monkeypatch.setattr(runner, "NativeToolCalling", constructor)
    with pytest.raises(RuntimeError) as caught:
        runner._approval_dispatch_inputs(None, None)
    assert caught.value is fault
    constructor.assert_called_once_with()


async def test_retry_keeps_source_row_traces_events_and_lineage(store):
    async with Agent(
        prompt="test", provider=MockLLM([LLMResponse(text="original")]), state_store=store
    ) as first:
        original = await first.run("question", tenant_id="tenant")
    source = await store.get_run(original.run_id)
    traces = await store.get_traces(original.run_id)
    events = await store.get_run_events(original.run_id)
    async with Agent(
        prompt="test", provider=MockLLM([LLMResponse(text="retried")]), state_store=store
    ) as second:
        retried = await second.retry(original.run_id, tenant_id="tenant")
    assert retried.status == RunStatus.SUCCESS
    assert retried.answer == "retried"
    assert retried.run_id != original.run_id
    assert (await store.get_run(retried.run_id)).retry_of_run_id == original.run_id
    assert await store.get_run(original.run_id) == source
    assert await store.get_traces(original.run_id) == traces
    assert await store.get_run_events(original.run_id) == events
