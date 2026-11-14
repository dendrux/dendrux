"""Characterize approval faults and lazy resumed-stream ownership before PR6."""

import asyncio
from contextvars import ContextVar
from unittest.mock import AsyncMock, Mock

import pytest

from dendrux import Agent
from dendrux.llm.mock import MockLLM
from dendrux.runtime import runner
from dendrux.types import LLMResponse, RunEventType, RunStatus, StreamEvent, StreamEventType
from tests._fixtures.storage import pin_storage_format
from tests._helpers.paused_runs import _pause
from tests._helpers.runtime_paths import client_read, client_result, make_agent
from tests._helpers.scripted_provider import tool_call
from tests.unit.test_runner_move_contract import store as store


@pytest.mark.parametrize("storage_format", [1, 2])
@pytest.mark.parametrize("ending", ["complete", "second_fault", "early_close", "aclose", "cancel"])
async def test_resumed_provider_created_pulled_and_closed_in_consumer_task(
    store, storage_format, ending
):
    await pin_storage_format(store._engine, storage_format)
    marker = ContextVar("pr6_resume_marker", default=None)
    value = object()
    seen = []
    entered = asyncio.Event()

    class Provider(MockLLM):
        def complete_stream(self, *args, **kwargs):
            seen.append(("create", asyncio.current_task(), marker.get()))
            return self.events()

        async def events(self):
            token = marker.set(value)
            try:
                seen.append(("pull", asyncio.current_task(), marker.get()))
                yield StreamEvent(type=StreamEventType.TEXT_DELTA, text="partial")
                if ending == "second_fault":
                    raise RuntimeError("pr6_second_iteration_fault")
                if ending == "cancel":
                    entered.set()
                    await asyncio.Event().wait()
                yield StreamEvent(type=StreamEventType.DONE, raw=LLMResponse(text="partial"))
            finally:
                seen.append(("close", asyncio.current_task(), marker.get()))
                marker.reset(token)

    seeder = make_agent(store, [tool_call("client_read", pid="c")], tools=[client_read])
    try:
        paused = await seeder.run("go", tenant_id="tenant")
    finally:
        await seeder.close()
    async with Agent(
        name="cross-path",
        prompt="test",
        tools=[client_read],
        provider=Provider([]),
        state_store=store,
    ) as agent:
        stream = agent.resume_stream(
            paused.run_id,
            tool_results=[client_result(paused)],
            expected_pause_id=paused.pause_id,
            tenant_id="tenant",
        )
        assert seen == []
        events = []

        async def consume():
            async with stream:
                async for event in stream:
                    events.append(event)
                    if (
                        ending in {"early_close", "aclose"}
                        and event.type == RunEventType.TEXT_DELTA
                    ):
                        if ending == "aclose":
                            await stream.aclose()
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
            "complete": "success",
            "second_fault": "error",
            "early_close": "cancelled",
            "aclose": "cancelled",
            "cancel": "cancelled",
        }[ending]
        assert (await store.get_run(paused.run_id)).status == expected
        if ending == "second_fault":
            assert events[-1].type == RunEventType.RUN_ERROR
            assert events[-1].error == "pr6_second_iteration_fault"
        assert agent._task_manager._owned == {}


async def test_unstarted_resume_stream_close_is_lazy(monkeypatch):
    resolver = AsyncMock()
    provider = MockLLM([])
    complete = Mock(side_effect=AssertionError("provider called before iteration"))
    monkeypatch.setattr(provider, "complete_stream", complete)
    stream = runner.resume_stream(
        "run", agent=Agent(prompt="test"), provider=provider, state_store_resolver=resolver
    )
    await stream.aclose()
    resolver.assert_not_awaited()
    complete.assert_not_called()


@pytest.mark.parametrize("storage_format", [1, 2])
@pytest.mark.parametrize("streaming", [False, True])
async def test_approval_fault_preserves_consumption_boundary_and_retry_behavior(
    store, monkeypatch, storage_format, streaming
):
    await pin_storage_format(store._engine, storage_format)
    case = await _pause("approval", store, "tenant")
    raw = await store.get_pause_state(case.run_id)
    before = await store.get_run(case.run_id)
    fault = RuntimeError("pr6_approval_fault")
    execute = AsyncMock(side_effect=fault)
    scope = dict(tenant_id="tenant", expected_pause_id=raw["pause_id"])

    async def invoke():
        if streaming:
            async with case.agent.resume_stream(case.run_id, **scope) as stream:
                events = [event async for event in stream]
            return stream.result, events
        return await case.agent.resume(case.run_id, **scope), []

    try:
        with monkeypatch.context() as patcher:
            patcher.setattr(runner, "_execute_approved_tools", execute)
            if streaming:
                result, events = await invoke()
                assert result.status == (
                    RunStatus.WAITING_APPROVAL if storage_format == 1 else RunStatus.ERROR
                )
                assert events[-1].error == "pr6_approval_fault"
            else:
                with pytest.raises(RuntimeError) as caught:
                    await invoke()
                assert caught.value is fault
            execute.assert_awaited_once()
        after = await store.get_run(case.run_id)
        assert before.status == "waiting_approval"
        assert after.status == ("waiting_approval" if storage_format == 1 else "error")
        assert after.claim_epoch == before.claim_epoch + 1
        assert await store.get_tool_calls(case.run_id) == []
        if storage_format == 2:
            return
        result, _ = await invoke()
        assert result.status == RunStatus.SUCCESS
        calls = await store.get_tool_calls(case.run_id)
        assert len(calls) == 1 and calls[0].outcome == "completed"
        assert case.agent._task_manager._owned == {}
    finally:
        await case.agent.close()
