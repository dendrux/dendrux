"""Fenced streams close their local lifecycle with the durable run outcome."""

import asyncio
import datetime as dt
from unittest.mock import AsyncMock

import pytest

from dendrux import Agent, tool
from dendrux.types import LLMResponse, RunStatus, ToolCall
from tests._helpers.scripted_provider import ScriptedLLM
from tests.integration.test_tenant_run_control import _pause


@pytest.mark.parametrize("resume", [False, True])
async def test_fenced_stream_yields_persisted_outcome(store, monkeypatch, resume):
    import dendrux.runtime.runner as runner

    finished = AsyncMock()
    failed = AsyncMock()
    monkeypatch.setattr(runner, "record_run_finished", finished)
    monkeypatch.setattr(runner, "record_run_failed", failed)

    @tool()
    async def never() -> str:
        raise AssertionError("stale owner executed a tool")

    provider = ScriptedLLM([LLMResponse(tool_calls=[ToolCall(name="never", params={})])], gate_at=0)
    agent = Agent(
        prompt="p", state_store=store, provider=provider, tools=[never], strict_tenancy=False
    )
    source = None
    if resume:
        source = await _pause("input", store, "A")
        agent.name = source.agent.name
        stream = agent.resume_stream(source.run_id, user_input="go", tenant_id="A")
    else:
        stream = agent.stream("go")
    finished.reset_mock()
    failed.reset_mock()

    async def consume():
        return [event async for event in stream]

    task = asyncio.create_task(consume())
    try:
        await asyncio.wait_for(provider.entered.wait(), 5)
        assert await store.sweep_stale_runs(dt.timedelta(0))
        provider.gate.set()
        events = await asyncio.wait_for(task, 5)
        result = events[-1].run_result
        assert result.status == RunStatus.ERROR
        assert "fenced" not in (result.error or "").lower()
        assert "fenced" not in (events[-1].error or "").lower()
        assert failed.await_count == 0
        assert finished.await_count == 1
        assert (await store.get_run(result.run_id)).error == result.error
    finally:
        provider.gate.set()
        await agent.close()
        if source:
            await source.agent.close()


@pytest.mark.parametrize("resume", [False, True])
@pytest.mark.parametrize("winner_status", ["running", "paused"])
async def test_fenced_stream_preserves_live_winner_status(
    store, monkeypatch, resume, winner_status
):
    from dendrux.runtime import runner
    from dendrux.types import RunEventType

    failed = AsyncMock()
    monkeypatch.setattr(runner, "record_run_failed", failed)

    @tool()
    async def never() -> str:
        raise AssertionError("stale owner executed a tool")

    provider = ScriptedLLM([LLMResponse(tool_calls=[ToolCall(name="never", params={})])], gate_at=0)
    agent = Agent(
        prompt="p", state_store=store, provider=provider, tools=[never], strict_tenancy=False
    )
    source = None
    if resume:
        source = await _pause("input", store, "A")
        agent.name = source.agent.name
        stream = agent.resume_stream(source.run_id, user_input="go", tenant_id="A")
    else:
        stream = agent.stream("go")

    async def consume():
        return [event async for event in stream]

    task = asyncio.create_task(consume())
    try:
        await asyncio.wait_for(provider.entered.wait(), 5)
        run_id = stream.run_id
        old = (await store.get_run_control(run_id)).claim_epoch
        assert await store.try_pause_run(
            run_id, status="paused", pause_data={"pause_id": "new"}, claim_epoch=old
        )
        new = await store.claim_paused_run(run_id, expected_status="paused")
        assert new == old + 1
        if winner_status == "paused":
            assert await store.try_pause_run(
                run_id, status="paused", pause_data={"pause_id": "next"}, claim_epoch=new
            )
        provider.gate.set()
        terminal = (await asyncio.wait_for(task, 5))[-1]
        assert terminal.run_result.status.value == winner_status
        assert terminal.type == (
            RunEventType.RUN_ERROR if winner_status == "running" else RunEventType.RUN_PAUSED
        )
        if winner_status == "running":
            assert terminal.run_result.error_info.category == "run_fenced"
            assert terminal.run_result.error_info.retryable is False
        assert failed.await_count == 0
        record = await store.get_run(run_id)
        assert record.status == winner_status and record.error is None
        assert (await store.get_run_control(run_id)).claim_epoch == new
        assert await store.get_tool_calls(run_id) == []
    finally:
        provider.gate.set()
        await agent.close()
        if source:
            await source.agent.close()
