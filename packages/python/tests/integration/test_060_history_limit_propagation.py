"""Configured recovery bounds apply to replay and genuine conversation edits."""

from dataclasses import replace

import pytest

from dendrux import Agent, tool
from dendrux.errors import RunNotReclaimableError
from dendrux.llm.mock import MockLLM
from dendrux.strategies.native import NativeToolCalling
from dendrux.types import Limits, LLMResponse, RunStatus, ToolCall, ToolResult


class EditingStrategy(NativeToolCalling):
    def __init__(self):
        self.revision = 0

    def build_messages(self, **kwargs):
        messages, tools = super().build_messages(**kwargs)
        self.revision += 1
        messages[0] = replace(messages[0], content=f"revision {self.revision}")
        return messages, tools


def install_editing_strategy(monkeypatch):
    from dendrux.runtime import runner

    strategy = EditingStrategy()
    monkeypatch.setattr(runner, "NativeToolCalling", lambda: strategy)


async def finish(call, streaming):
    if not streaming:
        return await call
    async with call as stream:
        events = [event async for event in stream]
    return events[-1].run_result


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("kind", ["replay", "edit"])
@pytest.mark.parametrize("bound", ["max_rebuild_rows", "max_rebuild_bytes"])
async def test_history_bounds_refuse_before_provider(store, monkeypatch, streaming, kind, bound):
    source = None
    if kind == "replay":
        async with Agent(
            prompt="p", provider=MockLLM([LLMResponse(text="source")]), state_store=store
        ) as first:
            source = await first.run("source", tenant_id="tenant")
    if kind == "edit":
        install_editing_strategy(monkeypatch)
    provider = MockLLM([LLMResponse(text="should not be called")])
    async with Agent(
        prompt="p",
        provider=provider,
        state_store=store,
        limits=Limits(**{bound: 1}),
    ) as agent:
        kwargs = dict(tenant_id="tenant")
        if source is not None:
            kwargs["runs"] = [source.run_id]
        call = agent.stream("go", **kwargs) if streaming else agent.run("go", **kwargs)
        if streaming:
            result = await finish(call, True)
            assert result.status == RunStatus.ERROR
            assert "history_too_large" in str(result.error)
        else:
            with pytest.raises(RunNotReclaimableError, match="history_too_large"):
                await call
        assert provider._call_count == 0


@tool(target="client")
async def read_value() -> str:
    return "unused"


@pytest.mark.parametrize("streaming", [False, True])
async def test_resumed_edit_keeps_reconstruction_limit(store, monkeypatch, streaming):
    install_editing_strategy(monkeypatch)
    provider = MockLLM(
        [
            LLMResponse(tool_calls=[ToolCall("read_value")]),
            LLMResponse(text="should not be called"),
        ]
    )
    async with Agent(
        prompt="p",
        provider=provider,
        tools=[read_value],
        state_store=store,
        limits=Limits(max_rebuild_rows=3),
    ) as agent:
        paused = await agent.run("go", tenant_id="tenant")
        pending = paused.pending.tool_calls[0]
        kwargs = dict(
            tenant_id="tenant",
            expected_pause_id=paused.pause_id,
            tool_results=[ToolResult(pending.name, pending.id, '"result"')],
        )
        call = (
            agent.resume_stream(paused.run_id, **kwargs)
            if streaming
            else agent.resume(paused.run_id, **kwargs)
        )
        if streaming:
            result = await finish(call, True)
            assert result.status == RunStatus.ERROR
        else:
            with pytest.raises(RunNotReclaimableError, match="history_too_large"):
                await call
        assert provider._call_count == 1


@pytest.mark.parametrize("kind", ["replay", "edit", "resume_edit"])
async def test_history_rereads_keep_explicit_large_limits(store, monkeypatch, kind):
    import dendrux.runtime.replay as replay

    limits = Limits(max_rebuild_rows=200_000, max_rebuild_bytes=160 * 1024 * 1024)
    source = None
    if kind == "replay":
        async with Agent(
            prompt="p", provider=MockLLM([LLMResponse(text="source")]), state_store=store
        ) as first:
            source = await first.run("source", tenant_id="tenant")
    if kind != "replay":
        install_editing_strategy(monkeypatch)
    original = replay.conversation_records
    observed = []

    async def capture(*args, **kwargs):
        observed.append(kwargs.get("limits"))
        return await original(*args, **kwargs)

    monkeypatch.setattr(replay, "conversation_records", capture)
    provider = MockLLM(
        [LLMResponse(tool_calls=[ToolCall("read_value")]), LLMResponse(text="done")]
        if kind == "resume_edit"
        else [LLMResponse(text="done")]
    )
    async with Agent(
        prompt="p",
        provider=provider,
        state_store=store,
        tools=[read_value],
        limits=limits,
    ) as agent:
        result = await agent.run(
            "go", tenant_id="tenant", **({"runs": [source.run_id]} if source else {})
        )
        if kind == "resume_edit":
            pending = result.pending.tool_calls[0]
            result = await agent.resume(
                result.run_id,
                tenant_id="tenant",
                expected_pause_id=result.pause_id,
                tool_results=[ToolResult(pending.name, pending.id, '"result"')],
            )
        assert result.status == RunStatus.SUCCESS
        assert observed and all(bound is limits for bound in observed)
