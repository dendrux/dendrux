"""Streaming approval refuses a held multi-child batch before server effects."""

from __future__ import annotations

import pytest

from dendrux import Agent, tool
from dendrux.types import LLMResponse, RunStatus, ToolCall
from tests._helpers.fanout_effects import FanoutEffects
from tests._helpers.scripted_provider import ScriptedLLM
from tests.integration.test_fanout_driver import round2_children

pytestmark = pytest.mark.parametrize("engine", ["sqlite_file", "postgres"], indirect=True)


@pytest.mark.parametrize("entry", ["stream", "resume_stream"])
async def test_resume_stream_refuses_two_held_children_until_streaming_fanout(store, entry):
    effects = FanoutEffects()
    children = round2_children(effects)

    authorized = []

    @tool()
    async def authorize() -> str:
        authorized.append(True)
        return "approved"

    calls = [ToolCall(c.name, {"task": "go"}) for c in children]
    llm = ScriptedLLM(
        [
            LLMResponse(tool_calls=[calls[0], ToolCall("authorize", {}), calls[1]]),
            LLMResponse(text="done"),
        ]
    )
    parent = Agent(
        name="lead",
        prompt="p",
        state_store=store,
        subagents=children,
        tools=[authorize],
        require_approval=["authorize"],
        provider=effects.provider("root", llm),
    )
    try:
        paused = None
        if entry == "resume_stream":
            paused = await parent.run("go", tenant_id="t1")
            assert paused.status == RunStatus.WAITING_APPROVAL
        events = []
        err = None
        manager = (
            parent.stream("go", tenant_id="t1")
            if paused is None
            else parent.resume_stream(
                paused.run_id, tenant_id="t1", expected_pause_id=paused.pause_id
            )
        )
        try:
            async with manager as stream:
                async for ev in stream:
                    events.append(ev)
        except Exception as e:  # noqa: BLE001 - capture stream failure to check durable state.
            err = e
        result = last_result(events)
        rec = await store.get_run(result.run_id)
        state = await store.get_loop_state(result.run_id)
        assert err is None
        assert last_result(events).status == RunStatus.SUCCESS
        assert rec.status == "success"
        assert not state.get("managed_admitted") and not state.get("managed_wait")
        assert [effects.counts().provider_calls(c.name) for c in children] == [0, 0]
        assert authorized == []
    finally:
        await parent.close()
        for child in children:
            await child.close()


def last_result(events):
    return next(e.run_result for e in reversed(events) if e.run_result is not None)
