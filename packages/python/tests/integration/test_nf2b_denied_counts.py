"""Denied observations must not consume the resumed tool execution budget."""

import pytest

from dendrux import tool
from dendrux._meta_keys import NOT_EXECUTED_META_KEY
from dendrux.types import LLMResponse, Role, RunStatus, ToolDecision
from tests._fixtures.storage import pin_storage_format
from tests._helpers.runtime_paths import client_read, client_result, make_agent
from tests._helpers.scripted_provider import tool_call


@pytest.mark.parametrize("storage_format", [1, 2])
@pytest.mark.parametrize("origin", ["run", "stream"])
@pytest.mark.parametrize("entry", ["resume", "resume_stream", "submit_tool_results"])
@pytest.mark.parametrize("denial", ["static", "hook", "hook_error"])
async def test_denial_preserves_one_execution_after_resume(
    store, engine, storage_format, origin, entry, denial
):
    await pin_storage_format(engine, storage_format)
    effects = []

    @tool(max_calls_per_run=1)
    async def work() -> str:
        """Record one execution."""
        effects.append(True)
        return "ok"

    def policy(ctx):
        if ctx.tool_name != "work":
            return ToolDecision.allow()
        if denial == "hook_error":
            raise RuntimeError("policy unavailable")
        return ToolDecision.deny("not yet")

    initial = make_agent(
        store,
        [tool_call("work", pid="denied"), tool_call("client_read", pid="pause")],
        tools=[work, client_read],
        deny=["work"] if denial == "static" else [],
        before_tool=policy if denial != "static" else None,
    )
    if origin == "run":
        paused = await initial.run("go")
    else:
        async with initial.stream("go") as stream:
            async for _ in stream:
                pass
        paused = stream.result
    assert paused.status == RunStatus.WAITING_CLIENT_TOOL
    assert effects == []

    resumed = make_agent(
        store,
        [
            tool_call("work", pid="allowed"),
            tool_call("work", pid="limited"),
            LLMResponse(text="done"),
        ],
        tools=[work, client_read],
        on_governance_change="warn",
    )
    results = [client_result(paused)]
    if entry == "resume":
        result = await resumed.resume(paused.run_id, tool_results=results)
    elif entry == "submit_tool_results":
        result = await resumed.submit_tool_results(paused.run_id, results)
    else:
        async with resumed.resume_stream(paused.run_id, tool_results=results) as stream:
            async for _ in stream:
                pass
        result = stream.result
    assert result.status == RunStatus.SUCCESS
    assert effects == [True]
    observations = [m for m in resumed.provider.messages[-1] if m.role == Role.TOOL]
    denied = next(m for m in observations if m.name == "work")
    assert denied.meta[NOT_EXECUTED_META_KEY] is True
