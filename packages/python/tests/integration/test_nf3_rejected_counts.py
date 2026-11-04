"""Pre-execution rejections do not spend the resumed call budget."""

import pytest

from dendrux import tool
from dendrux._meta_keys import NOT_EXECUTED_META_KEY
from dendrux.types import LLMResponse, Role, RunStatus, ToolCall
from tests._fixtures.storage import pin_storage_format
from tests._helpers.runtime_paths import client_read, client_result, make_agent
from tests._helpers.scripted_provider import tool_call


@pytest.mark.parametrize("storage_format", [1, 2])
@pytest.mark.parametrize("origin", ["run", "stream"])
@pytest.mark.parametrize("entry", ["resume", "resume_stream", "submit_tool_results"])
@pytest.mark.parametrize("rejection", ["arguments", "json", "unknown"])
async def test_rejection_preserves_one_execution_after_resume(
    store, engine, storage_format, origin, entry, rejection
):
    await pin_storage_format(engine, storage_format)
    effects = []

    @tool(max_calls_per_run=1)
    async def work(value: int) -> str:
        """Record one execution."""
        effects.append(value)
        return "ok"

    invalid = ToolCall(
        "work",
        {"value": "wrong"},
        provider_tool_call_id="rejected",
        parse_error="invalid JSON" if rejection == "json" else None,
    )
    initial = make_agent(
        store,
        [LLMResponse(tool_calls=[invalid]), tool_call("client_read", pid="pause")],
        tools=[client_read] if rejection == "unknown" else [work, client_read],
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
            tool_call("work", {"value": 1}, pid="allowed"),
            tool_call("work", {"value": 2}, pid="limited"),
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
    assert effects == [1]
    observations = [m for m in resumed.provider.messages[-1] if m.role == Role.TOOL]
    rejected = next(m for m in observations if m.name == "work")
    assert rejected.meta[NOT_EXECUTED_META_KEY] is True
    traces = await store.get_traces(paused.run_id)
    trace = next(
        t for t in traces if (t.meta or {}).get("call_id") == rejected.call_id and t.role == "tool"
    )
    assert trace.meta[NOT_EXECUTED_META_KEY] is True


@pytest.mark.parametrize("storage_format", [1, 2])
@pytest.mark.parametrize(
    "path",
    [
        "run",
        "run_stream",
        "resume",
        "resume_stream",
        "retry",
        "recovery",
        "tool_results",
        "input",
        "approve",
        "reject",
        "decisions",
        "submit_resume",
    ],
)
async def test_rejection_count_on_every_supported_entry(store, engine, storage_format, path):
    from tests._helpers.runtime_paths import path_case
    from tests._helpers.scripted_provider import ScriptedLLM

    if path == "recovery" and storage_format == 1:
        pytest.skip("Legacy runs cannot be reclaimed")
    await pin_storage_format(engine, storage_format)
    effects = []

    @tool(max_calls_per_run=1)
    async def work(value: int) -> str:
        """Record an actual execution."""
        effects.append(value)
        return "ok"

    provider = ScriptedLLM(
        [
            tool_call("work", {"value": "wrong"}, pid="rejected"),
            tool_call("client_read", pid="pause"),
            tool_call("work", {"value": 1}, pid="allowed"),
            tool_call("work", {"value": 2}, pid="limited"),
            LLMResponse(text="done"),
        ]
    )
    case = await path_case(store, engine, path, provider)
    case.agent.tools.append(work)
    if client_read not in case.agent.tools:
        case.agent.tools.append(client_read)
    try:
        paused = await case.invoke()
        assert paused.status == RunStatus.WAITING_CLIENT_TOOL
        assert effects == []
        result = await case.agent.resume(
            paused.run_id,
            tool_results=[client_result(paused)],
            tenant_id="tenant",
            expected_pause_id=paused.pause_id,
        )
        assert result.status == RunStatus.SUCCESS
        assert effects == [1]
    finally:
        await case.agent.close()
