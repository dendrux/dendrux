"""The store under a real ``Agent``: runs, pauses, continuations and a restart.

Scripted providers (no network) drive the public entry paths against the store, and each check
reads the result back through the ``StateStore`` protocol only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dendrux import Agent
from dendrux.llm.mock import MockLLM
from dendrux.testing.store_kit.contracts._marks import contract
from dendrux.tool import tool
from dendrux.types import LLMResponse, RunStatus, ToolCall, ToolResult, UsageStats

if TYPE_CHECKING:
    from dendrux.testing.store_kit._kit import Kit

flows = contract("agent_flows")
reopened_flows = contract("agent_flows", uses=("reopen",))


@tool()
async def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


@tool(target="client")
async def read_range(sheet: str) -> str:
    """Read a range on the client."""
    return ""


@tool()
async def refund(order_id: int) -> str:
    """Refund an order (needs approval)."""
    return f"refunded {order_id}"


def _usage() -> UsageStats:
    return UsageStats(input_tokens=10, output_tokens=5, total_tokens=15)


def say(body: str) -> LLMResponse:
    return LLMResponse(text=body, usage=_usage())


def ask(name: str, params: dict[str, Any], call_id: str) -> LLMResponse:
    call = ToolCall(name=name, params=params, provider_tool_call_id=call_id)
    return LLMResponse(text="working", tool_calls=[call], usage=_usage())


def agent_for(store: Any, *responses: LLMResponse, tools: Any = (), **kwargs: Any) -> Agent:
    # 0.5 behaviour with the choices made explicit, so the check does not depend on defaults
    return Agent(
        provider=MockLLM(list(responses)),
        prompt="You are the store kit agent.",
        name="KitFlow",
        tools=list(tools),
        state_store=store,
        strict_tenancy=False,
        on_governance_change="warn",
        **kwargs,
    )


@flows
async def test_a_plain_run_is_recorded_from_start_to_finish(dendrux_kit: Kit) -> None:
    store = dendrux_kit.store
    result = await agent_for(store, say("The answer is 42.")).run("what is it?")
    assert result.status == RunStatus.SUCCESS and result.answer == "The answer is 42."
    record = await store.get_run(result.run_id)
    assert record is not None
    assert (record.status, record.answer, record.agent_name) == (
        "success",
        "The answer is 42.",
        "KitFlow",
    )
    assert record.total_input_tokens == 10 and record.total_output_tokens == 5
    types = [e.event_type for e in await store.get_run_events(result.run_id)]
    assert types[0] == "run.started" and "llm.completed" in types
    assert [e.sequence_index for e in await store.get_run_events(result.run_id)] == list(
        range(len(types))
    )
    roles = [t.role for t in await store.get_traces(result.run_id)]
    assert "user" in roles and roles[-1] == "assistant"


@flows
async def test_a_server_tool_call_is_persisted_with_its_result(dendrux_kit: Kit) -> None:
    store = dendrux_kit.store
    agent = agent_for(store, ask("add", {"a": 2, "b": 3}, "t1"), say("2 + 3 = 5"), tools=[add])
    result = await agent.run("add")
    assert result.status == RunStatus.SUCCESS
    [row] = await store.get_tool_calls(result.run_id)
    assert (row.tool_name, row.target, row.success) == ("add", "server", True)
    assert row.params == {"a": 2, "b": 3}
    assert "tool.completed" in [e.event_type for e in await store.get_run_events(result.run_id)]


@flows
async def test_a_client_tool_pause_resumes_from_stored_state(dendrux_kit: Kit) -> None:
    store = dendrux_kit.store
    agent = agent_for(
        store, ask("read_range", {"sheet": "S1"}, "t1"), say("3 rows"), tools=[read_range]
    )
    paused = await agent.run("read the sheet")
    assert paused.status == RunStatus.WAITING_CLIENT_TOOL and paused.pending is not None
    record = await store.get_run(paused.run_id)
    assert record is not None and record.status == "waiting_client_tool"
    assert await store.get_pause_state(paused.run_id) is not None
    call_id = paused.pending.tool_calls[0].id
    done = await agent.submit_tool_results(
        paused.run_id,
        [ToolResult(name="read_range", call_id=call_id, payload='"3 rows"')],
        expected_pause_id=paused.pause_id,
    )
    assert done.status == RunStatus.SUCCESS and done.answer == "3 rows"
    final = await store.get_run(paused.run_id)
    assert final is not None and final.status == "success"
    assert await store.get_pause_state(paused.run_id) is None


@flows
async def test_an_approval_pause_can_be_approved(dendrux_kit: Kit) -> None:
    store = dendrux_kit.store
    agent = agent_for(
        store,
        ask("refund", {"order_id": 7}, "t1"),
        say("Refund processed."),
        tools=[refund],
        require_approval=["refund"],
    )
    paused = await agent.run("refund order 7")
    assert paused.status == RunStatus.WAITING_APPROVAL
    done = await agent.submit_approval(
        paused.run_id, approved=True, expected_pause_id=paused.pause_id
    )
    assert done.status == RunStatus.SUCCESS
    [row] = await store.get_tool_calls(paused.run_id)
    assert (row.tool_name, row.success) == ("refund", True)


@flows
async def test_a_rejected_approval_does_not_run_the_tool(dendrux_kit: Kit) -> None:
    store = dendrux_kit.store
    agent = agent_for(
        store,
        ask("refund", {"order_id": 7}, "t1"),
        say("Not refunded."),
        tools=[refund],
        require_approval=["refund"],
    )
    paused = await agent.run("refund order 7")
    done = await agent.submit_approval(
        paused.run_id,
        approved=False,
        rejection_reason="Not authorised.",
        expected_pause_id=paused.pause_id,
    )
    assert done.status == RunStatus.SUCCESS
    rows = await store.get_tool_calls(paused.run_id)
    assert all(r.result is None or "refunded 7" not in str(r.result) for r in rows)


@flows
async def test_a_paused_run_can_be_cancelled(dendrux_kit: Kit) -> None:
    store = dendrux_kit.store
    agent = agent_for(store, ask("read_range", {"sheet": "S1"}, "t1"), tools=[read_range])
    paused = await agent.run("read")
    cancelled = await agent.cancel_run(paused.run_id)
    assert cancelled.status == RunStatus.CANCELLED
    record = await store.get_run(paused.run_id)
    assert record is not None and record.status == "cancelled"


@flows
async def test_a_second_submission_to_a_resumed_pause_loses(dendrux_kit: Kit) -> None:
    store = dendrux_kit.store
    agent = agent_for(
        store, ask("read_range", {"sheet": "S1"}, "t1"), say("done"), tools=[read_range]
    )
    paused = await agent.run("read")
    assert paused.pending is not None
    results = [
        ToolResult(name="read_range", call_id=paused.pending.tool_calls[0].id, payload='"x"')
    ]
    await agent.submit_tool_results(paused.run_id, results, expected_pause_id=paused.pause_id)
    try:
        again = await agent.submit_tool_results(
            paused.run_id, results, expected_pause_id=paused.pause_id
        )
    except Exception:
        again = None  # a typed refusal is a loss as well
    assert again is None or again.status != RunStatus.WAITING_CLIENT_TOOL
    assert len(await store.get_tool_calls(paused.run_id)) <= 1


@flows
async def test_a_streamed_run_ends_in_the_same_stored_state(dendrux_kit: Kit) -> None:
    store = dendrux_kit.store
    agent = agent_for(store, say("streamed"))
    async with agent.stream("go") as stream:
        async for _ in stream:
            pass
        result = stream.result
    assert result is not None and result.status == RunStatus.SUCCESS
    record = await store.get_run(result.run_id)
    assert record is not None and (record.status, record.answer) == ("success", "streamed")


@reopened_flows
async def test_a_pause_resumes_on_a_store_reopened_after_a_restart(dendrux_kit: Kit) -> None:
    store = dendrux_kit.store
    first = agent_for(
        store, ask("read_range", {"sheet": "S1"}, "t1"), say("unused"), tools=[read_range]
    )
    paused = await first.run("read")
    assert paused.pending is not None
    reopened = await dendrux_kit.reopen()
    second = agent_for(reopened, say("resumed after restart"), tools=[read_range])
    done = await second.submit_tool_results(
        paused.run_id,
        [ToolResult(name="read_range", call_id=paused.pending.tool_calls[0].id, payload='"x"')],
        expected_pause_id=paused.pause_id,
    )
    assert done.status == RunStatus.SUCCESS and done.answer == "resumed after restart"
    record = await reopened.get_run(paused.run_id)
    assert record is not None and record.status == "success"
