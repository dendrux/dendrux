"""``tenant_id=`` on the run-control write methods (0.4.1, Oreo #31).

``submit_tool_results``, ``submit_input``, ``submit_approval``,
``submit_resume``, ``resume`` and ``resume_stream`` take the same optional
tenant scope as ``cancel_run`` / ``submit_note``: a mismatch is a
``RunNotFoundError`` raised before any write, and the run is left untouched.
SQLite + Postgres via the ``store`` matrix fixture.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import pytest

from dendrux.agent import Agent
from dendrux.errors import RunNotFoundError
from dendrux.runtime.context import get_delegation_context
from dendrux.tool import tool
from dendrux.types import (
    ApprovalDecision,
    LLMResponse,
    RunEventType,
    RunStatus,
    ToolResult,
)
from tests._helpers.scripted_provider import ScriptedLLM, tool_call

pytestmark = pytest.mark.asyncio


@tool(target="client")
async def read_range(sheet: str) -> str:
    return ""


@tool()
async def refund(order_id: int) -> str:
    return f"Refunded {order_id}"


@dataclass
class Paused:
    agent: Agent
    run_id: str


async def _pause(kind: str, store: Any, tenant: str | None) -> Paused:
    """Build an agent and run it to a pause of ``kind`` under ``tenant``."""
    holder: dict[str, Agent] = {}

    @tool()
    async def step() -> str:
        ctx = get_delegation_context()
        assert ctx is not None
        await holder["agent"].request_pause(ctx.run_id)
        return "stepped"

    if kind == "approval":
        first = tool_call("refund", {"order_id": 7}, pid="t1")
        tools = [refund]
    elif kind == "paused":
        first = tool_call("step", pid="t1")
        tools = [step]
    else:
        first = tool_call("read_range", {"sheet": "S1"}, pid="t1")
        tools = [read_range]
    llm = ScriptedLLM([first, LLMResponse(text="done")])
    agent = Agent(
        provider=llm,
        prompt="Test.",
        tools=tools,
        require_approval=["refund"] if kind == "approval" else None,
        state_store=store,
    )
    holder["agent"] = agent
    paused = await agent.run("go", tenant_id=tenant)
    if kind == "input":
        await store.pause_run(
            paused.run_id,
            status=RunStatus.WAITING_HUMAN_INPUT.value,
            pause_data=await store.get_pause_state(paused.run_id),
        )
    expected = {
        "client": RunStatus.WAITING_CLIENT_TOOL,
        "input": RunStatus.WAITING_HUMAN_INPUT,
        "approval": RunStatus.WAITING_APPROVAL,
        "paused": RunStatus.PAUSED,
    }[kind]
    record = await store.get_run(paused.run_id)
    assert record.status == expected.value
    return Paused(agent, paused.run_id)


async def _call_ids(store: Any, run_id: str) -> list[str]:
    pause = await store.get_pause_state(run_id)
    return [c["id"] for c in pause["pending_tool_calls"]]


async def _results(store: Any, run_id: str) -> list[ToolResult]:
    return [
        ToolResult(name="read_range", call_id=c, payload='"rows"')
        for c in await _call_ids(store, run_id)
    ]


async def _drain(stream: Any) -> Any:
    events = []
    async with stream:
        async for event in stream:
            events.append(event)
    assert events[0].type == RunEventType.RUN_RESUMED
    assert events[-1].type == RunEventType.RUN_COMPLETED
    return events[-1].run_result


Invoke = Callable[[Paused, Any, Any], Awaitable[Any]]


async def _submit_tool_results(p: Paused, store: Any, scope: dict[str, Any]) -> Any:
    return await p.agent.submit_tool_results(p.run_id, await _results(store, p.run_id), **scope)


async def _submit_input(p: Paused, store: Any, scope: dict[str, Any]) -> Any:
    return await p.agent.submit_input(p.run_id, "answer", **scope)


async def _submit_approval_bool(p: Paused, store: Any, scope: dict[str, Any]) -> Any:
    return await p.agent.submit_approval(p.run_id, approved=True, **scope)


async def _submit_approval_reject(p: Paused, store: Any, scope: dict[str, Any]) -> Any:
    return await p.agent.submit_approval(p.run_id, approved=False, **scope)


async def _submit_approval_decisions(p: Paused, store: Any, scope: dict[str, Any]) -> Any:
    ids = await _call_ids(store, p.run_id)
    return await p.agent.submit_approval(
        p.run_id, decisions={i: ApprovalDecision.approve() for i in ids}, **scope
    )


async def _submit_resume(p: Paused, store: Any, scope: dict[str, Any]) -> Any:
    return await p.agent.submit_resume(p.run_id, **scope)


async def _resume_tool_results(p: Paused, store: Any, scope: dict[str, Any]) -> Any:
    return await p.agent.resume(p.run_id, tool_results=await _results(store, p.run_id), **scope)


async def _resume_input(p: Paused, store: Any, scope: dict[str, Any]) -> Any:
    return await p.agent.resume(p.run_id, user_input="answer", **scope)


async def _resume_approve(p: Paused, store: Any, scope: dict[str, Any]) -> Any:
    return await p.agent.resume(p.run_id, **scope)


async def _resume_stream_tool_results(p: Paused, store: Any, scope: dict[str, Any]) -> Any:
    results = await _results(store, p.run_id)
    return await _drain(p.agent.resume_stream(p.run_id, tool_results=results, **scope))


async def _resume_stream_input(p: Paused, store: Any, scope: dict[str, Any]) -> Any:
    return await _drain(p.agent.resume_stream(p.run_id, user_input="answer", **scope))


async def _resume_stream_approve(p: Paused, store: Any, scope: dict[str, Any]) -> Any:
    return await _drain(p.agent.resume_stream(p.run_id, **scope))


CASES: dict[str, tuple[str, Invoke]] = {
    "submit_tool_results": ("client", _submit_tool_results),
    "submit_input": ("input", _submit_input),
    "submit_approval_approved": ("approval", _submit_approval_bool),
    "submit_approval_rejected": ("approval", _submit_approval_reject),
    "submit_approval_decisions": ("approval", _submit_approval_decisions),
    "submit_resume": ("paused", _submit_resume),
    "resume_tool_results": ("client", _resume_tool_results),
    "resume_user_input": ("input", _resume_input),
    "resume_approve": ("approval", _resume_approve),
    "resume_paused": ("paused", _resume_approve),
    "resume_stream_tool_results": ("client", _resume_stream_tool_results),
    "resume_stream_user_input": ("input", _resume_stream_input),
    "resume_stream_approve": ("approval", _resume_stream_approve),
    "resume_stream_paused": ("paused", _resume_stream_approve),
}

every_case = pytest.mark.parametrize("case", list(CASES))


async def _snapshot(store: Any, run_id: str) -> dict[str, Any]:
    record = await store.get_run(run_id)
    return {
        "status": record.status,
        "answer": record.answer,
        "iterations": record.iteration_count,
        "pause_data": await store.get_pause_state(run_id),
        "events": [(e.sequence_index, e.event_type) for e in await store.get_run_events(run_id)],
        "tool_calls": [(t.tool_name, t.success) for t in await store.get_tool_calls(run_id)],
    }


@every_case
@pytest.mark.parametrize("scope", ["other", "none"])
async def test_mismatch_raises_and_leaves_run_untouched(store, case, scope) -> None:
    kind, invoke = CASES[case]
    paused = await _pause(kind, store, "t1")
    before = await _snapshot(store, paused.run_id)

    tenant = "t2" if scope == "other" else None
    with pytest.raises(RunNotFoundError) as exc:
        await invoke(paused, store, {"tenant_id": tenant})
    assert exc.value.args == RunNotFoundError(paused.run_id).args

    assert await _snapshot(store, paused.run_id) == before


@every_case
async def test_matching_tenant_works(store, case) -> None:
    kind, invoke = CASES[case]
    paused = await _pause(kind, store, "t1")
    result = await invoke(paused, store, {"tenant_id": "t1"})
    assert result.status == RunStatus.SUCCESS
    assert (await store.get_run(paused.run_id)).status == "success"


@every_case
async def test_omitted_scope_is_unchanged(store, case) -> None:
    kind, invoke = CASES[case]
    paused = await _pause(kind, store, "t1")
    result = await invoke(paused, store, {})
    assert result.status == RunStatus.SUCCESS


@every_case
async def test_none_matches_untenanted_run_only(store, case) -> None:
    kind, invoke = CASES[case]
    paused = await _pause(kind, store, None)
    before = await _snapshot(store, paused.run_id)

    with pytest.raises(RunNotFoundError):
        await invoke(paused, store, {"tenant_id": "t1"})
    assert await _snapshot(store, paused.run_id) == before

    result = await invoke(paused, store, {"tenant_id": None})
    assert result.status == RunStatus.SUCCESS


async def test_missing_run_with_scope_is_not_found(store) -> None:
    agent = Agent(provider=ScriptedLLM([]), prompt="Test.", tools=[read_range], state_store=store)
    result = ToolResult(name="read_range", call_id="x", payload='""')
    calls = [
        agent.submit_tool_results("nope", [result], tenant_id="t1"),
        agent.submit_input("nope", "x", tenant_id="t1"),
        agent.submit_approval("nope", approved=True, tenant_id="t1"),
        agent.submit_resume("nope", tenant_id="t1"),
        agent.resume("nope", tenant_id="t1"),
    ]
    for call in calls:
        with pytest.raises(RunNotFoundError):
            await call
    with pytest.raises(RunNotFoundError):
        async for _ in agent.resume_stream("nope", tenant_id="t1"):
            raise AssertionError("yielded an event")


async def test_resume_stream_mismatch_raises_before_any_event(store) -> None:
    paused = await _pause("client", store, "t1")
    results = await _results(store, paused.run_id)
    stream = paused.agent.resume_stream(paused.run_id, tool_results=results, tenant_id="t2")
    events: list[Any] = []
    with pytest.raises(RunNotFoundError):
        async for event in stream:
            events.append(event)
    assert events == []
    assert stream.result is None


async def test_resume_stream_mismatch_does_not_cancel_a_running_run(store) -> None:
    """The stream's close-time cleanup must not touch another tenant's live run."""
    paused = await _pause("client", store, "t1")
    from sqlalchemy import update

    from dendrux.db.models import AgentRun

    async with store._session_factory() as session:
        await session.execute(
            update(AgentRun).where(AgentRun.id == paused.run_id).values(status="running")
        )
        await session.commit()
    before = await _snapshot(store, paused.run_id)

    async with paused.agent.resume_stream(paused.run_id, tenant_id="t2") as stream:
        with pytest.raises(RunNotFoundError):
            async for _ in stream:
                pass
    assert await _snapshot(store, paused.run_id) == before


async def test_validation_errors_on_scoped_calls_do_not_leak_existence(store) -> None:
    """A wrong-tenant caller sees not-found, not a status or argument error."""
    paused = await _pause("client", store, "t1")
    with pytest.raises(RunNotFoundError):
        await paused.agent.submit_input(paused.run_id, "x", tenant_id="t2")
    with pytest.raises(RunNotFoundError):
        await paused.agent.submit_resume(paused.run_id, tenant_id="t2")
