"""Managed subagents (0.7, B9): a SingleCall agent as a managed child.

A SingleCall child makes exactly one model call. It succeeds, fails, or (when its
process dies mid-call) ends failed through ``sweep()`` and is never run again; the
parent always sees an ordinary tool result. Runs on SQLite and on Postgres
(``DENDRUX_TEST_PG_URL``, a private ``*_test`` database).

The file name avoids the ``single_call`` area rule on purpose: these are managed
runtime tests (area ``runtime``).
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from dendrux import Agent
from dendrux.db.models import AgentRun, RunEvent, TokenUsage, ToolCallRecord
from dendrux.errors import PersistenceError
from dendrux.loops.single import SingleCall
from dendrux.runtime.context import get_delegation_context
from dendrux.runtime.stores._factory import store_for_engine
from dendrux.store import RunStore
from dendrux.types import (
    Budget,
    LLMResponse,
    ProviderCapabilities,
    RunStatus,
    ToolCall,
    UsageStats,
)
from tests._helpers.gates import wait_for_gate
from tests._helpers.scripted_provider import ScriptedLLM
from tests.integration.test_budget import _CostReportingLLM
from tests.integration.test_managed_budget import response, unknown_response
from tests.integration.test_managed_recovery import fresh_store

BRIEF = "SENTINEL-oneshot-brief-41c2"
REPORT = "SENTINEL-oneshot-report-9b07"
TENANT = "t1"


class ContextLLM(ScriptedLLM):
    """Scripted provider recording the run each call is served under (real ContextVar).

    Every response carries 10 input tokens so usage can be counted per call.
    """

    capabilities = ProviderCapabilities(
        supports_native_tools=True, supports_tool_call_ids=True, supports_structured_output=True
    )

    def __init__(self, responses: list[LLMResponse], **kwargs: Any) -> None:
        super().__init__(responses, **kwargs)
        self.served: list[str | None] = []

    async def complete(self, messages: Any, tools: Any = None, **kwargs: Any) -> LLMResponse:
        ctx = get_delegation_context()
        self.served.append(ctx.run_id if ctx else None)
        response = await super().complete(messages, tools, **kwargs)
        response.usage = UsageStats(input_tokens=10, output_tokens=1, total_tokens=11)
        return response


class FailingLLM(ContextLLM):
    async def complete(self, messages: Any, tools: Any = None, **kwargs: Any) -> LLMResponse:
        self.calls += 1
        raise RuntimeError("SENTINEL-provider-text-should-not-leak")


def call(name: str, params: dict[str, Any], pid: str) -> LLMResponse:
    return LLMResponse(tool_calls=[ToolCall(name=name, params=params, provider_tool_call_id=pid)])


class Summary(BaseModel):
    verdict: str
    score: int


def scorer(llm: Any, **kwargs: Any) -> Agent:
    return Agent(
        name="scorer",
        description="Scores one draft in a single model call.",
        prompt="You score.",
        provider=llm,
        loop=SingleCall(),
        **kwargs,
    )


def lead(store: Any, llm: Any, *children: Agent, **kwargs: Any) -> Agent:
    return Agent(
        name="lead",
        prompt="You lead.",
        provider=llm,
        state_store=store,
        subagents=list(children),
        **kwargs,
    )


async def rows(engine: Any, model: Any, *where: Any) -> list[Any]:
    async with AsyncSession(engine) as session:
        return list((await session.execute(select(model).where(*where))).scalars().all())


async def count(engine: Any, model: Any, *where: Any) -> int:
    async with AsyncSession(engine) as session:
        return int(await session.scalar(select(func.count()).select_from(model).where(*where)))


async def observed(root_llm: ContextLLM, turn: int) -> list[Any]:
    return [m for m in root_llm.messages[turn] if m.role.value == "tool"]


async def drive(root: Agent, entry: str) -> Any:
    if entry == "run":
        return await root.run("go", tenant_id=TENANT)
    stream = root.stream("go", tenant_id=TENANT)
    async for _ in stream:
        pass
    return stream.result


async def subtree_usage(engine: Any, root_id: str) -> int:
    async with RunStore.from_engine(engine) as reader:
        tree = await reader.get_subtree(root_id, tenant_id=TENANT, max_nodes=16)
    return tree.usage.input_tokens


# ---------------------------------------------------------------------------
# In-process outcomes: success, structured output, provider failure
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("entry", ["run", "stream"])
async def test_single_call_child_report_reaches_the_parent(
    store: Any, engine: Any, entry: str
) -> None:
    child_llm = ContextLLM([LLMResponse(text=REPORT)])
    root_llm = ContextLLM([call("scorer", {"task": BRIEF}, "r1"), LLMResponse(text="root done")])
    result = await drive(lead(store, root_llm, scorer(child_llm)), entry)

    assert result.status == RunStatus.SUCCESS and result.answer == "root done"
    [child] = await rows(engine, AgentRun, AgentRun.parent_run_id == result.run_id)
    assert child.status == "success" and child.delegation_level == 1
    assert child.loop_state["loop"] == "single_call"  # the marker sweep keys on (P2-3)
    assert child_llm.calls == 1 and child_llm.served == [child.id]
    assert [m.content for m in child_llm.messages[0] if m.role.value == "user"] == [BRIEF]
    [observation] = await observed(root_llm, 1)
    assert json.loads(observation.content) == REPORT
    [intent] = await rows(engine, ToolCallRecord, ToolCallRecord.agent_run_id == result.run_id)
    assert intent.outcome == "completed" and intent.meta["managed"]["child_run_id"] == child.id
    assert await count(engine, TokenUsage, TokenUsage.agent_run_id == child.id) == 1
    assert await subtree_usage(engine, result.run_id) == 30


async def test_structured_child_reports_the_compact_value(store: Any, engine: Any) -> None:
    child_llm = ContextLLM([LLMResponse(text='{"verdict": "ok", "score": 3}')])
    root_llm = ContextLLM([call("scorer", {"task": BRIEF}, "r1"), LLMResponse(text="root done")])
    result = await lead(store, root_llm, scorer(child_llm, output_type=Summary)).run(
        "go", tenant_id=TENANT
    )

    assert result.status == RunStatus.SUCCESS
    [observation] = await observed(root_llm, 1)
    assert observation.content == '{"verdict":"ok","score":3}'


async def test_structured_child_with_an_invalid_value_fails_the_child(
    store: Any, engine: Any
) -> None:
    """SingleCall validates its own output: a bad value fails the child, never the root."""
    child_llm = ContextLLM([LLMResponse(text='{"verdict": "ok"}')])
    root_llm = ContextLLM([call("scorer", {"task": BRIEF}, "r1"), LLMResponse(text="root done")])
    result = await lead(store, root_llm, scorer(child_llm, output_type=Summary)).run(
        "go", tenant_id=TENANT
    )

    assert result.status == RunStatus.SUCCESS and result.answer == "root done"
    [child] = await rows(engine, AgentRun, AgentRun.parent_run_id == result.run_id)
    assert child.status == "error"
    [observation] = await observed(root_llm, 1)
    body = json.loads(observation.content)
    assert (body["code"], body["child_run_id"]) == ("managed_child_failed", child.id)


@pytest.mark.parametrize("entry", ["run", "stream"])
async def test_provider_error_in_the_child_is_a_failed_child(
    store: Any, engine: Any, entry: str
) -> None:
    child_llm = FailingLLM([])
    root_llm = ContextLLM([call("scorer", {"task": BRIEF}, "r1"), LLMResponse(text="root done")])
    result = await drive(lead(store, root_llm, scorer(child_llm)), entry)

    assert result.status == RunStatus.SUCCESS and result.answer == "root done"
    [child] = await rows(engine, AgentRun, AgentRun.parent_run_id == result.run_id)
    assert child.status == "error" and child_llm.calls == 1
    [observation] = await observed(root_llm, 1)
    body = json.loads(observation.content)
    assert (body["code"], body["child_run_id"]) == ("managed_child_failed", child.id)
    assert "SENTINEL-provider-text" not in observation.content
    events = await rows(engine, RunEvent, RunEvent.agent_run_id == result.run_id)
    [completed] = [e for e in events if e.event_type == "tool.completed"]
    assert completed.data.get("managed_outcome") == "failed"


async def test_a_single_call_child_runs_in_the_callers_task(store: Any, engine: Any) -> None:
    """No helper task drives the child's provider (AGENTS §5)."""
    tasks: list[int] = []

    class TaskLLM(ContextLLM):
        async def complete(self, messages: Any, tools: Any = None, **kwargs: Any) -> LLMResponse:
            tasks.append(id(asyncio.current_task()))
            return await super().complete(messages, tools, **kwargs)

    root_llm = TaskLLM([call("scorer", {"task": BRIEF}, "r1"), LLMResponse(text="root done")])
    result = await lead(store, root_llm, scorer(TaskLLM([LLMResponse(text=REPORT)]))).run(
        "go", tenant_id=TENANT
    )
    assert result.status == RunStatus.SUCCESS and len(tasks) == 3 and len(set(tasks)) == 1


# ---------------------------------------------------------------------------
# Refusals: before any write, so the parent model never retries a hopeless call
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("entry", ["run", "stream"])
async def test_guardrails_with_output_type_child_is_refused_before_any_write(
    store: Any, engine: Any, entry: str
) -> None:
    """P1-2: SingleCall refuses this pair only inside its run (after the bind); the root
    refuses it at entry instead, so no child row exists and no model is called."""
    from dendrux.guardrails import SecretDetection

    child_llm = ContextLLM([LLMResponse(text='{"verdict": "ok", "score": 3}')])
    root_llm = ContextLLM([call("scorer", {"task": BRIEF}, "r1"), LLMResponse(text="root done")])
    child = scorer(child_llm, output_type=Summary, guardrails=[SecretDetection(action="block")])
    root = lead(store, root_llm, child)
    with pytest.raises(ValueError, match="scorer"):
        if entry == "run":
            await root.run("go", tenant_id=TENANT)
        else:
            async for _ in root.stream("go", tenant_id=TENANT):
                pass
    assert root_llm.calls == 0 and child_llm.calls == 0
    assert await rows(engine, AgentRun) == []


def test_single_call_parent_and_child_budget_stay_refused() -> None:
    from dendrux.errors import SubagentDefinitionError
    from dendrux.runtime import _managed_budget

    child = scorer(ContextLLM([]))
    with pytest.raises(SubagentDefinitionError) as info:
        Agent(
            name="lead", prompt="p", provider=ContextLLM([]), loop=SingleCall(), subagents=[child]
        )
    assert info.value.reason == "single_call_parent"
    budgeted = scorer(ContextLLM([]), budget=Budget(max_tokens=100))
    with pytest.raises(ValueError, match="child budget"):
        _managed_budget.validate_definitions(
            Agent(name="lead", prompt="p", provider=ContextLLM([]), subagents=[budgeted]),
            agent_type=Agent,
        )


async def test_streaming_a_structured_single_call_child_is_refused(store: Any, engine: Any) -> None:
    root = lead(store, ContextLLM([]), scorer(ContextLLM([]), output_type=Summary))
    with pytest.raises(ValueError, match="cannot stream subagent 'scorer'"):
        async for _ in root.stream("go", tenant_id=TENANT):
            pass
    assert await rows(engine, AgentRun) == []


# ---------------------------------------------------------------------------
# The pre-claim guard: a running SingleCall run is never claimed, on any store
# ---------------------------------------------------------------------------


class ProcessDeathError(BaseException):
    """Ends the caller like a killed process: no runner cleanup finalises the run."""


class DyingLLM(ContextLLM):
    async def complete(self, messages: Any, tools: Any = None, **kwargs: Any) -> LLMResponse:
        self.calls += 1
        raise ProcessDeathError


def claiming_store(store: Any, claims: list[str]) -> Any:
    """A custom store whose ``reclaim_run`` is a spy that would claim anything it is asked."""

    base = type(store)

    class ClaimingStore(base):  # type: ignore[misc, valid-type]
        async def reclaim_run(self, run_id: str, **kwargs: Any) -> Any:
            claims.append(run_id)
            return await base.reclaim_run(self, run_id, **kwargs)

    custom = store_for_engine(store._engine)
    custom.__class__ = ClaimingStore
    return custom


async def dead_single_call_child(store: Any, engine: Any) -> tuple[str, str]:
    """A root parked on a SingleCall child whose process died during its model call."""
    root_llm = ContextLLM([call("scorer", {"task": BRIEF}, "r1")])
    with pytest.raises(ProcessDeathError):
        await lead(store, root_llm, scorer(DyingLLM([]))).run("go", tenant_id=TENANT)
    [root] = await rows(engine, AgentRun, AgentRun.agent_name == "lead")
    [child] = await rows(engine, AgentRun, AgentRun.parent_run_id == root.id)
    assert (root.status, child.status, child.loop_state["loop"]) == (
        "paused",
        "running",
        "single_call",
    )
    return root.id, child.id


async def marks(engine: Any, *run_ids: str) -> list[tuple[Any, ...]]:
    found = []
    for run_id in run_ids:
        [row] = await rows(engine, AgentRun, AgentRun.id == run_id)
        events = await count(engine, RunEvent, RunEvent.agent_run_id == run_id)
        found.append(
            (row.status, row.claim_epoch, row.reclaim_attempts, row.lease_expired_at, events)
        )
    return found


@pytest.mark.parametrize("registered", ["single_call", "react"])
async def test_root_reclaim_never_claims_a_running_single_call_leaf_on_any_store(
    store: Any, engine: Any, registered: str, caplog: pytest.LogCaptureFixture
) -> None:
    """The saved ``single_call`` marker refuses before the store is asked, even when the
    registered child has since become ReAct (definition drift) and the store would claim."""
    from dendrux.errors import RunNotReclaimableError

    root_id, child_id = await dead_single_call_child(store, engine)
    before = await marks(engine, root_id, child_id)
    claims: list[str] = []
    child_llm = ContextLLM([LLMResponse(text=REPORT)])
    child = (
        scorer(child_llm)
        if registered == "single_call"
        else Agent(name="scorer", description="d", prompt="p", provider=child_llm)
    )
    root = lead(claiming_store(store, claims), ContextLLM([]), child)
    with pytest.raises(RunNotReclaimableError) as refused:
        await root.reclaim(root_id, tenant_id=TENANT)
    assert (refused.value.run_id, refused.value.reason) == (child_id, "loop_unsupported")
    assert claims == [] and child_llm.calls == 0
    assert await marks(engine, root_id, child_id) == before
    assert any(
        getattr(r, "code", None) == "loop_unsupported" and getattr(r, "run_id", None) == child_id
        for r in caplog.records
    )


async def test_a_single_call_agent_never_claims_on_any_store(store: Any, engine: Any) -> None:
    """The agent half (P2-2): a SingleCall agent refuses before ``reclaim_run``, standalone
    or as a managed leaf whose saved marker says ``react``."""
    from dendrux.errors import RunNotReclaimableError
    from tests._helpers.recovery_checkpoint import checkpoint

    claims: list[str] = []
    custom = claiming_store(store, claims)
    react = Agent(name="solo", prompt="p", provider=ContextLLM([]), state_store=custom)
    item = await checkpoint(react, store, engine)
    before = await marks(engine, item.run_id)
    llm = ContextLLM([LLMResponse(text=REPORT)])
    solo = Agent(name="solo", prompt="p", provider=llm, state_store=custom, loop=SingleCall())
    with pytest.raises(RunNotReclaimableError) as refused:
        await solo.reclaim(item.run_id, tenant_id="tenant", item=item)
    assert refused.value.reason == "loop_unsupported"
    assert claims == [] and llm.calls == 0
    assert await marks(engine, item.run_id) == before


async def test_a_react_leaf_now_registered_as_single_call_is_never_claimed(
    store: Any, engine: Any
) -> None:
    """Definition drift the other way: the saved marker is ``react`` but the registered
    child is now SingleCall. The agent half refuses through the root before any claim."""
    from dendrux.errors import RunNotReclaimableError
    from tests._helpers.recovery_checkpoint import expire

    react_child = Agent(name="scorer", description="d", prompt="p", provider=DyingLLM([]))
    with pytest.raises(ProcessDeathError):
        await lead(store, ContextLLM([call("scorer", {"task": BRIEF}, "r1")]), react_child).run(
            "go", tenant_id=TENANT
        )
    [root] = await rows(engine, AgentRun, AgentRun.agent_name == "lead")
    [child] = await rows(engine, AgentRun, AgentRun.parent_run_id == root.id)
    assert child.loop_state["loop"] == "react"
    await expire(store, engine, child.id)
    before = await marks(engine, root.id, child.id)
    claims: list[str] = []
    child_llm = ContextLLM([LLMResponse(text=REPORT)])
    drifted = lead(claiming_store(store, claims), ContextLLM([]), scorer(child_llm))
    with pytest.raises(RunNotReclaimableError) as refused:
        await drifted.reclaim(root.id, tenant_id=TENANT)
    assert (refused.value.run_id, refused.value.reason) == (child.id, "loop_unsupported")
    assert claims == [] and child_llm.calls == 0
    assert await marks(engine, root.id, child.id) == before


# ---------------------------------------------------------------------------
# Root aggregate budget: the child's one call counts toward the root
# ---------------------------------------------------------------------------


def budgeted(store: Any, root_responses: list[Any], child_responses: list[Any], budget: Any) -> Any:
    child_llm = _CostReportingLLM(child_responses)
    child = Agent(
        name="child", description="Worker", prompt="Work", provider=child_llm, loop=SingleCall()
    )
    root_llm = _CostReportingLLM(root_responses)
    root = Agent(
        name="root",
        prompt="Lead",
        provider=root_llm,
        subagents=[child],
        state_store=store,
        budget=budget,
        strict_tenancy=False,
    )
    return root, root_llm, child_llm


async def test_root_budget_counts_the_child_and_refuses_admission_when_spent(
    store: Any,
) -> None:

    root, root_llm, child_llm = budgeted(
        store,
        [response(3, "child"), response()],
        [response(2)],
        Budget(max_cost_usd=10, on_exceed="pause"),
    )
    result = await root.run("go")
    assert result.status == RunStatus.SUCCESS and child_llm.calls == 1
    usage = await store.subtree_usage(result.run_id)
    assert usage.known_cost_usd == 5 and usage.llm_calls == 3

    spent, spent_llm, spent_child = budgeted(
        store, [response(10, "child")], [response()], Budget(max_cost_usd=10, on_exceed="pause")
    )
    paused = await spent.run("go")
    assert paused.status == RunStatus.PAUSED and spent_llm.calls == 1
    assert spent_child.calls == 0, "admission measures the root before the bind"
    assert len((await store.get_subtree(paused.run_id)).nodes) == 1


@pytest.mark.parametrize("on_exceed", ["pause", "cancel"])
async def test_a_child_call_crossing_the_limit_stops_the_root_at_its_next_checkpoint(
    store: Any, on_exceed: str
) -> None:
    """The child's own budget events are advisory (``warn``); the parent enforces (Q3)."""

    root, root_llm, child_llm = budgeted(
        store,
        [response(9, "child"), response()],
        [response(2)],
        Budget(max_cost_usd=10, on_exceed=on_exceed),
    )
    result = await root.run("go")
    expected = RunStatus.PAUSED if on_exceed == "pause" else RunStatus.CANCELLED
    assert result.status == expected and root_llm.calls == 1 and child_llm.calls == 1
    tree = await store.get_subtree(result.run_id)
    child = tree.nodes[1].run
    assert child.status == "success"
    assert (await store.subtree_usage(result.run_id)).known_cost_usd == 11
    child_actions = [
        e.data.get("action")
        for e in await store.get_run_events(child.id)
        if e.event_type == "budget.exceeded"
    ]
    assert set(child_actions) <= {"warn"}
    root_exceeded = [
        e for e in await store.get_run_events(result.run_id) if e.event_type == "budget.exceeded"
    ]
    assert root_exceeded


@pytest.mark.parametrize("fault", ["unpriced", "unknown", "write"])
async def test_unpriced_or_unknown_child_usage_stops_an_enforcing_root(
    store: Any, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:

    child_response = {"unpriced": response(None), "unknown": unknown_response()}.get(
        fault, response()
    )
    root, root_llm, child_llm = budgeted(
        store,
        [response(1, "child"), response()],
        [child_response],
        Budget(max_cost_usd=10, on_exceed="pause"),
    )
    if fault == "write":
        original = store.save_llm_interaction

        async def write(*args: Any, **kwargs: Any) -> Any:
            ctx = get_delegation_context()
            if ctx and ctx.delegation_level == 1:
                raise PersistenceError()
            return await original(*args, **kwargs)

        monkeypatch.setattr(store, "save_llm_interaction", write)
    result = await root.run("go")
    assert result.status == RunStatus.PAUSED and root_llm.calls == 1 and child_llm.calls == 1
    pause = await store.get_pause_state(result.run_id)
    if fault == "unpriced":
        assert pause.get("pause_message") is None
        assert (await store.subtree_usage(result.run_id)).unpriced_calls == 1
    else:
        assert pause["pause_message"] == "managed_usage_unknown"


# ---------------------------------------------------------------------------
# Cancel cascade reaching a SingleCall child
# ---------------------------------------------------------------------------


async def test_root_cancel_before_the_child_call_makes_no_provider_call(
    store: Any, engine: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dendrux.runtime import _managed

    original = _managed.bind_child

    async def bind_then_cancel(invocation: Any, *args: Any, **kwargs: Any) -> Any:
        bound = await original(invocation, *args, **kwargs)
        await lead(store, ContextLLM([])).cancel_run(invocation.root_run_id, tenant_id=TENANT)
        return bound

    monkeypatch.setattr(_managed, "bind_child", bind_then_cancel)
    child_llm = ContextLLM([LLMResponse(text=REPORT)])
    root_llm = ContextLLM([call("scorer", {"task": BRIEF}, "r1"), LLMResponse(text="root done")])
    result = await lead(store, root_llm, scorer(child_llm)).run("go", tenant_id=TENANT)
    assert result.status == RunStatus.CANCELLED
    [child] = await rows(engine, AgentRun, AgentRun.parent_run_id == result.run_id)
    assert child.status == "cancelled" and child_llm.calls == 0 and root_llm.calls == 1


@pytest.mark.parametrize("entry", ["run", "stream"])
async def test_root_cancel_during_the_child_call_in_process_is_cooperative(
    store: Any, engine: Any, entry: str
) -> None:
    """``cancel_run`` on the root does not interrupt a subagent's in-flight provider call,
    even in the owning process (the child's task is not tracked by the root's Agent; the
    same holds for a ReAct child). The call finishes, adoption refuses under the cancel,
    and the root ends cancelled without another model call."""
    child_llm = ContextLLM([LLMResponse(text=REPORT)], gate_at=0)
    root_llm = ContextLLM([call("scorer", {"task": BRIEF}, "r1"), LLMResponse(text="root done")])
    root = lead(store, root_llm, scorer(child_llm))
    owner = asyncio.create_task(drive(root, entry))
    await wait_for_gate(child_llm.entered, task=owner)
    [row] = await rows(engine, AgentRun, AgentRun.agent_name == "lead")
    try:
        await root.cancel_run(row.id, tenant_id=TENANT)
        await asyncio.sleep(0.05)
        assert not owner.done() and not child_llm.cancelled
    finally:
        child_llm.gate.set()
    result = await asyncio.wait_for(owner, 30)
    assert result.status == RunStatus.CANCELLED and root_llm.calls == 1
    [child] = await rows(engine, AgentRun, AgentRun.parent_run_id == row.id)
    assert child.status == "success" and child_llm.calls == 1
    [intent] = await rows(engine, ToolCallRecord, ToolCallRecord.agent_run_id == row.id)
    assert intent.outcome != "completed", "a cancelled root never adopts the report"


async def test_caller_task_cancel_during_the_child_call_cancels_the_child(
    store: Any, engine: Any
) -> None:
    """A cancelled caller task is not a crash: the child ends ``cancelled`` (as a ReAct
    child does) and the root later adopts ``managed_child_cancelled``."""
    child_llm = ContextLLM([LLMResponse(text=REPORT)], gate_at=0)
    root_llm = ContextLLM([call("scorer", {"task": BRIEF}, "r1")])
    owner = asyncio.create_task(
        lead(store, root_llm, scorer(child_llm)).run("go", tenant_id=TENANT)
    )
    await wait_for_gate(child_llm.entered, task=owner)
    owner.cancel()
    with pytest.raises(asyncio.CancelledError):
        await owner
    assert child_llm.cancelled
    [row] = await rows(engine, AgentRun, AgentRun.agent_name == "lead")
    [child] = await rows(engine, AgentRun, AgentRun.parent_run_id == row.id)
    assert (row.status, child.status) == ("paused", "cancelled")
    later_llm = ContextLLM([LLMResponse(text="root done")])
    again = ContextLLM([])
    result = await lead(store, later_llm, scorer(again)).resume(row.id, tenant_id=TENANT)
    assert result.status == RunStatus.SUCCESS and again.calls == 0
    [observation] = await observed(later_llm, 0)
    assert json.loads(observation.content)["code"] == "managed_child_cancelled"


@pytest.mark.parametrize("engine", ["sqlite_file", "postgres"], indirect=True)
async def test_root_cancel_from_another_process_during_the_call_is_never_adopted(
    engine: Any,
) -> None:
    """Cooperative: the in-flight call finishes, but adoption refuses under the root's
    cancel, so the root ends cancelled and the parent's model is not called again."""

    child_llm = ContextLLM([LLMResponse(text=REPORT)], gate_at=0)
    root_llm = ContextLLM([call("scorer", {"task": BRIEF}, "r1"), LLMResponse(text="root done")])
    owner_root = lead(store_for_engine(engine), root_llm, scorer(child_llm))
    owner = asyncio.create_task(owner_root.run("go", tenant_id=TENANT))
    await wait_for_gate(child_llm.entered, task=owner)
    [row] = await rows(engine, AgentRun, AgentRun.agent_name == "lead")
    try:
        async with fresh_store(engine) as other:
            await lead(other, ContextLLM([])).cancel_run(row.id, tenant_id=TENANT)
    finally:
        child_llm.gate.set()
    result = await asyncio.wait_for(owner, 30)
    assert result.status == RunStatus.CANCELLED and root_llm.calls == 1
    assert child_llm.calls == 1 and not child_llm.cancelled
    [root_row] = await rows(engine, AgentRun, AgentRun.id == row.id)
    assert root_row.status == "cancelled"
    [intent] = await rows(engine, ToolCallRecord, ToolCallRecord.agent_run_id == row.id)
    assert intent.outcome != "completed", "a cancelled root never adopts the report"
