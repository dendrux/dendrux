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
from dendrux.loops.single import SingleCall
from dendrux.runtime.context import get_delegation_context
from dendrux.store import RunStore
from dendrux.types import LLMResponse, ProviderCapabilities, RunStatus, ToolCall, UsageStats
from tests._helpers.scripted_provider import ScriptedLLM

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
    from dendrux.types import Budget

    child = scorer(ContextLLM([]))
    with pytest.raises(SubagentDefinitionError) as info:
        Agent(name="lead", prompt="p", provider=ContextLLM([]), loop=SingleCall(), subagents=[child])
    assert info.value.reason == "single_call_parent"
    budgeted = scorer(ContextLLM([]), budget=Budget(max_tokens=100))
    with pytest.raises(ValueError, match="child budget"):
        _managed_budget.validate_definitions(
            Agent(name="lead", prompt="p", provider=ContextLLM([]), subagents=[budgeted]),
            agent_type=Agent,
        )


async def test_streaming_a_structured_single_call_child_is_refused(
    store: Any, engine: Any
) -> None:
    root = lead(store, ContextLLM([]), scorer(ContextLLM([]), output_type=Summary))
    with pytest.raises(ValueError, match="cannot stream subagent 'scorer'"):
        async for _ in root.stream("go", tenant_id=TENANT):
            pass
    assert await rows(engine, AgentRun) == []
