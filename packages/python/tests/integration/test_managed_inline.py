"""Managed subagents (0.7, T2b): a finished child returns inline through durable adoption.

``await root.run(brief)`` lets the model call a registered child. The child runs its
own provider and tools on a fresh brief in the same caller task; its terminal
answer becomes the parent's original tool observation through the persisted bind,
the child's terminal outcome, ``submit_and_claim`` and ``complete_tool_call``. The
root then continues to success. Runs on SQLite and on Postgres
(``DENDRUX_TEST_PG_URL``, a private ``*_test`` database).
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from dendrux import Agent, tool
from dendrux.db.models import AgentRun, RunEvent, ToolCallRecord
from dendrux.loops.base import BaseNotifier
from dendrux.loops.single import SingleCall
from dendrux.runtime.context import get_delegation_context
from dendrux.runtime.state import MANAGED_HEADER_KEY
from dendrux.store import RunStore
from dendrux.types import LLMResponse, ProviderCapabilities, ToolCall, UsageStats
from tests._helpers.scripted_provider import ScriptedLLM

BRIEF = "SENTINEL-managed-brief-5c1d"
ROOT_INPUT = "SENTINEL-root-input-90aa"
REPORT = "SENTINEL-child-report-77e3"
TENANT = "t1"


class ContextLLM(ScriptedLLM):
    """A scripted provider that records the run and task each call is served under.

    The delegation ContextVar is the real one the runner sets, so a call made from
    the wrong context or task is visible here as it would be to a real provider.
    """

    def __init__(self, responses: list[LLMResponse], *, usage: int = 0) -> None:
        super().__init__(responses)
        self.served: list[tuple[str | None, int]] = []
        self.usage = usage

    async def complete(self, messages: Any, tools: Any = None, **kwargs: Any) -> LLMResponse:
        ctx = get_delegation_context()
        self.served.append((ctx.run_id if ctx else None, id(asyncio.current_task())))
        response = await super().complete(messages, tools, **kwargs)
        if self.usage:
            response.usage = UsageStats(
                input_tokens=self.usage, output_tokens=self.usage, total_tokens=2 * self.usage
            )
        return response


def call(name: str, params: dict[str, Any], pid: str) -> LLMResponse:
    return LLMResponse(tool_calls=[ToolCall(name=name, params=params, provider_tool_call_id=pid)])


word_calls: list[str] = []


@tool()
async def count_words(text: str) -> int:
    """Count words."""
    word_calls.append(text)
    return len(text.split())


def child_agent(store: Any, llm: Any, name: str = "reviewer", **kwargs: Any) -> Agent:
    return Agent(
        name=name,
        description="Reviews one draft and reports.",
        prompt="You review.",
        provider=llm,
        tools=kwargs.pop("tools", [count_words]),
        **kwargs,
    )


def root_agent(store: Any, llm: Any, *children: Agent, **kwargs: Any) -> Agent:
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


async def test_finished_child_is_adopted_inline(store: Any, engine: Any) -> None:
    """Spec test 5: own provider and tools, fresh brief, report adopted, root succeeds."""
    word_calls.clear()
    child_llm = ContextLLM(
        [call("count_words", {"text": "a b c"}, "c1"), LLMResponse(text=REPORT)], usage=7
    )
    root_llm = ContextLLM(
        [call("reviewer", {"task": BRIEF}, "r1"), LLMResponse(text="root done")], usage=3
    )
    child = child_agent(store, child_llm)
    root = root_agent(store, root_llm, child)

    result = await root.run(ROOT_INPUT, tenant_id=TENANT)

    assert result.status.value == "success"
    assert result.answer == "root done"
    root_id = result.run_id
    children = await rows(engine, AgentRun, AgentRun.parent_run_id == root_id)
    assert len(children) == 1
    child_run = children[0]
    record = await store.get_run(child_run.id)
    assert child_run.status == "success" and record.answer == REPORT
    assert child_run.tenant_id == TENANT and child_run.delegation_level == 1
    assert word_calls == ["a b c"]

    # Fresh brief: the child's first request is its own prompt plus the task only.
    first = [(m.role.value, m.content) for m in child_llm.messages[0]]
    assert first[-1] == ("user", BRIEF)
    assert all(ROOT_INPUT not in content for _, content in first)
    assert [role for role, _ in first] == ["system", "user"]

    # The report is the original call's observation, once.
    observations = [m for m in root_llm.messages[1] if m.role.value == "tool"]
    assert len(observations) == 1
    assert json.loads(observations[0].content) == REPORT
    intent = (await rows(engine, ToolCallRecord, ToolCallRecord.agent_run_id == root_id))[0]
    assert intent.outcome == "completed" and intent.target == "agent"
    assert intent.meta["managed"]["child_run_id"] == child_run.id

    # Same caller task, each run under its own delegation context.
    tasks = {task for _, task in root_llm.served + child_llm.served}
    assert len(tasks) == 1
    assert {run for run, _ in root_llm.served} == {root_id}
    assert {run for run, _ in child_llm.served} == {child_run.id}

    # The root keeps its header; the child records its invocation.
    root_row = (await rows(engine, AgentRun, AgentRun.id == root_id))[0]
    assert root_row.loop_state[MANAGED_HEADER_KEY]["path"] == []
    assert root_row.loop_state.get("managed_wait") is None
    header = child_run.loop_state[MANAGED_HEADER_KEY]
    assert header["root_run_id"] == root_id and header["path"] == ["reviewer"]


async def test_two_calls_in_separate_turns_get_two_children(store: Any, engine: Any) -> None:
    child_llm = ContextLLM([LLMResponse(text="first report"), LLMResponse(text="second report")])
    root_llm = ContextLLM(
        [
            call("reviewer", {"task": "one"}, "r1"),
            call("reviewer", {"task": "two"}, "r2"),
            LLMResponse(text="both done"),
        ]
    )
    root = root_agent(store, root_llm, child_agent(store, child_llm, tools=[]))
    result = await root.run(ROOT_INPUT, tenant_id=TENANT)

    assert result.status.value == "success" and result.answer == "both done"
    children = await rows(engine, AgentRun, AgentRun.parent_run_id == result.run_id)
    answers = [(await store.get_run(c.id)).answer for c in children]
    assert sorted(answers) == ["first report", "second report"]
    assert len({c.id for c in children}) == 2
    observed = [json.loads(m.content) for m in root_llm.messages[2] if m.role.value == "tool"]
    assert observed == ["first report", "second report"]
    intents = await rows(engine, ToolCallRecord, ToolCallRecord.agent_run_id == result.run_id)
    assert sorted(i.outcome for i in intents) == ["completed", "completed"]


class Summary(BaseModel):
    verdict: str
    score: int


class StructuredLLM(ContextLLM):
    capabilities = ProviderCapabilities(
        supports_native_tools=True, supports_tool_call_ids=True, supports_structured_output=True
    )


async def test_structured_child_report(store: Any, engine: Any) -> None:
    child_llm = StructuredLLM([LLMResponse(text='{"verdict": "ok", "score": 3}')])
    child = Agent(
        name="scorer",
        description="Scores a draft.",
        prompt="Score.",
        provider=child_llm,
        loop=SingleCall(),
        output_type=Summary,
    )
    root_llm = ContextLLM([call("scorer", {"task": BRIEF}, "r1"), LLMResponse(text="ok")])
    root = root_agent(store, root_llm, child)
    result = await root.run(ROOT_INPUT, tenant_id=TENANT)
    assert result.status.value == "success"
    observation = next(m for m in root_llm.messages[1] if m.role.value == "tool")
    assert json.loads(observation.content) == {"verdict": "ok", "score": 3}


async def test_root_usage_is_local_and_subtree_usage_includes_children(
    store: Any, engine: Any
) -> None:
    """§5a usage: result.usage is the root's own; the subtree total adds the child once."""
    child_llm = ContextLLM([LLMResponse(text=REPORT)], usage=100)
    root_llm = ContextLLM(
        [call("reviewer", {"task": BRIEF}, "r1"), LLMResponse(text="root done")], usage=1
    )
    root = root_agent(store, root_llm, child_agent(store, child_llm, tools=[]))
    result = await root.run(ROOT_INPUT, tenant_id=TENANT)

    assert result.usage.input_tokens == 2  # two root calls, one token each
    tree = await RunStore(store).get_subtree(result.run_id, tenant_id=TENANT)
    assert tree.usage.input_tokens == 102  # adoption charges nothing
    assert len(tree.nodes) == 2


async def test_root_feed_records_delegation_start_and_outcome(store: Any, engine: Any) -> None:
    """§5a feed: start carries the child id and path; the outcome says succeeded."""
    child_llm = ContextLLM([LLMResponse(text=REPORT)])
    root_llm = ContextLLM([call("reviewer", {"task": BRIEF}, "r1"), LLMResponse(text="ok")])
    root = root_agent(store, root_llm, child_agent(store, child_llm, tools=[]))
    result = await root.run(ROOT_INPUT, tenant_id=TENANT)
    child_id = (await rows(engine, AgentRun, AgentRun.parent_run_id == result.run_id))[0].id

    events = await rows(engine, RunEvent, RunEvent.agent_run_id == result.run_id)
    started = [e for e in events if e.event_type == "tool.started"]
    completed = [e for e in events if e.event_type == "tool.completed"]
    assert len(started) == 1 and len(completed) == 1
    assert started[0].data["child_run_id"] == child_id
    assert started[0].data["agent_path"] == ["reviewer"]
    assert completed[0].data["managed_outcome"] == "succeeded"
    assert completed[0].data["child_run_id"] == child_id
    for event in events:
        assert BRIEF not in json.dumps(event.data or {})
        assert REPORT not in json.dumps(event.data or {})
    child_events = await rows(engine, RunEvent, RunEvent.agent_run_id == child_id)
    assert [e.event_type for e in child_events][0] == "run.started"


class CountingNotifier(BaseNotifier):
    def __init__(self) -> None:
        self.started: list[str] = []
        self.finished: list[str] = []

    async def on_run_started(self, run_id: str, **kwargs: Any) -> None:
        self.started.append(run_id)

    async def on_run_finished(self, run_id: str, *args: Any, **kwargs: Any) -> None:
        self.finished.append(run_id)


async def test_notifier_fires_for_the_child_once(store: Any, engine: Any) -> None:
    """§5a notifier: the call-level notifier sees the child's run id, never twice."""
    child_llm = ContextLLM([LLMResponse(text=REPORT)])
    root_llm = ContextLLM([call("reviewer", {"task": BRIEF}, "r1"), LLMResponse(text="ok")])
    root = root_agent(store, root_llm, child_agent(store, child_llm, tools=[]))
    notifier = CountingNotifier()
    result = await root.run(ROOT_INPUT, tenant_id=TENANT, notifier=notifier)
    child_id = (await rows(engine, AgentRun, AgentRun.parent_run_id == result.run_id))[0].id
    assert notifier.started.count(child_id) == 1
    assert notifier.finished.count(child_id) == 1
    assert notifier.finished.count(result.run_id) == 1


@pytest.mark.parametrize("missing", ["store", "child_store"])
async def test_managed_root_refuses_unsupported_configuration(
    store: Any, engine: Any, missing: str
) -> None:
    child_llm = ContextLLM([LLMResponse(text=REPORT)])
    root_llm = ContextLLM([call("reviewer", {"task": BRIEF}, "r1")])
    if missing == "store":
        root = Agent(
            name="lead",
            prompt="p",
            provider=root_llm,
            subagents=[child_agent(store, child_llm, tools=[])],
        )
        with pytest.raises(Exception, match="persist|Persist"):
            await root.run(ROOT_INPUT)
    else:
        other = type(store)(store._engine) if hasattr(store, "_engine") else object()
        child = child_agent(store, child_llm, tools=[], state_store=other)
        root = root_agent(store, root_llm, child)
        with pytest.raises(ValueError, match="store"):
            await root.run(ROOT_INPUT, tenant_id=TENANT)
    assert root_llm.calls == 0
