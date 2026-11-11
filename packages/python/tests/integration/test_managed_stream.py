"""Managed subagents (0.7, T3): streaming a workflow and its root-routed continuations.

``root.stream()`` drives the same transitions as ``run()``: each run's generator
advances in the caller's task; a subagent's progress is forwarded with its own
run id but never its text; a subagent's external pause ends the stream with the
root-facing pause; a finished subagent is adopted and its parent's continuation
streams on. ``root.resume_stream(root_id)`` routes to the active leaf. Runs on
SQLite and Postgres (``DENDRUX_TEST_PG_URL``, a private ``*_test`` database).
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from dendrux import Agent, tool
from dendrux.db.models import AgentRun, ReactTrace
from dendrux.errors import ManagedChildControlError, RunNotFoundError
from dendrux.loops.base import BaseNotifier
from dendrux.runtime import _managed
from dendrux.runtime.context import get_delegation_context
from dendrux.types import (
    InputValidationError,
    LLMResponse,
    RunEventType,
    RunStatus,
    ToolCall,
    ToolResult,
)
from tests._helpers.scripted_provider import ScriptedLLM

BRIEF = "SENTINEL-stream-brief-61f0"
CHILD_TEXT = "SENTINEL-child-text-a7d2"
GRAND_TEXT = "SENTINEL-grandchild-text-0b3c"
ROOT_FIRST = "Delegating now."
ROOT_DONE = "root done"
TENANT = "t1"


class TaskLLM(ScriptedLLM):
    """Records the run and the asyncio task each call is served in (real ContextVar)."""

    def __init__(self, responses: list[LLMResponse], on_call: Any = None) -> None:
        super().__init__(responses)
        self.served: list[tuple[str | None, int]] = []
        self.on_call = on_call

    async def complete(self, messages: Any, tools: Any = None, **kwargs: Any) -> LLMResponse:
        ctx = get_delegation_context()
        run_id = ctx.run_id if ctx else None
        self.served.append((run_id, id(asyncio.current_task())))
        if self.on_call is not None:
            await self.on_call(self.calls, run_id)
        return await super().complete(messages, tools, **kwargs)


def call(name: str, params: dict[str, Any], pid: str, text: str | None = None) -> LLMResponse:
    return LLMResponse(
        text=text, tool_calls=[ToolCall(name=name, params=params, provider_tool_call_id=pid)]
    )


published: list[str] = []


@tool()
async def publish(text: str) -> str:
    """Publish a draft."""
    published.append(text)
    return f"published {text}"


@tool()
async def count_words(text: str) -> int:
    """Count words."""
    return len(text.split())


@tool(target="client")
async def read_sheet(sheet: str) -> str:
    """Read a sheet in the user's browser."""
    return ""


def agent(name: str, llm: Any, **kwargs: Any) -> Agent:
    return Agent(
        name=name, description=f"The {name}.", prompt=f"You are {name}.", provider=llm, **kwargs
    )


def lead(store: Any, llm: Any, *children: Agent) -> Agent:
    return Agent(
        name="lead", prompt="You lead.", provider=llm, state_store=store, subagents=list(children)
    )


async def rows(engine: Any, model: Any, *where: Any) -> list[Any]:
    async with AsyncSession(engine) as session:
        return list((await session.execute(select(model).where(*where))).scalars().all())


async def child_of(engine: Any, parent_id: str) -> Any:
    found = await rows(engine, AgentRun, AgentRun.parent_run_id == parent_id)
    assert len(found) == 1
    return found[0]


async def observations(engine: Any, run_id: str) -> list[Any]:
    found = await rows(engine, ReactTrace, ReactTrace.agent_run_id == run_id)
    return [t for t in found if t.role == "tool"]


async def collect(stream: Any) -> list[Any]:
    async with stream:
        return [event async for event in stream]


def texts(events: list[Any]) -> str:
    return "".join(e.text or "" for e in events if e.type == RunEventType.TEXT_DELTA)


async def test_stream_drives_children_in_the_callers_task_and_ends_with_root_success(
    store: Any, engine: Any
) -> None:
    child_llm = TaskLLM(
        [call("count_words", {"text": "a b c"}, "c1"), LLMResponse(text=CHILD_TEXT)]
    )
    root_llm = TaskLLM(
        [call("reviewer", {"task": BRIEF}, "r1", text=ROOT_FIRST), LLMResponse(text=ROOT_DONE)]
    )
    root = lead(store, root_llm, agent("reviewer", child_llm, tools=[count_words]))

    stream = root.stream("go", tenant_id=TENANT)
    events = await collect(stream)

    root_id = events[0].run_id
    assert events[0].type == RunEventType.RUN_STARTED
    assert events[-1].type == RunEventType.RUN_COMPLETED
    final = events[-1].run_result
    assert (final.run_id, final.status, final.answer) == (root_id, RunStatus.SUCCESS, ROOT_DONE)
    assert stream.result is final and stream.run_id == root_id
    # Exactly one terminal event and one start: the workflow is one stream.
    lifecycle = [
        e.type
        for e in events
        if e.type
        in (
            RunEventType.RUN_STARTED,
            RunEventType.RUN_RESUMED,
            RunEventType.RUN_COMPLETED,
            RunEventType.RUN_PAUSED,
            RunEventType.RUN_ERROR,
            RunEventType.RUN_CANCELLED,
        )
    ]
    assert lifecycle == [RunEventType.RUN_STARTED, RunEventType.RUN_COMPLETED]

    # Root text streams live before and after the subagent; no child text is concatenated.
    assert texts(events) == ROOT_FIRST + ROOT_DONE
    assert CHILD_TEXT not in texts(events)

    # The child's progress is forwarded with its own run id.
    child = await child_of(engine, root_id)
    child_tools = [e for e in events if e.run_id == child.id]
    assert {e.type for e in child_tools} >= {RunEventType.TOOL_RESULT}
    assert all(e.type != RunEventType.TEXT_DELTA for e in child_tools)
    assert all(e.run_id in (None, root_id) for e in events if e.tool_name in ("reviewer",))

    # Same caller task for every provider call; adopted once.
    tasks = {task for _, task in root_llm.served + child_llm.served}
    assert len(tasks) == 1
    adopted = await observations(engine, root_id)
    assert len(adopted) == 1 and json.loads(adopted[0].content) == CHILD_TEXT
    assert child.status == "success"


def _depth_two(store: Any, *, restarted: bool = False) -> Agent:
    """lead -> reviewer -> auditor (approval). ``restarted``: a fresh process whose
    providers continue after the first turn of each run."""
    skip = 1 if restarted else 0
    auditor = agent(
        "auditor",
        TaskLLM([call("publish", {"text": "g"}, "g1"), LLMResponse(text=GRAND_TEXT)][skip:]),
        tools=[publish],
        require_approval=["publish"],
    )
    reviewer = agent(
        "reviewer",
        TaskLLM([call("auditor", {"task": "audit"}, "c1"), LLMResponse(text=CHILD_TEXT)][skip:]),
        subagents=[auditor],
    )
    return lead(
        store,
        TaskLLM([call("reviewer", {"task": BRIEF}, "r1"), LLMResponse(text=ROOT_DONE)][skip:]),
        reviewer,
    )


async def test_stream_pause_then_resume_stream_unwinds_depth_two(store: Any, engine: Any) -> None:
    published.clear()
    root = _depth_two(store)
    events = await collect(root.stream("go", tenant_id=TENANT))

    paused = events[-1]
    assert paused.type == RunEventType.RUN_PAUSED
    result = paused.run_result
    root_id = result.run_id
    middle = await child_of(engine, root_id)
    leaf = await child_of(engine, middle.id)
    assert result.status == RunStatus.WAITING_APPROVAL
    assert result.pending.origin_run_id == leaf.id
    assert result.pending.agent_path == ["lead", "reviewer", "auditor"]
    assert result.pause_id == leaf.pause_data["pause_id"]

    # A fresh Agent (a restart) resumes the workflow through the root id.
    restarted = _depth_two(store, restarted=True)
    resumed = await collect(
        restarted.resume_stream(root_id, tenant_id=TENANT, expected_pause_id=result.pause_id)
    )
    assert resumed[0].type == RunEventType.RUN_RESUMED and resumed[0].run_id == root_id
    final = resumed[-1]
    assert final.type == RunEventType.RUN_COMPLETED
    assert (final.run_result.run_id, final.run_result.status) == (root_id, RunStatus.SUCCESS)
    assert texts(resumed) == ROOT_DONE
    assert published == ["g"]
    for run_id, report in ((root_id, CHILD_TEXT), (middle.id, GRAND_TEXT)):
        adopted = await observations(engine, run_id)
        assert [json.loads(o.content) for o in adopted] == [report]


@pytest.mark.parametrize("mode", ["tool_results", "user_input"])
async def test_resume_stream_payloads_route_to_the_leaf(
    store: Any, engine: Any, mode: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    if mode == "user_input":
        from tests.integration.test_managed_routing import patch_questions

        patch_questions(monkeypatch)
        first = LLMResponse(text="Which draft?")
        tools: list[Any] = [count_words]
    else:
        first = call("read_sheet", {"sheet": "S1"}, "c1")
        tools = [read_sheet]
    child = agent("reviewer", TaskLLM([first, LLMResponse(text=CHILD_TEXT)]), tools=tools)
    root = lead(
        store,
        TaskLLM([call("reviewer", {"task": BRIEF}, "r1"), LLMResponse(text=ROOT_DONE)]),
        child,
    )
    paused = (await collect(root.stream("go", tenant_id=TENANT)))[-1].run_result
    assert paused.pending.origin_run_id is not None

    kwargs: dict[str, Any] = {"tenant_id": TENANT, "expected_pause_id": paused.pause_id}
    if mode == "tool_results":
        call_id = paused.pending.tool_calls[0].id
        kwargs["tool_results"] = [ToolResult(name="read_sheet", call_id=call_id, payload='"cells"')]
    else:
        kwargs["user_input"] = "the second"
    events = await collect(root.resume_stream(paused.run_id, **kwargs))
    assert events[-1].type == RunEventType.RUN_COMPLETED
    assert events[-1].run_result.answer == ROOT_DONE
    assert CHILD_TEXT not in texts(events)


class CrashError(Exception):
    """Stands in for a process dying at a chosen boundary."""


async def test_resume_stream_adopts_a_finished_child_without_a_token(
    store: Any, engine: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    child = agent("reviewer", TaskLLM([LLMResponse(text=CHILD_TEXT)]), tools=[])
    root = lead(
        store,
        TaskLLM([call("reviewer", {"task": BRIEF}, "r1"), LLMResponse(text=ROOT_DONE)]),
        child,
    )
    original = _managed.claim_adoption
    crashed: list[bool] = []

    async def crash_once(*args: Any, **kwargs: Any) -> Any:
        if not crashed:
            crashed.append(True)
            raise CrashError
        return await original(*args, **kwargs)

    monkeypatch.setattr(_managed, "claim_adoption", crash_once)
    with pytest.raises(CrashError):
        await root.run("go", tenant_id=TENANT)
    root_row = (await rows(engine, AgentRun, AgentRun.parent_run_id.is_(None)))[0]
    assert root_row.status == "paused"
    assert (await child_of(engine, root_row.id)).status == "success"

    events = await collect(root.resume_stream(root_row.id, tenant_id=TENANT))
    assert events[0].type == RunEventType.RUN_RESUMED and events[0].run_id == root_row.id
    assert events[-1].type == RunEventType.RUN_COMPLETED
    assert events[-1].run_result.answer == ROOT_DONE
    assert texts(events) == ROOT_DONE
    assert len(await observations(engine, root_row.id)) == 1


async def test_routing_refusals_raise_before_any_claim(store: Any, engine: Any) -> None:
    root = _depth_two(store)
    paused = (await collect(root.stream("go", tenant_id=TENANT)))[-1].run_result
    root_id = paused.run_id
    middle = await child_of(engine, root_id)
    leaf = await child_of(engine, middle.id)
    attempts = [
        (
            RunNotFoundError,
            root.resume_stream(root_id, tenant_id="other", expected_pause_id=paused.pause_id),
        ),
        (
            ManagedChildControlError,
            root.resume_stream(middle.id, tenant_id=TENANT, expected_pause_id=paused.pause_id),
        ),
        (InputValidationError, root.resume_stream(root_id, tenant_id=TENANT)),
    ]
    for error, stream in attempts:
        with pytest.raises(error):
            async with stream:
                async for _ in stream:
                    pass
    rows_after = {r.id: (r.status, r.claim_epoch) for r in await rows(engine, AgentRun)}
    assert rows_after[root_id][0] == "paused"
    assert rows_after[leaf.id][0] == "waiting_approval"
    assert rows_after[middle.id][0] == "paused"


async def test_a_subagent_with_guardrails_cannot_be_streamed(store: Any) -> None:
    from dendrux.guardrails import PII

    child = agent("reviewer", TaskLLM([]), guardrails=[PII()])
    root = lead(store, TaskLLM([]), child)
    with pytest.raises(ValueError):
        root.stream("go", tenant_id=TENANT)
    with pytest.raises(ValueError):
        root.resume_stream("01ZZZZZZZZZZZZZZZZZZZZZZZZ", tenant_id=TENANT)


async def test_closing_the_workflow_stream_cancels_the_root_and_active_child(
    store: Any, engine: Any
) -> None:
    child = agent(
        "reviewer",
        TaskLLM([call("count_words", {"text": "a"}, "c1"), LLMResponse(text=CHILD_TEXT)]),
        tools=[count_words],
    )
    root = lead(
        store,
        TaskLLM([call("reviewer", {"task": BRIEF}, "r1"), LLMResponse(text=ROOT_DONE)]),
        child,
    )
    stream = root.stream("go", tenant_id=TENANT)
    root_id = None
    async with stream:
        async for event in stream:
            root_id = root_id or event.run_id
            if event.type == RunEventType.TOOL_RESULT and event.run_id not in (None, root_id):
                break  # abandon while the child streams
    child_row = await child_of(engine, root_id)
    root_row = (await rows(engine, AgentRun, AgentRun.id == root_id))[0]
    assert child_row.status == "cancelled"
    assert root_row.status == "cancelled"
    assert stream.result is not None and stream.result.status == RunStatus.CANCELLED


class HookCounter(BaseNotifier):
    """Counts every hook per run; a second delivery route would double them."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, Any]] = []

    async def on_run_started(self, run_id: str, **kwargs: Any) -> None:
        self.calls.append(("run_started", run_id, None))

    async def on_llm_call_completed(
        self, run_id: str, response: Any, iteration: int, **kw: Any
    ) -> None:
        self.calls.append(("llm", run_id, iteration))

    async def on_tool_completed(self, run_id: str, call: Any, result: Any, iteration: int) -> None:
        self.calls.append(("tool", run_id, call.id))

    async def on_run_finished(self, run_id: str, *args: Any, **kwargs: Any) -> None:
        self.calls.append(("finished", run_id, None))


@pytest.mark.parametrize("entry", ["stream", "resume_stream", "resume", "submit_approval"])
async def test_call_level_notifier_reaches_each_child_once(
    store: Any, engine: Any, entry: str
) -> None:
    """§5a: stream and continuation calls fire for children with their own run ids."""
    published.clear()
    root = _depth_two(store)
    observer = HookCounter()
    if entry == "stream":
        events = await collect(root.stream("go", tenant_id=TENANT, notifier=observer))
        paused = events[-1].run_result
        runs = await rows(engine, AgentRun)
        assert {c[1] for c in observer.calls if c[0] == "run_started"} == {r.id for r in runs}
    else:
        paused = (await collect(root.stream("go", tenant_id=TENANT)))[-1].run_result
        kwargs = {"tenant_id": TENANT, "expected_pause_id": paused.pause_id, "notifier": observer}
        if entry == "resume_stream":
            final = (await collect(root.resume_stream(paused.run_id, **kwargs)))[-1].run_result
        elif entry == "resume":
            final = await root.resume(paused.run_id, **kwargs)
        else:
            final = await root.submit_approval(paused.run_id, approved=True, **kwargs)
        assert final.status == RunStatus.SUCCESS
        runs = await rows(engine, AgentRun)
        for run in runs:  # leaf, middle and root each continue once and finish once
            assert observer.calls.count(("run_started", run.id, None)) == 1
            assert observer.calls.count(("finished", run.id, None)) == 1
    duplicates = {c for c in observer.calls if observer.calls.count(c) > 1}
    assert {c for c in duplicates if c[0] != "run_started"} == set()
    assert paused.run_id in {c[1] for c in observer.calls}


async def test_a_continued_parent_streams_its_next_subagent(store: Any, engine: Any) -> None:
    """After adoption the root's continuation streams on and may hand off again."""
    reviewer = agent("reviewer", TaskLLM([LLMResponse(text=CHILD_TEXT)]), tools=[])
    writer = agent("writer", TaskLLM([LLMResponse(text="written")]), tools=[])
    root_llm = TaskLLM(
        [
            call("reviewer", {"task": BRIEF}, "r1"),
            call("writer", {"task": "write"}, "r2", text="Now writing."),
            LLMResponse(text=ROOT_DONE),
        ]
    )
    root = lead(store, root_llm, reviewer, writer)
    events = await collect(root.stream("go", tenant_id=TENANT))
    final = events[-1]
    assert final.type == RunEventType.RUN_COMPLETED
    assert final.run_result.answer == ROOT_DONE
    assert texts(events) == "Now writing." + ROOT_DONE
    root_id = final.run_result.run_id
    children = await rows(engine, AgentRun, AgentRun.parent_run_id == root_id)
    assert sorted(c.agent_name for c in children) == ["reviewer", "writer"]
    assert all(c.status == "success" for c in children)
    adopted = [json.loads(o.content) for o in await observations(engine, root_id)]
    assert sorted(adopted) == sorted([CHILD_TEXT, "written"])
    tasks = {task for _, task in root_llm.served + reviewer.provider.served}
    assert len(tasks) == 1


# ---------------------------------------------------------------------------
# Fix round 1, R2: closing right after the opening event never strands a claim
# ---------------------------------------------------------------------------


async def _first_event_then_close(stream: Any) -> Any:
    async with stream:
        async for event in stream:
            return event
    return None


async def test_close_after_opening_event_with_a_claimed_adoption_cancels_the_workflow(
    store: Any, engine: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R -> C -> terminal G not yet adopted. resume_stream(R) adopts G into C (C now
    claimed), yields RUN_RESUMED(R); the consumer closes: C is not left RUNNING."""
    original = _managed.claim_adoption
    crashed: list[bool] = []

    async def crash_once(*args: Any, **kwargs: Any) -> Any:
        if not crashed:
            crashed.append(True)
            raise CrashError
        return await original(*args, **kwargs)

    monkeypatch.setattr(_managed, "claim_adoption", crash_once)
    auditor = agent("auditor", TaskLLM([LLMResponse(text=GRAND_TEXT)]), tools=[])
    reviewer = agent(
        "reviewer",
        TaskLLM([call("auditor", {"task": "audit"}, "c1"), LLMResponse(text=CHILD_TEXT)]),
        subagents=[auditor],
    )
    root = lead(store, TaskLLM([call("reviewer", {"task": BRIEF}, "r1")]), reviewer)
    with pytest.raises(CrashError):
        await root.run("go", tenant_id=TENANT)
    root_row = (await rows(engine, AgentRun, AgentRun.parent_run_id.is_(None)))[0]
    middle = await child_of(engine, root_row.id)
    leaf = await child_of(engine, middle.id)
    assert (root_row.status, middle.status, leaf.status) == ("paused", "paused", "success")

    first = await _first_event_then_close(root.resume_stream(root_row.id, tenant_id=TENANT))

    assert first.type == RunEventType.RUN_RESUMED and first.run_id == root_row.id
    after = {r.id: r for r in await rows(engine, AgentRun)}
    assert after[middle.id].status == "cancelled"  # no live claim left without an owner
    assert after[root_row.id].status == "cancelled"
    assert len(await observations(engine, middle.id)) == 1  # the adoption itself stays
    assert reviewer.provider.calls == 1  # the claimed continuation never ran


async def test_close_after_opening_event_for_a_leaf_pause_cancels_the_workflow(
    store: Any, engine: Any
) -> None:
    root = _depth_two(store)
    paused = (await collect(root.stream("go", tenant_id=TENANT)))[-1].run_result
    restarted = _depth_two(store, restarted=True)
    before = list(published)

    first = await _first_event_then_close(
        restarted.resume_stream(paused.run_id, tenant_id=TENANT, expected_pause_id=paused.pause_id)
    )

    assert first.type == RunEventType.RUN_RESUMED
    statuses = {r.agent_name: r.status for r in await rows(engine, AgentRun)}
    assert statuses == {"lead": "cancelled", "reviewer": "cancelled", "auditor": "cancelled"}
    assert published == before  # the approved tool never ran
