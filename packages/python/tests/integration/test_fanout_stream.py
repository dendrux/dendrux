# ruff: noqa: F811 - the shared adoption fixtures are imported by name
"""0.8 streaming fan-out: owner tasks feed one bounded queue that the root stream drains.

The root's own generator stays in the caller's task. Each child of a fan-out runs in its
owner task with the non-streaming runner; its tool progress reaches the root stream as
structural events tagged with its own run id. The consumer clears ``wake``, drains the
queue, observes "every owner done", else waits on ``wake``: no queue waiter task, no
completion flag set by the waiting caller. Closing or cancelling the stream runs the
driver's exit path (custody closed, owners cancelled once, finalized and joined).
Runs on SQLite and Postgres (``DENDRUX_TEST_PG_URL``, a private ``*_test`` database).
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from dendrux import Agent, tool
from dendrux.loops.base import BaseNotifier
from dendrux.loops.single import SingleCall
from dendrux.runtime import _managed_fanout, _managed_stream
from dendrux.runtime.context import get_delegation_context
from dendrux.types import LLMResponse, RunEventType, RunStatus, ToolCall, ToolResult
from tests._helpers.fanout_effects import FanoutEffects
from tests._helpers.gated_provider import GatedStreamLLM
from tests._helpers.gates import wait_for_gate
from tests._helpers.scripted_provider import ScriptedLLM
from tests.integration.test_fanout_driver import (  # noqa: F401 - fixtures used by name
    assert_durable_reports,
    assert_parent_turn,
    live_adoption,
    round2_children,
    v4_continuation,
)

pytestmark = pytest.mark.parametrize("engine", ["sqlite_file", "postgres"], indirect=True)

TENANT = "t1"
DEADLOCK_S = 5
CHILD_EVENTS = frozenset({"tool_use_start", "tool_use_end", "tool_result"})


class TaskLLM(ScriptedLLM):
    """Records the run and the asyncio task each call is served in (real ContextVar)."""

    def __init__(self, responses: list[LLMResponse]) -> None:
        super().__init__(responses)
        self.served: list[tuple[str | None, asyncio.Task[Any] | None]] = []

    async def complete(self, messages: Any, tools: Any = None, **kwargs: Any) -> LLMResponse:
        ctx = get_delegation_context()
        self.served.append((ctx.run_id if ctx else None, asyncio.current_task()))
        return await super().complete(messages, tools, **kwargs)


@tool()
async def note(text: str) -> str:
    """Write a note."""
    return f"noted {text}"


@tool(target="client")
async def read() -> str:
    """Read something in the browser."""
    return ""


def worker(name: str, llm: Any, **kwargs: Any) -> Agent:
    return Agent(name=name, description=f"The {name}.", prompt="p", provider=llm, **kwargs)


def noting(name: str, calls: int = 1, *, provider: Any = None) -> Agent:
    """A ReAct child that writes ``calls`` notes in one turn, then answers its name."""
    turn = LLMResponse(
        tool_calls=[
            ToolCall("note", {"text": f"{name}-{i}"}, id=f"{name}-{i}") for i in range(calls)
        ]
    )
    return worker(name, provider or TaskLLM([turn, LLMResponse(text=name)]), tools=[note])


def lead(store: Any, llm: Any, children: list[Agent], **kwargs: Any) -> Agent:
    return Agent(
        name="lead", prompt="p", provider=llm, state_store=store, subagents=children, **kwargs
    )


def batch(children: list[Agent]) -> LLMResponse:
    return LLMResponse(tool_calls=[ToolCall(c.name, {"task": "go"}) for c in children])


async def collect(stream: Any, *, timeout: float = DEADLOCK_S) -> list[Any]:
    async def run() -> list[Any]:
        async with stream:
            return [event async for event in stream]

    return await asyncio.wait_for(run(), timeout)


async def close_all(*agents: Agent) -> None:
    for agent in agents:
        await agent.close()


def child_events(events: list[Any], root_id: str) -> list[Any]:
    kind = lambda e: getattr(e.type, "value", e.type)  # noqa: E731
    return [e for e in events if kind(e) in CHILD_EVENTS and e.run_id not in (None, root_id)]


async def child_ids(store: Any, parent_id: str, names: list[str]) -> list[str]:
    rows = await store.list_runs(tenant_id=TENANT, parent_run_id=parent_id)
    by_name = {r.agent_name: r.id for r in rows}
    return [by_name[n] for n in names]


def adopted_calls(adoptions: list[Any], index: int = 0) -> list[ToolCall]:
    """The call ids one observed adoption completed, in call order."""
    return [ToolCall("subagent", {}, id=call_id) for call_id in adoptions[index][1]]


def captured_runs(monkeypatch: Any) -> list[Any]:
    """Every FanoutRun the driver starts, in start order."""
    runs: list[Any] = []
    original = _managed_fanout.start_fanout

    async def start(*args: Any, **kwargs: Any) -> Any:
        run = await original(*args, **kwargs)
        runs.append(run)
        return run

    monkeypatch.setattr(_managed_fanout, "start_fanout", start)
    return runs


def owner_tasks_left() -> list[asyncio.Task[Any]]:
    return [t for t in asyncio.all_tasks() if t.get_name().startswith("fanout:") and not t.done()]


# ---------------------------------------------------------------------------
# Entry paths: stream, resume_stream, a streamed approval
# ---------------------------------------------------------------------------


async def test_stream_fans_out_in_owner_tasks_and_continues_the_root(store, live_adoption):
    effects = FanoutEffects()
    children = [noting("alpha"), noting("beta")]
    llm = TaskLLM([batch(children), LLMResponse(text="done")])
    root = lead(store, effects.provider("root", llm), children)
    caller = asyncio.current_task()
    try:
        events = await collect(root.stream("go", tenant_id=TENANT))
        final = events[-1]
        assert final.type == RunEventType.RUN_COMPLETED
        assert final.run_result.status == RunStatus.SUCCESS
        assert final.run_result.answer == "done"
        root_id = final.run_result.run_id
        assert events[0].type == RunEventType.RUN_STARTED and events[0].run_id == root_id
        ids = await child_ids(store, root_id, ["alpha", "beta"])
        progress = child_events(events, root_id)
        for child_id, name in zip(ids, ["alpha", "beta"], strict=True):
            own = [e for e in progress if e.run_id == child_id]
            assert [getattr(e.type, "value", e.type) for e in own] == [
                "tool_use_start",
                "tool_use_end",
                "tool_result",
            ]
            assert all(e.tool_call is None or e.tool_call.params == {} for e in own)
            assert own[-1].tool_result.payload == "" and own[-1].tool_call_id == f"{name}-0"
        # The root's provider stays in the caller's task; each child is served in its
        # own owner task, never the caller's.
        assert {task for _, task in llm.served} == {caller}
        tasks = [children[i].provider.served[0][1] for i in range(2)]
        assert caller not in tasks and tasks[0] is not tasks[1]
        assert [t for _, t in children[0].provider.served] == [tasks[0]] * 2
        assert live_adoption and live_adoption[0][0] == root_id and live_adoption[0][2] is caller
        await assert_durable_reports(
            store, root_id, adopted_calls(live_adoption), ["alpha", "beta"]
        )
        assert not owner_tasks_left()
    finally:
        await close_all(root, *children)


async def test_resume_stream_fans_out_after_a_client_pause(store, live_adoption):
    effects = FanoutEffects()
    children = round2_children(effects)
    llm = ScriptedLLM(
        [
            LLMResponse(tool_calls=[ToolCall("read", {})]),
            batch(children),
            LLMResponse(text="done"),
        ]
    )
    root = lead(store, effects.provider("root", llm), children, tools=[read])
    try:
        paused = (await collect(root.stream("go", tenant_id=TENANT)))[-1].run_result
        assert paused.status == RunStatus.WAITING_CLIENT_TOOL
        reply = [ToolResult("read", paused.pending.tool_calls[0].id, '"answer"')]
        events = await collect(
            root.resume_stream(
                paused.run_id,
                tool_results=reply,
                tenant_id=TENANT,
                expected_pause_id=paused.pause_id,
            )
        )
        final = events[-1].run_result
        assert events[0].type == RunEventType.RUN_RESUMED
        assert (final.status, final.answer) == (RunStatus.SUCCESS, "done")
        assert [effects.counts().provider_calls(c.name) for c in children] == [1, 1]
        assert final.pendings == () and final.pending is None and final.pause_id is None
        await assert_durable_reports(
            store, final.run_id, adopted_calls(live_adoption), ["alpha", "beta"]
        )
    finally:
        await close_all(root, *children)


@pytest.mark.parametrize("entry", ["stream", "resume_stream"])
async def test_streamed_approval_releases_two_held_children_after_the_server_call(
    store, live_adoption, entry
):
    """A held multi-child batch is no longer refused on the stream path."""
    effects = FanoutEffects()
    children = round2_children(effects)
    authorized: list[bool] = []

    @tool()
    async def authorize() -> str:
        assert sum(effects.counts().provider_calls(c.name) for c in children) == 0
        authorized.append(True)
        return "approved"

    calls = [ToolCall(c.name, {"task": "go"}) for c in children]
    llm = ScriptedLLM(
        [
            LLMResponse(tool_calls=[calls[0], ToolCall("authorize", {}), calls[1]]),
            LLMResponse(text="done"),
        ]
    )
    root = lead(
        store,
        effects.provider("root", llm),
        children,
        tools=[authorize],
        require_approval=["authorize"],
    )
    try:
        if entry == "stream":
            paused = (await collect(root.stream("go", tenant_id=TENANT)))[-1].run_result
        else:
            paused = await root.run("go", tenant_id=TENANT)
        assert paused.status == RunStatus.WAITING_APPROVAL
        assert sum(effects.counts().provider_calls(c.name) for c in children) == 0
        events = await collect(
            root.resume_stream(paused.run_id, tenant_id=TENANT, expected_pause_id=paused.pause_id)
        )
        final = events[-1].run_result
        assert (final.status, final.answer) == (RunStatus.SUCCESS, "done")
        assert authorized == [True]
        assert [effects.counts().provider_calls(c.name) for c in children] == [1, 1]
        assert live_adoption[0][1] == tuple(c.id for c in calls)
        await assert_durable_reports(store, final.run_id, calls, ["alpha", "beta"])
        assert_parent_turn(llm, calls, ["alpha", "beta"])
    finally:
        await close_all(root, *children)


# ---------------------------------------------------------------------------
# Queue protocol (Q1-Q6)
# ---------------------------------------------------------------------------


async def _revision2(self: Any, run: Any) -> Any:
    """The revision-2 schedule: a queue waiter, then a join that sets completion."""
    done = asyncio.Event()
    while not done.is_set():
        yield await self.queue.get()
    await asyncio.wait(run.owners)
    done.set()


async def _join_first(self: Any, run: Any) -> Any:
    """Join every owner before draining: producers blocked on a full queue never finish."""
    await asyncio.wait(run.owners)
    while not self.queue.empty():
        yield self.queue.get_nowait()


@pytest.mark.parametrize("schedule", ["current", "revision2"])
async def test_q1_zero_event_children_end_the_loop_and_fan_in(
    store, live_adoption, monkeypatch, schedule
):
    if schedule == "revision2":
        monkeypatch.setattr(_managed_stream.FanoutStream, "_pending", _revision2)
    effects = FanoutEffects()
    children = round2_children(effects)  # SingleCall children forward no events
    root = lead(
        store,
        effects.provider("root", ScriptedLLM([batch(children), LLMResponse(text="done")])),
        children,
    )
    try:
        if schedule == "revision2":
            with pytest.raises(TimeoutError):
                await collect(root.stream("go", tenant_id=TENANT))
            assert not owner_tasks_left()
            return
        events = await collect(root.stream("go", tenant_id=TENANT))
        final = events[-1].run_result
        assert (final.status, final.answer) == (RunStatus.SUCCESS, "done")
        assert child_events(events, final.run_id) == []
        assert len(live_adoption) == 1
        await assert_durable_reports(
            store, final.run_id, adopted_calls(live_adoption), ["alpha", "beta"]
        )
    finally:
        await close_all(root, *children)


def _full_gate(monkeypatch: Any) -> asyncio.Event:
    """Set once a producer finds the stream queue full (it is about to block)."""
    full = asyncio.Event()
    original = _managed_stream.FanoutStream._put

    async def put(self: Any, event: Any) -> None:
        if self.queue.full():
            full.set()
        await original(self, event)

    monkeypatch.setattr(_managed_stream.FanoutStream, "_put", put)
    return full


class CompletionOrder(BaseNotifier):
    """The order each run's tools completed in, as its owner task reported them."""

    def __init__(self) -> None:
        self.completed: dict[str, list[str]] = {}

    async def on_tool_completed(self, run_id: str, call: Any, result: Any, iteration: int) -> None:
        self.completed.setdefault(run_id, []).append(call.id)


Q2_CALLS = 50  # 2 children x 50 notes x 3 events = 300 > 256


@pytest.mark.parametrize("schedule", ["current", "join_first"])
async def test_q2_more_than_the_queue_bound_before_the_next_step(
    store, live_adoption, monkeypatch, schedule
):
    if schedule == "join_first":
        monkeypatch.setattr(_managed_stream.FanoutStream, "_pending", _join_first)
    full = _full_gate(monkeypatch)
    children = [noting("alpha", Q2_CALLS), noting("beta", Q2_CALLS)]
    root = lead(store, ScriptedLLM([batch(children), LLMResponse(text="done")]), children)
    observer = CompletionOrder()

    async def consume() -> list[Any]:
        events: list[Any] = []
        async with root.stream("go", tenant_id=TENANT, notifier=observer) as stream:
            async for event in stream:
                events.append(event)
                if len(child_events(events, events[0].run_id)) == 1:
                    # Hold the consumer until producers block on the bound.
                    await asyncio.wait_for(full.wait(), DEADLOCK_S)
        return events

    try:
        if schedule == "join_first":
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(consume(), DEADLOCK_S)
            assert not owner_tasks_left()
            return
        events = await asyncio.wait_for(consume(), 4 * DEADLOCK_S)
        assert full.is_set()
        final = events[-1].run_result
        assert (final.status, final.answer) == (RunStatus.SUCCESS, "done")
        progress = child_events(events, final.run_id)
        assert len(progress) == 2 * Q2_CALLS * 3
        # Nothing dropped, and each child's results arrive in the order it produced them.
        for child_id in await child_ids(store, final.run_id, ["alpha", "beta"]):
            results = [
                e.tool_call_id
                for e in progress
                if e.type == RunEventType.TOOL_RESULT and e.run_id == child_id
            ]
            assert results == observer.completed[child_id]
            assert len(results) == Q2_CALLS
    finally:
        await close_all(root, *children)


async def test_q3_a_final_producer_error_follows_every_event(store, live_adoption, monkeypatch):
    from dendrux.runtime import _run_execute

    children = [noting("alpha"), noting("beta")]
    real = _run_execute._run_once
    alpha_done = asyncio.Event()

    async def execute(*args: Any, **kwargs: Any) -> Any:
        branch = kwargs["_preclaimed"].branch
        if branch.child.name == "alpha":
            try:
                return await real(*args, **kwargs)
            finally:
                alpha_done.set()
        await alpha_done.wait()
        events = kwargs["extra_notifier"]
        call = ToolCall("note", {}, id="beta-last")
        await events.on_tool_started(branch.child_run_id, call, 1)
        raise ValueError("SENTINEL-q3-branch")

    monkeypatch.setattr(_run_execute, "_run_once", execute)
    root = lead(store, ScriptedLLM([batch(children), LLMResponse(text="done")]), children)
    try:
        events = await collect(root.stream("go", tenant_id=TENANT))
        final = events[-1]
        assert final.type == RunEventType.RUN_ERROR
        root_id = events[0].run_id
        assert final.run_result.run_id == root_id
        assert final.run_result.meta["error_type"] == "ValueError"
        progress = child_events(events, root_id)
        assert [e.tool_call_id for e in progress][-2:] == ["beta-last", "beta-last"]
        assert len(progress) == 5  # alpha's three, then beta's two before it failed
        assert events.index(progress[-1]) == len(events) - 2
        assert not owner_tasks_left()
    finally:
        await close_all(root, *children)


async def test_q4_nested_fanout_events_reach_the_root_queue(store, live_adoption, monkeypatch):
    runs = captured_runs(monkeypatch)
    inner = [noting("scout"), noting("analyst")]
    planner = worker(
        "planner",
        ScriptedLLM([batch(inner), LLMResponse(text="planned")]),
        subagents=inner,
    )
    writer = worker("writer", ScriptedLLM([LLMResponse(text="written")]), loop=SingleCall())
    children = [planner, writer]
    root = lead(store, ScriptedLLM([batch(children), LLMResponse(text="done")]), children)
    try:
        events = await collect(root.stream("go", tenant_id=TENANT))
        final = events[-1].run_result
        assert (final.status, final.answer) == (RunStatus.SUCCESS, "done")
        planner_id, _ = await child_ids(store, final.run_id, ["planner", "writer"])
        nested = await child_ids(store, planner_id, ["scout", "analyst"])
        progress = child_events(events, final.run_id)
        for child_id, name in zip(nested, ["scout", "analyst"], strict=True):
            own = [e for e in progress if e.run_id == child_id]
            assert [e.tool_call_id for e in own] == [f"{name}-0"] * 3
        # One driver per level, each joined before its parent settles.
        assert len(runs) == 2
        assert all(run.owners_done() and len(run.owners) == 2 for run in runs)
        assert {len(adopted[1]) for adopted in live_adoption} == {2}
        assert len(live_adoption) == 2
    finally:
        await close_all(root, *children, *inner)


async def test_q5_slow_consumer_receives_every_event(store, live_adoption):
    children = [noting("alpha", 5), noting("beta", 5)]
    root = lead(store, ScriptedLLM([batch(children), LLMResponse(text="done")]), children)
    try:
        events: list[Any] = []

        async def consume() -> None:
            async with root.stream("go", tenant_id=TENANT) as stream:
                async for event in stream:
                    events.append(event)
                    await asyncio.sleep(0.005)

        await asyncio.wait_for(consume(), 4 * DEADLOCK_S)
        final = events[-1].run_result
        assert final.status == RunStatus.SUCCESS
        assert len(child_events(events, final.run_id)) == 2 * 5 * 3
    finally:
        await close_all(root, *children)


def gated_child(name: str, *, before: int = 1) -> tuple[Agent, GatedStreamLLM]:
    """A ReAct child that notes ``before`` times, then blocks inside its next model call."""
    turns = [
        LLMResponse(tool_calls=[ToolCall("note", {"text": name}, id=f"{name}-{i}")])
        for i in range(before)
    ]
    provider = GatedStreamLLM(before=turns)
    provider.gate_complete = True
    return worker(name, provider, tools=[note]), provider


def closed_in_its_task(provider: GatedStreamLLM) -> bool:
    """A single child streams in the caller; a fan-out child's call opens, waits and
    closes in one owner task."""
    if not provider.complete_tasks:
        return provider.closed or not provider.first_sent.is_set()
    return provider.complete_closed and len(set(provider.complete_tasks)) == 1


async def _rows(store: Any, root_id: str) -> dict[str, str]:
    rows = [await store.get_run(root_id), *await store.list_runs(parent_run_id=root_id)]
    return {("root" if r.id == root_id else r.agent_name): r.status for r in rows}


@pytest.mark.parametrize("how", ["abandoned", "queue_full"])
async def test_q5_q6_close_runs_the_exit_path_without_awaiting_the_queue(
    store, live_adoption, monkeypatch, how
):
    runs = captured_runs(monkeypatch)
    full = _full_gate(monkeypatch)
    if how == "queue_full":
        # Enough events to fill the bound before the consumer advances again.
        children = [noting("alpha", Q2_CALLS), noting("beta", Q2_CALLS)]
        providers: list[GatedStreamLLM] = []
    else:
        pairs = [gated_child("alpha"), gated_child("beta")]
        children = [c for c, _ in pairs]
        providers = [p for _, p in pairs]
    root = lead(store, ScriptedLLM([batch(children), LLMResponse(text="done")]), children)
    stream = root.stream("go", tenant_id=TENANT)
    try:
        events: list[Any] = []
        async with stream:
            iterator = aiter(stream)  # held, so the stream stays open after the break
            async for event in iterator:
                events.append(event)
                if child_events(events, events[0].run_id):
                    break  # stop advancing once a child's progress arrived
            if how == "queue_full":
                await asyncio.wait_for(full.wait(), DEADLOCK_S)
            else:
                for provider in providers:
                    await wait_for_gate(provider.first_sent, timeout=DEADLOCK_S)
            assert not runs[0].owners_done()
            await asyncio.wait_for(stream.aclose(), DEADLOCK_S)
        root_id = events[0].run_id
        assert stream.result is not None and stream.result.status == RunStatus.CANCELLED
        assert runs[0].owners_done() and not owner_tasks_left()
        assert all(owner.cancelling() <= 1 for owner in runs[0].owners)
        statuses = await _rows(store, root_id)
        assert statuses == {"root": "cancelled", "alpha": "cancelled", "beta": "cancelled"}
        for child_id in await child_ids(store, root_id, ["alpha", "beta"]):
            assert not root._task_manager.is_running(child_id)
        for provider in providers:
            assert closed_in_its_task(provider)
    finally:
        for provider in providers:
            provider.gate.set()
        await close_all(root, *children)


# ---------------------------------------------------------------------------
# Pauses (A5), early close (A6) and caller cancellation
# ---------------------------------------------------------------------------


async def test_a5_subagent_paused_while_a_sibling_runs_then_resume_stream(store, live_adoption):
    asker = worker(
        "asker",
        ScriptedLLM([LLMResponse(tool_calls=[ToolCall("read", {})]), LLMResponse(text="asked")]),
        tools=[read],
    )
    slow = GatedStreamLLM(first="slow ", second="done")
    slow.gate_complete = True
    sibling = worker("sibling", slow, loop=SingleCall())
    children = [asker, sibling]
    root = lead(store, ScriptedLLM([batch(children), LLMResponse(text="done")]), children)
    try:
        events: list[Any] = []
        async with root.stream("go", tenant_id=TENANT) as stream:
            async for event in stream:
                events.append(event)
                if event.type == RunEventType.SUBAGENT_PAUSED:
                    assert not slow.complete_closed  # the sibling is still running
                    slow.gate.set()
        root_id = events[0].run_id
        asker_id, sibling_id = await child_ids(store, root_id, ["asker", "sibling"])
        paused_events = [e for e in events if e.type == RunEventType.SUBAGENT_PAUSED]
        assert len(paused_events) == 1 and paused_events[0].run_id == asker_id
        signal = paused_events[0].pending
        view = events[-1]
        assert view.type == RunEventType.RUN_PAUSED
        result = view.run_result
        assert result.run_id == root_id and result.status == RunStatus.WAITING_CLIENT_TOOL
        assert len(result.pendings) == 1 and result.pending == result.pendings[0]
        assert result.pause_id == result.pending.pause_id == signal.pause_id
        assert result.pending.origin_run_id == asker_id
        assert (await store.get_run(sibling_id)).status == "success"
        assert [c.id for c in signal.tool_calls] == [c.id for c in result.pending.tool_calls]

        reply = [ToolResult("read", result.pending.tool_calls[0].id, '"answer"')]
        resumed = await collect(
            root.resume_stream(
                root_id, tool_results=reply, tenant_id=TENANT, expected_pause_id=result.pause_id
            )
        )
        assert resumed[0].type == RunEventType.RUN_RESUMED and resumed[0].run_id == root_id
        final = resumed[-1].run_result
        assert (final.run_id, final.status, final.answer) == (root_id, RunStatus.SUCCESS, "done")
        assert final.pendings == () and final.pending is None and final.pause_id is None
        assert len(live_adoption) == 1 and live_adoption[0][0] == root_id
        assert {e.run_id for e in child_events(resumed, root_id)} <= {asker_id}
    finally:
        slow.gate.set()
        await close_all(root, *children)


async def test_resume_stream_on_a_fanout_root_refuses_a_stale_token(store, live_adoption):
    from dendrux.errors import PauseStatusMismatchError

    asker = worker(
        "asker",
        ScriptedLLM([LLMResponse(tool_calls=[ToolCall("read", {})]), LLMResponse(text="asked")]),
        tools=[read],
    )
    other = worker("other", ScriptedLLM([LLMResponse(text="other")]), loop=SingleCall())
    children = [asker, other]
    root = lead(store, ScriptedLLM([batch(children), LLMResponse(text="done")]), children)
    try:
        paused = (await collect(root.stream("go", tenant_id=TENANT)))[-1].run_result
        before = await _rows(store, paused.run_id)
        stream = root.resume_stream(
            paused.run_id, tool_results=[], tenant_id=TENANT, expected_pause_id="01STALE"
        )
        with pytest.raises(PauseStatusMismatchError) as caught:
            await collect(stream)
        assert caught.value.current_pause_id == paused.pause_id
        assert await _rows(store, paused.run_id) == before
    finally:
        await close_all(root, *children)


@pytest.mark.parametrize("n", [1, 2])
async def test_a6_early_close_matches_the_single_child_outcome(store, live_adoption, n):
    pairs = [gated_child(name) for name in ("alpha", "beta")[:n]]
    children = [c for c, _ in pairs]
    providers = [p for _, p in pairs]
    root = lead(store, ScriptedLLM([batch(children), LLMResponse(text="done")]), children)
    before = set(asyncio.all_tasks())
    stream = root.stream("go", tenant_id=TENANT)
    try:
        events: list[Any] = []
        async with stream:
            iterator = aiter(stream)  # held, so the stream stays open after the break
            async for event in iterator:
                events.append(event)
                if len({e.run_id for e in child_events(events, events[0].run_id)}) == n:
                    break
            # Owner tasks keep running while the consumer holds; a single child streams
            # only when the consumer advances, as in 0.7.
            for provider in providers if n > 1 else ():
                await wait_for_gate(provider.first_sent, timeout=DEADLOCK_S)
        root_id = events[0].run_id
        assert stream.result.status == RunStatus.CANCELLED
        statuses = await _rows(store, root_id)
        assert statuses == {"root": "cancelled", **{c.name: "cancelled" for c in children}}
        assert set(asyncio.all_tasks()) - before - {asyncio.current_task()} == set()
        for provider in providers:
            assert closed_in_its_task(provider)
        for child_id in await child_ids(store, root_id, [c.name for c in children]):
            assert not root._task_manager.is_running(child_id)
    finally:
        for provider in providers:
            provider.gate.set()
        await close_all(root, *children)


@pytest.mark.parametrize("n", [1, 2])
async def test_caller_cancel_propagates_and_matches_the_single_child_outcome(
    store, live_adoption, n
):
    pairs = [gated_child(name) for name in ("alpha", "beta")[:n]]
    children = [c for c, _ in pairs]
    providers = [p for _, p in pairs]
    root = lead(store, ScriptedLLM([batch(children), LLMResponse(text="done")]), children)
    seen: list[Any] = []

    async def consume() -> None:
        async with root.stream("go", tenant_id=TENANT) as stream:
            async for event in stream:
                seen.append(event)

    caller = asyncio.create_task(consume())
    try:
        for provider in providers:
            await wait_for_gate(provider.first_sent, task=caller, timeout=DEADLOCK_S)
        caller.cancel("first")
        with pytest.raises(asyncio.CancelledError) as caught:
            await caller
        assert caught.value.args == ("first",)
        statuses = await _rows(store, seen[0].run_id)
        assert statuses == {"root": "cancelled", **{c.name: "cancelled" for c in children}}
        assert not owner_tasks_left()
        for provider in providers:
            assert closed_in_its_task(provider)
    finally:
        for provider in providers:
            provider.gate.set()
        await asyncio.gather(caller, return_exceptions=True)
        await close_all(root, *children)


async def test_single_child_stream_keeps_its_provider_stream_in_the_caller(store, monkeypatch):
    """N = 1 keeps 0.7: the child streams in the caller's task and no driver starts."""
    runs = captured_runs(monkeypatch)
    child = noting("alpha")
    llm = TaskLLM([batch([child]), LLMResponse(text="done")])
    root = lead(store, llm, [child])
    caller = asyncio.current_task()
    try:
        events = await collect(root.stream("go", tenant_id=TENANT))
        assert events[-1].run_result.status == RunStatus.SUCCESS
        assert runs == []
        assert {task for _, task in child.provider.served} == {caller}
        assert [e.type for e in child_events(events, events[0].run_id)] == [
            RunEventType.TOOL_USE_END,
            RunEventType.TOOL_RESULT,
        ]
    finally:
        await close_all(root, child)


def asking(name: str, questions: int) -> Agent:
    """A ReAct child that asks the browser ``questions`` times, then answers its name."""
    turns = [LLMResponse(tool_calls=[ToolCall("read", {})]) for _ in range(questions)]
    return worker(name, ScriptedLLM([*turns, LLMResponse(text=name)]), tools=[read])


async def _answer(root: Agent, view: Any) -> list[Any]:
    reply = [ToolResult("read", view.pending.tool_calls[0].id, '"answer"')]
    return await collect(
        root.resume_stream(
            view.run_id, tool_results=reply, tenant_id=TENANT, expected_pause_id=view.pause_id
        )
    )


async def test_resume_stream_answers_paused_children_one_at_a_time(store, live_adoption):
    """A leaf that pauses again, and a finished leaf whose sibling still waits, both end
    with the root view; the last answer adopts every report and finishes the root."""
    first, second = asking("first", 2), asking("second", 1)
    children = [first, second]
    root = lead(store, ScriptedLLM([batch(children), LLMResponse(text="done")]), children)
    try:
        view = (await collect(root.stream("go", tenant_id=TENANT)))[-1].run_result
        first_id, second_id = await child_ids(store, view.run_id, ["first", "second"])
        assert [p.origin_run_id for p in view.pendings] == [first_id, second_id]

        again = (await _answer(root, view))[-1]  # the first child asks again
        assert again.type == RunEventType.RUN_PAUSED
        view = again.run_result
        assert [p.origin_run_id for p in view.pendings] == [first_id, second_id]
        assert view.pending == view.pendings[0] and view.pause_id == view.pending.pause_id

        view = replace_first(view)  # answer the second child: it finishes, the first waits
        done_second = (await _answer(root, view))[-1].run_result
        assert (await store.get_run(second_id)).status == "success"
        assert [p.origin_run_id for p in done_second.pendings] == [first_id]
        assert live_adoption == []

        final = (await _answer(root, done_second))[-1].run_result
        assert (final.status, final.answer) == (RunStatus.SUCCESS, "done")
        assert final.pendings == () and final.pause_id is None
        assert len(live_adoption) == 1
    finally:
        await close_all(root, *children)


def replace_first(view: Any) -> Any:
    """The same root view addressed to its second pending pause."""
    from dataclasses import replace

    return replace(view, pending=view.pendings[1], pause_id=view.pendings[1].pause_id)


async def test_a_cancel_before_the_bind_ends_the_stream_cancelled(store, monkeypatch):
    effects = FanoutEffects()
    children = round2_children(effects)
    root = lead(store, ScriptedLLM([batch(children), LLMResponse(text="done")]), children)
    original = _managed_fanout._prepare_batch

    async def prepare(store_: Any, branches: Any) -> Any:
        assert await store.request_cancel(branches[0].parent_run_id)
        return await original(store_, branches)

    monkeypatch.setattr(_managed_fanout, "_prepare_batch", prepare)
    try:
        events = await collect(root.stream("go", tenant_id=TENANT))
        final = events[-1]
        assert final.type == RunEventType.RUN_CANCELLED
        assert final.run_result.status == RunStatus.CANCELLED
        assert not effects.counts().provider_calls("alpha")
        assert await store.list_runs(parent_run_id=final.run_result.run_id) == []
    finally:
        await close_all(root, *children)


async def test_a_stream_that_does_not_own_the_bind_drives_nothing(store, monkeypatch):
    """A replayed bind returns the existing children: no owner, no adoption, root view."""
    effects = FanoutEffects()
    children = round2_children(effects)
    root = lead(store, ScriptedLLM([batch(children), LLMResponse(text="done")]), children)
    original = _managed_fanout.start_fanout

    async def replayed(invocation: Any, **kwargs: Any) -> Any:
        first = await original(invocation)
        await first.wait()
        return await original(invocation, **kwargs)

    monkeypatch.setattr(_managed_fanout, "start_fanout", replayed)
    try:
        events = await collect(root.stream("go", tenant_id=TENANT))
        view = events[-1].run_result
        assert events[-1].type == RunEventType.RUN_PAUSED
        assert view.status == RunStatus.RUNNING and view.pendings == ()
        assert [effects.counts().provider_calls(c.name) for c in children] == [1, 1]
        state = await store.get_loop_state(view.run_id)
        assert state.get("managed_admitted")  # nobody adopted: recovery takes it from here
    finally:
        await close_all(root, *children)
