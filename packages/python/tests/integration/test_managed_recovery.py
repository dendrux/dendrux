"""Managed subagents (0.7, T4): full recovery of a managed workflow from its root.

One test per crash cut of spec v2 §3, recovered by a fresh process view (new Agent
objects and providers; on Postgres also a new engine and pool) through the root id:
bind committed before the child starts, a started child before its first effect,
a leaf submission claimed but not consumed, a terminal child not yet adopted, a
parent claimed for adoption but not completed, adoption complete but the parent not
continued, unknown ordinary effects, and stale owner conflicts. Each asserts the
persisted ids and generations, one adoption, one observation, no rerun of the child
and usage counted once. Also: the cached root view never shows the root's private
wait (B1), and recovery of a pending managed call (B4/B5).

Runs on SQLite and Postgres (``DENDRUX_TEST_PG_URL``, a private ``*_test`` database);
contender races run on Postgres only.
"""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from typing import Any

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from dendrux import Agent, tool
from dendrux.db.models import AgentRun, ReactTrace, RunEvent, TokenUsage, ToolCallRecord
from dendrux.runtime import _managed
from dendrux.runtime.context import get_delegation_context
from dendrux.runtime.stores._factory import store_for_engine
from dendrux.types import LLMResponse, RunStatus, ToolCall
from tests._helpers.recovery_checkpoint import expire
from tests._helpers.scripted_provider import ScriptedLLM

BRIEF = "SENTINEL-recover-brief-5a1d"
REPORT = "SENTINEL-recover-report-8c2e"
TENANT = "t1"


class ProcessDeathError(BaseException):
    """Ends the caller like a killed process: no runner cleanup finalizes the run."""


class ContextLLM(ScriptedLLM):
    """Scripted provider recording the run each call is served under (real ContextVar).

    ``on_call(index, run_id)`` (optional) runs before the scripted response.
    """

    def __init__(self, responses: list[LLMResponse], on_call: Any = None) -> None:
        super().__init__(responses)
        self.served: list[str | None] = []
        self.on_call = on_call

    async def complete(self, messages: Any, tools: Any = None, **kwargs: Any) -> LLMResponse:
        ctx = get_delegation_context()
        run_id = ctx.run_id if ctx else None
        if self.on_call is not None:
            await self.on_call(self.calls, run_id)
        self.served.append(run_id)
        return await super().complete(messages, tools, **kwargs)


def call(name: str, params: dict[str, Any], pid: str) -> LLMResponse:
    return LLMResponse(tool_calls=[ToolCall(name=name, params=params, provider_tool_call_id=pid)])


effects: list[str] = []


@tool()
async def publish(text: str) -> str:
    """Publish a draft (an external effect)."""
    effects.append(f"publish:{text}")
    return f"published {text}"


@tool()
async def note(text: str) -> str:
    """Take a note."""
    effects.append(f"note:{text}")
    return "noted"


@tool()
async def fetch(x: int) -> str:
    """Fetch (a server effect)."""
    effects.append(f"fetch:{x}")
    return "fetched"


def reviewer(llm: Any, **kwargs: Any) -> Agent:
    return Agent(
        name="reviewer",
        description="Reviews one draft and reports.",
        prompt="You review.",
        provider=llm,
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


async def run_row(engine: Any, run_id: str) -> Any:
    return (await rows(engine, AgentRun, AgentRun.id == run_id))[0]


async def root_id_of(engine: Any) -> str:
    return (await rows(engine, AgentRun, AgentRun.agent_name == "lead"))[0].id


async def children_of(engine: Any, run_id: str) -> list[Any]:
    return await rows(engine, AgentRun, AgentRun.parent_run_id == run_id)


async def observations(engine: Any, run_id: str) -> list[Any]:
    traces = await rows(engine, ReactTrace, ReactTrace.agent_run_id == run_id)
    return [t for t in traces if t.role == "tool"]


async def tool_events(engine: Any, run_id: str, kind: str) -> int:
    events = await rows(engine, RunEvent, RunEvent.agent_run_id == run_id)
    return [e.event_type for e in events].count(kind)


@asynccontextmanager
async def fresh_store(engine: Any) -> Any:
    """A fresh process's store: on Postgres a new engine and pool; SQLite in-memory
    shares the one connection (a new engine would be a new, empty database)."""
    if engine.dialect.name == "postgresql":
        other = create_async_engine(engine.url, pool_size=4)
        try:
            yield store_for_engine(other)
        finally:
            await other.dispose()
    else:
        yield store_for_engine(engine)


def _postgres(engine: Any) -> None:
    if engine.dialect.name != "postgresql":
        pytest.skip("independent-session race: Postgres only")


# ---------------------------------------------------------------------------
# B1: the cached root view never shows the root's private wait
# ---------------------------------------------------------------------------


def _approval_workflow(store: Any) -> Agent:
    child = reviewer(
        ContextLLM([call("publish", {"text": "d"}, "c1")]),
        tools=[publish],
        require_approval=["publish"],
    )
    return lead(store, ContextLLM([call("reviewer", {"task": BRIEF}, "r1")]), child)


def _requested_solo(store: Any) -> Agent:
    async def pause_me(index: int, run_id: str | None) -> None:
        if index == 0:
            await store.request_pause(run_id)

    return Agent(
        name="solo",
        prompt="p",
        provider=ContextLLM([call("note", {"text": "a"}, "n1")], on_call=pause_me),
        tools=[note],
        state_store=store,
    )


@pytest.mark.parametrize("kind", ["legacy_requested", "managed_root"])
async def test_cached_view_never_shows_a_stale_status_after_a_cancel_between_reads(
    store: Any, engine: Any, kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """B1: ``root_view`` / the cached result re-read a run that moved between its reads."""
    from dendrux.runtime._pending import _PendingReader

    agent = _requested_solo(store) if kind == "legacy_requested" else _approval_workflow(store)
    paused = await agent.run("go", tenant_id=TENANT)
    expected = RunStatus.PAUSED if kind == "legacy_requested" else RunStatus.WAITING_APPROVAL
    assert paused.status == expected
    run_id = paused.run_id
    original = _PendingReader._summary
    fired: list[bool] = []

    async def cancel_first(self: Any, target: str, tenant_id: Any) -> Any:
        if not fired:  # between the cached record read and the pending read
            fired.append(True)
            await agent.cancel_run(run_id, tenant_id=TENANT, cascade=True)
        return await original(self, target, tenant_id)

    monkeypatch.setattr(_PendingReader, "_summary", cancel_first)
    cached = await _managed.root_view(store, run_id)

    assert fired
    assert cached.status == RunStatus.CANCELLED
    assert cached.pending is None and cached.pause_id is None
    assert (await run_row(engine, run_id)).status == "cancelled"


async def test_cached_view_of_a_stable_wait_reads_once(
    store: Any, engine: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """B1: an agreeing read makes no extra record read."""
    agent = _approval_workflow(store)
    paused = await agent.run("go", tenant_id=TENANT)
    reads: list[str] = []
    original = type(store).get_run

    async def counted(self: Any, run_id: str, **kwargs: Any) -> Any:
        reads.append(run_id)
        return await original(self, run_id, **kwargs)

    monkeypatch.setattr(type(store), "get_run", counted)
    cached = await _managed.root_view(store, paused.run_id)
    assert cached.status == RunStatus.WAITING_APPROVAL
    assert cached.pause_id == paused.pause_id
    assert reads.count(paused.run_id) == 1


# ---------------------------------------------------------------------------
# B6: a custom store without the pre-claim child refusal still refuses safely
# ---------------------------------------------------------------------------


def _nonconforming(store: Any) -> Any:
    """A custom store whose reclaim_run lacks the kit's pre-claim managed-child refusal."""
    from dendrux.runtime.context import managed_root_reclaim_scope

    base = type(store)

    class NonconformingStore(base):  # type: ignore[misc, valid-type]
        async def reclaim_run(self, run_id: str, **kwargs: Any) -> Any:
            with managed_root_reclaim_scope(run_id):  # skips the native refusal
                return await base.reclaim_run(self, run_id, **kwargs)

    custom = store_for_engine(store._engine)
    custom.__class__ = NonconformingStore
    return custom


async def _dead_child(store: Any, engine: Any) -> tuple[str, Any]:
    class DyingLLM(ContextLLM):
        async def complete(self, messages: Any, tools: Any = None, **kw: Any) -> LLMResponse:
            raise ProcessDeathError

    first = ContextLLM([call("reviewer", {"task": BRIEF}, "r1")])
    root = lead(store, first, reviewer(DyingLLM([])))
    with pytest.raises(ProcessDeathError):
        await root.run("go", tenant_id=TENANT)
    root_id = await root_id_of(engine)
    [child] = await children_of(engine, root_id)
    assert child.status == "running"
    return root_id, child


async def _recovery_marks(engine: Any, run_id: str) -> tuple[Any, ...]:
    row = await run_row(engine, run_id)
    events = await count(engine, RunEvent, RunEvent.agent_run_id == run_id)
    return (row.status, row.claim_epoch, row.reclaim_attempts, row.lease_expired_at, events)


async def test_custom_store_without_pre_claim_refusal_refuses_a_direct_child_reclaim(
    store: Any, engine: Any
) -> None:
    """B6: the runtime refuses a direct child reclaim before such a store's claim."""
    from dendrux.errors import ManagedChildControlError, RunNotFoundError

    root_id, child = await _dead_child(store, engine)
    item = await expire(store, engine, child.id)
    before = await _recovery_marks(engine, child.id)
    custom = _nonconforming(store)
    direct = reviewer(ContextLLM([]), state_store=custom)
    with pytest.raises(RunNotFoundError):  # another tenant learns nothing
        await direct.reclaim(child.id, tenant_id="other", item=item)
    assert await _recovery_marks(engine, child.id) == before
    for _ in range(2):
        with pytest.raises(ManagedChildControlError):
            await direct.reclaim(child.id, tenant_id=TENANT, item=item)
        assert await _recovery_marks(engine, child.id) == before

    child_llm = ContextLLM([LLMResponse(text=REPORT)])
    root_llm = ContextLLM([LLMResponse(text="root done")])
    result = await lead(custom, root_llm, reviewer(child_llm)).reclaim(root_id, tenant_id=TENANT)
    assert result.status == RunStatus.SUCCESS and child_llm.calls == 1
    assert len(await children_of(engine, root_id)) == 1


# ---------------------------------------------------------------------------
# The crash matrix (spec v2 §3, §8 T4): phase one runs in a child process that
# dies with ``os._exit`` at the cut (no cleanup, no finally); phase two recovers
# through the root id with new Agents, providers and a new engine.
# ---------------------------------------------------------------------------

CRASHED = 17
"""Exit code of a phase-one process that reached its crash cut."""
_CRASH_IN_TOOL: bool | str = False


def _usage() -> Any:
    from dendrux.types import UsageStats

    return UsageStats(input_tokens=10, output_tokens=1, total_tokens=11)


def _said(text: str) -> LLMResponse:
    return LLMResponse(text=text, usage=_usage())


def _asked(name: str, params: dict[str, Any], pid: str) -> LLMResponse:
    return LLMResponse(
        tool_calls=[ToolCall(name=name, params=params, provider_tool_call_id=pid)], usage=_usage()
    )


@tool()
async def send(text: str) -> str:
    """Send a message (non-idempotent external effect)."""
    if _CRASH_IN_TOOL == "raise":
        raise ProcessDeathError  # in-process crash; the started intent stays unresolved
    if _CRASH_IN_TOOL:
        import os

        os._exit(CRASHED)  # the effect started; its outcome is unknown
    effects.append(f"send:{text}")
    return "sent"


def _child(llm: Any, cut: str) -> Agent:
    if cut == "leaf_submitted":
        return reviewer(llm, tools=[publish], require_approval=["publish"])
    if cut == "unknown_effect":
        return reviewer(llm, tools=[send])
    return reviewer(llm)


def _child_script(cut: str) -> list[LLMResponse]:
    if cut == "leaf_submitted":
        return [_asked("publish", {"text": "d"}, "c1"), _said(REPORT)]
    if cut == "unknown_effect":
        return [_asked("send", {"text": "d"}, "c1"), _said(REPORT)]
    return [_said(REPORT)]


def _crash_now(*args: Any, **kwargs: Any) -> Any:
    import os

    os._exit(CRASHED)


async def _crash_async(*args: Any, **kwargs: Any) -> Any:
    _crash_now()


def _install_cut(cut: str) -> None:
    """Make the phase-one process die exactly at ``cut``."""
    global _CRASH_IN_TOOL
    from dendrux.runtime import _reclaim, persistence

    if cut == "pre_bind":
        _managed.bind_child = _crash_async
    elif cut == "bind_committed":
        original = _managed.bind_child

        async def bound_then_die(*args: Any, **kwargs: Any) -> Any:
            await original(*args, **kwargs)
            _crash_now()

        _managed.bind_child = bound_then_die
    elif cut == "child_terminal":
        _managed.adopt = _crash_async
    elif cut == "adoption_claimed":
        persistence.adopt_managed_result = _crash_async
    elif cut == "adoption_complete":
        _reclaim.continue_claimed = _crash_async
    elif cut == "unknown_effect":
        _CRASH_IN_TOOL = True


async def _phase_one(cut: str, url: str) -> None:
    """The first process: run until the cut, where it dies."""
    from dendrux.runtime.persistence import PersistenceRecorder

    engine = create_async_engine(url)
    store = store_for_engine(engine)
    child_llm = ContextLLM(
        _child_script(cut),
        on_call=(lambda i, r: _crash_async()) if cut == "child_started" else None,
    )
    root = lead(
        store, ContextLLM([_asked("reviewer", {"task": BRIEF}, "r1")]), _child(child_llm, cut)
    )
    if cut != "leaf_submitted":
        _install_cut(cut)
        await root.run("go", tenant_id=TENANT)
        raise SystemExit(3)  # the cut was never reached
    paused = await root.run("go", tenant_id=TENANT)
    assert paused.status == RunStatus.WAITING_APPROVAL, paused.status
    PersistenceRecorder.consume_resume = _crash_async  # type: ignore[method-assign]
    await root.submit_approval(
        paused.run_id, approved=True, expected_pause_id=paused.pause_id, tenant_id=TENANT
    )
    raise SystemExit(3)


def _crash_in_new_process(cut: str, engine: Any) -> None:
    import os
    import subprocess
    import sys
    from pathlib import Path

    here = Path(__file__).resolve()
    package = here.parents[2]
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(package / "src"), str(package)]),
        "T4_CRASH_URL": engine.url.render_as_string(hide_password=False),
    }
    done = subprocess.run(
        [sys.executable, "-m", "tests.integration.test_managed_recovery", cut],
        cwd=package,
        env=env,
        capture_output=True,
        timeout=120,
        check=False,
    )
    assert done.returncode == CRASHED, done.stderr.decode()[-2000:]


@asynccontextmanager
async def _fresh_process(engine: Any) -> Any:
    """Phase two's store: a new engine and pool on the same database."""
    other = create_async_engine(engine.url.render_as_string(hide_password=False))
    try:
        yield store_for_engine(other)
    finally:
        await other.dispose()


async def _snapshot(engine: Any, root_id: str) -> dict[str, Any]:
    root = await run_row(engine, root_id)
    children = await children_of(engine, root_id)
    return {
        "root": (root.status, root.claim_epoch),
        "wait": (root.pause_data or {}).get("pause_id"),
        "children": [(c.id, c.status, c.claim_epoch) for c in children],
    }


async def _usage_rows(engine: Any, run_id: str) -> int:
    return await count(engine, TokenUsage, TokenUsage.agent_run_id == run_id)


async def _assert_adopted_once(
    engine: Any, store: Any, root_id: str, child_id: str, *, root_calls: int, child_calls: int
) -> None:
    """One child, one adoption, one observation; usage counted once per provider call."""
    from dendrux.store import RunStore

    [child] = await children_of(engine, root_id)
    assert child.id == child_id and child.status == "success"
    [intent] = await rows(engine, ToolCallRecord, ToolCallRecord.agent_run_id == root_id)
    assert intent.outcome == "completed"
    assert intent.meta["managed"]["child_run_id"] == child_id
    observed = await observations(engine, root_id)
    assert len(observed) == 1 and json.loads(observed[0].content) == REPORT
    assert await tool_events(engine, root_id, "tool.completed") == 1
    assert await _usage_rows(engine, root_id) == root_calls
    assert await _usage_rows(engine, child_id) == child_calls
    root = await run_row(engine, root_id)
    assert root.status == "success" and root.total_input_tokens == 10 * root_calls
    assert (await run_row(engine, child_id)).total_input_tokens == 10 * child_calls
    async with RunStore.from_engine(engine) as reader:
        subtree = await reader.get_subtree(root_id, tenant_id=TENANT, max_nodes=16)
    assert subtree.usage.input_tokens == 10 * (root_calls + child_calls)


def _restarted(store: Any, cut: str, child_script: list[LLMResponse]) -> tuple[Agent, Any, Any]:
    child_llm = ContextLLM(child_script)
    root_llm = ContextLLM([_said("root done")])
    return lead(store, root_llm, _child(child_llm, cut)), root_llm, child_llm


crash_engines = pytest.mark.parametrize("engine", ["sqlite_file", "postgres"], indirect=True)


@crash_engines
async def test_crash_after_bind_before_the_child_starts(engine: Any) -> None:
    """Cut: bind committed, child never started. No stealing; after expiry the same child."""
    from dendrux.errors import RunNotReclaimableError

    _crash_in_new_process("bind_committed", engine)
    root_id = await root_id_of(engine)
    before = await _snapshot(engine, root_id)
    [(child_id, status, epoch)] = before["children"]
    assert before["root"] == ("paused", 1) and (status, epoch) == ("running", 0)
    assert before["wait"] is not None

    async with _fresh_process(engine) as store:
        root, root_llm, child_llm = _restarted(store, "bind_committed", [_said(REPORT)])
        with pytest.raises(RunNotReclaimableError):  # a healthy owner is never stolen from
            await root.reclaim(root_id, tenant_id=TENANT)
        assert await _snapshot(engine, root_id) == before
        await expire(store, engine, child_id)
        result = await root.reclaim(root_id, tenant_id=TENANT)
        assert result.status == RunStatus.SUCCESS and result.run_id == root_id
        assert child_llm.served == [child_id] and root_llm.calls == 1
        first = [m for m in child_llm.messages[0] if m.role.value == "user"]
        assert [m.content for m in first] == [BRIEF]  # the bound brief, not new state
        await _assert_adopted_once(engine, store, root_id, child_id, root_calls=2, child_calls=1)


@crash_engines
async def test_crash_in_a_started_child_before_its_first_effect(engine: Any) -> None:
    _crash_in_new_process("child_started", engine)
    root_id = await root_id_of(engine)
    [(child_id, status, _)] = (await _snapshot(engine, root_id))["children"]
    assert status == "running" and await _usage_rows(engine, child_id) == 0

    async with _fresh_process(engine) as store:
        root, root_llm, child_llm = _restarted(store, "child_started", [_said(REPORT)])
        await expire(store, engine, child_id)
        result = await root.reclaim(root_id, tenant_id=TENANT)
        assert result.status == RunStatus.SUCCESS
        assert child_llm.served == [child_id] and root_llm.calls == 1
        await _assert_adopted_once(engine, store, root_id, child_id, root_calls=2, child_calls=1)


@crash_engines
async def test_crash_after_a_leaf_submission_is_claimed_before_it_is_consumed(
    engine: Any,
) -> None:
    """Cut: the root-id approval claimed the leaf; recovery consumes it once."""
    effects.clear()
    _crash_in_new_process("leaf_submitted", engine)
    root_id = await root_id_of(engine)
    [(child_id, status, epoch)] = (await _snapshot(engine, root_id))["children"]
    leaf = await run_row(engine, child_id)
    assert status == "running" and leaf.loop_state["submission"] is not None
    assert effects == []

    async with _fresh_process(engine) as store:
        root, root_llm, child_llm = _restarted(store, "leaf_submitted", [_said(REPORT)])
        await expire(store, engine, child_id)
        result = await root.reclaim(root_id, tenant_id=TENANT)
        assert result.status == RunStatus.SUCCESS
        assert effects == ["publish:d"]  # the approved effect ran once
        assert child_llm.served == [child_id] and root_llm.calls == 1
        decided = await tool_events(engine, child_id, "approval.decided")
        assert decided == 1
        assert (await run_row(engine, child_id)).claim_epoch > epoch
        await _assert_adopted_once(engine, store, root_id, child_id, root_calls=2, child_calls=2)


@crash_engines
async def test_crash_after_the_child_finished_before_its_parent_accepted(engine: Any) -> None:
    _crash_in_new_process("child_terminal", engine)
    root_id = await root_id_of(engine)
    before = await _snapshot(engine, root_id)
    [(child_id, status, _)] = before["children"]
    assert before["root"] == ("paused", 1) and status == "success"

    async with _fresh_process(engine) as store:
        root, root_llm, child_llm = _restarted(store, "child_terminal", [])
        result = await root.reclaim(root_id, tenant_id=TENANT)
        assert result.status == RunStatus.SUCCESS
        assert child_llm.calls == 0 and root_llm.calls == 1
        assert (await run_row(engine, root_id)).claim_epoch == 2  # one adoption claim
        await _assert_adopted_once(engine, store, root_id, child_id, root_calls=2, child_calls=1)


@crash_engines
async def test_crash_after_the_parent_accepted_before_its_tool_row_completed(
    engine: Any,
) -> None:
    """Cut: submit_and_claim done, complete_tool_call not. A live claim is not stolen."""
    from dendrux.errors import RunNotReclaimableError

    _crash_in_new_process("adoption_claimed", engine)
    root_id = await root_id_of(engine)
    root_row = await run_row(engine, root_id)
    [child] = await children_of(engine, root_id)
    assert (root_row.status, root_row.claim_epoch) == ("running", 2)
    assert root_row.loop_state["submission"] is not None
    [intent] = await rows(engine, ToolCallRecord, ToolCallRecord.agent_run_id == root_id)
    assert intent.outcome == "started"

    async with _fresh_process(engine) as store:
        root, root_llm, child_llm = _restarted(store, "adoption_claimed", [])
        with pytest.raises(RunNotReclaimableError):
            await root.reclaim(root_id, tenant_id=TENANT)
        assert (await run_row(engine, root_id)).claim_epoch == 2
        await expire(store, engine, root_id)
        result = await root.reclaim(root_id, tenant_id=TENANT)
        assert result.status == RunStatus.SUCCESS
        assert child_llm.calls == 0 and root_llm.calls == 1
        await _assert_adopted_once(engine, store, root_id, child.id, root_calls=2, child_calls=1)


@crash_engines
async def test_crash_after_adoption_completed_before_the_parent_continued(
    engine: Any,
) -> None:
    _crash_in_new_process("adoption_complete", engine)
    root_id = await root_id_of(engine)
    root_row = await run_row(engine, root_id)
    [child] = await children_of(engine, root_id)
    assert root_row.status == "running" and root_row.loop_state.get("submission") is None
    assert root_row.loop_state.get("managed_wait") is None

    async with _fresh_process(engine) as store:
        root, root_llm, child_llm = _restarted(store, "adoption_complete", [])
        await expire(store, engine, root_id)
        result = await root.reclaim(root_id, tenant_id=TENANT)
        assert result.status == RunStatus.SUCCESS
        assert child_llm.calls == 0 and root_llm.calls == 1
        await _assert_adopted_once(engine, store, root_id, child.id, root_calls=2, child_calls=1)


@crash_engines
async def test_an_unknown_child_effect_follows_the_uncertain_outcome_policy(engine: Any) -> None:
    """A non-idempotent effect started before the crash is never replayed silently:
    recovery asks for approval through the root, and only that decision replays it."""
    effects.clear()
    _crash_in_new_process("unknown_effect", engine)
    root_id = await root_id_of(engine)
    [(child_id, status, _)] = (await _snapshot(engine, root_id))["children"]
    [started] = await rows(engine, ToolCallRecord, ToolCallRecord.agent_run_id == child_id)
    assert status == "running" and started.outcome == "started"

    async with _fresh_process(engine) as store:
        root, root_llm, child_llm = _restarted(store, "unknown_effect", [_said(REPORT)])
        await expire(store, engine, child_id)
        waiting = await root.reclaim(root_id, tenant_id=TENANT)
        assert waiting.status == RunStatus.WAITING_APPROVAL and waiting.run_id == root_id
        assert waiting.pending.origin_run_id == child_id and effects == []
        assert child_llm.calls == 0
        result = await root.submit_approval(
            root_id, approved=True, expected_pause_id=waiting.pause_id, tenant_id=TENANT
        )
        assert result.status == RunStatus.SUCCESS
        assert effects == ["send:d"] and child_llm.served == [child_id]
        await _assert_adopted_once(engine, store, root_id, child_id, root_calls=2, child_calls=2)


@crash_engines
async def test_crash_between_handoff_and_bind_binds_the_same_invocation_once(
    engine: Any,
) -> None:
    """B5: the parent stays running until bind; recovery binds the saved call once."""
    _crash_in_new_process("pre_bind", engine)
    root_id = await root_id_of(engine)
    row = await run_row(engine, root_id)
    assert row.status == "running" and not await children_of(engine, root_id)
    [pending] = row.loop_state["pending"]
    assert pending["target"] == "agent" and row.loop_state["managed_admitted"] == pending["call_id"]
    assert not await rows(engine, ToolCallRecord, ToolCallRecord.agent_run_id == root_id)

    async with _fresh_process(engine) as store:
        root, root_llm, child_llm = _restarted(store, "pre_bind", [_said(REPORT)])
        await expire(store, engine, root_id)
        result = await root.reclaim(root_id, tenant_id=TENANT)
        assert result.status == RunStatus.SUCCESS
        [child] = await children_of(engine, root_id)
        [intent] = await rows(engine, ToolCallRecord, ToolCallRecord.agent_run_id == root_id)
        assert intent.tool_call_id == pending["call_id"]  # the same invocation
        assert intent.params == {"task": BRIEF}
        header = (await run_row(engine, child.id)).loop_state["managed"]
        assert header["invocation"]["tool_call_id"] == pending["call_id"]
        assert child_llm.served == [child.id] and root_llm.calls == 1
        await _assert_adopted_once(engine, store, root_id, child.id, root_calls=2, child_calls=1)


# ---------------------------------------------------------------------------
# Stale owner conflicts on Postgres (independent engines and pools)
# ---------------------------------------------------------------------------


async def test_pg_root_controls_never_steal_a_healthy_running_child(engine: Any) -> None:
    """While a child runs under its live owner, root reclaim/resume refuse and change
    nothing; the owner then finishes the workflow with one child."""
    from dendrux.errors import RunNotPausedError, RunNotReclaimableError

    _postgres(engine)
    gated = ScriptedLLM([_said(REPORT)], gate_at=0)
    root_llm = ContextLLM([_asked("reviewer", {"task": BRIEF}, "r1"), _said("root done")])
    store = store_for_engine(engine)
    workflow = lead(store, root_llm, reviewer(gated))
    owner = asyncio.create_task(workflow.run("go", tenant_id=TENANT))
    await asyncio.wait_for(gated.entered.wait(), 30)
    root_id = await root_id_of(engine)
    before = await _snapshot(engine, root_id)
    try:
        async with _fresh_process(engine) as other:
            stranger_llm = ContextLLM([])
            stranger = lead(other, ContextLLM([]), reviewer(stranger_llm))
            with pytest.raises(RunNotReclaimableError):
                await stranger.reclaim(root_id, tenant_id=TENANT)
            with pytest.raises(RunNotPausedError):
                await stranger.resume(root_id, tenant_id=TENANT, expected_pause_id="stale")
            seen = await stranger.get_result(root_id, tenant_id=TENANT)
            assert seen.status == RunStatus.RUNNING and seen.pending is None
            assert stranger_llm.calls == 0
        assert await _snapshot(engine, root_id) == before
    finally:
        gated.gate.set()
    result = await asyncio.wait_for(owner, 30)
    assert result.status == RunStatus.SUCCESS and result.answer == "root done"
    assert gated.calls == 1 and len(await children_of(engine, root_id)) == 1


@pytest.mark.parametrize("cut", ["child_started", "pre_bind"])
@pytest.mark.parametrize("round_", [1, 2, 3])
async def test_pg_concurrent_root_recoveries_drive_one_child_once(
    engine: Any, cut: str, round_: int
) -> None:
    """After lease expiry, contending root reclaims on two pools: the exact bound child
    (or, before the bind, one new child) runs once; never a second child."""
    _postgres(engine)
    _crash_in_new_process(cut, engine)
    root_id = await root_id_of(engine)
    if cut == "child_started":
        [child] = await children_of(engine, root_id)
        expired = child.id
    else:
        expired = root_id
    await expire(store_for_engine(engine), engine, expired)
    child_llms = [ContextLLM([_said(REPORT)]) for _ in range(4)]
    root_llms = [ContextLLM([_said("root done")]) for _ in range(4)]
    async with _fresh_process(engine) as other:
        stores = [store_for_engine(engine), other]
        outcomes = await asyncio.gather(
            *(
                lead(stores[i % 2], root_llms[i], reviewer(child_llms[i])).reclaim(
                    root_id, tenant_id=TENANT
                )
                for i in range(4)
            ),
            return_exceptions=True,
        )
    from dendrux.errors import RunNotReclaimableError

    for outcome in outcomes:
        assert not isinstance(outcome, BaseException) or isinstance(
            outcome, RunNotReclaimableError
        ), outcome
    [child] = await children_of(engine, root_id)
    assert sum(llm.calls for llm in child_llms) == 1
    assert sum(llm.calls for llm in root_llms) == 1
    assert {s for llm in child_llms for s in llm.served} == {child.id}
    assert any(not isinstance(o, BaseException) and o.status == RunStatus.SUCCESS for o in outcomes)
    await _assert_adopted_once(
        engine, store_for_engine(engine), root_id, child.id, root_calls=2, child_calls=1
    )


# ---------------------------------------------------------------------------
# B4: recovery completes a pending managed call (mixed batch, unadmitted call)
# ---------------------------------------------------------------------------


def _hooked(
    store: Any, root_llm: Any, child: Agent, *, deny: str | None = None, hook: Any = None, **kw: Any
) -> Agent:
    async def deny_hook(ctx: Any) -> Any:
        from dendrux.types import ToolDecision

        effects.append(f"hook:{ctx.tool_name}")
        return ToolDecision.deny("not now") if ctx.tool_name == deny else None

    return Agent(
        name="lead",
        prompt="You lead.",
        provider=root_llm,
        state_store=store,
        tools=[fetch, send],
        subagents=[child],
        before_tool=hook or deny_hook,
        **kw,
    )


def _mixed(*names: str) -> LLMResponse:
    params = {"reviewer": {"task": BRIEF}, "fetch": {"x": 1}, "send": {"text": "s"}}
    return LLMResponse(
        tool_calls=[
            ToolCall(name=n, params=params[n], provider_tool_call_id=f"p{i}")
            for i, n in enumerate(names)
        ],
        usage=_usage(),
    )


async def _die_at(
    monkeypatch: Any, target: Any, name: str, run: Any, *, after: bool = False
) -> None:
    import inspect

    original = getattr(target, name)

    async def die(*args: Any, **kwargs: Any) -> Any:
        if after:
            await original(*args, **kwargs)
        raise ProcessDeathError

    def die_sync(*args: Any, **kwargs: Any) -> Any:
        raise ProcessDeathError

    with monkeypatch.context() as patch:
        patch.setattr(target, name, die if inspect.iscoroutinefunction(original) else die_sync)
        with pytest.raises(ProcessDeathError):
            await run()


class _ChildLLM(ContextLLM):
    """Records the child's first call in the shared effect log (order evidence)."""

    async def complete(self, messages: Any, tools: Any = None, **kwargs: Any) -> LLMResponse:
        effects.append("child")
        return await super().complete(messages, tools, **kwargs)


@pytest.mark.parametrize("order", [("reviewer", "fetch"), ("fetch", "reviewer")])
async def test_recovered_mixed_batch_runs_the_server_call_then_binds_once(
    store: Any, engine: Any, monkeypatch: Any, order: tuple[str, str]
) -> None:
    """Crash before the batch was prepared: recovery prepares both calls (hooks once each),
    runs the server call, then binds the managed call once. No refusal."""
    from dendrux.loops import react

    first = _hooked(store, ContextLLM([_mixed(*order)]), reviewer(ContextLLM([])))
    await _die_at(
        monkeypatch, react, "_prepare_tool_calls", lambda: first.run("go", tenant_id=TENANT)
    )
    root_id = await root_id_of(engine)
    effects.clear()
    item = await expire(store, engine, root_id)
    child_llm = _ChildLLM([_said(REPORT)])
    root_llm = ContextLLM([_said("root done")])
    result = await _hooked(store, root_llm, reviewer(child_llm)).reclaim(
        root_id, tenant_id=TENANT, item=item
    )

    assert result.status == RunStatus.SUCCESS and result.answer == "root done"
    assert sorted(effects[:2]) == ["hook:fetch", "hook:reviewer"]
    assert effects[2:] == ["fetch:1", "child"]  # the server call finished before the bind
    [child] = await children_of(engine, root_id)
    assert child_llm.served == [child.id]
    seen = {m.name: m.content for m in root_llm.messages[0] if m.role.value == "tool"}
    assert json.loads(seen["reviewer"]) == REPORT and json.loads(seen["fetch"]) == "fetched"
    intents = await rows(engine, ToolCallRecord, ToolCallRecord.agent_run_id == root_id)
    assert sorted((r.tool_name, r.outcome) for r in intents) == [
        ("fetch", "completed"),
        ("reviewer", "completed"),
    ]


async def test_recovered_unadmitted_call_after_its_sibling_finished_binds_without_repeat(
    store: Any, engine: Any, monkeypatch: Any
) -> None:
    """Crash after the server sibling completed, before the handoff wrote admission:
    the lone managed call is admitted again by its own hook, the sibling never repeats."""
    first = _hooked(store, ContextLLM([_mixed("reviewer", "fetch")]), reviewer(ContextLLM([])))
    await _die_at(
        monkeypatch, _managed, "prepare_handoff", lambda: first.run("go", tenant_id=TENANT)
    )
    root_id = await root_id_of(engine)
    row = await run_row(engine, root_id)
    assert [p["tool"] for p in row.loop_state["pending"]] == ["reviewer"]
    assert row.loop_state.get("managed_admitted") is None
    effects.clear()
    item = await expire(store, engine, root_id)
    child_llm = _ChildLLM([_said(REPORT)])
    root_llm = ContextLLM([_said("root done")])
    result = await _hooked(store, root_llm, reviewer(child_llm)).reclaim(
        root_id, tenant_id=TENANT, item=item
    )
    assert result.status == RunStatus.SUCCESS
    assert effects == ["hook:reviewer", "child"]  # no second fetch
    assert len(await children_of(engine, root_id)) == 1


@pytest.mark.parametrize("decision", ["deny", "approval_hook"])
async def test_recovered_unadmitted_call_denied_by_policy_runs_no_child(
    store: Any, engine: Any, monkeypatch: Any, decision: str
) -> None:
    """The recovered admission applies the parent's policy: a denial is an ordinary
    observation and no child exists."""
    from dendrux.loops import react
    from dendrux.types import ToolDecision

    async def approval_hook(ctx: Any) -> Any:
        effects.append(f"hook:{ctx.tool_name}")
        return ToolDecision.require_approval("check")

    def build(llm: Any, child: Agent) -> Agent:
        if decision == "deny":
            return _hooked(store, llm, child, deny="reviewer")
        return _hooked(store, llm, child, hook=approval_hook)

    first = build(ContextLLM([_asked("reviewer", {"task": BRIEF}, "r1")]), reviewer(ContextLLM([])))
    await _die_at(
        monkeypatch, react, "_prepare_tool_calls", lambda: first.run("go", tenant_id=TENANT)
    )
    root_id = await root_id_of(engine)
    effects.clear()
    item = await expire(store, engine, root_id)
    child_llm = ContextLLM([_said(REPORT)])
    root_llm = ContextLLM([_said("handled")])
    result = await build(root_llm, reviewer(child_llm)).reclaim(
        root_id, tenant_id=TENANT, item=item
    )
    assert result.status == RunStatus.SUCCESS and result.answer == "handled"
    assert effects == ["hook:reviewer"] and child_llm.calls == 0
    assert not await children_of(engine, root_id)
    [seen] = [m for m in root_llm.messages[0] if m.role.value == "tool"]
    assert REPORT not in seen.content


async def test_recovered_unadmitted_oversize_brief_is_refused_before_bind(
    store: Any, engine: Any, monkeypatch: Any
) -> None:
    from dendrux.loops import react
    from dendrux.types import Limits

    def child(llm: Any) -> Agent:
        return reviewer(llm, limits=Limits(max_input_bytes=8))

    first = _hooked(
        store, ContextLLM([_asked("reviewer", {"task": BRIEF}, "r1")]), child(ContextLLM([]))
    )
    await _die_at(
        monkeypatch, react, "_prepare_tool_calls", lambda: first.run("go", tenant_id=TENANT)
    )
    root_id = await root_id_of(engine)
    item = await expire(store, engine, root_id)
    child_llm = ContextLLM([])
    root_llm = ContextLLM([_said("handled")])
    result = await _hooked(store, root_llm, child(child_llm)).reclaim(
        root_id, tenant_id=TENANT, item=item
    )
    assert result.status == RunStatus.SUCCESS and child_llm.calls == 0
    assert not await children_of(engine, root_id)
    [seen] = [m for m in root_llm.messages[0] if m.role.value == "tool"]
    assert "input limit" in seen.content


@pytest.mark.parametrize("gate", ["gated_sibling", "unknown_sibling"])
async def test_recovered_managed_call_needing_approval_is_refused_before_effects(
    store: Any, engine: Any, monkeypatch: Any, gate: str
) -> None:
    """Kept refusal: a recovered approval pause holding a managed call would continue
    through today's approval path, which pauses that call for a client (see report)."""
    global _CRASH_IN_TOOL
    from dendrux.errors import ManagedUnsupportedError
    from dendrux.loops import react

    kw: dict[str, Any] = {"require_approval": ["fetch"]} if gate == "gated_sibling" else {}
    names = ("reviewer", "fetch") if gate == "gated_sibling" else ("send", "reviewer")
    first = _hooked(store, ContextLLM([_mixed(*names)]), reviewer(ContextLLM([])), **kw)
    if gate == "gated_sibling":  # the gate is recorded; the approval pause is not
        await _die_at(
            monkeypatch, react, "_pause_result", lambda: first.run("go", tenant_id=TENANT)
        )
    else:  # the non-idempotent sibling started; its outcome is unknown
        _CRASH_IN_TOOL = "raise"
        try:
            with pytest.raises(ProcessDeathError):
                await first.run("go", tenant_id=TENANT)
        finally:
            _CRASH_IN_TOOL = False
    root_id = await root_id_of(engine)
    effects.clear()
    item = await expire(store, engine, root_id)
    before = await _snapshot(engine, root_id)
    child_llm = ContextLLM([_said(REPORT)])
    fresh = _hooked(store, ContextLLM([]), reviewer(child_llm), **kw)
    with pytest.raises(ManagedUnsupportedError) as info:
        await fresh.reclaim(root_id, tenant_id=TENANT, item=item)
    assert info.value.operation == "recover_managed_approval"
    assert effects == [] and child_llm.calls == 0
    assert not await children_of(engine, root_id)
    assert (await _snapshot(engine, root_id))["children"] == before["children"]


async def test_recovered_mixed_batch_honours_a_requested_pause_then_binds_on_resume(
    store: Any, engine: Any, monkeypatch: Any
) -> None:
    """A pause requested before recovery stops before any effect; the root-id resume
    then prepares, runs the server call and binds the managed call once."""
    from dendrux.loops import react

    first = _hooked(store, ContextLLM([_mixed("reviewer", "fetch")]), reviewer(ContextLLM([])))
    await _die_at(
        monkeypatch, react, "_prepare_tool_calls", lambda: first.run("go", tenant_id=TENANT)
    )
    root_id = await root_id_of(engine)
    effects.clear()
    item = await expire(store, engine, root_id)
    await store.request_pause(root_id)
    child_llm = ContextLLM([_said(REPORT)])
    agent = _hooked(store, ContextLLM([_said("root done")]), reviewer(child_llm))
    paused = await agent.reclaim(root_id, tenant_id=TENANT, item=item)
    assert paused.status == RunStatus.PAUSED and effects == []
    assert sorted(c.name for c in paused.pending.tool_calls) == ["fetch", "reviewer"]
    assert not await children_of(engine, root_id)
    result = await agent.resume(root_id, tenant_id=TENANT, expected_pause_id=paused.pause_id)
    assert result.status == RunStatus.SUCCESS
    assert sorted(effects) == ["fetch:1", "hook:fetch", "hook:reviewer"]
    [child] = await children_of(engine, root_id)
    assert child_llm.served == [child.id]


# ---------------------------------------------------------------------------
# Fix round 1, R3: the bound child receives exactly the admitted brief
# ---------------------------------------------------------------------------

EMAIL = "review@example.com"
RESTORED = f"Contact {EMAIL}"


@pytest.mark.parametrize("cut", ["live", "unprepared", "pre_handoff", "pre_bind"])
async def test_recovered_child_gets_the_brief_its_admission_saw(
    store: Any, engine: Any, monkeypatch: Any, cut: str
) -> None:
    """With PII restore, the hook, the brief limit and the child see the restored
    brief, live and after a crash at each pre-bind cut; the limit is exactly its size,
    so a different brief could not pass. No raw value reaches the root's model."""
    from dendrux.guardrails import PII
    from dendrux.loops import react
    from dendrux.types import Limits

    hooked: list[Any] = []

    async def hook(ctx: Any) -> None:
        hooked.append(dict(ctx.params))

    def root(llm: Any, child_llm: Any) -> Agent:
        child = reviewer(child_llm, limits=Limits(max_input_bytes=len(RESTORED.encode())))
        return Agent(
            name="lead",
            prompt="You lead.",
            provider=llm,
            state_store=store,
            subagents=[child],
            before_tool=hook,
            guardrails=[PII(action="redact", restore="all")],
        )

    asked = _asked("reviewer", {"task": "Contact <<EMAIL_ADDRESS_1>>"}, "r1")
    child_llm = ContextLLM([_said(REPORT)])
    root_llm = ContextLLM([asked, _said("root done")])
    first = root(root_llm, child_llm)
    if cut == "live":
        result = await first.run(f"Contact {EMAIL}", tenant_id=TENANT)
        model_calls = root_llm.messages
    else:
        target, name = {
            "unprepared": (react, "_prepare_tool_calls"),
            "pre_handoff": (_managed, "prepare_handoff"),
            "pre_bind": (_managed, "bind_child"),
        }[cut]
        await _die_at(
            monkeypatch, target, name, lambda: first.run(f"Contact {EMAIL}", tenant_id=TENANT)
        )
        root_id = await root_id_of(engine)
        item = await expire(store, engine, root_id)
        fresh_root_llm = ContextLLM([_said("root done")])
        result = await root(fresh_root_llm, child_llm).reclaim(root_id, tenant_id=TENANT, item=item)
        model_calls = root_llm.messages + fresh_root_llm.messages
    assert result.status == RunStatus.SUCCESS
    assert hooked and all(h == {"task": RESTORED} for h in hooked)
    [child] = await children_of(engine, result.run_id)
    assert child_llm.served == [child.id]
    briefs = [m.content for m in child_llm.messages[0] if m.role.value == "user"]
    assert briefs == [RESTORED]
    assert EMAIL not in repr(model_calls)
    traces = await rows(engine, ReactTrace, ReactTrace.agent_run_id == result.run_id)
    assert EMAIL not in " ".join(t.content or "" for t in traces)


if __name__ == "__main__":
    import asyncio
    import os
    import sys

    asyncio.run(_phase_one(sys.argv[1], os.environ["T4_CRASH_URL"]))
