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

from contextlib import asynccontextmanager
from typing import Any

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from dendrux import Agent, tool
from dendrux.db.models import AgentRun, ReactTrace, RunEvent
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
