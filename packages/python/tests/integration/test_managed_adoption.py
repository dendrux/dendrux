"""Managed subagents (0.7, T2b): durable adoption, its crash cuts, failures and refusals.

Adoption is ``submit_and_claim`` on the parent's private wait generation, then one
``complete_tool_call`` that resolves the started intent, writes the reserved
observation and clears the binding. A restart reaches the same path through
``Agent.reclaim(root_id)``. Runs on SQLite and Postgres (``DENDRUX_TEST_PG_URL``,
a private ``*_test`` database); contender races run on Postgres only.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from dendrux import Agent, tool
from dendrux.db.models import AgentRun, ReactTrace, RunEvent, TokenUsage, ToolCallRecord
from dendrux.errors import (
    ManagedChildControlError,
    ManagedUnsupportedError,
    RunFencedError,
    RunNotReclaimableError,
)
from dendrux.runtime import _managed, _reclaim
from dendrux.runtime.context import get_delegation_context
from dendrux.runtime.stores._factory import store_for_engine
from dendrux.types import Limits, LLMResponse, RunStatus, ToolCall
from tests._helpers.recovery_checkpoint import expire
from tests._helpers.scripted_provider import ScriptedLLM

BRIEF = "SENTINEL-adopt-brief-1b7c"
REPORT = "SENTINEL-adopt-report-3e9f"
TENANT = "t1"


class CrashError(Exception):
    """Stands in for a process dying at a chosen boundary."""


class ContextLLM(ScriptedLLM):
    """Scripted provider that records the run each call is served under (real ContextVar)."""

    def __init__(self, responses: list[LLMResponse]) -> None:
        super().__init__(responses)
        self.served: list[str | None] = []

    async def complete(self, messages: Any, tools: Any = None, **kwargs: Any) -> LLMResponse:
        ctx = get_delegation_context()
        self.served.append(ctx.run_id if ctx else None)
        return await super().complete(messages, tools, **kwargs)


def call(name: str, params: dict[str, Any], pid: str) -> LLMResponse:
    return LLMResponse(tool_calls=[ToolCall(name=name, params=params, provider_tool_call_id=pid)])


def child_agent(llm: Any, **kwargs: Any) -> Agent:
    return Agent(
        name="reviewer",
        description="Reviews one draft and reports.",
        prompt="You review.",
        provider=llm,
        **kwargs,
    )


def root_agent(store: Any, llm: Any, child: Agent, **kwargs: Any) -> Agent:
    return Agent(
        name="lead",
        prompt="You lead.",
        provider=llm,
        state_store=store,
        subagents=[child],
        on_governance_change="raise",
        **kwargs,
    )


async def count(engine: Any, model: Any, *where: Any) -> int:
    async with AsyncSession(engine) as session:
        return int(await session.scalar(select(func.count()).select_from(model).where(*where)))


async def rows(engine: Any, model: Any, *where: Any) -> list[Any]:
    async with AsyncSession(engine) as session:
        return list((await session.execute(select(model).where(*where))).scalars().all())


async def root_id_of(engine: Any) -> str:
    return (await rows(engine, AgentRun, AgentRun.agent_name == "lead"))[0].id


async def child_of(engine: Any, root_id: str) -> Any:
    return (await rows(engine, AgentRun, AgentRun.parent_run_id == root_id))[0]


async def observations(engine: Any, root_id: str) -> list[Any]:
    traces = await rows(engine, ReactTrace, ReactTrace.agent_run_id == root_id)
    return [t for t in traces if t.role == "tool"]


async def crash_before_adoption(store: Any, engine: Any, monkeypatch: Any) -> tuple[str, str]:
    """Run until the child is terminal, then die before the parent adopts it."""

    async def die(*args: Any, **kwargs: Any) -> Any:
        raise CrashError

    child_llm = ContextLLM([LLMResponse(text=REPORT)])
    root_llm = ContextLLM([call("reviewer", {"task": BRIEF}, "r1")])
    root = root_agent(store, root_llm, child_agent(child_llm))
    with monkeypatch.context() as patch:
        patch.setattr(_managed, "adopt", die)
        with pytest.raises(CrashError):
            await root.run("go", tenant_id=TENANT)
    root_id = await root_id_of(engine)
    return root_id, (await child_of(engine, root_id)).id


def fresh_root(store: Any, *responses: LLMResponse) -> tuple[Agent, ContextLLM, ContextLLM]:
    """A new process: new Agent objects and providers, same store."""
    child_llm = ContextLLM([])
    root_llm = ContextLLM(list(responses))
    return root_agent(store, root_llm, child_agent(child_llm)), root_llm, child_llm


# ---------------------------------------------------------------------------
# Spec test 7: the adoption commit and its crash cuts
# ---------------------------------------------------------------------------


async def test_adoption_commit_contains_tool_outcome_trace_and_marker(
    store: Any, engine: Any, monkeypatch: Any
) -> None:
    """Cut 1: child terminal, parent not adopted. A fresh driver adopts; no child rerun."""
    root_id, child_id = await crash_before_adoption(store, engine, monkeypatch)
    root_row = (await rows(engine, AgentRun, AgentRun.id == root_id))[0]
    assert root_row.status == "paused" and root_row.pause_data["schema_version"] == 3
    intent = (await rows(engine, ToolCallRecord, ToolCallRecord.agent_run_id == root_id))[0]
    assert intent.outcome == "started"
    usage_before = await count(engine, TokenUsage, TokenUsage.agent_run_id == child_id)

    root, root_llm, child_llm = fresh_root(store, LLMResponse(text="root done"))
    result = await root.reclaim(root_id, tenant_id=TENANT)

    assert result.status == RunStatus.SUCCESS and result.answer == "root done"
    assert child_llm.calls == 0 and root_llm.calls == 1
    seen = [m for m in root_llm.messages[0] if m.role.value == "tool"]
    assert [json.loads(m.content) for m in seen] == [REPORT]
    intent = (await rows(engine, ToolCallRecord, ToolCallRecord.agent_run_id == root_id))[0]
    assert intent.outcome == "completed" and intent.meta["managed"]["child_run_id"] == child_id
    assert len(await observations(engine, root_id)) == 1
    assert await count(engine, TokenUsage, TokenUsage.agent_run_id == child_id) == usage_before
    root_row = (await rows(engine, AgentRun, AgentRun.id == root_id))[0]
    assert root_row.loop_state.get("managed_wait") is None
    assert root_row.loop_state.get("submission") is None


async def test_crash_after_adoption_claim_is_consumed_through_tool_completion_once(
    store: Any, engine: Any, monkeypatch: Any
) -> None:
    """Cut 2: the report is durably accepted, the tool row is not yet completed."""
    from dendrux.runtime import persistence

    root_id, child_id = await crash_before_adoption(store, engine, monkeypatch)
    root, _, _ = fresh_root(store)

    async def die(*args: Any, **kwargs: Any) -> Any:
        raise CrashError

    with monkeypatch.context() as patch:
        patch.setattr(persistence, "adopt_managed_result", die)
        with pytest.raises(CrashError):
            await root.reclaim(root_id, tenant_id=TENANT)
    row = (await rows(engine, AgentRun, AgentRun.id == root_id))[0]
    assert row.status == "running" and row.loop_state["submission"] is not None
    assert row.loop_state["managed_wait"]["child_run_id"] == child_id
    intent = (await rows(engine, ToolCallRecord, ToolCallRecord.agent_run_id == root_id))[0]
    assert intent.outcome == "started"

    item = await expire(store, engine, root_id)
    root, root_llm, child_llm = fresh_root(store, LLMResponse(text="root done"))
    result = await root.reclaim(root_id, tenant_id=TENANT, item=item)

    assert result.status == RunStatus.SUCCESS
    assert child_llm.calls == 0 and root_llm.calls == 1
    intent = (await rows(engine, ToolCallRecord, ToolCallRecord.agent_run_id == root_id))[0]
    assert intent.outcome == "completed"
    assert len(await observations(engine, root_id)) == 1
    events = await rows(engine, RunEvent, RunEvent.agent_run_id == root_id)
    assert [e.event_type for e in events].count("tool.completed") == 1


async def test_lost_adoption_acknowledgement_never_appends_twice(
    store: Any, engine: Any, monkeypatch: Any
) -> None:
    """Cut 3: the completion committed, the caller died. Retry and restart add nothing."""
    from dendrux.runtime.persistence import adopt_managed_result

    root_id, child_id = await crash_before_adoption(store, engine, monkeypatch)
    claims: list[Any] = []
    original = _reclaim.continue_claimed

    async def die(agent: Any, store_: Any, run_id: str, claim: Any, **kwargs: Any) -> Any:
        claims.append(claim)
        raise CrashError

    root, _, _ = fresh_root(store)
    with monkeypatch.context() as patch:
        patch.setattr(_reclaim, "continue_claimed", die)
        with pytest.raises(CrashError):
            await root.reclaim(root_id, tenant_id=TENANT)
    assert _reclaim.continue_claimed is original
    traces_before = await count(engine, ReactTrace, ReactTrace.agent_run_id == root_id)

    # Replaying the same completion (a lost acknowledgement's retry) is a no-op.
    claim = claims[0]
    row = (await rows(engine, ToolCallRecord, ToolCallRecord.agent_run_id == root_id))[0]
    assert row.outcome == "completed"
    before_adoption = SimpleNamespace(
        claim_epoch=claim.claim_epoch,
        loop_state={
            **claim.loop_state,
            "managed_wait": {"tool_call_id": row.tool_call_id, "child_run_id": child_id},
            "pending": [{"call_id": row.tool_call_id, "order_index": 3, "tool": "reviewer"}],
        },
    )
    from dendrux.runtime.state import claimed_store

    await adopt_managed_result(
        claimed_store(store, root_id, claim.claim_epoch),
        root_id,
        claim=before_adoption,
        call_id=row.tool_call_id,
        report=await _report(engine, root_id),
        child_run_id=child_id,
        agent_path=("reviewer",),
    )
    assert await count(engine, ReactTrace, ReactTrace.agent_run_id == root_id) == traces_before

    item = await expire(store, engine, root_id)
    root, root_llm, child_llm = fresh_root(store, LLMResponse(text="root done"))
    result = await root.reclaim(root_id, tenant_id=TENANT, item=item)
    assert result.status == RunStatus.SUCCESS
    assert child_llm.calls == 0 and root_llm.calls == 1
    assert len(await observations(engine, root_id)) == 1
    tool_rows = await rows(engine, ToolCallRecord, ToolCallRecord.agent_run_id == root_id)
    assert [r.outcome for r in tool_rows] == ["completed"]


async def _report(engine: Any, root_id: str) -> Any:
    from dendrux.types import ToolResult

    row = (await rows(engine, ToolCallRecord, ToolCallRecord.agent_run_id == root_id))[0]
    return ToolResult(
        name=row.tool_name,
        call_id=row.tool_call_id,
        payload=json.dumps(row.result),
        success=row.success,
        error=row.error_message,
    )


# ---------------------------------------------------------------------------
# Spec test 8: contenders, stale parent writer, cancel versus claim (Postgres)
# ---------------------------------------------------------------------------


def _postgres(engine: Any) -> None:
    if engine.dialect.name != "postgresql":
        pytest.skip("independent-session contenders run on Postgres")


async def test_pg_adoption_contenders_and_stale_parent(
    store: Any, engine: Any, monkeypatch: Any
) -> None:
    _postgres(engine)
    for _ in range(3):
        root_id, child_id = await crash_before_adoption(store, engine, monkeypatch)

        # The bind retired the parent's creation epoch: its old writer is fenced.
        with pytest.raises(RunFencedError):
            await store.save_trace(
                root_id,
                "assistant",
                "stale",
                order_index=99,
                claim_epoch=0,
            )

        second = create_async_engine(engine.url, pool_size=6)
        try:
            other = store_for_engine(second)
            contenders = []
            for i in range(8):
                target = store if i % 2 == 0 else other
                agent, llm, child_llm = fresh_root(target, LLMResponse(text="root done"))
                contenders.append((agent, llm, child_llm))
            results = await asyncio.gather(
                *(a.reclaim(root_id, tenant_id=TENANT) for a, _, _ in contenders),
                return_exceptions=True,
            )
        finally:
            await second.dispose()
        # Losers either read the root back or meet its live owner (lease_fresh).
        for r in results:
            if isinstance(r, BaseException):
                assert isinstance(r, RunNotReclaimableError), r
                assert r.reason in {"lease_fresh", "already_reclaimed", "not_running"}
        assert sum(llm.calls for _, llm, _ in contenders) == 1
        assert sum(c.calls for _, _, c in contenders) == 0
        assert len(await observations(engine, root_id)) == 1
        events = await rows(engine, RunEvent, RunEvent.agent_run_id == root_id)
        assert [e.event_type for e in events].count("tool.completed") == 1
        final = (await rows(engine, AgentRun, AgentRun.id == root_id))[0]
        assert final.status == "success"
        async with engine.begin() as conn:
            await conn.execute(AgentRun.__table__.delete().where(AgentRun.agent_name == "lead"))


async def test_pg_cancel_versus_adoption_claim_never_revives_the_root(
    store: Any, engine: Any, monkeypatch: Any
) -> None:
    _postgres(engine)
    outcomes = set()
    for _ in range(4):
        root_id, _ = await crash_before_adoption(store, engine, monkeypatch)
        root, root_llm, child_llm = fresh_root(store, LLMResponse(text="root done"))
        canceller, _, _ = fresh_root(store)
        results = await asyncio.gather(
            root.reclaim(root_id, tenant_id=TENANT),
            canceller.cancel_run(root_id, tenant_id=TENANT),
            return_exceptions=True,
        )
        for r in results:  # only the ordinary live-owner refusal is acceptable
            if isinstance(r, BaseException):
                assert isinstance(r, RunNotReclaimableError), r
                assert r.reason in {"lease_fresh", "cancelled", "not_running"}, r.reason
        final = (await rows(engine, AgentRun, AgentRun.id == root_id))[0]
        intent = (await rows(engine, ToolCallRecord, ToolCallRecord.agent_run_id == root_id))[0]
        assert final.status in {"cancelled", "success"}
        assert child_llm.calls == 0
        if final.status == "cancelled" and intent.outcome == "started":
            assert root_llm.calls == 0 and not await observations(engine, root_id)
        outcomes.add((final.status, intent.outcome))
        # A cancelled root is never revived by a later adoption attempt.
        if final.status == "cancelled":
            again, again_llm, _ = fresh_root(store, LLMResponse(text="x"))
            await again.reclaim(root_id, tenant_id=TENANT)
            assert again_llm.calls == 0
            assert (await rows(engine, AgentRun, AgentRun.id == root_id))[0].status == "cancelled"
        async with engine.begin() as conn:
            await conn.execute(AgentRun.__table__.delete().where(AgentRun.agent_name == "lead"))
    assert outcomes


# ---------------------------------------------------------------------------
# Spec test 9: failures and oversize keep the child's outcome
# ---------------------------------------------------------------------------


class BoomError(Exception):
    pass


class FailingLLM(ContextLLM):
    async def complete(self, messages: Any, tools: Any = None, **kwargs: Any) -> LLMResponse:
        self.calls += 1
        raise BoomError("SENTINEL-vendor-text-should-not-leak")


@tool()
async def spin() -> str:
    """Loop."""
    return "again"


@pytest.mark.parametrize("kind", ["error", "max_iterations", "oversize", "cancelled"])
async def test_child_failure_and_oversize_preserve_outcome(
    store: Any, engine: Any, kind: str
) -> None:
    root_llm = ContextLLM([call("reviewer", {"task": BRIEF}, "r1"), LLMResponse(text="handled")])
    limits = None
    if kind == "error":
        child_llm: ContextLLM = FailingLLM([])
        child = child_agent(child_llm)
    elif kind == "max_iterations":
        child_llm = ContextLLM([call("spin", {}, "s1")])
        child = child_agent(child_llm, tools=[spin], max_iterations=1)
    elif kind == "oversize":
        child_llm = ContextLLM([LLMResponse(text=REPORT * 40)])
        child = child_agent(child_llm)
        limits = Limits(max_tool_result_bytes=256)
    else:

        class CancellingLLM(ContextLLM):
            async def complete(self, messages: Any, tools: Any = None, **kw: Any) -> LLMResponse:
                ctx = get_delegation_context()
                assert ctx is not None
                await store.request_cancel(ctx.run_id)
                return await super().complete(messages, tools, **kw)

        child_llm = CancellingLLM([call("spin", {}, "s1"), LLMResponse(text="never")])
        child = child_agent(child_llm, tools=[spin])
    root = root_agent(store, root_llm, child, **({"limits": limits} if limits else {}))

    result = await root.run("go", tenant_id=TENANT)

    assert result.status == RunStatus.SUCCESS and result.answer == "handled"
    child_row = await child_of(engine, result.run_id)
    expected_child = {
        "error": "error",
        "max_iterations": "max_iterations",
        "oversize": "success",
        "cancelled": "cancelled",
    }[kind]
    assert child_row.status == expected_child
    code = {
        "error": "managed_child_failed",
        "max_iterations": "managed_child_failed",
        "oversize": "managed_report_too_large",
        "cancelled": "managed_child_cancelled",
    }[kind]
    seen = [json.loads(m.content) for m in root_llm.messages[1] if m.role.value == "tool"]
    assert len(seen) == 1 and seen[0]["code"] == code
    assert seen[0]["child_run_id"] == child_row.id
    text = json.dumps(seen)
    assert "SENTINEL-vendor-text" not in text and REPORT not in text
    if kind == "oversize":
        assert seen[0]["limit"] == 256 and seen[0]["bytes"] > 256
    intent = (await rows(engine, ToolCallRecord, ToolCallRecord.agent_run_id == result.run_id))[0]
    assert intent.outcome == "failed"
    completed = [
        e
        for e in await rows(engine, RunEvent, RunEvent.agent_run_id == result.run_id)
        if e.event_type == "tool.completed"
    ]
    assert completed[0].data["managed_outcome"] == "failed"
    assert completed[0].data["code"] == code
    calls_before = child_llm.calls
    # Reconnecting a finished workflow never reruns the child to fix delivery.
    again, _, again_child = fresh_root(store)
    assert (await again.reclaim(result.run_id, tenant_id=TENANT)).status == RunStatus.SUCCESS
    assert child_llm.calls == calls_before and again_child.calls == 0


class Verdict(BaseModel):
    verdict: str


def test_child_report_mapping_is_safe() -> None:
    record = SimpleNamespace(id="01" + "A" * 24, status="success", answer='{"wrong": 1}')
    typed = SimpleNamespace(name="scorer", _output_type=Verdict)
    report = _managed.child_report(record, typed, limit=1024)  # type: ignore[arg-type]
    assert not report.success and report.meta["code"] == "managed_report_invalid"
    assert "wrong" not in report.payload
    exact = json.dumps("x" * 10)
    text = SimpleNamespace(name="r", _output_type=None)
    ok = SimpleNamespace(id=record.id, status="success", answer="x" * 10)
    assert _managed.child_report(ok, text, limit=len(exact)).success  # type: ignore[arg-type]
    over = _managed.child_report(ok, text, limit=len(exact) - 1)  # type: ignore[arg-type]
    assert over.meta["code"] == "managed_report_too_large"


async def test_oversize_brief_is_refused_before_bind(store: Any, engine: Any) -> None:
    child_llm = ContextLLM([LLMResponse(text="never")])
    child = child_agent(child_llm, limits=Limits(max_input_bytes=16))
    root_llm = ContextLLM([call("reviewer", {"task": BRIEF * 4}, "r1"), LLMResponse(text="ok")])
    root = root_agent(store, root_llm, child)
    result = await root.run("go", tenant_id=TENANT)
    assert result.status == RunStatus.SUCCESS and child_llm.calls == 0
    assert not await rows(engine, AgentRun, AgentRun.parent_run_id == result.run_id)
    seen = [json.loads(m.content) for m in root_llm.messages[1] if m.role.value == "tool"]
    assert seen == [{"error": seen[0]["error"], "bytes": len(BRIEF * 4), "limit": 16}]


# ---------------------------------------------------------------------------
# Spec test 10: a child's external pause, direct child control, shared templates
# ---------------------------------------------------------------------------


@tool()
async def publish(text: str) -> str:
    """Publish."""
    return "published"


async def test_child_pause_is_saved_without_fake_parent_result(store: Any, engine: Any) -> None:
    child_llm = ContextLLM([call("publish", {"text": "t"}, "c1")])
    child = child_agent(child_llm, tools=[publish], require_approval=["publish"])
    root_llm = ContextLLM([call("reviewer", {"task": BRIEF}, "r1")])
    root = root_agent(store, root_llm, child)

    result = await root.run("go", tenant_id=TENANT)

    root_id = result.run_id
    assert root_id == await root_id_of(engine)
    child_row = await child_of(engine, root_id)
    assert result.status == RunStatus.WAITING_APPROVAL
    assert result.pause_id == child_row.pause_data["pause_id"]
    pending = result.pending
    assert (pending.origin_run_id, pending.agent_name, pending.agent_path) == (
        child_row.id,
        "reviewer",
        ["lead", "reviewer"],
    )
    assert child_row.status == "waiting_approval"
    root_row = (await rows(engine, AgentRun, AgentRun.id == root_id))[0]
    assert root_row.status == "paused" and root_row.pause_data["schema_version"] == 3
    intent = (await rows(engine, ToolCallRecord, ToolCallRecord.agent_run_id == root_id))[0]
    assert intent.outcome == "started"
    assert not await observations(engine, root_id)

    # Root-facing continuations are explicitly unsupported in this build (T3).
    with pytest.raises(ManagedUnsupportedError):
        await root.submit_approval(
            root_id, approved=True, expected_pause_id=result.pause_id, tenant_id=TENANT
        )
    with pytest.raises(ManagedUnsupportedError):
        root.stream("go", tenant_id=TENANT)

    # Direct control of the child is refused after the tenant check; reads work.
    direct = child_agent(ContextLLM([]), tools=[publish], state_store=store)
    for attempt in (
        lambda: direct.submit_approval(
            child_row.id, approved=True, expected_pause_id=result.pause_id, tenant_id=TENANT
        ),
        lambda: direct.cancel_run(child_row.id, tenant_id=TENANT),
        lambda: direct.request_pause(child_row.id, tenant_id=TENANT),
        lambda: direct.resume(child_row.id, tenant_id=TENANT, expected_pause_id=result.pause_id),
    ):
        with pytest.raises(ManagedChildControlError) as caught:
            await attempt()
        assert caught.value.code == "managed_root_required"
    from dendrux.errors import RunNotFoundError

    with pytest.raises(RunNotFoundError):
        await direct.cancel_run(child_row.id, tenant_id="other-tenant")
    loaded = await direct.get_result(child_row.id, tenant_id=TENANT)
    assert loaded.status == RunStatus.WAITING_APPROVAL
    assert (await rows(engine, AgentRun, AgentRun.id == child_row.id))[0].status == (
        "waiting_approval"
    )


async def test_shared_template_across_tenants_does_not_leak(store: Any, engine: Any) -> None:
    template = child_agent(ContextLLM([LLMResponse(text="a"), LLMResponse(text="b")]))
    before = (template.name, template.tools, template.provider, template._state_store)
    roots = [
        root_agent(
            store, ContextLLM([call("reviewer", {"task": t}, "r1"), LLMResponse(text=t)]), template
        )
        for t in ("A", "B")
    ]
    if engine.dialect.name == "postgresql":
        results = await asyncio.gather(
            roots[0].run("go", tenant_id="tenant-a"), roots[1].run("go", tenant_id="tenant-b")
        )
    else:  # in-memory SQLite shares one connection: run one after the other
        results = [
            await roots[0].run("go", tenant_id="tenant-a"),
            await roots[1].run("go", tenant_id="tenant-b"),
        ]
    for result, tenant in zip(results, ("tenant-a", "tenant-b"), strict=True):
        assert result.status == RunStatus.SUCCESS
        child = await child_of(engine, result.run_id)
        assert child.tenant_id == tenant
    assert (template.name, template.tools, template.provider, template._state_store) == before


class ProcessDeathError(BaseException):
    """Ends the caller like a killed process: no runner cleanup finalizes the run."""


async def test_running_child_is_reclaimed_only_through_its_root(store: Any, engine: Any) -> None:
    """Narrow reconnect: a child left running after a crash resumes via the root id."""

    class DyingLLM(ContextLLM):
        async def complete(self, messages: Any, tools: Any = None, **kw: Any) -> LLMResponse:
            if self.calls == 1:
                self.calls += 1
                raise ProcessDeathError
            return await super().complete(messages, tools, **kw)

    @tool()
    async def note(text: str) -> str:
        """Note."""
        return "noted"

    child_llm = DyingLLM([call("note", {"text": "x"}, "c1")])
    root_llm = ContextLLM([call("reviewer", {"task": BRIEF}, "r1")])
    root = root_agent(store, root_llm, child_agent(child_llm, tools=[note]))
    with pytest.raises(ProcessDeathError):
        await root.run("go", tenant_id=TENANT)
    root_id = await root_id_of(engine)
    child = await child_of(engine, root_id)
    assert child.status == "running"

    item = await expire(store, engine, child.id)
    # The child is on its last allowed reclaim attempt.
    async with engine.begin() as conn:
        await conn.execute(
            AgentRun.__table__.update()
            .where(AgentRun.id == child.id)
            .values(reclaim_limit=AgentRun.reclaim_attempts + 1)
        )
    before = await _recovery_state(engine, child.id)
    direct = child_agent(ContextLLM([]), tools=[note], state_store=store)
    with pytest.raises(ManagedChildControlError):
        await direct.reclaim(child.id, tenant_id=TENANT, item=item)
    # Refused before the claim: epoch, lease mark, attempts and events unchanged.
    assert await _recovery_state(engine, child.id) == before

    child_llm2 = ContextLLM([LLMResponse(text=REPORT)])
    root_llm2 = ContextLLM([LLMResponse(text="root done")])
    restarted = root_agent(store, root_llm2, child_agent(child_llm2, tools=[note]))
    result = await restarted.reclaim(root_id, tenant_id=TENANT)
    assert result.status == RunStatus.SUCCESS and result.answer == "root done"
    assert child_llm2.calls == 1 and root_llm2.calls == 1
    assert set(child_llm2.served) == {child.id}
    seen = [json.loads(m.content) for m in root_llm2.messages[0] if m.role.value == "tool"]
    assert seen == [REPORT]


async def test_duplicate_bind_never_drives_a_second_child(
    store: Any, engine: Any, monkeypatch: Any
) -> None:
    """Spec test 3 at the driver: a repeat bind reads back and never runs a child."""
    from dataclasses import replace

    from dendrux.types import generate_ulid

    repeats: list[Any] = []
    original = _managed._run_child

    async def twice(invocation: Any, notifier: Any) -> Any:
        first = await original(invocation, notifier)
        repeats.append(await original(replace(invocation, child_run_id=generate_ulid()), notifier))
        return first

    monkeypatch.setattr(_managed, "_run_child", twice)
    child_llm = ContextLLM([LLMResponse(text=REPORT)])
    root_llm = ContextLLM([call("reviewer", {"task": BRIEF}, "r1"), LLMResponse(text="ok")])
    result = await root_agent(store, root_llm, child_agent(child_llm)).run("go", tenant_id=TENANT)

    assert result.status == RunStatus.SUCCESS
    assert child_llm.calls == 1
    ((repeat, bound),) = repeats
    assert bound is False and repeat.meta.get(_managed.NOT_OWNER_KEY)
    assert len(await rows(engine, AgentRun, AgentRun.parent_run_id == result.run_id)) == 1


async def test_blocked_child_entry_keeps_the_native_block_mark(store: Any, engine: Any) -> None:
    """A brief the child's guardrail blocks: the bound child carries the store's block mark."""
    from dendrux._meta_keys import BLOCKED_META_KEY
    from dendrux.guardrails import SecretDetection

    secret = "AKIAIOSFODNN7EXAMPLE"
    child_llm = ContextLLM([LLMResponse(text="never")])
    child = child_agent(child_llm, guardrails=[SecretDetection(action="block")])
    root_llm = ContextLLM(
        [call("reviewer", {"task": f"use {secret}"}, "r1"), LLMResponse(text="handled")]
    )
    result = await root_agent(store, root_llm, child).run("go", tenant_id=TENANT)

    assert result.status == RunStatus.SUCCESS and child_llm.calls == 0
    child_row = await child_of(engine, result.run_id)
    assert child_row.status == "error"
    assert child_row.meta.get(BLOCKED_META_KEY) is True
    seen = [json.loads(m.content) for m in root_llm.messages[1] if m.role.value == "tool"]
    assert seen[0]["code"] == "managed_child_failed"
    assert secret not in json.dumps(seen)


async def _recovery_state(engine: Any, run_id: str) -> tuple[Any, ...]:
    row = (await rows(engine, AgentRun, AgentRun.id == run_id))[0]
    events = await count(engine, RunEvent, RunEvent.agent_run_id == run_id)
    return (
        row.claim_epoch,
        row.reclaim_attempts,
        row.lease_expired_at is not None,
        row.status,
        events,
    )


async def test_pg_direct_child_reclaim_never_consumes_root_recovery(
    store: Any, engine: Any
) -> None:
    """Independent sessions: a direct child reclaim refuses; the root recovers at once."""
    _postgres(engine)

    @tool()
    async def note(text: str) -> str:
        """Note."""
        return "noted"

    for _ in range(3):

        class DyingLLM(ContextLLM):
            async def complete(self, messages: Any, tools: Any = None, **kw: Any) -> LLMResponse:
                if self.calls == 1:
                    self.calls += 1
                    raise ProcessDeathError
                return await super().complete(messages, tools, **kw)

        root = root_agent(
            store,
            ContextLLM([call("reviewer", {"task": BRIEF}, "r1")]),
            child_agent(DyingLLM([call("note", {"text": "x"}, "c1")]), tools=[note]),
        )
        with pytest.raises(ProcessDeathError):
            await root.run("go", tenant_id=TENANT)
        root_id = await root_id_of(engine)
        child = await child_of(engine, root_id)
        item = await expire(store, engine, child.id)
        async with engine.begin() as conn:
            await conn.execute(
                AgentRun.__table__.update()
                .where(AgentRun.id == child.id)
                .values(reclaim_limit=AgentRun.reclaim_attempts + 1)
            )
        second = create_async_engine(engine.url, pool_size=4)
        try:
            direct = child_agent(ContextLLM([]), tools=[note], state_store=store_for_engine(second))
            child_llm = ContextLLM([LLMResponse(text=REPORT)])
            restarted = root_agent(
                store,
                ContextLLM([LLMResponse(text="root done")]),
                child_agent(child_llm, tools=[note]),
            )
            refused, recovered = await asyncio.gather(
                direct.reclaim(child.id, tenant_id=TENANT, item=item),
                restarted.reclaim(root_id, tenant_id=TENANT),
                return_exceptions=True,
            )
        finally:
            await second.dispose()
        assert isinstance(refused, ManagedChildControlError), refused
        assert not isinstance(recovered, BaseException), recovered
        assert recovered.status == RunStatus.SUCCESS and child_llm.calls == 1
        async with engine.begin() as conn:
            await conn.execute(AgentRun.__table__.delete().where(AgentRun.agent_name == "lead"))


# ---------------------------------------------------------------------------
# R4: recovery of a pending managed batch never dispatches ordinarily
# ---------------------------------------------------------------------------

effects: list[str] = []


@tool()
async def fetch(x: int) -> str:
    """Server effect."""
    effects.append("fetch")
    return "fetched"


def _recovery_root(store: Any, root_llm: Any, *children: Agent, **kwargs: Any) -> Agent:
    async def hook(ctx: Any) -> None:
        effects.append(f"hook:{ctx.tool_name}")

    return Agent(
        name="lead",
        prompt="You lead.",
        provider=root_llm,
        state_store=store,
        tools=[fetch],
        subagents=list(children),
        before_tool=hook,
        **kwargs,
    )


def _named(name: str, llm: Any) -> Agent:
    return Agent(name=name, description=f"{name} child.", prompt="c", provider=llm)


async def _crash_at(monkeypatch: Any, module: Any, name: str, run: Any) -> None:
    async def die(*args: Any, **kwargs: Any) -> Any:
        raise ProcessDeathError

    with monkeypatch.context() as patch:
        patch.setattr(module, name, die)
        with pytest.raises(ProcessDeathError):
            await run()


async def test_recovered_unsupported_batch_is_refused_with_zero_effects(
    store: Any, engine: Any, monkeypatch: Any
) -> None:
    """Crash at the assistant checkpoint (before live preflight), then recover."""
    from dendrux.loops import react

    batch = LLMResponse(
        tool_calls=[
            ToolCall(name="reviewer", params={"task": BRIEF}, provider_tool_call_id="p1"),
            ToolCall(name="auditor", params={"task": BRIEF}, provider_tool_call_id="p2"),
            ToolCall(name="fetch", params={"x": 1}, provider_tool_call_id="p3"),
        ]
    )
    first = _recovery_root(
        store,
        ContextLLM([batch]),
        _named("reviewer", ContextLLM([])),
        _named("auditor", ContextLLM([])),
    )
    await _crash_at(
        monkeypatch, react, "_refuse_managed_batch", lambda: first.run("go", tenant_id=TENANT)
    )
    root_id = await root_id_of(engine)
    effects.clear()
    item = await expire(store, engine, root_id)
    reviewer_llm, auditor_llm = ContextLLM([]), ContextLLM([])
    root_llm = ContextLLM([LLMResponse(text="regrouped")])
    fresh = _recovery_root(
        store, root_llm, _named("reviewer", reviewer_llm), _named("auditor", auditor_llm)
    )
    result = await fresh.reclaim(root_id, tenant_id=TENANT, item=item)

    assert result.status == RunStatus.SUCCESS and result.answer == "regrouped"
    assert effects == [] and reviewer_llm.calls == 0 and auditor_llm.calls == 0
    assert not await rows(engine, AgentRun, AgentRun.parent_run_id == root_id)
    seen = [json.loads(m.content) for m in root_llm.messages[0] if m.role.value == "tool"]
    assert len(seen) == 3 and all("subagent" in o["error"] for o in seen)
    tool_rows = await rows(engine, ToolCallRecord, ToolCallRecord.agent_run_id == root_id)
    assert {r.meta.get("code") for r in tool_rows} == {"managed_batch_unsupported"}


async def test_recovered_pre_bind_handoff_binds_once_not_a_client_pause(
    store: Any, engine: Any, monkeypatch: Any
) -> None:
    """Crash after the handoff checkpoint, before the bind, then recover."""
    first = _recovery_root(
        store,
        ContextLLM([call("reviewer", {"task": BRIEF}, "r1")]),
        _named("reviewer", ContextLLM([])),
    )
    await _crash_at(monkeypatch, _managed, "bind_child", lambda: first.run("go", tenant_id=TENANT))
    root_id = await root_id_of(engine)
    row = (await rows(engine, AgentRun, AgentRun.id == root_id))[0]
    assert row.status == "running" and row.loop_state.get("managed_wait") is None
    assert [p["target"] for p in row.loop_state["pending"]] == ["agent"]
    effects.clear()
    item = await expire(store, engine, root_id)
    child_llm = ContextLLM([LLMResponse(text=REPORT)])
    root_llm = ContextLLM([LLMResponse(text="root done")])
    fresh = _recovery_root(store, root_llm, _named("reviewer", child_llm))
    result = await fresh.reclaim(root_id, tenant_id=TENANT, item=item)

    assert result.status == RunStatus.SUCCESS and result.answer == "root done"
    assert child_llm.calls == 1 and effects == []
    assert len(await rows(engine, AgentRun, AgentRun.parent_run_id == root_id)) == 1
    seen = [json.loads(m.content) for m in root_llm.messages[0] if m.role.value == "tool"]
    assert seen == [REPORT]


async def test_recovered_mixed_pending_batch_is_refused_before_effects(
    store: Any, engine: Any, monkeypatch: Any
) -> None:
    """A managed call still pending beside an ordinary one: explicit staged refusal."""
    from dendrux.loops import react

    batch = LLMResponse(
        tool_calls=[
            ToolCall(name="reviewer", params={"task": BRIEF}, provider_tool_call_id="p1"),
            ToolCall(name="fetch", params={"x": 1}, provider_tool_call_id="p2"),
        ]
    )
    first = _recovery_root(store, ContextLLM([batch]), _named("reviewer", ContextLLM([])))
    await _crash_at(
        monkeypatch, react, "_prepare_tool_calls", lambda: first.run("go", tenant_id=TENANT)
    )
    root_id = await root_id_of(engine)
    effects.clear()
    item = await expire(store, engine, root_id)
    child_llm, root_llm = ContextLLM([]), ContextLLM([])
    fresh = _recovery_root(store, root_llm, _named("reviewer", child_llm))
    with pytest.raises(ManagedUnsupportedError) as info:
        await fresh.reclaim(root_id, tenant_id=TENANT, item=item)
    assert info.value.operation == "recover_mixed_batch"
    assert effects == [] and child_llm.calls == 0 and root_llm.calls == 0
    assert not await rows(engine, AgentRun, AgentRun.parent_run_id == root_id)
    row = (await rows(engine, AgentRun, AgentRun.id == root_id))[0]
    assert row.status != "waiting_client_tool"


# ---------------------------------------------------------------------------
# R2: terminal-child adoption keeps governance and skill-continuity admission
# ---------------------------------------------------------------------------


async def _untouched(engine: Any, root_id: str) -> tuple[Any, ...]:
    row = (await rows(engine, AgentRun, AgentRun.id == root_id))[0]
    intent = (await rows(engine, ToolCallRecord, ToolCallRecord.agent_run_id == root_id))[0]
    return (
        row.status,
        row.claim_epoch,
        "submission" in (row.pause_data or {}),
        intent.outcome,
        len(await observations(engine, root_id)),
    )


@pytest.mark.parametrize("change", ["guardrails", "approval", "skills"])
async def test_changed_parent_definition_is_refused_before_adoption(
    store: Any, engine: Any, monkeypatch: Any, change: str
) -> None:
    from dendrux.errors import GovernanceMismatchError
    from dendrux.guardrails import SecretDetection
    from dendrux.skills import Skill

    root_id, _ = await crash_before_adoption(store, engine, monkeypatch)
    before = await _untouched(engine, root_id)
    assert before == ("paused", 1, False, "started", 0)
    extra: dict[str, Any] = {
        "guardrails": {"guardrails": [SecretDetection(action="block")]},
        "approval": {"tools": [publish], "require_approval": ["publish"]},
        "skills": {"skills": [Skill(name="demo", description="d", body="b")]},
    }[change]
    child_llm = ContextLLM([])
    root_llm = ContextLLM([LLMResponse(text="never")])
    changed = root_agent(store, root_llm, child_agent(child_llm), **extra)
    expected = RunNotReclaimableError if change == "skills" else GovernanceMismatchError
    with pytest.raises(expected) as info:
        await changed.reclaim(root_id, tenant_id=TENANT)
    if change == "skills":
        assert info.value.reason == "skill_mismatch"
    assert await _untouched(engine, root_id) == before
    assert root_llm.calls == 0 and child_llm.calls == 0

    # The unchanged definition still adopts and finishes.
    same, same_llm, _ = fresh_root(store, LLMResponse(text="root done"))
    assert (await same.reclaim(root_id, tenant_id=TENANT)).status == RunStatus.SUCCESS
    assert same_llm.calls == 1


async def test_changed_governance_with_warn_policy_adopts_and_reports(
    store: Any, engine: Any, monkeypatch: Any
) -> None:
    from dendrux.guardrails import SecretDetection

    root_id, _ = await crash_before_adoption(store, engine, monkeypatch)
    root_llm = ContextLLM([LLMResponse(text="root done")])
    warned = Agent(
        name="lead",
        prompt="You lead.",
        provider=root_llm,
        state_store=store,
        subagents=[child_agent(ContextLLM([]))],
        guardrails=[SecretDetection(action="block")],
        on_governance_change="warn",
    )
    assert (await warned.reclaim(root_id, tenant_id=TENANT)).status == RunStatus.SUCCESS
    events = await rows(engine, RunEvent, RunEvent.agent_run_id == root_id)
    assert "governance.changed" in [e.event_type for e in events]


# ---------------------------------------------------------------------------
# R1: the parent admits the report like any tool result before it is durable
# ---------------------------------------------------------------------------

CONFIDENTIAL = "SENTINEL-confidential-6a1f"
EMAIL = "jane.sentinel@example.com"
AWS_KEY = "AKIAIOSFODNN7EXAMPLE"


async def _durable_text(engine: Any, root_id: str) -> str:
    """Every parent row a report can reach: pause/submission, tool row, traces, events."""
    run = (await rows(engine, AgentRun, AgentRun.id == root_id))[0]
    tools = await rows(engine, ToolCallRecord, ToolCallRecord.agent_run_id == root_id)
    traces = await rows(engine, ReactTrace, ReactTrace.agent_run_id == root_id)
    events = await rows(engine, RunEvent, RunEvent.agent_run_id == root_id)
    return json.dumps(
        [
            run.pause_data,
            run.loop_state,
            run.error,
            [(t.result, t.error_message, t.meta) for t in tools],
            [t.content for t in traces],
            [e.data for e in events],
        ],
        default=str,
    )


def _provider_text(llm: ContextLLM) -> str:
    return json.dumps([[m.content for m in call] for call in llm.messages])


transform_calls: list[str] = []


def _drop_confidential(ctx: Any) -> Any:
    transform_calls.append(ctx.call_id)
    assert ctx.tool_name == "reviewer" and ctx.params == {"task": BRIEF}
    return {"summary": "kept", "len": len(ctx.result)}


@pytest.mark.parametrize("cut", ["none", "after_claim", "after_completion"])
async def test_parent_transform_shapes_the_accepted_report_once(
    store: Any, engine: Any, monkeypatch: Any, cut: str
) -> None:
    from dendrux.runtime import persistence

    transform_calls.clear()
    child_llm = ContextLLM([LLMResponse(text=f"report {CONFIDENTIAL}")])
    root_llm = ContextLLM([call("reviewer", {"task": BRIEF}, "r1"), LLMResponse(text="ok")])
    root = root_agent(
        store, root_llm, child_agent(child_llm), tool_result_transform=_drop_confidential
    )
    if cut == "none":
        result = await root.run("go", tenant_id=TENANT)
        final_llm = root_llm
    else:
        target, name = (
            (persistence, "adopt_managed_result")
            if cut == "after_claim"
            else (_reclaim, "continue_claimed")
        )
        await _crash_at(monkeypatch, target, name, lambda: root.run("go", tenant_id=TENANT))
        root_id = await root_id_of(engine)
        item = await expire(store, engine, root_id)
        final_llm = ContextLLM([LLMResponse(text="ok")])
        again = root_agent(
            store, final_llm, child_agent(ContextLLM([])), tool_result_transform=_drop_confidential
        )
        result = await again.reclaim(root_id, tenant_id=TENANT, item=item)
    assert result.status == RunStatus.SUCCESS
    assert len(transform_calls) == 1  # recovery uses the saved accepted report
    assert CONFIDENTIAL not in await _durable_text(engine, result.run_id)
    assert CONFIDENTIAL not in _provider_text(final_llm)
    seen = [json.loads(m.content) for m in final_llm.messages[-1] if m.role.value == "tool"]
    assert seen == [{"summary": "kept", "len": len(f"report {CONFIDENTIAL}")}]
    assert child_llm.calls == 1


async def test_transformed_oversize_report_is_refused_without_rerunning_the_child(
    store: Any, engine: Any
) -> None:
    child_llm = ContextLLM([LLMResponse(text="short")])
    root_llm = ContextLLM([call("reviewer", {"task": BRIEF}, "r1"), LLMResponse(text="ok")])
    root = root_agent(
        store,
        root_llm,
        child_agent(child_llm),
        tool_result_transform=lambda ctx: "x" * 400,
        limits=Limits(max_tool_result_bytes=256),
    )
    result = await root.run("go", tenant_id=TENANT)
    assert result.status == RunStatus.SUCCESS and child_llm.calls == 1
    seen = [json.loads(m.content) for m in root_llm.messages[1] if m.role.value == "tool"]
    assert seen[0]["code"] == "managed_report_too_large" and seen[0]["limit"] == 256


async def test_parent_pii_redaction_applies_to_the_adopted_observation(
    store: Any, engine: Any
) -> None:
    from dendrux.guardrails import PII

    child_llm = ContextLLM([LLMResponse(text=f"contact {EMAIL}")])
    root_llm = ContextLLM([call("reviewer", {"task": BRIEF}, "r1"), LLMResponse(text="ok")])
    root = root_agent(store, root_llm, child_agent(child_llm), guardrails=[PII(action="redact")])
    result = await root.run("go", tenant_id=TENANT)
    assert result.status == RunStatus.SUCCESS
    assert EMAIL not in _provider_text(root_llm)
    traces = await rows(engine, ReactTrace, ReactTrace.agent_run_id == result.run_id)
    assert EMAIL not in json.dumps([t.content for t in traces])
    mapping = await store.get_pii_mapping(result.run_id)
    assert EMAIL in (mapping or {}).values()


@pytest.mark.parametrize("cut", ["none", "after_claim", "after_completion"])
async def test_parent_block_policy_stops_the_parent_and_keeps_no_blocked_text(
    store: Any, engine: Any, monkeypatch: Any, cut: str
) -> None:
    from dendrux._meta_keys import BLOCKED_META_KEY
    from dendrux.guardrails import SecretDetection
    from dendrux.runtime import persistence

    child_llm = ContextLLM([LLMResponse(text=f"key {AWS_KEY}")])
    root_llm = ContextLLM([call("reviewer", {"task": BRIEF}, "r1"), LLMResponse(text="never")])

    def make(llm: Any, child: Agent) -> Agent:
        return root_agent(store, llm, child, guardrails=[SecretDetection(action="block")])

    root = make(root_llm, child_agent(child_llm))
    if cut == "none":
        result = await root.run("go", tenant_id=TENANT)
        root_id = result.run_id
    else:
        target, name = (
            (persistence, "adopt_managed_result")
            if cut == "after_claim"
            else (_reclaim, "continue_claimed")
        )
        await _crash_at(monkeypatch, target, name, lambda: root.run("go", tenant_id=TENANT))
        root_id = await root_id_of(engine)
        # The blocked report marked the parent in its claim transaction already.
        marked = (await rows(engine, AgentRun, AgentRun.id == root_id))[0]
        assert (marked.meta or {}).get(BLOCKED_META_KEY) is True
        assert AWS_KEY not in await _durable_text(engine, root_id)
        item = await expire(store, engine, root_id)
        result = await make(ContextLLM([]), child_agent(ContextLLM([]))).reclaim(
            root_id, tenant_id=TENANT, item=item
        )
    assert result.status == RunStatus.ERROR
    assert root_llm.calls == 1 and child_llm.calls == 1
    row = (await rows(engine, AgentRun, AgentRun.id == root_id))[0]
    assert row.failure_reason == "guardrail_blocked"
    assert AWS_KEY not in await _durable_text(engine, root_id)
    assert AWS_KEY not in (result.error or "")
    assert (row.meta or {}).get(BLOCKED_META_KEY) is True


# ---------------------------------------------------------------------------
# Fix round 2
# ---------------------------------------------------------------------------


def _hooked_root(store: Any, root_llm: Any, child: Agent, decision: str) -> Agent:
    async def hook(ctx: Any) -> Any:
        from dendrux.types import ToolDecision

        effects.append(f"hook:{ctx.tool_name}")
        if decision == "deny":
            return ToolDecision.deny("not now")
        return ToolDecision.require_approval("check")

    return Agent(
        name="lead",
        prompt="You lead.",
        provider=root_llm,
        state_store=store,
        subagents=[child],
        before_tool=hook,
    )


def _die_sync(*args: Any, **kwargs: Any) -> Any:
    raise ProcessDeathError


@pytest.mark.parametrize("decision", ["deny", "approval"])
async def test_r4a_unadmitted_pending_call_is_refused_before_policy_or_child(
    store: Any, engine: Any, monkeypatch: Any, decision: str
) -> None:
    """Crash before preparation: no admission evidence, so no bind and no hook."""
    from dendrux.loops import react

    first = _hooked_root(
        store,
        ContextLLM([call("reviewer", {"task": BRIEF}, "r1")]),
        child_agent(ContextLLM([])),
        decision,
    )
    await _crash_at(
        monkeypatch, react, "_prepare_tool_calls", lambda: first.run("go", tenant_id=TENANT)
    )
    root_id = await root_id_of(engine)
    effects.clear()
    item = await expire(store, engine, root_id)
    child_llm = ContextLLM([LLMResponse(text=REPORT)])
    fresh = _hooked_root(store, ContextLLM([]), child_agent(child_llm), decision)
    with pytest.raises(ManagedUnsupportedError) as info:
        await fresh.reclaim(root_id, tenant_id=TENANT, item=item)
    assert info.value.operation == "recover_unadmitted_call"
    assert effects == [] and child_llm.calls == 0
    assert not await rows(engine, AgentRun, AgentRun.parent_run_id == root_id)


async def test_r4a_recorded_denial_is_kept_and_no_child_runs(
    store: Any, engine: Any, monkeypatch: Any
) -> None:
    from dendrux.loops import react

    first = _hooked_root(
        store,
        ContextLLM([call("reviewer", {"task": BRIEF}, "r1")]),
        child_agent(ContextLLM([])),
        "deny",
    )
    with monkeypatch.context() as patch:
        patch.setattr(react, "_split_targets", _die_sync)
        with pytest.raises(ProcessDeathError):
            await first.run("go", tenant_id=TENANT)
    root_id = await root_id_of(engine)
    effects.clear()
    item = await expire(store, engine, root_id)
    child_llm = ContextLLM([LLMResponse(text=REPORT)])
    root_llm = ContextLLM([LLMResponse(text="ok")])
    fresh = _hooked_root(store, root_llm, child_agent(child_llm), "deny")
    result = await fresh.reclaim(root_id, tenant_id=TENANT, item=item)
    assert result.status == RunStatus.SUCCESS
    assert child_llm.calls == 0 and effects == []
    assert not await rows(engine, AgentRun, AgentRun.parent_run_id == root_id)
    seen = [m for m in root_llm.messages[0] if m.role.value == "tool"]
    assert len(seen) == 1 and REPORT not in seen[0].content


async def test_r4a_recorded_approval_gate_is_not_bound_on_recovery(
    store: Any, engine: Any, monkeypatch: Any
) -> None:
    """A gate recorded for the batch (here on the server sibling; an approval hook on a
    managed call is an ordinary denial, as for client targets) is never bypassed."""
    from dendrux.loops import react

    batch = LLMResponse(
        tool_calls=[
            ToolCall(name="reviewer", params={"task": BRIEF}, provider_tool_call_id="p1"),
            ToolCall(name="fetch", params={"x": 1}, provider_tool_call_id="p2"),
        ]
    )
    first = _recovery_root(
        store, ContextLLM([batch]), _named("reviewer", ContextLLM([])), require_approval=["fetch"]
    )
    with monkeypatch.context() as patch:
        patch.setattr(react, "_pause_result", _die_sync)
        with pytest.raises(ProcessDeathError):
            await first.run("go", tenant_id=TENANT)
    root_id = await root_id_of(engine)
    row = (await rows(engine, AgentRun, AgentRun.id == root_id))[0]
    assert sorted(p["phase"] for p in row.loop_state["pending"]) == ["dispatchable", "gated"]
    effects.clear()
    item = await expire(store, engine, root_id)
    child_llm = ContextLLM([LLMResponse(text=REPORT)])
    fresh = _recovery_root(
        store, ContextLLM([]), _named("reviewer", child_llm), require_approval=["fetch"]
    )
    with pytest.raises(ManagedUnsupportedError):
        await fresh.reclaim(root_id, tenant_id=TENANT, item=item)
    assert child_llm.calls == 0 and effects == []
    assert not await rows(engine, AgentRun, AgentRun.parent_run_id == root_id)
    gated = (await rows(engine, AgentRun, AgentRun.id == root_id))[0].loop_state["pending"]
    assert "gated" in [p["phase"] for p in gated]  # the saved gate is preserved


@pytest.mark.parametrize("refused_before_crash", [1, 2])
async def test_r4b_partial_batch_refusal_refuses_the_rest_on_recovery(
    store: Any, engine: Any, monkeypatch: Any, refused_before_crash: int
) -> None:
    """Crash after each refusal, incl. the last managed one before the server call."""
    from dendrux.loops import react

    batch = LLMResponse(
        tool_calls=[
            ToolCall(name="reviewer", params={"task": BRIEF}, provider_tool_call_id="p1"),
            ToolCall(name="auditor", params={"task": BRIEF}, provider_tool_call_id="p2"),
            ToolCall(name="fetch", params={"x": 1}, provider_tool_call_id="p3"),
        ]
    )
    first = _recovery_root(
        store,
        ContextLLM([batch]),
        _named("reviewer", ContextLLM([])),
        _named("auditor", ContextLLM([])),
    )
    original = react._ToolBatch.reject
    done = {"n": 0}

    async def reject(self: Any, tc: Any, result: Any) -> None:
        await original(self, tc, result)
        done["n"] += 1
        if done["n"] == refused_before_crash:
            raise ProcessDeathError

    with monkeypatch.context() as patch:
        patch.setattr(react._ToolBatch, "reject", reject)
        with pytest.raises(ProcessDeathError):
            await first.run("go", tenant_id=TENANT)
    root_id = await root_id_of(engine)
    effects.clear()
    item = await expire(store, engine, root_id)
    root_llm = ContextLLM([LLMResponse(text="regrouped")])
    fresh = _recovery_root(
        store, root_llm, _named("reviewer", ContextLLM([])), _named("auditor", ContextLLM([]))
    )
    result = await fresh.reclaim(root_id, tenant_id=TENANT, item=item)
    assert result.status == RunStatus.SUCCESS and effects == []
    tool_rows = await rows(engine, ToolCallRecord, ToolCallRecord.agent_run_id == root_id)
    assert len(tool_rows) == 3
    assert {r.meta.get("code") for r in tool_rows} == {"managed_batch_unsupported"}
    assert not await rows(engine, AgentRun, AgentRun.parent_run_id == root_id)


async def test_r5_recovered_bind_refuses_a_newly_single_call_child(
    store: Any, engine: Any, monkeypatch: Any
) -> None:
    from dendrux.errors import SubagentDefinitionError
    from dendrux.loops.single import SingleCall

    first = _recovery_root(
        store,
        ContextLLM([call("reviewer", {"task": BRIEF}, "r1")]),
        _named("reviewer", ContextLLM([])),
    )
    await _crash_at(monkeypatch, _managed, "bind_child", lambda: first.run("go", tenant_id=TENANT))
    root_id = await root_id_of(engine)
    item = await expire(store, engine, root_id)
    child_llm = ContextLLM([LLMResponse(text=REPORT)])
    changed = Agent(
        name="reviewer", description="r", prompt="c", provider=child_llm, loop=SingleCall()
    )
    fresh = _recovery_root(store, ContextLLM([]), changed)
    with pytest.raises(SubagentDefinitionError) as info:
        await fresh.reclaim(root_id, tenant_id=TENANT, item=item)
    assert info.value.reason == "loop_unsupported" and child_llm.calls == 0
    assert not await rows(engine, AgentRun, AgentRun.parent_run_id == root_id)


async def test_r5_recovered_descendant_bind_refuses_a_single_call_grandchild(
    store: Any, engine: Any, monkeypatch: Any
) -> None:
    from dendrux.errors import SubagentDefinitionError
    from dendrux.loops.single import SingleCall

    def reviewer(llm: Any, auditor: Agent) -> Agent:
        return Agent(
            name="reviewer", description="r", prompt="c", provider=llm, subagents=[auditor]
        )

    first = _recovery_root(
        store,
        ContextLLM([call("reviewer", {"task": BRIEF}, "r1")]),
        reviewer(
            ContextLLM([call("auditor", {"task": "check"}, "c1")]),
            _named("auditor", ContextLLM([])),
        ),
    )
    calls_seen = {"n": 0}
    original = _managed.bind_child

    async def die_second(*args: Any, **kwargs: Any) -> Any:
        calls_seen["n"] += 1
        if calls_seen["n"] == 2:
            raise ProcessDeathError
        return await original(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(_managed, "bind_child", die_second)
        with pytest.raises(ProcessDeathError):
            await first.run("go", tenant_id=TENANT)
    root_id = await root_id_of(engine)
    middle = await child_of(engine, root_id)
    await expire(store, engine, middle.id)
    grandchild_llm = ContextLLM([LLMResponse(text="leaf")])
    changed = Agent(
        name="auditor", description="a", prompt="c", provider=grandchild_llm, loop=SingleCall()
    )
    fresh = _recovery_root(store, ContextLLM([]), reviewer(ContextLLM([]), changed))
    with pytest.raises(SubagentDefinitionError):
        await fresh.reclaim(root_id, tenant_id=TENANT)
    assert grandchild_llm.calls == 0
    assert not await rows(engine, AgentRun, AgentRun.parent_run_id == middle.id)


async def test_r6_reconnect_finds_the_intent_beyond_one_page_of_tool_rows(
    store: Any, engine: Any, monkeypatch: Any
) -> None:
    from datetime import UTC, datetime, timedelta

    from dendrux.types import generate_ulid

    root_id, child_id = await crash_before_adoption(store, engine, monkeypatch)
    intent = (await rows(engine, ToolCallRecord, ToolCallRecord.agent_run_id == root_id))[0]
    older = datetime.now(UTC) - timedelta(hours=1)
    async with engine.begin() as conn:
        await conn.execute(
            ToolCallRecord.__table__.insert(),
            [
                dict(
                    id=generate_ulid(),
                    agent_run_id=root_id,
                    tool_call_id=generate_ulid(),
                    tool_name="fetch",
                    target="server",
                    params={"x": i},
                    result={"ok": i},
                    success=True,
                    duration_ms=0,
                    iteration_index=intent.iteration_index,
                    outcome="completed",
                    idempotent=False,
                    created_at=older,
                    updated_at=older,
                )
                for i in range(1001)
            ],
        )
    root, root_llm, child_llm = fresh_root(store, LLMResponse(text="root done"))
    result = await root.reclaim(root_id, tenant_id=TENANT)
    assert result.status == RunStatus.SUCCESS and root_llm.calls == 1 and child_llm.calls == 0
    assert len(await observations(engine, root_id)) == 1
    adopted = (await rows(engine, ToolCallRecord, ToolCallRecord.id == intent.id))[0]
    assert adopted.outcome == "completed"
