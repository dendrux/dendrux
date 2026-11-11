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
from dendrux.runtime.persistence import message_trace
from dendrux.runtime.stores._factory import store_for_engine
from dendrux.types import Limits, LLMResponse, Message, Role, RunStatus, ToolCall
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
        await asyncio.gather(
            root.reclaim(root_id, tenant_id=TENANT),
            canceller.cancel_run(root_id, tenant_id=TENANT),
            return_exceptions=True,
        )
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
    origin = result.meta["managed_origin"]
    assert origin == {
        "origin_run_id": child_row.id,
        "agent_name": "reviewer",
        "agent_path": ["lead", "reviewer"],
    }
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


async def test_seeded_trace_helper_is_importable() -> None:
    # Keeps message_trace / Message imports honest for the store-kit helpers.
    assert message_trace(Message(role=Role.USER, content="x"), 0).role == "user"


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
    direct = child_agent(ContextLLM([]), tools=[note], state_store=store)
    with pytest.raises(ManagedChildControlError):
        await direct.reclaim(child.id, tenant_id=TENANT, item=item)
    # The refusal comes after the store's claim (no extra read on ordinary reclaims),
    # so it used that recovery mark: the root reclaims after the next expiry.
    await expire(store, engine, child.id)

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
