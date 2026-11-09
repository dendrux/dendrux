"""Recovery preserves the approval boundary of an entire tool batch."""

import asyncio
from unittest.mock import patch

import pytest
from sqlalchemy import event

from dendrux import tool
from dendrux.errors import RunNotReclaimableError
from dendrux.runtime.stores._factory import store_for_engine
from dendrux.types import ApprovalDecision, ToolCall, ToolDecision
from tests._helpers.recovery_checkpoint import checkpoint, expire
from tests.integration.test_060_reclaim_agent import make_agent


class Crash(BaseException):
    pass


def batch_agent(store, effects, *, hook=False):
    @tool()
    async def write_region() -> str:
        """Write the approved region."""
        effects.append("write_region")
        return "written"

    @tool()
    async def post_review_notice() -> str:
        """Post the review notice."""
        effects.append("post_review_notice")
        return "posted"

    async def before(context):
        if context.tool_name == "write_region":
            return ToolDecision.require_approval("Review region")
        return None

    return make_agent(
        store,
        tools=[write_region, post_review_notice],
        **({"before_tool": before} if hook else {"require_approval": ["write_region"]}),
    )


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("hook", [False, True])
@pytest.mark.parametrize("decision", ["approve", "decline", "per_call"])
async def test_061_reclaim_gate_holds_ungated_first(store, engine, reverse, hook, decision):
    effects = []
    agent = batch_agent(store, effects, hook=hook)
    write, notice = ToolCall("write_region"), ToolCall("post_review_notice")
    calls = [notice, write] if not reverse else [write, notice]
    item = await checkpoint(
        agent,
        store,
        engine,
        calls=calls,
        phases=["gated" if c == write and not hook else "dispatchable" for c in calls],
    )
    statements = []

    def capture(conn, cursor, statement, *args):
        statements.append(statement)

    event.listen(engine.sync_engine, "before_cursor_execute", capture)
    try:
        paused = await agent.reclaim(item.run_id, tenant_id="tenant", item=item)
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", capture)
        print("reclaim_statements", engine.dialect.name, reverse, hook, len(statements))
    assert len(statements) <= {"sqlite": 31, "postgresql": 31}[engine.dialect.name]
    assert paused.status.value == "waiting_approval"
    assert effects == []
    assert await store.get_tool_calls(item.run_id) == []
    raw = await store.get_pause_state(item.run_id)
    assert raw["pause_id"] == paused.pause_id and paused.pause_id
    assert [c["id"] for c in raw["pending_tool_calls"]] == [c.id for c in calls]
    assert set(raw["approval_gates"]) == {write.id}
    assert raw["approval_gates"][write.id]["source"] == ("hook" if hook else "static")
    kwargs = (
        {"decisions": {write.id: ApprovalDecision(approved=False)}}
        if decision == "per_call"
        else {"approved": decision == "approve"}
    )
    result = await agent.submit_approval(
        item.run_id, tenant_id="tenant", expected_pause_id=paused.pause_id, **kwargs
    )
    assert result.status.value == "success"
    assert effects == (
        [c.name for c in calls]
        if decision == "approve"
        else [notice.name]
        if decision == "per_call"
        else []
    )
    events = await store.get_run_events(item.run_id)
    names = [e.event_type for e in events]
    selected = [
        n
        for n in names
        if n
        in {
            "run.lease_expired",
            "run.reclaimed",
            "approval.requested",
            "run.paused",
            "run.resumed",
            "tool.started",
            "tool.completed",
            "approval.decided",
            "run.completed",
        }
    ]
    expected = [
        "run.lease_expired",
        "run.reclaimed",
        "run.resumed",
        "approval.requested",
        "run.paused",
        "run.resumed",
    ]
    for call in calls:
        if call.name in effects:
            expected.append("tool.started")
        expected.append("tool.completed")
    expected += ["approval.decided", "run.completed"]
    assert selected == expected
    audit = next(e.data for e in events if e.event_type == "approval.decided")
    assert {d["call_id"]: d["decision"] for d in audit["decisions"]} == {
        write.id: "approved" if decision == "approve" else "denied",
        notice.id: "denied" if decision == "decline" else "not_gated_executed",
    }
    if decision == "approve":
        assert audit == {
            "decision": "approved",
            "run_id": item.run_id,
            "decisions": [
                {
                    "call_id": c.id,
                    "tool_name": c.name,
                    "decision": "approved" if c == write else "not_gated_executed",
                    "reason": None,
                }
                for c in calls
            ],
        }
        requested = next(e.data for e in events if e.event_type == "approval.requested")
        assert requested["call_id"] == write.id and requested["tool_name"] == write.name
        assert requested["params"] == {}
        paused_event = next(e.data for e in events if e.event_type == "run.paused")
        assert paused_event["pause_id"] == paused.pause_id
        assert paused_event["status"] == "waiting_approval"
        assert [c["id"] for c in paused_event["pending_tool_calls"]] == [c.id for c in calls]
    await agent.close()


async def test_061_recovery_batch_second_crash(store, engine):
    effects = []
    agent = batch_agent(store, effects)
    calls = [ToolCall("post_review_notice"), ToolCall("write_region")]
    item = await checkpoint(agent, store, engine, calls=calls, phases=["dispatchable", "gated"])
    from dendrux.runtime import _reclaim

    async def crash(*args, **kwargs):
        raise Crash()

    with patch.object(_reclaim, "_recovery_pause_result", crash), pytest.raises(Crash):
        await agent.reclaim(item.run_id, tenant_id="tenant", item=item)
    assert effects == [] and await store.get_tool_calls(item.run_id) == []
    item = await expire(store, engine, item.run_id)
    result = await agent.reclaim(item.run_id, tenant_id="tenant", item=item)
    assert result.status.value == "waiting_approval" and effects == []
    result = await agent.submit_approval(
        item.run_id, tenant_id="tenant", expected_pause_id=result.pause_id, approved=True
    )
    assert result.status.value == "success" and effects == [c.name for c in calls]
    await agent.close()


async def test_061_recovery_batch_prior_approval(store, engine):
    effects = []
    agent = batch_agent(store, effects)
    calls = [ToolCall("write_region"), ToolCall("post_review_notice")]
    item = await checkpoint(agent, store, engine, calls=calls, phases=["gated", "dispatchable"])
    paused = await agent.reclaim(item.run_id, tenant_id="tenant", item=item)
    submit = store.submit_and_claim

    async def crash(*args, **kwargs):
        await submit(*args, **kwargs)
        raise Crash()

    with patch.object(store, "submit_and_claim", crash), pytest.raises(Crash):
        await agent.submit_approval(
            item.run_id, tenant_id="tenant", expected_pause_id=paused.pause_id, approved=True
        )
    assert effects == []
    item = await expire(store, engine, item.run_id)
    result = await agent.reclaim(item.run_id, tenant_id="tenant", item=item)
    assert result.status.value == "success" and effects == [c.name for c in calls]
    await agent.close()


async def test_061_recovery_batch_two_reclaimers(store, engine, monkeypatch):
    if engine.dialect.name != "postgresql":
        pytest.skip("Independent claim transactions require PostgreSQL")
    effects = []
    agents = [batch_agent(store_for_engine(engine), effects) for _ in range(2)]
    calls = [ToolCall("post_review_notice"), ToolCall("write_region")]
    item = await checkpoint(agents[0], store, engine, calls=calls, phases=["dispatchable", "gated"])
    barrier = asyncio.Barrier(2)
    for agent in agents:
        original = agent._state_store.reclaim_run

        async def race(*args, _claim=original, **kwargs):
            await asyncio.wait_for(barrier.wait(), 10)
            return await _claim(*args, **kwargs)

        monkeypatch.setattr(agent._state_store, "reclaim_run", race)
    try:
        results = await asyncio.wait_for(
            asyncio.gather(
                *(a.reclaim(item.run_id, tenant_id="tenant", item=item) for a in agents),
                return_exceptions=True,
            ),
            20,
        )
        winners = [r for r in results if not isinstance(r, BaseException)]
        losers = [r for r in results if isinstance(r, BaseException)]
        assert len(winners) == len(losers) == 1
        assert isinstance(losers[0], RunNotReclaimableError)
        assert winners[0].status.value == "waiting_approval"
        assert effects == [] and await store.get_tool_calls(item.run_id) == []
        row = await store.get_run(item.run_id)
        assert row.claim_epoch == item.claim_epoch + 1 and row.reclaim_attempts == 1
        state = await store.get_loop_state(item.run_id)
        assert [p["call_id"] for p in state["pending"]] == [c.id for c in calls]
        events = await store.get_run_events(item.run_id)
        assert sum(e.event_type == "run.reclaimed" for e in events) == 1
        assert sum(e.event_type == "run.paused" for e in events) == 1
        final = await agents[0].submit_approval(
            item.run_id, tenant_id="tenant", expected_pause_id=winners[0].pause_id, approved=True
        )
        assert final.status.value == "success" and effects == [c.name for c in calls]
        assert len(await store.get_tool_calls(item.run_id)) == 2
    finally:
        await asyncio.gather(*(a.close() for a in agents))


async def test_061_recovery_batch_completed_sibling(store, engine):
    from sqlalchemy import update

    from dendrux.db.models import ToolCallRecord
    from dendrux.runtime.state import EventWrite
    from dendrux.types import generate_ulid

    effects = []
    agent = batch_agent(store, effects)
    done, notice, write = (
        ToolCall("post_review_notice"),
        ToolCall("post_review_notice"),
        ToolCall("write_region"),
    )
    item = await checkpoint(
        agent,
        store,
        engine,
        calls=[done, notice, write],
        phases=["dispatchable", "dispatchable", "gated"],
    )
    await store.start_tool_call(
        item.run_id,
        claim_epoch=item.claim_epoch,
        row_id=generate_ulid(),
        tool_call_id=done.id,
        provider_tool_call_id=None,
        tool_name=done.name,
        target="server",
        params=None,
        iteration_index=1,
        idempotent=False,
        event=EventWrite(generate_ulid(), "tool.started", 1, {}, done.id, None),
    )
    async with engine.begin() as conn:
        await conn.execute(
            update(ToolCallRecord)
            .where(
                ToolCallRecord.agent_run_id == item.run_id,
                ToolCallRecord.tool_call_id == done.id,
            )
            .values(outcome="completed", result="already posted", success=True)
        )
    paused = await agent.reclaim(item.run_id, tenant_id="tenant", item=item)
    assert effects == []
    raw = await store.get_pause_state(item.run_id)
    assert [c["id"] for c in raw["pending_tool_calls"]] == [notice.id, write.id]
    entries = (await store.conversation_traces(item.run_id, after_position=None)).entries
    saved = [e for e in entries if e.role == "tool"]
    assert len(saved) == 1 and saved[0].content == '"already posted"'
    final = await agent.submit_approval(
        item.run_id,
        tenant_id="tenant",
        expected_pause_id=paused.pause_id,
        approved=True,
    )
    assert final.status.value == "success" and effects == [notice.name, write.name]
    rows = await store.get_tool_calls(item.run_id)
    assert len(rows) == 3 and all(r.outcome == "completed" for r in rows)
    entries = (await store.conversation_traces(item.run_id, after_position=None)).entries
    assert len([e for e in entries if e.role == "tool"]) == 3
    await agent.close()


@pytest.mark.parametrize("approved", [True, False])
async def test_061_recovery_batch_client_sibling(store, engine, approved):
    from dendrux.types import ToolResult

    effects = []

    @tool()
    async def write_region() -> str:
        """Write the approved region."""
        effects.append("write_region")
        return "written"

    @tool(target="client")
    async def client() -> str:
        """Run exclusively on the client."""
        pytest.fail("client sibling dispatched on server")

    agent = make_agent(store, tools=[client, write_region], require_approval=["write_region"])
    calls = [ToolCall("client"), ToolCall("write_region")]
    item = await checkpoint(agent, store, engine, calls=calls, phases=["client", "gated"])
    paused = await agent.reclaim(item.run_id, tenant_id="tenant", item=item)
    assert paused.status.value == "waiting_approval" and effects == []
    raw = await store.get_pause_state(item.run_id)
    assert [c["id"] for c in raw["pending_tool_calls"]] == [c.id for c in calls]
    assert set(raw["approval_gates"]) == {calls[1].id}
    assert await store.get_tool_calls(item.run_id) == []
    final = await agent.submit_approval(
        item.run_id,
        tenant_id="tenant",
        expected_pause_id=paused.pause_id,
        approved=approved,
    )
    assert effects == (["write_region"] if approved else [])
    if approved:
        assert final.status.value == "waiting_client_tool"
        assert final.pause_id != paused.pause_id
        final = await agent.submit_tool_results(
            item.run_id,
            [ToolResult(name="client", call_id=calls[0].id, payload='"ok"')],
            expected_pause_id=final.pause_id,
            tenant_id="tenant",
        )
    assert final.status.value == "success"
    await agent.close()


@pytest.mark.parametrize("decision", ["approve", "decline", "per_call"])
async def test_061_recovery_batch_uncertain_and_approved_same_name(store, engine, decision):
    effects = []
    agent = batch_agent(store, effects)
    calls = [ToolCall("write_region"), ToolCall("write_region")]
    item = await checkpoint(agent, store, engine, calls=calls, phases=["gated", "gated"])
    paused = await agent.reclaim(item.run_id, tenant_id="tenant", item=item)
    start = store.start_tool_call

    async def crash(*args, **kwargs):
        await start(*args, **kwargs)
        raise Crash()

    with patch.object(store, "start_tool_call", crash), pytest.raises(Crash):
        await agent.submit_approval(
            item.run_id,
            tenant_id="tenant",
            expected_pause_id=paused.pause_id,
            approved=True,
        )
    assert effects == []
    state = await store.get_loop_state(item.run_id)
    assert [p["phase"] for p in state["pending"]] == ["dispatchable", "approved"]
    item = await expire(store, engine, item.run_id)
    paused = await agent.reclaim(item.run_id, tenant_id="tenant", item=item)
    assert paused.status.value == "waiting_approval" and effects == []
    raw = await store.get_pause_state(item.run_id)
    assert [c["id"] for c in raw["pending_tool_calls"]] == [c.id for c in calls]
    assert raw["approval_gates"] == {
        calls[0].id: {"source": "recovery", "reason": "replay_after_crash"},
    }
    kwargs = (
        {"decisions": {calls[0].id: ApprovalDecision(approved=False)}}
        if decision == "per_call"
        else {"approved": decision == "approve"}
    )
    final = await agent.submit_approval(
        item.run_id,
        tenant_id="tenant",
        expected_pause_id=paused.pause_id,
        **kwargs,
    )
    assert final.status.value == "success"
    expected = 2 if decision == "approve" else 1 if decision == "per_call" else 0
    assert effects == ["write_region"] * expected
    rows = {r.tool_call_id: r for r in await store.get_tool_calls(item.run_id)}
    assert rows[calls[0].id].outcome == ("completed" if decision == "approve" else "unknown")
    if decision != "approve":
        assert rows[calls[0].id].error_message == "outcome_unknown"
    await agent.close()


async def test_061_recovery_batch_prepares_hooks_once_before_effects(store, engine):
    effects, checked = [], []

    @tool()
    async def work(index: int) -> str:
        """Record a prepared call."""
        effects.append(index)
        return "ok"

    async def before(context):
        assert effects == []
        checked.append(context.call_id)

    agent = make_agent(store, tools=[work], before_tool=before)
    calls = [ToolCall("work", params={"index": i}) for i in range(2)]
    item = await checkpoint(agent, store, engine, calls=calls, phases=["dispatchable"] * 2)
    result = await agent.reclaim(item.run_id, tenant_id="tenant", item=item)
    assert result.status.value == "success"
    assert checked == [c.id for c in calls] and effects == [0, 1]
    await agent.close()


@pytest.mark.parametrize("crash_again", [False, True])
async def test_061_recovery_batch_decline_held_intent(store, engine, crash_again):
    from dendrux.runtime.state import EventWrite
    from dendrux.types import generate_ulid

    effects = []

    @tool(idempotent=True)
    async def notice() -> str:
        """Replay only after the batch decision."""
        effects.append("notice")
        return "ok"

    @tool()
    async def write() -> str:
        """Write after approval."""
        effects.append("write")
        return "ok"

    agent = make_agent(store, tools=[notice, write], require_approval=["write"])
    calls = [ToolCall("notice"), ToolCall("write")]
    item = await checkpoint(agent, store, engine, calls=calls, phases=["dispatchable", "gated"])
    await store.start_tool_call(
        item.run_id,
        claim_epoch=item.claim_epoch,
        row_id=generate_ulid(),
        tool_call_id=calls[0].id,
        provider_tool_call_id=None,
        tool_name="notice",
        target="server",
        params=None,
        iteration_index=1,
        idempotent=True,
        event=EventWrite(generate_ulid(), "tool.started", 1, {}, calls[0].id, None),
    )
    paused = await agent.reclaim(item.run_id, tenant_id="tenant", item=item)
    assert effects == []
    assert set((await store.get_pause_state(item.run_id))["approval_gates"]) == {calls[1].id}
    complete = store.complete_tool_call

    async def crash(*args, **kwargs):
        await complete(*args, **kwargs)
        raise Crash()

    if crash_again:
        with patch.object(store, "complete_tool_call", crash), pytest.raises(Crash):
            await agent.submit_approval(
                item.run_id,
                tenant_id="tenant",
                expected_pause_id=paused.pause_id,
                approved=False,
            )
        item = await expire(store, engine, item.run_id)
        final = await agent.reclaim(item.run_id, tenant_id="tenant", item=item)
    else:
        final = await agent.submit_approval(
            item.run_id,
            tenant_id="tenant",
            expected_pause_id=paused.pause_id,
            approved=False,
        )
    assert final.status.value == "success" and effects == []
    rows = await store.get_tool_calls(item.run_id)
    assert len(rows) == 1 and rows[0].outcome == "unknown"
    assert rows[0].error_message == "outcome_unknown"
    assert (await store.get_loop_state(item.run_id))["pending"] == []
    await agent.close()


@pytest.mark.parametrize("gated", [False, True])
async def test_061_recovery_batch_skill_activation_precedes_read(store, engine, gated):
    from dendrux import Skill

    effects = []

    @tool()
    async def write() -> str:
        """Write after the prepared skill calls."""
        effects.append("write")
        return "ok"

    agent = make_agent(
        store,
        tools=[write],
        require_approval=["write"] if gated else [],
        skills=[
            Skill(name="guide", description="guide", body="body", files={"notes.txt": "notes"})
        ],
    )
    calls = [
        ToolCall("use_skill", params={"name": "guide"}),
        ToolCall("read_skill_resource", params={"skill": "guide", "path": "notes.txt"}),
        ToolCall("write"),
    ]
    item = await checkpoint(
        agent,
        store,
        engine,
        calls=calls,
        phases=["dispatchable", "dispatchable", "gated" if gated else "dispatchable"],
    )
    from sqlalchemy import update

    from dendrux.db.models import AgentRun
    from dendrux.skills._session import SkillSession

    state = await store.get_loop_state(item.run_id)
    state["skills"]["pins"] = {
        skill.name: skill.digest for skill in SkillSession.prepare(agent).available.values()
    }
    async with engine.begin() as conn:
        await conn.execute(
            update(AgentRun).where(AgentRun.id == item.run_id).values(loop_state=state)
        )
    result = await agent.reclaim(item.run_id, tenant_id="tenant", item=item)
    if gated:
        assert result.status.value == "waiting_approval" and effects == []
        raw = await store.get_pause_state(item.run_id)
        assert [c["id"] for c in raw["pending_tool_calls"]] == [c.id for c in calls]
        assert await store.get_tool_calls(item.run_id) == []
        assert not any(
            e.event_type == "skill.activated" for e in await store.get_run_events(item.run_id)
        )
        result = await agent.submit_approval(
            item.run_id,
            tenant_id="tenant",
            expected_pause_id=result.pause_id,
            approved=True,
        )
    assert result.status.value == "success" and effects == ["write"]
    rows = await store.get_tool_calls(item.run_id)
    assert len(rows) == 3 and all(row.success for row in rows)
    await agent.close()


async def test_061_recovery_batch_client_target_stays_client(store, engine):
    from dendrux import Agent
    from dendrux.llm.mock import MockLLM
    from dendrux.types import LLMResponse

    effects = []

    def client_tool(target):
        @tool(target=target)
        async def client() -> str:
            """Honor the original client target during recovery."""
            effects.append("client")
            return "ok"

        return client

    original = make_agent(store, tools=[client_tool("client")])
    call = ToolCall("client")
    item = await checkpoint(original, store, engine, calls=[call], phases=["client"])
    await original.close()
    agent = Agent(
        name="recovery",
        prompt="p",
        state_store=store,
        provider=MockLLM([LLMResponse(text="done")]),
        tools=[client_tool("server")],
        strict_tenancy=True,
        on_governance_change="warn",
    )
    result = await agent.reclaim(item.run_id, tenant_id="tenant", item=item)
    assert result.status.value == "waiting_client_tool" and effects == []
    assert await store.get_tool_calls(item.run_id) == []
    await agent.close()


async def test_061_recovery_batch_pause_during_preparation(store, engine):
    effects = []

    @tool()
    async def work(index: int) -> str:
        """Execute only after the requested pause is resumed."""
        effects.append(index)
        return "ok"

    requested = False

    async def before(context):
        nonlocal requested
        if context.params["index"] == 1 and not requested:
            requested = True
            assert await store.request_pause(context.run_id)

    agent = make_agent(store, tools=[work], before_tool=before)
    calls = [ToolCall("work", params={"index": i}) for i in range(2)]
    item = await checkpoint(agent, store, engine, calls=calls, phases=["dispatchable"] * 2)
    paused = await agent.reclaim(item.run_id, tenant_id="tenant", item=item)
    assert paused.status.value == "paused" and effects == []
    assert await store.get_tool_calls(item.run_id) == []
    raw = await store.get_pause_state(item.run_id)
    assert [c["id"] for c in raw["pending_tool_calls"]] == [c.id for c in calls]
    final = await agent.submit_resume(
        item.run_id,
        tenant_id="tenant",
        expected_pause_id=paused.pause_id,
    )
    assert final.status.value == "success" and effects == [0, 1]
    await agent.close()
