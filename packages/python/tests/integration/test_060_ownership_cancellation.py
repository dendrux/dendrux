"""Ownership survives cancellation between COMMIT and runner setup."""

import asyncio

import pytest
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from dendrux import Agent
from dendrux.db.models import AgentRun
from dendrux.llm.mock import MockLLM
from dendrux.runtime.submit import governance_snapshot
from dendrux.store import LiveUpdates
from dendrux.types import ApprovalDecision, LLMResponse, ToolResult
from tests._helpers.gates import wait_for_gate
from tests._helpers.paused_runs import _pause
from tests._helpers.recovery_checkpoint import checkpoint, expire
from tests.integration.test_060_reclaim_agent import make_agent

ENTRIES = (
    "run",
    "stream",
    "resume",
    "resume_stream",
    "submit_tool_results",
    "submit_input",
    "submit_approval",
    "reclaim",
)


BUDGET_ENTRIES = ENTRIES
ENTRIES += (
    "retry",
    "runs",
    "submit_resume",
    "resume_paused",
    "approval_rejected",
    "approval_decisions",
)


class GateBus(LiveUpdates):
    def __init__(self):
        super().__init__(send_timeout_s=5, flush_deadline_s=5)
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.enabled = False

    async def send(self, messages):
        if self.enabled:
            self.enabled = False
            self.entered.set()
            await self.release.wait()

    async def receive(self):
        yield None


async def consume(stream):
    async with stream:
        return [event async for event in stream]


async def invocation(entry, store, engine):
    if entry in ("run", "stream", "retry", "runs"):
        provider = MockLLM([LLMResponse(text="done"), LLMResponse(text="again")])
        agent = Agent(name="handoff", prompt="p", provider=provider, state_store=store)
        call = (
            agent.run("go", tenant_id="tenant")
            if entry == "run"
            else consume(agent.stream("go", tenant_id="tenant"))
        )
        if entry in ("retry", "runs"):
            call.close()
            source = await agent.run("source", tenant_id="tenant")
            if entry == "retry":
                async with engine.begin() as conn:
                    await conn.execute(
                        update(AgentRun).where(AgentRun.id == source.run_id).values(status="error")
                    )
                call = agent.retry(source.run_id, tenant_id="tenant")
            else:
                call = agent.run("again", runs=[source.run_id], tenant_id="tenant")
        return agent, call, None, 0, lambda: provider._call_count
    if entry == "reclaim":
        agent = make_agent(store)
        item = await checkpoint(agent, store, engine)
        return (
            agent,
            agent.reclaim(item.run_id, tenant_id="tenant", item=item),
            item.run_id,
            item.claim_epoch + 1,
            lambda: agent.provider._call_count,
        )
    kind = (
        "input"
        if entry == "submit_input"
        else "approval"
        if entry in ("submit_approval", "approval_rejected", "approval_decisions")
        else "paused"
        if entry in ("submit_resume", "resume_paused")
        else "client"
    )
    paused = await _pause(kind, store, "tenant")
    raw = await store.get_pause_state(paused.run_id)
    return (
        paused.agent,
        continuation(entry, paused.agent, paused.run_id, raw),
        paused.run_id,
        1,
        lambda: paused.provider.calls,
    )


def continuation(entry, agent, run_id, raw):
    kwargs = dict(tenant_id="tenant", expected_pause_id=raw["pause_id"])
    results = [ToolResult(c["name"], c["id"], '"ok"') for c in raw["pending_tool_calls"]]
    if entry == "submit_resume":
        call = agent.submit_resume(run_id, **kwargs)
    elif entry == "resume_paused":
        call = agent.resume(run_id, **kwargs)
    elif entry == "approval_rejected":
        call = agent.submit_approval(run_id, approved=False, **kwargs)
    elif entry == "approval_decisions":
        call = agent.submit_approval(
            run_id,
            decisions={c["id"]: ApprovalDecision.approve() for c in raw["pending_tool_calls"]},
            **kwargs,
        )
    elif entry == "submit_input":
        call = agent.submit_input(run_id, "answer", **kwargs)
    elif entry == "submit_approval":
        call = agent.submit_approval(run_id, approved=True, **kwargs)
    elif entry == "submit_tool_results":
        call = agent.submit_tool_results(run_id, results, **kwargs)
    elif entry == "resume":
        call = agent.resume(run_id, tool_results=results, **kwargs)
    else:
        call = consume(agent.resume_stream(run_id, tool_results=results, **kwargs))
    return call


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("boundary", ["hint", "commit_issued", "commit_after", "return"])
async def test_cancel_ownership_handoff(store, engine, monkeypatch, entry, boundary):
    bus = GateBus()
    monkeypatch.setattr(store, "_live_updates", bus)
    agent, call, run_id, epoch, calls = await invocation(entry, store, engine)
    before_calls = calls()
    before_tool_events = (
        [
            e.event_type
            for e in await store.get_run_events(run_id)
            if e.event_type.startswith("tool.")
        ]
        if run_id
        else []
    )
    entered, release = asyncio.Event(), asyncio.Event()
    if boundary == "hint":
        bus.enabled = True
        entered, release = bus.entered, bus.release
    elif boundary.startswith("commit"):
        original = AsyncSession.commit
        armed = True

        async def commit(session):
            nonlocal armed
            gate = armed
            armed = False
            if gate and boundary == "commit_issued":
                entered.set()
                await release.wait()
            await original(session)
            if gate and boundary == "commit_after":
                entered.set()
                await release.wait()

        monkeypatch.setattr(AsyncSession, "commit", commit)
    else:
        method = (
            "create_run"
            if entry in ("run", "stream", "retry", "runs")
            else "reclaim_run"
            if entry == "reclaim"
            else "claim_paused_run"
            if entry in ("submit_resume", "resume_paused")
            else "submit_and_claim"
        )
        original = getattr(store, method)

        async def returned(*args, **kwargs):
            result = await original(*args, **kwargs)
            entered.set()
            await release.wait()
            return result

        monkeypatch.setattr(store, method, returned)
    task = asyncio.create_task(call)
    try:
        await wait_for_gate(entered, task=task)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()  # A second cancellation must not discard the committed token.
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 10)
        if run_id is None:
            async with engine.connect() as conn:
                run_id = (
                    await conn.execute(
                        select(AgentRun.id)
                        .order_by(AgentRun.created_at.desc(), AgentRun.id.desc())
                        .limit(1)
                    )
                ).scalar_one_or_none()
        row = await store.get_run(run_id)
        assert row.status == "cancelled"
        assert row.claim_epoch == epoch
        assert calls() == before_calls
        events = await store.get_run_events(run_id)
        assert sum(e.event_type == "run.cancelled" for e in events) == 1
        assert [
            e.event_type for e in events if e.event_type.startswith("tool.")
        ] == before_tool_events
        assert not agent._task_manager.is_running(run_id)
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        await agent.close()


@pytest.mark.parametrize("stage", ["history", "session", "state"])
@pytest.mark.parametrize("superseded", [False, True])
async def test_reclaim_setup_cancel_is_fenced_and_drained(
    store, engine, monkeypatch, stage, superseded
):
    import dendrux.runtime.runner as runner

    agent = make_agent(store)
    item = await checkpoint(agent, store, engine)
    entered = asyncio.Event()
    if stage == "history":
        target, name = store, "conversation_traces"
    else:
        target, name = runner, "_resume_session" if stage == "session" else "_read_resume_state"
    original = getattr(target, name)

    async def gate(*args, **kwargs):
        entered.set()
        await asyncio.Event().wait()
        return await original(*args, **kwargs)

    monkeypatch.setattr(target, name, gate)
    task = asyncio.create_task(agent.reclaim(item.run_id, tenant_id="tenant", item=item))
    try:
        await wait_for_gate(entered, task=task)
        claimed_epoch = (await store.get_run(item.run_id)).claim_epoch
        if superseded:
            next_item = await expire(store, engine, item.run_id)
            next_claim = await store.reclaim_run(
                item.run_id,
                tenant_id="tenant",
                expected_claim_epoch=next_item.claim_epoch,
                agent_name=agent.name,
                governance=governance_snapshot(agent),
                governance_policy="raise",
                skill_pins={},
                loop_type="react",
            )
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 10)
        row = await store.get_run(item.run_id)
        events = await store.get_run_events(item.run_id)
        cancellations = [e for e in events if e.event_type == "run.cancelled"]
        if superseded:
            assert row.status == "running"
            assert row.claim_epoch == next_claim.claim_epoch
            assert cancellations == []
        else:
            assert row.status == "cancelled"
            assert row.claim_epoch == claimed_epoch == item.claim_epoch + 1
            assert len(cancellations) == 1
        assert agent.provider._call_count == 0
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await agent.close()


async def test_parallel_cancel_at_handoff_has_one_finalize(store, engine, monkeypatch):
    if engine.dialect.name != "postgresql":
        pytest.skip("Independent concurrent transactions require Postgres")
    bus = GateBus()
    monkeypatch.setattr(store, "_live_updates", bus)
    agent, call, run_id, epoch, calls = await invocation("submit_tool_results", store, engine)
    bus.enabled = True
    task = asyncio.create_task(call)
    try:
        await wait_for_gate(bus.entered, task=task)
        start = asyncio.Event()

        from dendrux.runtime.runner import _persist_task_cancellation
        from dendrux.runtime.state import claimed_store

        original = store.finalize_with_event
        entered = 0
        ready = asyncio.Event()
        outcomes = []

        async def finalize(rid, *, claim_epoch=None, **kwargs):
            nonlocal entered
            entered += 1
            if entered == 9:
                ready.set()
            await wait_for_gate(ready)
            result = await original(rid, claim_epoch=claim_epoch, **kwargs)
            outcomes.append(result)
            return result

        monkeypatch.setattr(store, "finalize_with_event", finalize)

        async def cancel():
            await start.wait()
            await _persist_task_cancellation(claimed_store(store, run_id, epoch), run_id)

        cancellers = [asyncio.create_task(cancel()) for _ in range(8)]
        start.set()
        task.cancel()
        await asyncio.gather(*cancellers)
        bus.release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        row = await store.get_run(run_id)
        assert row.status == "cancelled" and row.claim_epoch == epoch
        assert entered == 9 and sum(bool(result) for result in outcomes) == 1
        assert calls() == 1
        events = await store.get_run_events(run_id)
        assert sum(e.event_type == "run.cancelled" for e in events) == 1
    finally:
        bus.release.set()
        await asyncio.gather(task, return_exceptions=True)
        await agent.close()


@pytest.mark.parametrize("entry", ENTRIES)
async def test_repeated_cancel_during_exact_epoch_cleanup(store, engine, monkeypatch, entry):
    bus = GateBus()
    monkeypatch.setattr(store, "_live_updates", bus)
    agent, call, run_id, epoch, calls = await invocation(entry, store, engine)
    before_calls = calls()
    cleanup_entered, cleanup_release = asyncio.Event(), asyncio.Event()
    original = store.finalize_with_event
    finalizations = []

    async def finalize(rid, *, claim_epoch=None, **kwargs):
        cleanup_entered.set()
        await cleanup_release.wait()
        finalizations.append(claim_epoch)
        return await original(rid, claim_epoch=claim_epoch, **kwargs)

    monkeypatch.setattr(store, "finalize_with_event", finalize)
    bus.enabled = True
    task = asyncio.create_task(call)
    observer = asyncio.create_task(cleanup_entered.wait())
    try:
        await wait_for_gate(bus.entered, task=task)
        task.cancel()
        done, _ = await asyncio.wait(
            {observer, task}, timeout=5, return_when=asyncio.FIRST_COMPLETED
        )
        assert observer in done, "Cancellation returned without entering owned cleanup"
        for _ in range(3):
            task.cancel()
            await asyncio.sleep(0)
        cleanup_release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 10)
        if run_id is None:
            async with engine.connect() as conn:
                run_id = (
                    await conn.execute(
                        select(AgentRun.id)
                        .order_by(AgentRun.created_at.desc(), AgentRun.id.desc())
                        .limit(1)
                    )
                ).scalar_one()
        row = await store.get_run(run_id)
        assert row.status == "cancelled" and row.claim_epoch == epoch
        assert finalizations == [epoch]
        assert calls() == before_calls
    finally:
        bus.release.set()
        cleanup_release.set()
        observer.cancel()
        await asyncio.gather(task, observer, return_exceptions=True)
        await agent.close()


@pytest.mark.parametrize("entry", BUDGET_ENTRIES)
@pytest.mark.parametrize("live", [False, True])
async def test_ownership_handoff_query_count(
    store, engine, monkeypatch, entry, live, record_property
):
    from sqlalchemy import event

    if live:
        monkeypatch.setattr(store, "_live_updates", GateBus())
    agent, call, _, _, _ = await invocation(entry, store, engine)
    statements = []

    def capture(conn, cursor, statement, *args):
        statements.append(statement)

    event.listen(engine.sync_engine, "before_cursor_execute", capture)
    try:
        await call
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", capture)
        await agent.close()
    assert (
        len(statements)
        == {
            "run": 25,
            "stream": 25,
            "resume": 43,
            "resume_stream": 42,
            "submit_tool_results": 45,
            "submit_input": 41,
            "submit_approval": 51 if engine.dialect.name == "postgresql" else 53,
            "reclaim": 30,  # +1: scoped blocked-content lookup (R1.10, task 7 ruling)
        }[entry]
    )
    record_property("statements", len(statements))
    print(f"QUERY_COUNT {engine.dialect.name} {entry} live={live} {len(statements)}")


@pytest.mark.parametrize("terminal", [False, True])
async def test_cancelled_idempotent_create_does_not_own_existing_run(store, monkeypatch, terminal):
    from dendrux.loops.react import ReActLoop
    from dendrux.runtime.runner import _create_runtime_run
    from dendrux.types import generate_ulid

    run_id = generate_ulid()
    kwargs = dict(idempotency_key="same", idempotency_fingerprint="same", tenant_id="tenant")
    await store.create_run(run_id, "handoff", **kwargs)
    if terminal:
        await store.finalize_run(run_id, status="success")
    before = await store.get_run(run_id)
    bus = GateBus()
    monkeypatch.setattr(store, "_live_updates", bus)
    bus.enqueue([run_id])
    bus.enabled = True
    entered, release = bus.entered, bus.release
    task = asyncio.create_task(
        _create_runtime_run(store, ReActLoop(), generate_ulid(), "handoff", **kwargs)
    )
    try:
        await wait_for_gate(entered, task=task)
        task.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert await store.get_run(run_id) == before
        assert not any(e.event_type == "run.cancelled" for e in await store.get_run_events(run_id))
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize("entry", ENTRIES)
async def test_handoff_cancellation_cannot_finalize_newer_epoch(store, engine, monkeypatch, entry):
    bus = GateBus()
    monkeypatch.setattr(store, "_live_updates", bus)
    agent, call, run_id, epoch, calls = await invocation(entry, store, engine)
    before_calls = calls()
    bus.enabled = True
    task = asyncio.create_task(call)
    try:
        await wait_for_gate(bus.entered, task=task)
        if run_id is None:
            async with engine.connect() as conn:
                run_id = (
                    await conn.execute(
                        select(AgentRun.id)
                        .order_by(AgentRun.created_at.desc(), AgentRun.id.desc())
                        .limit(1)
                    )
                ).scalar_one()
        item = await expire(store, engine, run_id)
        claimed = await store.reclaim_run(
            run_id,
            tenant_id="tenant",
            agent_name=agent.name,
            expected_claim_epoch=item.claim_epoch,
            governance=governance_snapshot(agent),
            governance_policy="raise",
            skill_pins={},
            loop_type="react",
        )
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 10)
        row = await store.get_run(run_id)
        assert row.status == "running" and row.claim_epoch == claimed.claim_epoch == epoch + 2
        assert calls() == before_calls
        assert not any(e.event_type == "run.cancelled" for e in await store.get_run_events(run_id))
    finally:
        bus.release.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await agent.close()


@pytest.mark.parametrize("entry", ENTRIES[2:7])
async def test_cancel_during_postclaim_governance_is_terminal(store, engine, monkeypatch, entry):
    import dendrux.runtime.runner as runner
    import dendrux.runtime.submit as submit

    agent, call, run_id, epoch, calls = await invocation(entry, store, engine)
    before_calls = calls()
    entered = asyncio.Event()

    async def report(*args, **kwargs):
        entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(
        runner if entry.startswith("resume") else submit, "report_governance_change", report
    )
    task = asyncio.create_task(call)
    try:
        await wait_for_gate(entered, task=task)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 10)
        row = await store.get_run(run_id)
        assert row.status == "cancelled" and row.claim_epoch == epoch
        assert calls() == before_calls
        events = await store.get_run_events(run_id)
        assert sum(e.event_type == "run.cancelled" for e in events) == 1
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await agent.close()


@pytest.mark.parametrize("stage", ["rebuild_error", "session_error"])
async def test_cancel_during_reclaim_error_cleanup_never_loses_epoch(
    store, engine, monkeypatch, stage
):
    import dendrux.runtime._reclaim as reclaim
    import dendrux.runtime.runner as runner
    from dendrux.errors import RunNotReclaimableError

    agent = make_agent(store)
    item = await checkpoint(agent, store, engine)
    target, name = (
        (reclaim, "rebuild_pause") if stage == "rebuild_error" else (runner, "_resume_session")
    )

    async def failed(*args, **kwargs):
        if stage == "rebuild_error":
            raise RunNotReclaimableError(item.run_id, "history_too_large")
        raise ValueError("invalid setup")

    monkeypatch.setattr(target, name, failed)
    entered, release = asyncio.Event(), asyncio.Event()
    method = "finalize_with_event" if stage == "rebuild_error" else "release_run_claim"
    original = getattr(store, method)
    armed = True

    async def cleanup(run_id, *, claim_epoch=None, **kwargs):
        nonlocal armed
        if armed:
            armed = False
            entered.set()
            await release.wait()
        return await original(run_id, claim_epoch=claim_epoch, **kwargs)

    monkeypatch.setattr(store, method, cleanup)
    task = asyncio.create_task(agent.reclaim(item.run_id, tenant_id="tenant", item=item))
    try:
        await wait_for_gate(entered, task=task)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 10)
        row = await store.get_run(item.run_id)
        assert row.status in ("error", "cancelled")
        assert row.claim_epoch == item.claim_epoch + 1
        assert agent.provider._call_count == 0
    finally:
        release.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await agent.close()
