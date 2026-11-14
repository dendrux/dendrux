"""Reclaim and adoption stop identified blocked runs before any call (R1.7, task 7)."""

import asyncio
import json
import logging
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import update

from dendrux import Agent, sweep, tool
from dendrux.db.models import AgentRun
from dendrux.errors import RunNotReclaimableError
from dendrux.guardrails import SecretDetection
from dendrux.llm.mock import MockLLM
from dendrux.runtime import _run_start
from dendrux.runtime._blocked_content import _blocked_write
from dendrux.types import LLMResponse, RunStatus, ToolCall, ToolResult
from tests._fixtures.storage import pin_storage_format
from tests._helpers.recovery_checkpoint import checkpoint, expire
from tests.integration.test_061_secrets_continuation import (
    SECRET,
    paused_agent,
    rows_are_clean,
)


class Crash(BaseException):
    pass


def _agent(store, effects, provider=None, **kwargs):
    @tool()
    async def write_region() -> str:
        """Write the region."""
        effects.append("write_region")
        return "written"

    return Agent(
        name="recovery",
        prompt="p",
        state_store=store,
        provider=provider or MockLLM([LLMResponse(text="unreachable")]),
        strict_tenancy=True,
        on_governance_change="warn",
        tools=[write_region],
        **kwargs,
    )


async def _mark(store, run_id):
    with _blocked_write(store, run_id):
        await store.save_run_event(run_id, event_type="guardrail.detected", data={})


async def _run_errors(store, run_id):
    return [e for e in await store.get_run_events(run_id) if e.event_type in ("run.error",)]


@pytest.mark.parametrize("started", [False, True])
async def test_061_reclaim_marked_run_finalizes_error(store, engine, started, caplog):
    effects = []
    provider = MockLLM([LLMResponse(text="unreachable")])
    agent = _agent(store, effects, provider)
    call = ToolCall("write_region")
    item = await checkpoint(
        agent, store, engine, calls=[call], phases=["dispatchable"], started=started
    )
    await _mark(store, item.run_id)
    try:
        with caplog.at_level(logging.WARNING):
            result = await agent.reclaim(item.run_id, tenant_id="tenant", item=item)
    finally:
        await agent.close()
    assert result.status == RunStatus.ERROR
    assert result.meta["failure_reason"] == "guardrail_blocked"
    assert provider._call_count == 0 and effects == []
    record = await store.get_run(item.run_id)
    assert record.status == "error" and record.failure_reason == "guardrail_blocked"
    assert record.meta["dendrux.blocked"] is True
    assert len(await _run_errors(store, item.run_id)) == 1
    coded = [r for r in caplog.records if getattr(r, "code", None) == "blocked_content"]
    assert [r.run_id for r in coded] == [item.run_id]


async def test_reclaim_finishes_a_crash_between_mark_and_terminal_commit(
    store, engine, monkeypatch
):
    # A blocked submission committed its claim and mark, then the process died before
    # the terminal COMMIT. The lease expires; reclaim finishes it with zero calls.
    agent, provider, run_id, raw, effects = await paused_agent(store, engine, 2, guarded=True)
    original = store.submit_and_claim

    async def crash_after_commit(*args, **kwargs):
        result = await original(*args, **kwargs)
        if result:
            raise Crash()
        return result

    monkeypatch.setattr(store, "submit_and_claim", crash_after_commit)
    try:
        with pytest.raises(Crash):
            await agent.submit_tool_results(
                run_id,
                [
                    ToolResult(
                        name="client_read",
                        call_id=raw["pending_tool_calls"][0]["id"],
                        payload=json.dumps(SECRET),
                    )
                ],
                tenant_id="t",
                expected_pause_id=raw["pause_id"],
            )
    finally:
        await agent.close()
    monkeypatch.undo()
    record = await store.get_run(run_id)
    assert record.status == "running" and record.meta["dendrux.blocked"] is True
    item = await expire(store, engine, run_id)
    restarted_provider = MockLLM([LLMResponse(text="unreachable")])
    restarted = Agent(
        name="continuation",
        prompt="Answer briefly.",
        provider=restarted_provider,
        state_store=store,
        tools=agent._tools if hasattr(agent, "_tools") else [],
        on_governance_change="warn",
    )
    from tests.integration.test_061_secrets_continuation import client_read

    restarted = Agent(
        name="continuation",
        prompt="Answer briefly.",
        provider=restarted_provider,
        state_store=store,
        tools=[client_read],
        on_governance_change="warn",
    )
    try:
        result = await restarted.reclaim(run_id, tenant_id="t", item=item)
    finally:
        await restarted.close()
    assert result.status == RunStatus.ERROR
    assert restarted_provider._call_count == 0 and effects == []
    assert (await store.get_run(run_id)).failure_reason == "guardrail_blocked"
    await rows_are_clean(engine)


async def test_reclaim_finishes_a_crash_inside_the_stop(store, engine, monkeypatch):
    # The stop's own terminal COMMIT is lost (crash after the mark and the block event);
    # the next reclaim finishes it once.
    from dendrux.runtime import runner

    effects = []
    agent = _agent(store, effects)
    item = await checkpoint(
        agent, store, engine, calls=[ToolCall("write_region")], phases=["dispatchable"]
    )
    await _mark(store, item.run_id)
    original = runner._persist_run_failure

    async def crash(*args, **kwargs):
        raise Crash()

    monkeypatch.setattr(_run_start, "_persist_run_failure", crash)
    try:
        with pytest.raises(Crash):
            await agent.reclaim(item.run_id, tenant_id="tenant", item=item)
    finally:
        await agent.close()
    monkeypatch.setattr(_run_start, "_persist_run_failure", original)
    assert (await store.get_run(item.run_id)).status == "running"
    again = await expire(store, engine, item.run_id)
    provider = MockLLM([LLMResponse(text="unreachable")])
    agent = _agent(store, effects, provider)
    try:
        result = await agent.reclaim(item.run_id, tenant_id="tenant", item=again)
    finally:
        await agent.close()
    assert result.status == RunStatus.ERROR and provider._call_count == 0 and effects == []
    assert len(await _run_errors(store, item.run_id)) == 1


async def test_reclaim_never_overwrites_a_terminal_winner(store, engine, monkeypatch):
    # Another process finalizes the run after the reclaim claim and just before the stop's
    # terminal write: reclaim returns the stored winner, with no second terminal write.
    from dendrux.runtime import runner

    effects = []
    agent = _agent(store, effects)
    item = await checkpoint(
        agent, store, engine, calls=[ToolCall("write_region")], phases=["dispatchable"]
    )
    await _mark(store, item.run_id)
    terminal_write = runner._persist_run_failure
    attempts = []

    async def winner_first(store_view, run_id, *args, **kwargs):
        attempts.append(run_id)
        async with engine.begin() as connection:
            await connection.execute(
                update(AgentRun).where(AgentRun.id == run_id).values(status="cancelled")
            )
        return await terminal_write(store_view, run_id, *args, **kwargs)

    monkeypatch.setattr(_run_start, "_persist_run_failure", winner_first)
    try:
        result = await agent.reclaim(item.run_id, tenant_id="tenant", item=item)
    finally:
        await agent.close()
    record = await store.get_run(item.run_id)
    assert record.status == "cancelled" and record.failure_reason != "guardrail_blocked"
    assert result.status == RunStatus.CANCELLED and effects == [] and attempts == [item.run_id]
    assert await _run_errors(store, item.run_id) == []


async def test_two_reclaimers_make_exactly_one_terminal_write(store, engine):
    if engine.dialect.name != "postgresql":
        pytest.skip("Postgres two-process race")
    from sqlalchemy.ext.asyncio import create_async_engine

    from dendrux.runtime.stores._factory import store_for_engine

    effects = []
    seed_agent = _agent(store, effects)
    item = await checkpoint(
        seed_agent, store, engine, calls=[ToolCall("write_region")], phases=["dispatchable"]
    )
    await seed_agent.close()
    await _mark(store, item.run_id)
    other_engine = create_async_engine(engine.url)
    other_store = store_for_engine(other_engine)
    providers = [MockLLM([LLMResponse(text="x")]), MockLLM([LLMResponse(text="x")])]
    agents = [_agent(store, effects, providers[0]), _agent(other_store, effects, providers[1])]
    try:
        outcomes = await asyncio.gather(
            *(a.reclaim(item.run_id, tenant_id="tenant", item=item) for a in agents),
            return_exceptions=True,
        )
    finally:
        for a in agents:
            await a.close()
        await other_engine.dispose()
    results = [o for o in outcomes if not isinstance(o, BaseException)]
    refused = [o for o in outcomes if isinstance(o, RunNotReclaimableError)]
    assert len(results) == 1 and len(refused) == 1, outcomes
    assert results[0].status == RunStatus.ERROR
    assert sum(p._call_count for p in providers) == 0 and effects == []
    assert len(await _run_errors(store, item.run_id)) == 1


async def test_format1_stale_sweep_keeps_classification_and_mark(store, engine):
    effects = []
    agent = _agent(store, effects)
    await pin_storage_format(engine, 1)
    run_id = (await agent.run("go", tenant_id="tenant")).run_id
    await agent.close()
    async with engine.begin() as connection:
        await connection.execute(
            update(AgentRun)
            .where(AgentRun.id == run_id)
            .values(status="running", last_progress_at=datetime.now(UTC) - timedelta(minutes=10))
        )
    await _mark(store, run_id)
    await sweep(state_store=store, stale_running=timedelta(minutes=1), on_stale="reclaim")
    record = await store.get_run(run_id)
    assert record.meta["dendrux.blocked"] is True
    assert record.status in ("error", "running")


@pytest.mark.parametrize("fmt", [1, 2])
async def test_adopted_unprojected_input_never_reaches_events_or_traces(
    store, engine, fmt, monkeypatch
):
    # input claimed without projection (raw store, guardrails added later) is
    # projected before the run.resumed event and the user trace are written.
    from dendrux.runtime.runner import resume_claimed

    agent, _, run_id, raw, _ = await paused_agent(store, engine, fmt)
    raw["pending_tool_calls"] = []
    await store.pause_run(run_id, status="waiting_human_input", pause_data=raw)
    original = store.submit_and_claim

    async def crash_after_commit(*args, **kwargs):
        result = await original(*args, **kwargs)
        if result:
            raise Crash()
        return result

    monkeypatch.setattr(store, "submit_and_claim", crash_after_commit)
    try:
        with pytest.raises(Crash):
            await agent.submit_input(
                run_id, SECRET, tenant_id="t", expected_pause_id=raw["pause_id"]
            )
    finally:
        await agent.close()
    monkeypatch.undo()
    from tests.integration.test_061_secrets_continuation import client_read

    adopter_provider = MockLLM([LLMResponse(text="unreachable")])
    adopter = Agent(
        name="continuation",
        prompt="Answer briefly.",
        provider=adopter_provider,
        state_store=store,
        tools=[client_read],
        guardrails=[SecretDetection()],
        on_governance_change="warn",
    )
    record = await store.get_run(run_id)
    try:
        result = await resume_claimed(
            run_id,
            state_store=store,
            agent=adopter,
            provider=adopter._require_provider(),
            pre_claim_status="waiting_human_input",
            claim_epoch=record.claim_epoch,
        )
    finally:
        await adopter.close()
    assert result.status == RunStatus.ERROR and adopter_provider._call_count == 0
    assert SECRET not in (result.error or "")
    record = await store.get_run(run_id)
    assert record.failure_reason == "guardrail_blocked" and record.meta["dendrux.blocked"]
    events = await store.get_run_events(run_id)
    assert SECRET not in repr([(e.event_type, e.data) for e in events])
    traces = await store.get_traces(run_id)
    assert SECRET not in repr([t.content for t in traces])


@pytest.mark.parametrize("fmt", [1, 2])
async def test_failure_handler_survives_an_exception_whose_str_raises(store, engine, fmt):
    # deriving the failure payload must not escape the handler.
    from dendrux.runtime.runner import _persist_run_failure

    class BadStrError(Exception):
        def __str__(self):
            raise RuntimeError("no text")

    await pin_storage_format(engine, fmt)
    effects = []
    agent = _agent(store, effects)
    run_id = (await agent.run("go", tenant_id="tenant")).run_id
    await agent.close()
    async with engine.begin() as connection:
        await connection.execute(
            update(AgentRun).where(AgentRun.id == run_id).values(status="running")
        )
    assert await _persist_run_failure(store, run_id, BadStrError()) is False


async def _legacy_checkpoint(agent, store, engine):
    # A 0.6.0-shaped run: raw text in its stored history, a guardrail.blocked event but no
    # mark (0.6.0 never wrote one), still running after a crash, lease expired.
    from dendrux.runtime.persistence import message_trace
    from dendrux.runtime.submit import governance_snapshot
    from dendrux.types import Message, Role, UsageStats, _usage_to_dict, generate_ulid

    run_id = generate_ulid()
    await store.create_run(
        run_id,
        agent.name,
        tenant_id="tenant",
        seed_traces=[
            message_trace(Message(Role.SYSTEM, agent.prompt), 0, kind="seed"),
            message_trace(Message(Role.USER, f"secret {SECRET}"), 1, kind="seed"),
        ],
        meta={"dendrux.governance": governance_snapshot(agent)},
    )
    await store.checkpoint_step(
        run_id,
        claim_epoch=0,
        loop_state=dict(
            v=1,
            loop="react",
            phase="model_next",
            iteration=0,
            trace_order_offset=2,
            steps_count=0,
            usage=_usage_to_dict(UsageStats()),
            tool_counts={},
            provider_kwargs={},
            pending=[],
            submission=None,
            skills={"pins": {}},
        ),
        pii_mapping=None,
        trace=None,
    )
    await store.save_run_event(run_id, event_type="guardrail.blocked", data={})
    record = await store.get_run(run_id)
    meta = {k: v for k, v in (record.meta or {}).items() if k != "dendrux.blocked"}
    async with engine.begin() as connection:
        await connection.execute(
            update(AgentRun)
            .where(AgentRun.id == run_id)
            .values(meta=meta, last_progress_at=datetime.now(UTC) - timedelta(minutes=10))
        )
    page = await sweep(state_store=store, stale_running=timedelta(minutes=1), on_stale="reclaim")
    return next(item for item in page.reclaimable if item.run_id == run_id)


async def test_reclaim_of_legacy_event_only_run_finalizes_error(store, engine, caplog):
    import importlib.util
    from pathlib import Path

    effects = []
    provider = MockLLM([LLMResponse(text="unreachable")])
    agent = _agent(store, effects, provider)
    item = await _legacy_checkpoint(agent, store, engine)
    assert "dendrux.blocked" not in ((await store.get_run(item.run_id)).meta or {})
    try:
        with caplog.at_level(logging.DEBUG):
            caplog.clear()
            result = await agent.reclaim(item.run_id, tenant_id="tenant", item=item)
    finally:
        await agent.close()
    assert result.status == RunStatus.ERROR
    assert result.meta["failure_reason"] == "guardrail_blocked"
    assert provider._call_count == 0 and effects == []
    record = await store.get_run(item.run_id)
    assert record.status == "error" and record.failure_reason == "guardrail_blocked"
    # Sentinel scan of everything the reclaim could have sent or emitted. The legacy row
    # itself keeps its raw text (no erasure, R1.11), so the database is not scanned here.
    scanner_path = (
        Path(__file__).resolve().parents[5]
        / "owner-docs/design-0.6.x/work/build-0.6.1/live/surface_scanner.py"
    )
    if scanner_path.exists():
        spec = importlib.util.spec_from_file_location("surface_scanner", scanner_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        scanner = module.SurfaceScanner([SECRET])
        scanner.scan_value(
            {"result": result.to_dict(), "logs": caplog.text, "wire": provider.wire_messages},
            "reclaim",
        )
        assert not scanner.hits
    assert SECRET not in caplog.text and SECRET not in str(result.to_dict())
    assert provider.wire_messages == []


async def test_failed_reclaim_lookup_fails_closed(store, engine, monkeypatch):
    effects = []
    provider = MockLLM([LLMResponse(text="unreachable")])
    agent = _agent(store, effects, provider)
    item = await checkpoint(agent, store, engine)

    async def unavailable(*args, **kwargs):
        raise ValueError(SECRET)

    monkeypatch.setattr(store, "_blocked_run_ids", unavailable)
    from dendrux.errors import PersistenceError

    try:
        with pytest.raises(PersistenceError) as caught:
            await agent.reclaim(item.run_id, tenant_id="tenant", item=item)
    finally:
        await agent.close()
    assert SECRET not in str(caught.value)
    assert caught.value.__context__ is None and caught.value.__cause__ is None
    assert provider._call_count == 0 and effects == []
