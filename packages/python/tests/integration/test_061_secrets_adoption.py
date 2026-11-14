"""Adopted submissions keep their first verdict through durable evidence."""

import hashlib
import json
from functools import wraps

import pytest
from sqlalchemy import select

from dendrux import Agent
from dendrux.db.models import Base
from dendrux.guardrails._protocol import Finding
from dendrux.llm.mock import MockLLM
from dendrux.runtime import _run_events, runner
from dendrux.strategies.native import NativeToolCalling
from dendrux.types import (
    AgentStep,
    ApprovalDecision,
    Clarification,
    LLMResponse,
    RunStatus,
    ToolResult,
)
from tests._fixtures.storage import pin_storage_format
from tests._helpers.recovery_checkpoint import expire
from tests.integration.test_061_secrets_continuation import SECRET, client_read, paused_agent


class Crash(BaseException):
    pass


class AskInput(NativeToolCalling):
    def parse_response(self, response):
        return AgentStep(reasoning=None, action=Clarification(question="Which value?"))


class FirstVerdict:
    action = "block"

    def __init__(self, transient):
        self.transient = transient
        self.calls = 0

    async def scan(self, text):
        if SECRET not in text:
            return []
        self.calls += 1
        if self.transient:
            if self.calls == 1:
                raise RuntimeError("Detector temporarily unavailable.")
            return []
        start = text.index(SECRET)
        return [Finding("SECRET", start, start + len(SECRET), 1.0, SECRET)]


async def _raw_claim(store, engine, monkeypatch, fmt, kind):
    if kind == "input":
        await pin_storage_format(engine, fmt)
        agent = Agent(
            name="continuation",
            prompt="Answer briefly.",
            tools=[client_read],
            provider=MockLLM([LLMResponse(text="Which value?")]),
            state_store=store,
        )
        paused = await agent.run("go", tenant_id="t", strategy=AskInput())
        assert paused.status == RunStatus.WAITING_HUMAN_INPUT
        run_id = paused.run_id
        raw = await store.get_pause_state(run_id)
    else:
        agent, _, run_id, raw, _ = await paused_agent(
            store, engine, fmt, approval=kind in ("rejection", "approve", "deny")
        )
    status = (await store.get_run(run_id)).status
    kwargs = {"tenant_id": "t", "expected_pause_id": raw["pause_id"]}
    call_id = raw["pending_tool_calls"][0]["id"] if raw["pending_tool_calls"] else None
    if kind == "note":
        await agent.submit_note(run_id, SECRET, tenant_id="t")
    original = store.submit_and_claim

    @wraps(original)
    async def cut(*args, **kwargs):
        won = await original(*args, **kwargs)
        if won:
            raise Crash()
        return won

    try:
        with monkeypatch.context() as patch:
            patch.setattr(store, "submit_and_claim", cut)
            with pytest.raises(Crash):
                if kind == "input":
                    await agent.submit_input(run_id, SECRET, **kwargs)
                elif kind == "rejection":
                    await agent.submit_approval(
                        run_id, approved=False, rejection_reason=SECRET, **kwargs
                    )
                elif kind in ("approve", "deny"):
                    await agent.submit_approval(
                        run_id,
                        decisions={call_id: ApprovalDecision(kind == "approve", SECRET)},
                        **kwargs,
                    )
                else:
                    await agent.submit_tool_results(
                        run_id,
                        [
                            ToolResult(
                                name="client_read",
                                call_id=call_id,
                                payload=json.dumps("okay" if kind == "note" else SECRET),
                            )
                        ],
                        **kwargs,
                    )
    finally:
        await agent.close()
    return agent, run_id, status


async def _retained_values(engine):
    # Existing unguarded rows are the starting condition. New columns/rows must
    # not copy their text or raw-derived identities after a rejected verdict.
    candidates = [
        SECRET,
        hashlib.sha256(SECRET.encode()).hexdigest(),
        hashlib.sha256(json.dumps(SECRET).encode()).hexdigest(),
    ]
    retained = set()
    async with engine.connect() as conn:
        for table in Base.metadata.sorted_tables:
            for row in (await conn.execute(select(table))).mappings():
                identity = tuple(str(row[c.name]) for c in table.primary_key.columns)
                for key, value in row.items():
                    encoded = repr(value)
                    if any(candidate in encoded for candidate in candidates):
                        retained.add((table.name, identity, key, encoded))
    return retained


def _adopter(store, original, guard=None):
    return Agent(
        name=original.name,
        prompt=original.prompt,
        tools=original.tools,
        require_approval=original.require_approval,
        provider=MockLLM([LLMResponse(text="unreachable")]),
        state_store=store,
        guardrails=[guard] if guard is not None else None,
        on_governance_change="warn",
    )


@pytest.mark.parametrize("fmt", [1, 2])
@pytest.mark.parametrize("kind", ["tool", "input", "rejection", "approve", "deny", "note"])
@pytest.mark.parametrize("transient", [False, True])
async def test_adopted_submission_preserves_first_verdict(
    store, engine, monkeypatch, caplog, fmt, kind, transient
):
    original, run_id, status = await _raw_claim(store, engine, monkeypatch, fmt, kind)
    before = await _retained_values(engine)
    guard = FirstVerdict(transient)
    adopter = _adopter(store, original, guard)
    record = await store.get_run(run_id)
    try:
        result = await runner.resume_claimed(
            run_id,
            state_store=store,
            agent=adopter,
            provider=adopter._require_provider(),
            pre_claim_status=status,
            claim_epoch=record.claim_epoch,
        )
        assert result.status == RunStatus.ERROR
        assert result.meta["failure_reason"] == "guardrail_blocked"
        assert ("UNSCANNABLE" if transient else "SECRET") in result.error
        assert adopter._require_provider()._call_count == 0
        assert guard.calls == 1
        assert SECRET not in repr(result) and SECRET not in caplog.text
        assert await _retained_values(engine) <= before
        assert (await store.get_run(run_id)).meta["dendrux.blocked"] is True
    finally:
        await adopter.close()


@pytest.mark.parametrize("fmt", [1, 2])
@pytest.mark.parametrize("kind", ["tool", "input", "rejection", "approve", "deny"])
async def test_adopted_verdict_survives_first_event_commit(
    store, engine, monkeypatch, caplog, fmt, kind
):
    original, run_id, status = await _raw_claim(store, engine, monkeypatch, fmt, kind)
    before = await _retained_values(engine)
    guard = FirstVerdict(True)
    adopter = _adopter(store, original, guard)
    emit = runner._emit_event

    async def cut(store_view, rid, event_type, data, **kwargs):
        await emit(store_view, rid, event_type, data, **kwargs)
        if event_type == "run.resumed":
            assert SECRET not in repr(data)
            raise Crash()

    record = await store.get_run(run_id)
    try:
        with monkeypatch.context() as patch:
            patch.setattr(_run_events, "_emit_event", cut)
            with pytest.raises(Crash):
                await runner.resume_claimed(
                    run_id,
                    state_store=store,
                    agent=adopter,
                    provider=adopter._require_provider(),
                    pre_claim_status=status,
                    claim_epoch=record.claim_epoch,
                )
    finally:
        await adopter.close()
    assert guard.calls == 1
    assert (await store.get_run(run_id)).meta.get("dendrux.blocked") is True
    assert await _retained_values(engine) <= before
    restarted = _adopter(store, original)
    try:
        if fmt == 2:
            item = await expire(store, engine, run_id)
            result = await restarted.reclaim(run_id, tenant_id="t", item=item)
        else:
            result = await runner.resume_claimed(
                run_id,
                state_store=store,
                agent=restarted,
                provider=restarted._require_provider(),
                pre_claim_status=status,
                claim_epoch=record.claim_epoch,
            )
        assert result.status == RunStatus.ERROR
        assert result.meta["failure_reason"] == "guardrail_blocked"
        assert restarted._require_provider()._call_count == 0
        assert SECRET not in repr(result) and SECRET not in caplog.text
        assert await _retained_values(engine) <= before
    finally:
        await restarted.close()


@pytest.mark.parametrize("kind", ["tool", "input", "approve"])
@pytest.mark.parametrize("honors_mark", [False, True])
async def test_custom_adoption_marks_before_first_event(
    store, engine, monkeypatch, kind, honors_mark
):
    from dendrux.errors import PersistenceError
    from dendrux.runtime._blocked_content import _blocked_write
    from tests._fixtures.stores import LegacyInMemoryStore

    original, run_id, status = await _raw_claim(store, engine, monkeypatch, 1, kind)
    before = await _retained_values(engine)

    class ContractStore(LegacyInMemoryStore):
        async def update_run_meta(self, run, meta, **kwargs):
            if honors_mark:
                with _blocked_write(self.wrapped, run):
                    return await self.wrapped.update_run_meta(run, meta, **kwargs)
            return await self.wrapped.update_run_meta(run, meta, **kwargs)

    custom = ContractStore(store)
    guard = FirstVerdict(True)
    adopter = _adopter(custom, original, guard)
    emit = runner._emit_event
    resumed = []

    async def inspect(view, rid, event_type, data, **kwargs):
        if event_type == "run.resumed":
            assert (await store.get_run(rid)).meta.get("dendrux.blocked") is True
            resumed.append(True)
        return await emit(view, rid, event_type, data, **kwargs)

    monkeypatch.setattr(_run_events, "_emit_event", inspect)
    try:
        call = runner.resume_claimed(
            run_id,
            state_store=custom,
            agent=adopter,
            provider=adopter._require_provider(),
            pre_claim_status=status,
        )
        if honors_mark:
            result = await call
            assert result.status == RunStatus.ERROR and resumed == [True]
            assert result.meta["failure_reason"] == "guardrail_blocked"
        else:
            with pytest.raises(PersistenceError):
                await call
            assert resumed == []
        assert guard.calls == 1 and adopter._require_provider()._call_count == 0
        assert await _retained_values(engine) <= before
    finally:
        await adopter.close()


@pytest.mark.parametrize("fmt", [1, 2])
async def test_adopted_note_mark_survives_first_trace_commit(store, engine, monkeypatch, fmt):
    from dendrux.runtime.persistence import PersistenceRecorder
    from dendrux.types import Role

    original, run_id, status = await _raw_claim(store, engine, monkeypatch, fmt, "note")
    before = await _retained_values(engine)
    guard = FirstVerdict(True)
    adopter = _adopter(store, original, guard)
    append = PersistenceRecorder.on_message_appended

    async def cut(recorder, rid, message, iteration):
        await append(recorder, rid, message, iteration)
        if message.role == Role.USER and guard.calls:
            assert SECRET not in repr(message)
            raise Crash()

    record = await store.get_run(run_id)
    try:
        with monkeypatch.context() as patch:
            patch.setattr(PersistenceRecorder, "on_message_appended", cut)
            with pytest.raises(Crash):
                await runner.resume_claimed(
                    run_id,
                    state_store=store,
                    agent=adopter,
                    provider=adopter._require_provider(),
                    pre_claim_status=status,
                    claim_epoch=record.claim_epoch,
                )
    finally:
        await adopter.close()
    assert guard.calls == 1
    assert (await store.get_run(run_id)).meta.get("dendrux.blocked") is True
    assert await _retained_values(engine) <= before
    restarted = _adopter(store, original)
    try:
        if fmt == 2:
            item = await expire(store, engine, run_id)
            result = await restarted.reclaim(run_id, tenant_id="t", item=item)
        else:
            result = await runner.resume_claimed(
                run_id,
                state_store=store,
                agent=restarted,
                provider=restarted._require_provider(),
                pre_claim_status=status,
                claim_epoch=record.claim_epoch,
            )
        assert result.status == RunStatus.ERROR
        assert result.meta["failure_reason"] == "guardrail_blocked"
        assert restarted._require_provider()._call_count == 0
        assert await _retained_values(engine) <= before
    finally:
        await restarted.close()
