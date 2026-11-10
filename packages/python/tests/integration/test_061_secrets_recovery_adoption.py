"""Adoption retains rejection and clean history across recovery approval pauses."""

from functools import wraps

import pytest

from dendrux.llm.mock import MockLLM
from dendrux.runtime import _reclaim, runner
from dendrux.types import ApprovalDecision, LLMResponse, RunStatus, ToolCall
from tests._fixtures.storage import pin_storage_format
from tests._helpers.recovery_checkpoint import checkpoint
from tests.integration.test_061_secrets_adoption import (
    Crash,
    FirstVerdict,
    _adopter,
    _retained_values,
)
from tests.integration.test_061_secrets_continuation import SECRET
from tests.integration.test_061_secrets_reclaim import _agent


async def _claimed_approval(store, engine, monkeypatch, fmt, kind, reason, effects):
    await pin_storage_format(engine, fmt)
    provider = MockLLM([LLMResponse(tool_calls=[ToolCall("write_region")])])
    agent = _agent(
        store, effects, provider, **({"require_approval": ["write_region"]} if fmt == 1 else {})
    )
    if fmt == 2:
        item = await checkpoint(
            agent,
            store,
            engine,
            calls=[ToolCall("write_region")],
            phases=["dispatchable"],
            started=True,
        )
        paused = await agent.reclaim(item.run_id, tenant_id="tenant", item=item)
    else:
        # Legacy runs support approval adoption, but not checkpoint recovery.
        paused = await agent.run("go", tenant_id="tenant")
    assert paused.status == RunStatus.WAITING_APPROVAL
    raw = await store.get_pause_state(paused.run_id)
    call_id = raw["pending_tool_calls"][0]["id"]
    if fmt == 2:
        assert raw["approval_gates"][call_id]["source"] == "recovery"
    submit = store.submit_and_claim

    @wraps(submit)
    async def cut(*args, **kwargs):
        won = await submit(*args, **kwargs)
        if won:
            raise Crash()
        return won

    with monkeypatch.context() as patch:
        patch.setattr(store, "submit_and_claim", cut)
        with pytest.raises(Crash):
            kwargs = {"tenant_id": "tenant", "expected_pause_id": paused.pause_id}
            if kind == "batch":
                await agent.submit_approval(
                    paused.run_id, approved=False, rejection_reason=reason, **kwargs
                )
            else:
                await agent.submit_approval(
                    paused.run_id,
                    decisions={call_id: ApprovalDecision(kind == "approve", reason)},
                    **kwargs,
                )
    await agent.close()
    return agent, paused.run_id


@pytest.mark.parametrize("fmt", [1, 2])
@pytest.mark.parametrize("kind", ["deny", "approve", "batch"])
@pytest.mark.parametrize("transient", [False, True])
async def test_adopted_approval_block_precedes_resolution(
    store, engine, monkeypatch, caplog, fmt, kind, transient
):
    effects = []
    original, run_id = await _claimed_approval(
        store, engine, monkeypatch, fmt, kind, f"key {SECRET}", effects
    )
    before = await _retained_values(engine)
    guard = FirstVerdict(transient)
    agent = _adopter(store, original, guard)

    async def forbidden(*args, **kwargs):
        raise AssertionError("Blocked adoption reached approval resolution.")

    monkeypatch.setattr(_reclaim, "resolve", forbidden)
    monkeypatch.setattr(runner, "_resume_approval_work", forbidden)
    try:
        result = await runner.resume_claimed(
            run_id,
            state_store=store,
            agent=agent,
            provider=agent._require_provider(),
            pre_claim_status="waiting_approval",
        )
        assert result.status == RunStatus.ERROR
        assert result.meta["failure_reason"] == "guardrail_blocked"
        assert ("UNSCANNABLE" if transient else "SECRET") in result.error
        assert guard.calls == 1
        assert agent._require_provider()._call_count == 0 and effects == []
        row = await store.get_run(run_id)
        assert row.failure_reason == "guardrail_blocked" and row.meta["dendrux.blocked"] is True
        assert await _retained_values(engine) <= before
        assert SECRET not in repr(result) and SECRET not in caplog.text
    finally:
        await agent.close()


async def _clean_wire(store, engine, monkeypatch, fmt, kind, guarded):
    effects = []
    original, run_id = await _claimed_approval(
        store, engine, monkeypatch, fmt, kind, "not now", effects
    )
    agent = _adopter(store, original, FirstVerdict(False) if guarded else None)
    try:
        result = await runner.resume_claimed(
            run_id,
            state_store=store,
            agent=agent,
            provider=agent._require_provider(),
            pre_claim_status="waiting_approval",
        )
        assert result.status == RunStatus.SUCCESS
        assert agent._require_provider()._call_count == 1
        messages = agent._require_provider().call_history[0]["messages"]
        wire = [
            (
                message.role,
                message.content,
                message.name,
                bool(message.call_id),
                [(call.name, call.params) for call in message.tool_calls or ()],
            )
            for message in messages
        ]
        return wire, effects
    finally:
        await agent.close()


@pytest.mark.parametrize("fmt", [1, 2])
@pytest.mark.parametrize("kind", ["deny", "approve", "batch"])
async def test_clean_adoption_preserves_recovery_history(store, engine, monkeypatch, fmt, kind):
    baseline = await _clean_wire(store, engine, monkeypatch, fmt, kind, False)
    guarded = await _clean_wire(store, engine, monkeypatch, fmt, kind, True)
    assert guarded == baseline
