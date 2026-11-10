"""Reclaim projects an unconsumed stored submission before recovery work (R1.1, R1.2, R1.7)."""

from functools import wraps

import pytest

from dendrux import Agent, tool
from dendrux.errors import RunNotReclaimableError
from dendrux.guardrails._protocol import Finding
from dendrux.llm.mock import MockLLM
from dendrux.types import (
    ApprovalDecision,
    LLMResponse,
    RunStatus,
    ToolCall,
    _usage_from_dict,
)
from tests._fixtures.storage import pin_storage_format
from tests._helpers.recovery_checkpoint import expire
from tests.integration.test_061_secrets_adoption import (
    Crash,
    FirstVerdict,
    _adopter,
    _raw_claim,
    _retained_values,
)
from tests.integration.test_061_secrets_continuation import SECRET

KINDS = ["tool", "input", "rejection", "approve", "deny"]


class AllowAll(FirstVerdict):
    """Scans like the blocking detector but never blocks (clean control)."""

    async def scan(self, text):
        if SECRET in text:
            self.calls += 1
        return []


async def _events(store, run_id):
    return [e.event_type for e in await store.get_run_events(run_id)]


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("transient", [False, True])
async def test_reclaim_projects_unconsumed_submission_before_recovery(
    store, engine, monkeypatch, kind, transient
):
    agent, run_id, _ = await _raw_claim(store, engine, monkeypatch, 2, kind)
    checkpoint = await store.get_loop_state(run_id)
    assert checkpoint.get("submission"), "the crashed claim left its submission unconsumed"
    item = await expire(store, engine, run_id)
    before = await _retained_values(engine)
    marker = len(await _events(store, run_id))
    guard = FirstVerdict(transient)
    adopter = _adopter(store, agent, guard)
    provider = adopter._require_provider()
    try:
        result = await adopter.reclaim(run_id, tenant_id="t", item=item)
    finally:
        await adopter.close()
    row = await store.get_run(run_id)
    assert result.status == RunStatus.ERROR
    assert result.meta["failure_reason"] == "guardrail_blocked"
    assert row.status == "error" and row.failure_reason == "guardrail_blocked"
    assert row.meta["dendrux.blocked"] is True
    assert provider._call_count == 0
    new_events = (await _events(store, run_id))[marker:]
    # The stop happens before recovery resolution: no decision event, no tool work.
    assert "approval.decided" not in new_events and "tool.started" not in new_events
    assert new_events[-2:] == ["guardrail.blocked", "run.error"]
    # One raw scan; a transient detector failure stays a block (never a second raw scan).
    assert guard.calls == 1
    # Sentinel scan of every table: no new row or column holds the raw text or its digests.
    assert (await _retained_values(engine)) - before == set()
    # Committed accounting comes from the checkpoint; like adoption through resume_claimed,
    # the stop reports the turn that would have consumed the submission.
    assert result.iteration_count == row.iteration_count == checkpoint["iteration"] + 1
    committed = _usage_from_dict(checkpoint["usage"])
    tokens = ("input_tokens", "output_tokens", "total_tokens")
    assert [getattr(result.usage, k) for k in tokens] == [getattr(committed, k) for k in tokens]


@pytest.mark.parametrize("kind", KINDS)
async def test_reclaim_of_clean_unconsumed_submission_is_unchanged(
    store, engine, monkeypatch, kind
):
    agent, run_id, _ = await _raw_claim(store, engine, monkeypatch, 2, kind)
    item = await expire(store, engine, run_id)
    guard = AllowAll(False)
    adopter = _adopter(store, agent, guard)
    try:
        result = await adopter.reclaim(run_id, tenant_id="t", item=item)
    finally:
        await adopter.close()
    row = await store.get_run(run_id)
    assert result.meta.get("failure_reason") != "guardrail_blocked"
    assert "dendrux.blocked" not in (row.meta or {})


@pytest.mark.parametrize("kind", KINDS)
async def test_format1_claim_is_not_reclaimable_and_untouched(store, engine, monkeypatch, kind):
    agent, run_id, _ = await _raw_claim(store, engine, monkeypatch, 1, kind)
    before = await _retained_values(engine)
    row_before = await store.get_run(run_id)
    guard = FirstVerdict(False)
    adopter = _adopter(store, agent, guard)
    provider = adopter._require_provider()
    try:
        with pytest.raises(RunNotReclaimableError) as caught:
            await adopter.reclaim(run_id, tenant_id="t", item=None)
    finally:
        await adopter.close()
    assert caught.value.reason == "legacy_run"
    assert provider._call_count == 0 and guard.calls == 0
    row = await store.get_run(run_id)
    assert (row.status, row.meta, row.failure_reason) == (
        row_before.status,
        row_before.meta,
        row_before.failure_reason,
    )
    assert (await _retained_values(engine)) == before


TRANSITIONS = ["unavailable", "finding"]


class LaterVerdict:
    """Allows its first scan of the text; any later scan blocks (unavailable or a finding)."""

    action = "block"

    def __init__(self, later):
        self.later = later
        self.calls = 0

    async def scan(self, text):
        if SECRET not in text:
            return []
        self.calls += 1
        if self.calls == 1:
            return []
        if self.later == "unavailable":
            raise RuntimeError("Detector temporarily unavailable.")
        start = text.index(SECRET)
        return [Finding("SECRET", start, start + len(SECRET), 1.0, SECRET)]


def _rescan_in_recovery(monkeypatch):
    """Keep nothing in the engine's bounded projection cache, so recovery projects the
    submitted messages again (as it does once more than the cache holds came between)."""
    import dendrux.guardrails._engine as guard_engine

    monkeypatch.setattr(guard_engine, "_PROJECTION_CACHE_LIMIT", 0)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("later", TRANSITIONS)
async def test_reclaim_scans_each_stored_reason_once(store, engine, monkeypatch, kind, later):
    agent, run_id, _ = await _raw_claim(store, engine, monkeypatch, 2, kind)
    item = await expire(store, engine, run_id)
    before = await _retained_values(engine)
    guard = LaterVerdict(later)
    adopter = _adopter(store, agent, guard)
    try:
        result = await adopter.reclaim(run_id, tenant_id="t", item=item)
    finally:
        await adopter.close()
    row = await store.get_run(run_id)
    # The verdict for the form recovery stores and sends is taken once and reused.
    assert guard.calls == 1
    assert result.meta.get("failure_reason") != "guardrail_blocked"
    assert "dendrux.blocked" not in (row.meta or {})
    new = (await _retained_values(engine)) - before
    assert not (result.meta.get("failure_reason") == "guardrail_blocked" and new)


async def _sibling_claim(store, engine, monkeypatch, effects, guard):
    await pin_storage_format(engine, 2)

    @tool(parallel=False)
    async def held() -> str:
        effects.append(("held", guard.calls))
        return "ok"

    @tool(parallel=False)
    async def sibling() -> str:
        effects.append(("sibling", guard.calls))
        return "ok"

    original = Agent(
        name="continuation",
        prompt="Answer briefly.",
        tools=[held, sibling],
        require_approval=["held"],
        state_store=store,
        provider=MockLLM([LLMResponse(tool_calls=[ToolCall("held"), ToolCall("sibling")])]),
    )
    paused = await original.run("go", tenant_id="t")
    run_id = paused.run_id
    raw = await store.get_pause_state(run_id)
    call_id = next(tc["id"] for tc in raw["pending_tool_calls"] if tc["name"] == "held")
    submit = store.submit_and_claim

    @wraps(submit)
    async def cut(*args, **kwargs):
        await submit(*args, **kwargs)
        raise Crash()

    try:
        with monkeypatch.context() as patch:
            patch.setattr(store, "submit_and_claim", cut)
            with pytest.raises(Crash):
                await original.submit_approval(
                    run_id,
                    decisions={call_id: ApprovalDecision(False, SECRET)},
                    tenant_id="t",
                    expected_pause_id=raw["pause_id"],
                )
    finally:
        await original.close()
    assert effects == []
    return original, run_id


@pytest.mark.parametrize("later", TRANSITIONS)
async def test_allowed_denial_lets_its_sibling_run_after_one_scan(
    store, engine, monkeypatch, later
):
    effects = []
    guard = LaterVerdict(later)
    original, run_id = await _sibling_claim(store, engine, monkeypatch, effects, guard)
    item = await expire(store, engine, run_id)
    adopter = _adopter(store, original, guard)
    try:
        result = await adopter.reclaim(run_id, tenant_id="t", item=item)
    finally:
        await adopter.close()
    assert guard.calls == 1
    assert effects == [("sibling", 1)]
    assert result.meta.get("failure_reason") != "guardrail_blocked"


@pytest.mark.parametrize("later", TRANSITIONS)
async def test_later_recovery_block_stops_before_sibling(store, engine, monkeypatch, later):
    effects = []
    guard = LaterVerdict(later)
    original, run_id = await _sibling_claim(store, engine, monkeypatch, effects, guard)
    item = await expire(store, engine, run_id)
    marker = len(await _events(store, run_id))
    _rescan_in_recovery(monkeypatch)
    adopter = _adopter(store, original, guard)
    provider = adopter._require_provider()
    try:
        result = await adopter.reclaim(run_id, tenant_id="t", item=item)
    finally:
        await adopter.close()
    row = await store.get_run(run_id)
    assert guard.calls == 2
    assert effects == []
    assert result.status == RunStatus.ERROR
    assert result.meta["failure_reason"] == "guardrail_blocked"
    assert row.meta["dendrux.blocked"] is True and row.failure_reason == "guardrail_blocked"
    assert provider._call_count == 0
    new_events = (await _events(store, run_id))[marker:]
    assert "tool.started" not in new_events and "tool.completed" not in new_events
    assert new_events[-2:] == ["guardrail.blocked", "run.error"]


@pytest.mark.parametrize("later", TRANSITIONS)
async def test_later_recovery_block_never_consumes_an_unmarked_placeholder(
    store, engine, monkeypatch, later
):
    from dendrux.runtime.persistence import PersistenceRecorder

    agent, run_id, _ = await _raw_claim(store, engine, monkeypatch, 2, "deny")
    item = await expire(store, engine, run_id)
    guard = LaterVerdict(later)
    adopter = _adopter(store, agent, guard)
    provider = adopter._require_provider()
    consume = PersistenceRecorder.consume_resume
    consumed = []

    async def cut(recorder, messages):
        await consume(recorder, messages)
        consumed.append((await store.get_run(run_id)).meta.get("dendrux.blocked"))
        raise Crash()

    _rescan_in_recovery(monkeypatch)
    try:
        with monkeypatch.context() as patch:
            patch.setattr(PersistenceRecorder, "consume_resume", cut)
            result = await adopter.reclaim(run_id, tenant_id="t", item=item)
    finally:
        await adopter.close()
    # The later block stops the run before recovery consumes the submission.
    assert consumed == []
    assert guard.calls == 2
    assert result.meta["failure_reason"] == "guardrail_blocked"
    row = await store.get_run(run_id)
    assert row.meta["dendrux.blocked"] is True and row.status == "error"
    assert provider._call_count == 0
    # A guardrail-free process can never finish the run afterwards.
    restarted = _adopter(store, agent)
    try:
        with pytest.raises(RunNotReclaimableError):
            await restarted.reclaim(run_id, tenant_id="t", item=None)
    finally:
        await restarted.close()
    assert restarted._require_provider()._call_count == 0
