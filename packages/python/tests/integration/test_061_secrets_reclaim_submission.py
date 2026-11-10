"""Reclaim projects an unconsumed stored submission before recovery work (R1.1, R1.2, R1.7)."""

import pytest

from dendrux.errors import RunNotReclaimableError
from dendrux.types import RunStatus, _usage_from_dict
from tests._helpers.recovery_checkpoint import expire
from tests.integration.test_061_secrets_adoption import (
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
