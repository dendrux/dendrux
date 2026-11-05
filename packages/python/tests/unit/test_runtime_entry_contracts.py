"""Runtime entry preconditions and frozen-corpus metadata compatibility."""

import pytest

from dendrux import Agent, run
from dendrux.runtime.runner import run_stream
from dendrux.types import LLMResponse, RunEventType
from tests._helpers.scripted_provider import ScriptedLLM


@pytest.mark.parametrize("entry", ["agent", "run", "run_stream"])
async def test_idempotency_without_store_refuses_before_provider(entry):
    provider = ScriptedLLM([LLMResponse(text="unused")])
    agent = Agent(prompt="test", provider=provider)
    message = (
        "idempotency_key requires persistence (database_url or state_store). "
        "Idempotency cannot work without durable state."
    )
    if entry == "run_stream":
        stream = run_stream(agent, provider=provider, user_input="go", idempotency_key="key")
        events = [event async for event in stream]
        assert events[-1].type == RunEventType.RUN_ERROR
        assert events[-1].error == message
    else:
        with pytest.raises(ValueError) as caught:
            if entry == "agent":
                await agent.run("go", idempotency_key="key")
            else:
                await run(agent, provider=provider, user_input="go", idempotency_key="key")
        assert str(caught.value) == message
    assert provider.calls == 0


async def test_governance_report_failure_preserves_original_if_release_fails(monkeypatch, caplog):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from dendrux.runtime import submit
    from dendrux.runtime.state import RunRecord
    from dendrux.types import PauseState, RunStatus, UsageStats, generate_ulid

    agent = Agent(prompt="test", provider=ScriptedLLM([]), strict_tenancy=False)
    run_id = generate_ulid()
    pause = PauseState(
        agent_name=agent.name,
        pending_tool_calls=[],
        history=[],
        steps=[],
        iteration=0,
        trace_order_offset=0,
        usage=UsageStats(),
    )
    store = SimpleNamespace(
        get_run=AsyncMock(return_value=RunRecord(run_id, agent.name, "waiting_human_input")),
        get_pause_state=AsyncMock(return_value=pause.to_dict()),
        submit_and_claim=AsyncMock(return_value=1),
        release_run_claim=AsyncMock(side_effect=RuntimeError("secondary_private_sentinel")),
    )
    original = RuntimeError("governance_private_sentinel")
    monkeypatch.setattr(submit, "report_governance_change", AsyncMock(side_effect=original))
    with pytest.raises(RuntimeError) as caught:
        await submit.claim_or_raise(
            store,
            run_id,
            expected_status=RunStatus.WAITING_HUMAN_INPUT,
            submitted_data={"submitted_user_input": "answer"},
            agent=agent,
        )
    assert caught.value is original
    store.release_run_claim.assert_awaited_once_with(
        run_id,
        status="waiting_human_input",
        claim_epoch=1,
    )
    assert any(getattr(r, "code", None) == "persistence_error" for r in caplog.records)
    assert "private_sentinel" not in caplog.text


def test_retry_corpus_projection_changes_only_depth_without_mutating_input():
    from copy import deepcopy

    from tests.conformance.harness import normalised_corpus

    meta = {"dendrux.max_delegation_depth": 7, "keep": "retained"}
    view = {
        "rows": {
            "agent_runs": [
                {"retry_of_run_id": "prior", "meta": dict(meta)},
                {"retry_of_run_id": None, "meta": dict(meta)},
                {"meta": dict(meta)},
            ]
        }
    }
    before = deepcopy(view)
    rows = normalised_corpus(view)["rows"]["agent_runs"]
    assert rows[0]["meta"] == {"keep": "retained"}
    assert rows[1]["meta"] == meta
    assert rows[2]["meta"] == meta
    assert view == before


@pytest.mark.parametrize("missing", ["epoch", "method"])
async def test_unavailable_claim_release_logs_stable_code_with_run_id(missing, caplog):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from dendrux.runtime.submit import release_claim

    release = AsyncMock(return_value=True)
    run_id = "00000000000000000000000105"
    store = SimpleNamespace(**({"release_run_claim": release} if missing == "epoch" else {}))
    await release_claim(
        store, run_id, "waiting_approval", claim_epoch=None if missing == "epoch" else 2
    )
    release.assert_not_awaited()
    assert any(
        getattr(record, "code", None) == "persistence_error"
        and getattr(record, "run_id", None) == run_id
        for record in caplog.records
    )


def test_resume_aborted_view_keeps_only_closed_failure_kinds():
    from dendrux.store import StoredEvent, redact_event
    from dendrux.store._redact import PAUSE_CHANGING_EVENTS

    event = StoredEvent(
        1,
        0,
        "run.resume_aborted",
        None,
        {"failure_kind": "setup_error", "error": "private_abort_sentinel"},
        None,
    )
    assert redact_event(event).data == {"failure_kind": "setup_error"}
    assert "run.resume_aborted" in PAUSE_CHANGING_EVENTS
    from dataclasses import replace

    assert redact_event(replace(event, data={"failure_kind": "private_abort_sentinel"})).data == {}


@pytest.mark.parametrize("failure_kind", ["validation_error", "persistence_error", "setup_error"])
@pytest.mark.parametrize("view", ["raw", "claimed", "kwargs"])
async def test_legacy_claim_release_publishes_only_safe_failure_kind(failure_kind, view, caplog):
    import json
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from dendrux.errors import PersistenceError
    from dendrux.runtime.state import claimed_store
    from dendrux.runtime.submit import release_claim

    failures = {
        "validation_error": ValueError("private_abort_sentinel"),
        "persistence_error": PersistenceError(),
        "setup_error": RuntimeError("private_abort_sentinel"),
    }
    released = []

    async def legacy_release(run_id, *, status, claim_epoch):
        released.append((run_id, status, claim_epoch))
        return True

    async def loose_release(run_id, *, status, claim_epoch, **extras):
        return await legacy_release(run_id, status=status, claim_epoch=claim_epoch)

    run_id = "00000000000000000000000105"
    store = SimpleNamespace(
        release_run_claim=loose_release if view == "kwargs" else legacy_release,
        save_run_event=AsyncMock(),
    )
    target = claimed_store(store, run_id, 7) if view != "raw" else store
    await release_claim(
        target, run_id, "waiting_approval", claim_epoch=7, failure=failures[failure_kind]
    )
    assert released == [(run_id, "waiting_approval", 7)]
    store.save_run_event.assert_awaited_once()
    call = store.save_run_event.call_args
    assert call.args == (run_id,)
    assert call.kwargs["event_type"] == "run.resume_aborted"
    assert call.kwargs["data"] == {"failure_kind": failure_kind}
    assert call.kwargs["claim_epoch"] == 7
    assert "private_abort_sentinel" not in json.dumps(call.kwargs) + caplog.text


async def test_cleanup_failure_cannot_replace_the_callers_cancellation():
    import asyncio

    from dendrux.runtime.runner import _shielded_cleanup

    entered, finish, drained = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def cleanup():
        entered.set()
        await finish.wait()
        drained.set()
        raise asyncio.CancelledError("cleanup cancellation")

    task = asyncio.create_task(_shielded_cleanup(cleanup))
    try:
        await asyncio.wait_for(entered.wait(), 5)
        task.cancel("caller cancellation")
        await asyncio.sleep(0)
        assert not task.done()
        finish.set()
        with pytest.raises(asyncio.CancelledError) as caught:
            await asyncio.wait_for(task, 5)
        assert caught.value.args == ("caller cancellation",)
        assert drained.is_set()
    finally:
        finish.set()
        await asyncio.gather(task, return_exceptions=True)
