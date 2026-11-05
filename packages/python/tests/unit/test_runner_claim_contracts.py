"""Claim boundaries pinned before sharing request and failure decisions."""

import inspect
import json
from collections import Counter
from dataclasses import replace
from functools import wraps
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import event, update

from dendrux.db.models import AgentRun
from dendrux.errors import RunAlreadyTerminalError, StoreOutcomeUnknown
from dendrux.runtime import runner, submit
from dendrux.types import ApprovalDecision, LLMResponse, RunStatus, TextPart
from tests._fixtures.storage import pin_storage_format
from tests._helpers.runtime_paths import SUBMIT_KINDS, path_case
from tests._helpers.scripted_provider import ScriptedLLM
from tests.unit.test_runner_failure_save import store as store  # noqa: F401

PATHS = ["resume", "resume_stream", *SUBMIT_KINDS]


@pytest.mark.parametrize(
    "status,reason,text",
    [
        (
            "error",
            "abandoned_waiting",
            "was abandoned (swept as expired waiting run). It cannot be resumed.",
        ),
        ("error", "stale_running", "was swept as stale (stale_running). It cannot be resumed."),
        ("error", "never_started", "was swept as stale (never_started). It cannot be resumed."),
        *[
            (
                status,
                reason,
                "is not in status 'waiting_human_input' — cannot resume. "
                "It may have been claimed by another caller.",
            )
            for status, reason in [
                (None, None),
                ("error", None),
                ("error", "other"),
                ("success", "stale_running"),
                ("cancelled", None),
                ("running", None),
                ("paused", None),
            ]
        ],
    ],
)
async def test_resume_claim_failure_exact_text(status, reason, text):
    store = SimpleNamespace(
        get_run=AsyncMock(
            return_value=None
            if status is None
            else SimpleNamespace(status=status, failure_reason=reason)
        )
    )
    with pytest.raises(ValueError) as caught:
        await runner._raise_resume_claim_failure(store, "run", "waiting_human_input")
    assert type(caught.value) is ValueError
    assert str(caught.value) == f"Run 'run' {text}"
    store.get_run.assert_awaited_once_with("run")


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("storage_format", [1, 2])
async def test_cancel_requested_claim_policy(store, path, storage_format):
    await pin_storage_format(store._engine, storage_format)
    case = await path_case(store, store._engine, path, ScriptedLLM([LLMResponse(text="done")]))
    before = await store.get_run(case.run_id)
    async with store._engine.begin() as conn:
        await conn.execute(
            update(AgentRun).where(AgentRun.id == case.run_id).values(cancel_requested=True)
        )
    try:
        if path in SUBMIT_KINDS:
            with pytest.raises(RunAlreadyTerminalError) as caught:
                await case.invoke()
            assert caught.value.current_status == RunStatus.CANCELLED
            assert (await store.get_run(case.run_id)).claim_epoch == before.claim_epoch
        else:
            result = await case.invoke()
            assert result.status == RunStatus.CANCELLED
            assert (await store.get_run(case.run_id)).claim_epoch == before.claim_epoch + 1
    finally:
        await case.agent.close()


@pytest.mark.parametrize("path", ["resume", "resume_stream", "tool_results"])
@pytest.mark.parametrize("storage_format", [1, 2])
async def test_unknown_claim_outcome_never_releases_or_finalizes(
    store, monkeypatch, path, storage_format
):
    await pin_storage_format(store._engine, storage_format)
    case = await path_case(store, store._engine, path, ScriptedLLM([LLMResponse(text="done")]))
    original = StoreOutcomeUnknown()
    method = (
        "claim_paused_run" if storage_format == 1 and path != "tool_results" else "submit_and_claim"
    )
    monkeypatch.setattr(store, method, AsyncMock(side_effect=original))
    release, finalize = AsyncMock(), AsyncMock()
    monkeypatch.setattr(store, "release_run_claim", release)
    monkeypatch.setattr(store, "finalize_run", finalize)
    monkeypatch.setattr(store, "finalize_with_event", finalize)
    try:
        if path == "resume_stream":
            await case.invoke()
            assert case.events[-1].error is not None
        else:
            with pytest.raises(StoreOutcomeUnknown) as caught:
                await case.invoke()
            assert caught.value is original
        release.assert_not_awaited()
        finalize.assert_not_awaited()
    finally:
        await case.agent.close()


@pytest.mark.parametrize("path", ["resume", "resume_stream", "tool_results"])
@pytest.mark.parametrize("fails", [False, True])
async def test_governance_claim_target_and_release(store, monkeypatch, path, fails):
    case = await path_case(store, store._engine, path, ScriptedLLM([LLMResponse(text="done")]))
    before = await store.get_run(case.run_id)
    original = RuntimeError("governance report failed")
    seen = []

    async def report(agent, target, run_id, raw):
        seen.append(target)
        assert (target is store) == (path == "tool_results")
        if path != "tool_results":
            assert target.claim_epoch == before.claim_epoch + 1
        if fails:
            raise original

    monkeypatch.setattr(
        submit if path == "tool_results" else runner, "report_governance_change", report
    )
    try:
        if fails and path != "resume_stream":
            with pytest.raises(RuntimeError) as caught:
                await case.invoke()
            assert caught.value is original
        else:
            await case.invoke()
        assert len(seen) == 1
        if fails:
            assert (await store.get_run(case.run_id)).status == before.status
            assert "run.resume_aborted" in [
                e.event_type for e in await store.get_run_events(case.run_id)
            ]
    finally:
        await case.agent.close()


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("storage_format", [1, 2])
async def test_claim_transaction_bytes(store, monkeypatch, path, storage_format):
    await pin_storage_format(store._engine, storage_format)
    case = await path_case(store, store._engine, path, ScriptedLLM([LLMResponse(text="done")]))
    active = False
    statements, counts = [], Counter()

    def instrument(name, method):
        @wraps(method)
        async def wrapped(*args, **kwargs):
            if active:
                counts[name] += 1
            return await method(*args, **kwargs)

        return wrapped

    for name in dir(store):
        method = getattr(store, name)
        if not name.startswith("_") and inspect.iscoroutinefunction(method):
            monkeypatch.setattr(store, name, instrument(name, method))
    module, name = (
        (runner, "_claim_resume") if path.startswith("resume") else (submit, "claim_or_raise")
    )
    claim = getattr(module, name)

    @wraps(claim)
    async def capture(*args, **kwargs):
        nonlocal active
        active = True
        try:
            return await claim(*args, **kwargs)
        finally:
            active = False

    monkeypatch.setattr(module, name, capture)

    def boundary(name):
        def listener(conn):
            if active:
                statements.append(name)

        return listener

    def execute(conn, cursor, statement, parameters, context, executemany):
        if active:
            statements.append(statement)

    engine = store._engine.sync_engine
    listeners = [(name, boundary(name.upper())) for name in ("begin", "commit", "rollback")]
    listeners.append(("before_cursor_execute", execute))
    for name, listener in listeners:
        event.listen(engine, name, listener)
    try:
        await case.invoke()
    finally:
        for name, listener in listeners:
            event.remove(engine, name, listener)
        await case.agent.close()
    assert counts
    wire = (
        json.dumps({"calls": dict(sorted(counts.items())), "transactions": statements}, indent=2)
        + "\n"
    )
    expected = Path(__file__).with_name("fixtures") / f"claim_{path}_{storage_format}.json"
    assert wire == expected.read_text()


@pytest.mark.parametrize("path", ["resume", "submit"])
@pytest.mark.parametrize("kind", ["approve", "decisions", "reject", "input_parts"])
async def test_claim_envelope_approval_and_parts(store, path, kind):
    case = await path_case(
        store, store._engine, "input" if kind == "input_parts" else "approve", ScriptedLLM([])
    )
    raw = await store.get_pause_state(case.run_id)
    checkpoint = await store.get_loop_state(case.run_id)
    status = RunStatus.WAITING_HUMAN_INPUT if kind == "input_parts" else RunStatus.WAITING_APPROVAL
    decisions = (
        {c["id"]: ApprovalDecision.approve() for c in raw["pending_tool_calls"]}
        if kind == "decisions"
        else None
    )
    results = (
        submit.build_rejection_results(runner.PauseState.from_dict(raw), None)
        if kind == "reject"
        else None
    )
    try:
        if path == "resume":
            _, claimed_status, projected = await runner._claim_resume(
                store,
                case.run_id,
                raw,
                case.agent,
                status.value,
                expected_pause_id=raw["pause_id"],
                results=results,
                decisions=list(decisions.items()) if decisions is not None else None,
                user_input="answer" if kind == "input_parts" else None,
                user_parts=(TextPart("answer"),) if kind == "input_parts" else None,
                checkpoint=checkpoint,
            )
            assert claimed_status == status.value
        else:
            payload = (
                {
                    "submitted_decisions": submit.serialize_decisions(
                        runner.PauseState.from_dict(raw), decisions
                    )
                }
                if decisions is not None
                else {"submitted_tool_results": submit.serialize_tool_results(results)}
                if results is not None
                else {
                    "submitted_user_input": "answer",
                    "submitted_user_parts": [{"type": "text", "text": "answer"}],
                }
                if kind == "input_parts"
                else None
            )
            await submit.claim_or_raise(
                store,
                case.run_id,
                expected_status=status,
                submitted_data=payload,
                agent=case.agent,
                expected_pause_id=raw["pause_id"],
                approval=kind == "approve",
                call_results=results,
            )
        durable = await store.get_pause_state(case.run_id)
        envelope = durable["submission"]
        if path == "resume":
            assert projected["submission"] == envelope
        assert envelope["kind"] == ("input" if kind == "input_parts" else "approval")
        payload = envelope["payload"]
        if kind == "approve":
            assert payload == {"approved": True}
        elif kind == "reject" or (kind == "decisions" and path == "submit"):
            assert payload["approved"] is False
        else:
            assert "approved" not in payload
        if kind == "input_parts":
            assert payload["submitted_user_parts"] == [{"type": "text", "text": "answer"}]
    finally:
        await case.agent.close()


@pytest.mark.parametrize(
    "results,expected,statuses",
    [
        ([], "waiting_client_tool", ["waiting_client_tool", "waiting_approval"]),
        (None, "waiting_approval", ["waiting_approval", "paused"]),
        ([], "waiting_approval", ["waiting_approval"]),
        (None, "waiting_human_input", ["waiting_human_input"]),
    ],
)
async def test_resume_status_fallbacks_and_release_status(monkeypatch, results, expected, statuses):
    raw = {"schema_version": 1, "pending_tool_calls": []}
    calls = []
    original = RuntimeError("report failed")
    store = SimpleNamespace(get_run=AsyncMock(return_value=None))
    view = object()

    async def claim(target, run_id, *, expected_status, validate):
        calls.append(expected_status)
        return 3 if len(calls) == len(statuses) else False

    monkeypatch.setattr(runner, "submission_validator", lambda *a, **kw: lambda raw: None)
    monkeypatch.setattr(runner, "require_pause_id", lambda *a, **kw: None)
    monkeypatch.setattr(runner, "_canonicalize_legacy_denials", lambda *a: None)
    # The legacy store seam is exercised without depending on helper placement.

    async def method(run_id, *, expected_status, validate):
        return await claim(store, run_id, expected_status=expected_status, validate=validate)

    store.claim_paused_run = method
    monkeypatch.setattr(runner, "claimed_store", lambda *a: view)
    monkeypatch.setattr(runner, "report_governance_change", AsyncMock(side_effect=original))
    release = AsyncMock()
    monkeypatch.setattr(runner, "_release_claim", release)
    with pytest.raises(RuntimeError) as caught:
        await runner._claim_resume(
            store, "run", raw, object(), expected, expected_pause_id=None, results=results
        )
    assert caught.value is original
    assert calls == statuses
    release.assert_awaited_once_with(view, "run", statuses[-1])


@pytest.mark.parametrize("path", ["resume", "submit"])
async def test_token_validation_order_is_preserved(monkeypatch, path):
    order = []
    raw = {"schema_version": 1, "pause_id": "observed"}
    store = SimpleNamespace(get_pause_state=AsyncMock(return_value=raw))

    async def get_run(run_id):
        order.append("read")
        return SimpleNamespace(status="waiting_approval", cancel_requested=False)

    store.get_run = get_run

    def validator(*a, **kw):
        order.append("validate")
        return lambda raw: None

    def token(*a, **kw):
        order.append("token")
        raise ValueError("token required")

    module = runner if path == "resume" else submit
    monkeypatch.setattr(module, "submission_validator", validator)
    monkeypatch.setattr(module, "require_pause_id", token)
    with pytest.raises(ValueError, match="token required"):
        if path == "resume":
            await runner._claim_resume(
                store,
                "run",
                raw,
                object(),
                "waiting_approval",
                expected_pause_id=None,
                results=None,
            )
        else:
            await submit.claim_or_raise(
                store,
                "run",
                expected_status=RunStatus.WAITING_APPROVAL,
                agent=object(),
                approval=True,
            )
    assert order == (["validate", "read", "token"] if path == "resume" else ["read", "token"])


@pytest.mark.parametrize("path", ["resume", "resume_stream", "tool_results"])
async def test_governance_changed_event_epoch(store, monkeypatch, path):
    case = await path_case(store, store._engine, path, ScriptedLLM([LLMResponse(text="done")]))
    case.agent._on_governance_change = "warn"
    case.agent._limits = replace(case.agent._limits, max_tool_result_bytes=2_000_000)
    save = store.save_run_event
    epochs = []

    @wraps(save)
    async def tracked(run_id, **kwargs):
        if kwargs.get("event_type") == "governance.changed":
            epochs.append(kwargs.get("claim_epoch"))
        return await save(run_id, **kwargs)

    monkeypatch.setattr(store, "save_run_event", tracked)
    try:
        await case.invoke()
        assert len(epochs) == 1
        assert (epochs[0] is None) == (path == "tool_results")
    finally:
        await case.agent.close()


@pytest.mark.parametrize(
    "status,cancel,error",
    [
        (None, False, "RunNotFoundError"),
        ("running", False, "RunNotPausedError"),
        ("pending", False, "RunNotPausedError"),
        ("error", False, "RunAlreadyTerminalError"),
        ("waiting_approval", True, "PauseStatusMismatchError"),
        ("paused", True, "RunAlreadyTerminalError"),
        ("paused", False, None),
    ],
)
async def test_preflight_status_precedence(status, cancel, error):
    from dendrux import errors

    record = None if status is None else SimpleNamespace(status=status, cancel_requested=cancel)
    store = SimpleNamespace(get_run=AsyncMock(return_value=record))
    if error is None:
        await submit.preflight_status(store, "run", RunStatus.PAUSED)
    else:
        with pytest.raises(getattr(errors, error)) as caught:
            await submit.preflight_status(store, "run", RunStatus.PAUSED)
        assert type(caught.value).__name__ == error


@pytest.mark.parametrize(
    "decisions,results,user_input,key",
    [
        (True, True, "answer", "submitted_decisions"),
        (False, True, "answer", "submitted_tool_results"),
        (False, False, "answer", "submitted_user_input"),
        (False, False, None, "approved"),
    ],
)
async def test_resume_payload_priority(store, monkeypatch, decisions, results, user_input, key):
    case = await path_case(store, store._engine, "approve", ScriptedLLM([]))
    raw = await store.get_pause_state(case.run_id)
    checkpoint = await store.get_loop_state(case.run_id)
    monkeypatch.setattr(runner, "submission_validator", lambda *a, **kw: lambda raw: None)
    monkeypatch.setattr(runner, "_canonicalize_legacy_denials", lambda *a: None)
    payloads = []

    async def claim(run_id, **kwargs):
        payloads.append(kwargs["submitted_data"])
        return 3

    monkeypatch.setattr(store, "submit_and_claim", claim)
    monkeypatch.setattr(runner, "claimed_store", lambda *a: store)
    monkeypatch.setattr(runner, "report_governance_change", AsyncMock())
    try:
        await runner._claim_resume(
            store,
            case.run_id,
            raw,
            case.agent,
            "waiting_approval",
            expected_pause_id=raw["pause_id"],
            results=[] if results else None,
            user_input=user_input,
            decisions=[(raw["pending_tool_calls"][0]["id"], ApprovalDecision.approve())]
            if decisions
            else None,
            checkpoint=checkpoint,
        )
        assert set(payloads[0]) - {"approved"} == ({key} if key != "approved" else set())
        if results:
            assert payloads[0]["approved"] is False
    finally:
        await case.agent.close()


@pytest.mark.parametrize("path", ["resume", "submit"])
async def test_descriptor_capability_gate_is_caller_specific(monkeypatch, path):
    from dendrux.runtime.state import StoreCapabilities

    raw = {"schema_version": 2, "pause_id": "observed", "pending_tool_calls": []}
    store = SimpleNamespace(
        capabilities=StoreCapabilities(),
        get_run=AsyncMock(
            return_value=SimpleNamespace(status="waiting_client_tool", cancel_requested=False)
        ),
        get_pause_state=AsyncMock(return_value=raw),
        get_loop_state=AsyncMock(return_value={}),
        submit_and_claim=AsyncMock(return_value=3),
    )
    if path == "resume":
        monkeypatch.setattr(runner, "submission_validator", lambda *a, **kw: lambda raw: None)
        monkeypatch.setattr(runner, "require_pause_id", lambda *a, **kw: None)
        monkeypatch.setattr(runner, "claimed_store", lambda *a: store)
        monkeypatch.setattr(runner, "report_governance_change", AsyncMock())
        await runner._claim_resume(
            store, "run", raw, object(), "waiting_client_tool", expected_pause_id=None, results=[]
        )
    else:
        await submit.claim_or_raise(
            store,
            "run",
            expected_status=RunStatus.WAITING_CLIENT_TOOL,
            submitted_data={"submitted_tool_results": []},
        )
    kwargs = store.submit_and_claim.call_args.kwargs
    assert ("submission" in kwargs) == (path == "resume")
    assert store.get_loop_state.await_count == (1 if path == "resume" else 0)


async def test_submit_blob_edges_and_extra_validator_stay_inside_claim():
    raw = {"schema_version": 1, "pause_id": "observed"}
    validations = []
    edge = object()

    async def claim(run_id, **kwargs):
        assert validations == [raw]
        assert kwargs["blob_refs"] == [edge]
        assert kwargs["blob_refs_scope"] == "tenant"
        kwargs["validate"](raw)
        return 3

    store = SimpleNamespace(
        get_run=AsyncMock(
            return_value=SimpleNamespace(status="waiting_human_input", cancel_requested=False)
        ),
        get_pause_state=AsyncMock(return_value=raw),
        submit_and_claim=claim,
    )
    await submit.claim_or_raise(
        store,
        "run",
        expected_status=RunStatus.WAITING_HUMAN_INPUT,
        submitted_data={"submitted_user_input": "answer"},
        blob_refs=(edge,),
        blob_refs_scope="tenant",
        validate=validations.append,
    )
    assert validations == [raw, raw]


def test_pause_id_error_keeps_blob_status_fields():
    with pytest.raises(submit.PauseStatusMismatchError) as caught:
        submit.pause_validator("00000000000000000000000001", "old")({"pause_id": "new"})
    assert caught.value.current_status == RunStatus.PAUSED
    assert caught.value.expected_status == RunStatus.PAUSED
    assert caught.value.reason == "pause_id"


@pytest.mark.parametrize("token", [None, "explicit"])
@pytest.mark.parametrize("fallbacks", [(), ("waiting_approval",)])
def test_shared_request_keeps_status_order_and_generation(token, fallbacks):
    raw = {"pause_id": "observed"}
    request = submit._claim_request(raw, "waiting_client_tool", token, fallbacks=fallbacks)
    assert request.raw is raw
    assert request.statuses == ("waiting_client_tool", *fallbacks)
    assert request.generation == ("observed" if token is None else "explicit")


def test_shared_claim_projection_is_closed_and_does_not_mutate_raw():
    raw = {"pause_id": "observed"}
    payload = {"submitted_user_input": "answer"}
    descriptor = submit.submission_descriptor(
        raw, {"trace_order_offset": 7}, payload, "waiting_human_input"
    )
    projected = submit._claimed_submission_view(raw, descriptor, payload)
    assert set(projected["submission"]) == {
        "submission_id",
        "kind",
        "pause_id",
        "call_ids",
        "payload",
    }
    assert projected["submission"]["payload"] is payload
    assert raw == {"pause_id": "observed"}
    assert submit._claimed_submission_view(raw, None, None) is raw
