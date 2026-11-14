"""Task cancellation and stream close keep their distinct persistence contracts."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from dendrux import _codes
from dendrux._meta_keys import CANCEL_PERSISTED_META_KEY
from dendrux.loops import _helpers
from dendrux.runtime import _run_events, runner
from dendrux.runtime.persistence import PersistenceRecorder
from dendrux.types import RunEvent, RunEventType, RunStatus
from tests._helpers.gates import wait_for_gate
from tests.unit.test_runner_failure_save import entry_case, observer
from tests.unit.test_runner_failure_save import store as store  # noqa: F401


@pytest.mark.parametrize("entry", ["run", "retry", "resume"])
@pytest.mark.parametrize("interrupted", [False, True])
@pytest.mark.parametrize("outcome", ["won", "lost", "cancelled", "raises", "event_raises"])
async def test_task_cancel_save(store, monkeypatch, caplog, entry, interrupted, outcome):
    agent, invoke = await entry_case(store, entry)
    notifier = observer()
    original = asyncio.CancelledError("caller_cancel")
    finalize = store.finalize_run
    run_ids = []

    async def cancel_loop(**kwargs):
        run_ids.append(kwargs["run_id"])
        if outcome in {"lost", "cancelled"}:
            await finalize(
                kwargs["run_id"],
                status="success" if outcome == "lost" else "cancelled",
                answer="winner",
                expected_current_status="running",
            )
        raise original

    monkeypatch.setattr(agent._loop, "run", cancel_loop)
    monkeypatch.setattr(_run_events, "_take_task_interrupt", lambda *args: (interrupted, False))
    method = "finalize_with_event"
    if outcome == "event_raises":
        monkeypatch.setattr(store, "finalize_with_event", None)  # Legacy split-write contract.
        method = "finalize_run"
    save = AsyncMock(
        side_effect=RuntimeError("save_failed") if outcome == "raises" else getattr(store, method)
    )
    monkeypatch.setattr(store, method, save)
    emit = runner._emit_event

    async def event(*args, **kwargs):
        if args[2] == "run.cancelled" and outcome == "event_raises":
            raise RuntimeError("event_failed")
        return await emit(*args, **kwargs)

    monkeypatch.setattr(_run_events, "_emit_event", event)
    finished = AsyncMock()
    monkeypatch.setattr(PersistenceRecorder, "on_run_finished", finished)
    with pytest.raises(asyncio.CancelledError) as caught:
        await invoke(notifier)
    assert caught.value is original
    run_id = run_ids[0]
    save.assert_awaited_once()
    assert save.await_args.kwargs["status"] == "cancelled"
    if method == "finalize_run":
        assert save.await_args.kwargs["expected_current_status"] == "running"
    else:
        assert save.await_args.kwargs["allowed_current_statuses"] == ["running"]
    assert "answer" not in save.await_args.kwargs
    finished.assert_awaited_once()
    notifier.on_run_finished.assert_awaited_once()
    notifier.on_run_failed.assert_not_awaited()
    result = notifier.on_run_finished.await_args.args[1]
    assert result.status == (RunStatus.SUCCESS if outcome == "lost" else RunStatus.CANCELLED)
    assert result.meta.get(CANCEL_PERSISTED_META_KEY, False) == (
        outcome in {"won", "cancelled", "event_raises"}
    )
    if outcome in {"lost", "cancelled"}:
        assert result.answer == "winner"
    else:
        assert result.meta.get("interrupted", False) == interrupted
        if interrupted:
            assert result.meta["partial_output"] is False
    events = [e for e in await store.get_run_events(run_id) if e.event_type == "run.cancelled"]
    assert len(events) == int(outcome == "won")
    if events:
        assert events[0].data == (
            {"reason": "cancel_requested", "interrupted": True, "partial_output": False}
            if interrupted
            else {"reason": "task_cancelled"}
        )
    if outcome in {"raises", "event_raises"}:
        assert any(
            getattr(r, "code", None) == _codes.CANCEL_PERSIST_FAILED.name for r in caplog.records
        )


@pytest.mark.parametrize("entry", ["run_stream", "resume_stream"])
@pytest.mark.parametrize(
    "outcome", ["won", "lost", "cancelled", "raises", "event_raises", "read_raises"]
)
@pytest.mark.parametrize("partial", [None, "partial answer"])
async def test_stream_close_save(store, monkeypatch, caplog, entry, outcome, partial):
    agent, invoke = await entry_case(store, entry)
    notifier = observer()

    async def loop_stream(**kwargs):
        yield RunEvent(
            type=RunEventType.TEXT_DELTA if partial is not None else RunEventType.TOOL_USE_START,
            text=partial,
            phase="answer",
        )
        pytest.fail("close must not pull again")

    monkeypatch.setattr(agent._loop, "run_stream", loop_stream)
    finished = AsyncMock()
    monkeypatch.setattr(PersistenceRecorder, "on_run_finished", finished)
    stream = invoke(notifier)
    iterator = aiter(stream)
    async for event in iterator:
        if event.type in {RunEventType.TEXT_DELTA, RunEventType.TOOL_USE_START}:
            break
    run_id = stream.run_id
    if outcome in {"lost", "cancelled", "read_raises"}:
        await store.finalize_run(
            run_id,
            status="cancelled" if outcome == "cancelled" else "success",
            answer="winner",
            expected_current_status="running",
        )
    method = "finalize_with_event"
    if outcome == "event_raises":
        monkeypatch.setattr(store, "finalize_with_event", None)  # Legacy split-write contract.
        method = "finalize_run"
    save = AsyncMock(
        side_effect=RuntimeError("save_failed") if outcome == "raises" else getattr(store, method)
    )
    monkeypatch.setattr(store, method, save)
    emit = runner._emit_event

    async def event_write(*args, **kwargs):
        if args[2] == "run.cancelled" and outcome == "event_raises":
            raise RuntimeError("event_failed")
        return await emit(*args, **kwargs)

    monkeypatch.setattr(_run_events, "_emit_event", event_write)
    read = AsyncMock(
        side_effect=RuntimeError("read_failed") if outcome == "read_raises" else store.get_run
    )
    monkeypatch.setattr(store, "get_run", read)
    await stream.aclose()
    await iterator.aclose()
    save.assert_awaited_once()
    assert save.await_args.kwargs["answer"] == partial
    assert save.await_args.kwargs["status"] == "cancelled"
    if method == "finalize_run":
        assert save.await_args.kwargs["expected_current_status"] == "running"
    else:
        assert save.await_args.kwargs["allowed_current_statuses"] == ["running"]
    finished.assert_awaited_once()
    notifier.on_run_finished.assert_awaited_once()
    result = notifier.on_run_finished.await_args.args[1]
    assert result.status == {"lost": RunStatus.SUCCESS, "raises": RunStatus.RUNNING}.get(
        outcome, RunStatus.CANCELLED
    )
    assert result.meta[CANCEL_PERSISTED_META_KEY] == (
        outcome in {"won", "cancelled", "event_raises"}
    )
    assert read.await_count == int(outcome in {"lost", "cancelled", "raises", "read_raises"})
    events = [e for e in await store.get_run_events(run_id) if e.event_type == "run.cancelled"]
    assert len(events) == int(outcome == "won")
    if events:
        assert events[0].data == {"reason": "stream_closed", "partial_output": partial is not None}
    if outcome == "read_raises":
        assert "Failed to cancel run" not in caplog.text


@pytest.mark.parametrize("won", [False, True])
async def test_resume_close_before_context_does_not_emit_or_finish(store, monkeypatch, won):
    agent, invoke = await entry_case(store, "resume_stream")
    notifier = observer()
    entered = asyncio.Event()

    async def preparing(*args, **kwargs):
        entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(runner, "_prepare_resume", preparing)
    stream = invoke(notifier)
    iterator = aiter(stream)
    pulling = asyncio.create_task(anext(iterator))
    await wait_for_gate(entered, task=pulling)
    if not won:
        await store.finalize_run(stream.run_id, status="success", expected_current_status="running")
    read = AsyncMock(wraps=store.get_run)
    monkeypatch.setattr(store, "get_run", read)
    pulling.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pulling
    await stream.aclose()
    read.assert_not_awaited()
    assert (await store.get_run(stream.run_id)).status == ("cancelled" if won else "success")
    assert not any(
        e.event_type == "run.cancelled" for e in await store.get_run_events(stream.run_id)
    )
    notifier.on_run_finished.assert_not_awaited()


@pytest.mark.parametrize(
    "kind", ["cancel", "interrupt", "budget", "success", "error", "max_iterations"]
)
@pytest.mark.parametrize("won", [False, True])
@pytest.mark.parametrize("storage_format", [1, 2])
async def test_loop_outcome_payload_and_winner(store, kind, won, storage_format):
    from dendrux.runtime.state import ClaimResult, claimed_store
    from dendrux.types import RunResult, UsageStats
    from tests._fixtures.storage import pin_storage_format

    await pin_storage_format(store._engine, storage_format)
    created = await store.create_run("outcome", agent_name="a", input_data={})
    owned = claimed_store(store, "outcome", ClaimResult(0, created.storage_format, None, None))
    status = RunStatus.CANCELLED if kind in {"cancel", "interrupt", "budget"} else RunStatus(kind)
    meta = (
        {"interrupted": True, "partial_output": True}
        if kind == "interrupt"
        else {"budget_exceeded": ["tokens"]}
        if kind == "budget"
        else {}
    )
    proposal = RunResult(
        run_id="outcome",
        status=status,
        answer="answer",
        error="error" if kind == "error" else None,
        usage=UsageStats(input_tokens=2),
        meta=meta,
        stop_reason="max_tokens" if kind == "success" else None,
    )
    if not won:
        await store.finalize_run(
            "outcome", status="success", answer="winner", expected_current_status="running"
        )
    result = await runner._persist_loop_outcome(
        state_store=owned, run_id="outcome", result=proposal
    )
    assert result.meta["_finalize_won"] is won
    assert result.answer == ("answer" if won else "winner")
    assert result.status == (status if won else RunStatus.SUCCESS)
    events = await store.get_run_events("outcome")
    assert len(events) == int(won)
    if not won:
        return
    if kind == "budget":
        expected = {"reason": "budget_exceeded", "exceeded": ["tokens"]}
    elif kind in {"cancel", "interrupt"}:
        expected = {"reason": "cancel_requested", **meta}
    else:
        expected = {
            "status": status.value,
            "stop_reason": proposal.stop_reason,
            "truncated": kind == "success",
        }
        if kind == "error":
            expected["error"] = "error"
    assert events[0].data == expected
    row = await store.get_run("outcome")
    assert row.failure_reason == ("budget_exceeded" if kind == "budget" else None)


@pytest.mark.parametrize("has_store", [False, True])
async def test_task_cancel_without_row(has_store):
    from types import SimpleNamespace

    store = SimpleNamespace(
        finalize_run=AsyncMock(return_value=False),
        get_run=AsyncMock(return_value=None),
        save_run_event=AsyncMock(),
    )
    result = await runner._persist_task_cancellation(store if has_store else None, "missing")
    assert result.status == RunStatus.CANCELLED
    assert result.meta[CANCEL_PERSISTED_META_KEY] is False
    assert store.finalize_run.await_count == int(has_store)
    assert store.get_run.await_count == int(has_store)
    store.save_run_event.assert_not_awaited()


@pytest.mark.parametrize("entry", ["batch", "resume"])
@pytest.mark.parametrize("notifier_fails", [False, True])
async def test_cancel_hook_error_boundaries(monkeypatch, caplog, entry, notifier_fails):
    from types import SimpleNamespace

    from dendrux.types import RunResult

    result = RunResult(run_id="run", status=RunStatus.CANCELLED)
    recorder_error = RuntimeError("recorder_finish_failed")
    notifier_error = RuntimeError("notifier_finish_failed")
    recorded = AsyncMock(side_effect=recorder_error)
    notified = AsyncMock(side_effect=notifier_error if notifier_fails else None)
    monkeypatch.setattr(_helpers, "record_run_finished", recorded)
    monkeypatch.setattr(_helpers, "notify_run_finished", notified)
    monkeypatch.setattr(_run_events, "_drain_task_cancellation", AsyncMock(return_value=result))
    ctx = SimpleNamespace(recorder=object(), notifier=object())
    if entry == "batch":
        action = runner._RunLifecycle("run", None, None)._cancel_finish(
            ctx.recorder, ctx.notifier, result
        )
    else:
        action = runner._cancel_resumed_run(
            SimpleNamespace(_task_manager=None), object(), "run", ctx
        )
    if notifier_fails:
        with pytest.raises(RuntimeError) as caught:
            await action
        assert caught.value is notifier_error
    else:
        await action
    recorded.assert_awaited_once()
    notified.assert_awaited_once()
    assert "Recorder finish failed" in caplog.text


@pytest.mark.parametrize("budget", [False, True])
async def test_loop_cancel_allowed_statuses(monkeypatch, budget):
    from types import SimpleNamespace

    from dendrux._lifecycle import CANCEL_FINALIZE_STATUS_VALUES
    from dendrux.types import RunResult

    transition = AsyncMock(return_value=True)
    monkeypatch.setattr(_run_events, "_transition_with_event", transition)
    monkeypatch.setattr(_run_events, "_attach_undelivered_notes", AsyncMock())
    proposal = RunResult(
        run_id="run",
        status=RunStatus.CANCELLED,
        meta={"budget_exceeded": ["tokens"]} if budget else {},
    )
    await runner._persist_loop_outcome(state_store=SimpleNamespace(), run_id="run", result=proposal)
    assert transition.await_args.kwargs["allowed_current_statuses"] == (
        ["running"] if budget else list(CANCEL_FINALIZE_STATUS_VALUES)
    )
    assert "paused" not in transition.await_args.kwargs["allowed_current_statuses"]


@pytest.mark.parametrize("entry", ["run_stream", "resume_stream"])
async def test_stream_terminal_result_truthiness_is_path_specific(store, monkeypatch, entry):
    from dendrux.types import RunResult

    class FalseResult(RunResult):
        def __bool__(self):
            return False

    agent, invoke = await entry_case(store, entry)
    notifier = observer()

    async def loop_stream(**kwargs):
        yield RunEvent(
            type=RunEventType.RUN_COMPLETED,
            run_result=FalseResult(
                run_id=kwargs["run_id"], status=RunStatus.SUCCESS, answer="answer"
            ),
        )

    monkeypatch.setattr(agent._loop, "run_stream", loop_stream)
    async with invoke(notifier) as stream:
        events = [event async for event in stream]
    assert events[-1].type == RunEventType.RUN_COMPLETED
    assert (await store.get_run(stream.run_id)).status == (
        "success" if entry == "run_stream" else "running"
    )
    assert notifier.on_run_finished.await_count == int(entry == "run_stream")
