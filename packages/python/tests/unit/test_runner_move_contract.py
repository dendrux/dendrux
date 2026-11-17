"""Characterize the runner support seams before their owner modules move."""

import inspect
import logging
from dataclasses import fields
from types import SimpleNamespace
from typing import get_args
from unittest.mock import AsyncMock, Mock

import pytest
from sqlalchemy.ext.asyncio import create_async_engine

from dendrux import Agent
from dendrux._sentinel import _OMITTED
from dendrux.db._sqlite_sync import create_sqlite_schema
from dendrux.errors import ProviderAuthError, StoreOutcomeUnknown
from dendrux.llm.mock import MockLLM
from dendrux.loops.base import BaseNotifier
from dendrux.loops.react import ReActLoop
from dendrux.runtime import _run_images, _run_observers, _run_pending, _run_resume_prep, runner
from dendrux.runtime import context as context_mod
from dendrux.runtime.context import DelegationContext
from dendrux.runtime.persistence import PersistenceRecorder
from dendrux.store import SQLiteStateStore
from dendrux.types import LLMResponse, RunEventType, RunResult, RunStatus, StopReason, ToolCall


@pytest.fixture
async def store():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    await create_sqlite_schema(engine)
    try:
        yield SQLiteStateStore(engine)
    finally:
        await engine.dispose()


async def invoke(entry, store, *, key=None, notifier=None):
    options = dict(
        agent=Agent(name="move", prompt="prompt"),
        provider=MockLLM([LLMResponse(text="done")]),
        state_store=store,
        tenant_id=None,
        user_input="hello",
        idempotency_key=key,
        extra_notifier=notifier,
    )
    if entry == "run":
        return await runner.run(**options)
    async with runner.run_stream(**options) as stream:
        return [event async for event in stream]


async def test_cached_owner_fault_fanout_nested_restore_and_second_invocation(store, monkeypatch):
    original = await invoke("run", store, key="same")
    before = await store.get_run_events(original.run_id)
    cached = runner._build_cached_result
    failure = RuntimeError("cached_owner_fault")
    outer = AsyncMock(side_effect=failure)
    with monkeypatch.context() as patcher:
        patcher.setattr(_run_pending, "_build_cached_result", outer)
        with pytest.raises(RuntimeError) as caught:
            await invoke("run", store, key="same")
        assert caught.value is failure
        with monkeypatch.context() as inner:
            inner.setattr(_run_pending, "_build_cached_result", cached)
            assert (await invoke("run", store, key="same")).run_id == original.run_id
        events = await invoke("stream", store, key="same")
        assert events[-1].type == RunEventType.RUN_ERROR
        assert events[-1].error == "cached_owner_fault"
        assert outer.await_count == 2
    assert (await invoke("run", store, key="same")).run_id == original.run_id
    assert (await store.get_run(original.run_id)).status == "success"
    assert await store.get_run_events(original.run_id) == before


@pytest.mark.parametrize("entry", ["run", "stream"])
async def test_trace_fault_fires_and_finishes_owned_row(store, monkeypatch, entry):
    fault = RuntimeError("trace_owner_fault")
    traced = AsyncMock(side_effect=fault)
    notifier = BaseNotifier()
    notifier.on_run_failed = AsyncMock()
    finalize = AsyncMock(wraps=store.finalize_with_event)
    monkeypatch.setattr(store, "finalize_with_event", finalize)
    monkeypatch.setattr(_run_observers, "_start_trace", traced)
    if entry == "run":
        with pytest.raises(RuntimeError) as caught:
            await invoke(entry, store, notifier=notifier)
        assert caught.value is fault
    else:
        events = await invoke(entry, store, notifier=notifier)
        assert events[-1].error == "trace_owner_fault"
    traced.assert_awaited_once()
    finalize.assert_awaited_once()
    run_id = traced.await_args.args[2]
    assert (await store.get_run(run_id)).status == "error"
    notifier.on_run_failed.assert_awaited_once()


async def test_patch_between_stream_construction_and_iteration(monkeypatch):
    manager = runner.run_stream(
        agent=Agent(name="move", prompt="prompt"),
        provider=MockLLM([LLMResponse(text="done")]),
        user_input="hello",
        tenant_id=None,
    )
    traced = AsyncMock(side_effect=RuntimeError("late_trace_fault"))
    monkeypatch.setattr(_run_observers, "_start_trace", traced)
    async with manager as stream:
        events = [event async for event in stream]
    traced.assert_awaited_once()
    assert events[-1].error == "late_trace_fault"


@pytest.mark.parametrize("results", [False, True])
async def test_claim_refusal_owner_fault_fires_before_image_work(monkeypatch, results):
    fault = RuntimeError("claim_owner_fault")
    refusing = AsyncMock(side_effect=fault)
    monkeypatch.setattr(_run_pending, "_raise_resume_claim_failure", refusing)
    image = Mock(side_effect=AssertionError("image work ran before refusal"))
    monkeypatch.setattr(_run_images, "_image_runtime", image)
    store = SimpleNamespace(get_run=AsyncMock(return_value=SimpleNamespace(status="success")))
    with pytest.raises(RuntimeError) as caught:
        await runner._ingest_resume_images(
            None,
            None,
            store,
            "run",
            loop=ReActLoop(),
            tool_results=[] if results else None,
            input_items=() if results else ("hello",),
        )
    assert caught.value is fault
    refusing.assert_awaited_once()
    image.assert_not_called()
    monkeypatch.setattr(
        _run_resume_prep, "_claim_submission", AsyncMock(return_value=(0, None, None))
    )
    with pytest.raises(RuntimeError) as caught:
        await runner._claim_resume(
            store,
            "run",
            {},
            Agent(name="move", prompt="prompt", strict_tenancy=False),
            "waiting_human_input",
            expected_pause_id=None,
            results=None,
            user_input="hello",
        )
    assert caught.value is fault
    assert refusing.await_count == 2


async def test_real_context_seam_reaches_entry_and_tool_observer(monkeypatch):
    fault = RuntimeError("context_owner_fault")
    context = Mock(side_effect=fault)
    monkeypatch.setattr(context_mod, "get_delegation_context", context)
    with pytest.raises(RuntimeError) as caught:
        await invoke("run", None)
    assert caught.value is fault
    notifier = BaseNotifier()
    with pytest.raises(RuntimeError) as caught:
        await runner._TraceNotifier(notifier).on_tool_started(
            "run", ToolCall(id="call", name="tool", params={}), 0
        )
    assert caught.value is fault
    assert context.call_count == 2


def test_sentinels_depth_record_defaults_and_metadata():
    assert runner._OMITTED is _OMITTED
    assert frozenset(get_args(StopReason)) == runner._STOP_REASONS
    assert (
        inspect.signature(runner.run).parameters["max_delegation_depth"].default
        is runner._UNSET_DEPTH
    )
    assert runner._resolve_max_delegation_depth(runner._UNSET_DEPTH, None) == 10
    assert runner._resolve_max_delegation_depth(None, None) is None
    parent = DelegationContext("parent", max_delegation_depth=3)
    assert runner._resolve_max_delegation_depth(runner._UNSET_DEPTH, parent) == 3
    assert runner._resolve_max_delegation_depth(0, parent) == 0
    assert [field.name for field in fields(runner._PauseReads)] == [
        "record",
        "checkpoint",
        "history_records",
    ]
    assert runner._PauseReads(None) == runner._PauseReads(None, None, None)
    assert runner._ApprovalResume([], [], {}, None).remaining == ()
    assert runner._ApprovalResume([], [], {}, None).handoff is None
    for cls in (
        runner._TraceNotifier,
        runner._ResumeContext,
        runner._PauseReads,
        runner._ApprovalResume,
        runner._ResumeProgress,
    ):
        assert cls.__module__ == "dendrux.runtime.runner"
    history = []
    context = runner._ResumeContext(
        history=history,
        recorder=Mock(),
        notifier=None,
        pause_state=Mock(),
        resolved_loop=ReActLoop(),
        resolved_strategy=Mock(),
    )
    other = runner._ResumeContext(
        history=[],
        recorder=Mock(),
        notifier=None,
        pause_state=Mock(),
        resolved_loop=ReActLoop(),
        resolved_strategy=Mock(),
    )
    assert context.history is history
    context.notifier_warnings.append("warning")
    assert other.notifier_warnings == []
    progress = runner._ResumeProgress()
    assert progress._can_release(None, RuntimeError())
    assert not progress._can_release(None, StoreOutcomeUnknown())
    progress._start()
    assert not progress._can_release(context, RuntimeError())


async def test_recovery_best_effort_and_terminal_winner(caplog):
    fault = RuntimeError("recovery_read_fault")
    store = SimpleNamespace(get_replayed_tool_calls=AsyncMock(side_effect=fault))
    record = SimpleNamespace(
        id="01J00000000000000000000000", tenant_id=None, last_reclaimed_at=1, reclaim_attempts=2
    )
    assert await runner._recovery_info(store, record) is None
    assert "recovery_read_fault" not in caplog.text
    with pytest.raises(RuntimeError) as caught:
        await runner._recovery_info(store, record, best_effort=False)
    assert caught.value is fault
    result = RunResult(run_id="run", status=RunStatus.PAUSED)
    store.get_run = AsyncMock(return_value=SimpleNamespace(status="success"))
    winner = RunResult(run_id="run", status=RunStatus.SUCCESS, answer="winner")
    persisted = AsyncMock(return_value=winner)
    assert await runner._committed_pause_result(store, result, persisted) is winner
    persisted.assert_awaited_once_with(RunStatus.SUCCESS)


async def test_cached_hit_has_no_workflow_or_sinks(store, monkeypatch):
    original = await invoke("run", store, key="same")
    recorder = Mock(side_effect=AssertionError("cached recorder"))
    monkeypatch.setattr(PersistenceRecorder, "__init__", recorder)
    notifier = BaseNotifier()
    notifier.on_run_started = AsyncMock()
    result = await invoke("run", store, key="same", notifier=notifier)
    assert (result.answer, result.steps, result.iteration_count) == (
        "done",
        [],
        original.iteration_count,
    )
    recorder.assert_not_called()
    notifier.on_run_started.assert_not_awaited()


def test_safe_error_meta_logger_and_cause(caplog):
    sentinel = "private_vendor_pii_sentinel"
    cause = ValueError(sentinel)
    error = ProviderAuthError(provider="mock", message=sentinel)
    error.__cause__ = cause
    assert isinstance(error, RuntimeError)
    assert runner._cause_type(error) == "ValueError"
    payload = runner._error_payload(error)
    assert sentinel not in repr(payload)
    assert payload["error"] == error.safe_message
    meta = {
        "pause_state": sentinel,
        "pii_mapping": {"token": "masked", "__dendrux_origins__": sentinel},
    }
    assert runner._public_meta(meta) == {"pii_mapping": {"token": "masked"}}
    assert meta["pause_state"] == sentinel
    attached = {"nested": []}
    runner._attach_run_error_meta(error, "run", attached)
    attached["nested"].append(sentinel)
    assert error.run_meta == {"nested": []}
    assert error.__cause__ is cause
    assert runner.logger is logging.getLogger("dendrux.runtime.runner")
    assert sentinel not in caplog.text


@pytest.mark.parametrize("managed", [False, True])
def test_pending_read_status_keeps_a_coherent_later_view(managed):
    view = SimpleNamespace(managed=managed, status="success")
    status, moved = runner._read_status("paused", "paused", view)
    assert status == "success"
    assert moved is (not managed)


async def test_stale_pause_rereads_terminal_answer():
    result = RunResult(run_id="run", status=RunStatus.WAITING_HUMAN_INPUT)
    store = SimpleNamespace(
        get_run=AsyncMock(
            return_value=SimpleNamespace(
                status="success",
                answer="terminal winner",
                error=None,
                meta={},
            )
        )
    )
    await runner._reread_moved(store, result)
    assert (result.status, result.answer, result.pending, result.pause_id) == (
        RunStatus.SUCCESS,
        "terminal winner",
        None,
        None,
    )
    store.get_run.assert_awaited_once_with("run")


def test_cached_output_schema_failure_preserves_raw_answer_without_leaking(caplog):
    from pydantic import BaseModel

    class Answer(BaseModel):
        answer: str

    assert runner._cached_output('{"answer":"safe"}', Answer, "run").answer == "safe"
    assert runner._cached_output("private_invalid_cached_answer", Answer, "run") is None
    assert "private_invalid_cached_answer" not in caplog.text
    assert "failed validation" in caplog.text


async def test_missing_cached_record_refuses():
    store = SimpleNamespace(get_run=AsyncMock(return_value=None))
    with pytest.raises(RuntimeError, match="Idempotent run run not found in database"):
        await runner._build_cached_result(store, "run")
    store.get_run.assert_awaited_once_with("run")


def test_completed_memory_result_has_no_pending_state():
    result = RunResult(run_id="run", status=RunStatus.SUCCESS)
    assert runner._in_memory_pending(result) is None


async def test_unreadable_observer_marker_preserves_parent_trace():
    class Observer(BaseNotifier):
        @property
        def _dendrux_otel(self):
            raise RuntimeError("private_marker_fault")

    observer = Observer()
    observer.on_run_started = AsyncMock()
    started = Mock()
    trace = "00-" + "1" * 32 + "-" + "2" * 16 + "-01"
    link, notifier = await runner._start_trace(
        observer,
        None,
        "run",
        agent_name=None,
        agent_model=None,
        parent=DelegationContext("parent", trace_link=trace),
        on_started=started,
    )
    assert link is None and notifier.inner is observer
    observer.on_run_started.assert_awaited_once()
    assert observer.on_run_started.await_args.kwargs["trace_link"] == trace
    started.assert_called_once_with()
