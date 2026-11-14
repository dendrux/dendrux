"""Continuation observers close after an unconsumed claim is released."""

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from dendrux.loops.base import BaseNotifier
from dendrux.notifiers.otel import OpenTelemetryNotifier
from dendrux.runtime import _run_observers, runner
from dendrux.runtime.persistence import PersistenceRecorder
from dendrux.types import LLMResponse, RunStatus
from tests._fixtures.storage import pin_storage_format
from tests._helpers.gates import wait_for_gate
from tests.integration.test_060_reclaim_agent import checkpoint
from tests.integration.test_cross_path_invariants import SUBMIT_KINDS, make_agent, submission_case


@pytest.mark.parametrize(
    "kind,storage_format",
    [(kind, fmt) for kind in SUBMIT_KINDS for fmt in (1, 2)] + [("reclaim", 2)],
)
@pytest.mark.parametrize("cancel", [False, True])
async def test_post_trace_release_closes_observers(
    store, engine, monkeypatch, kind, cancel, storage_format
):
    await pin_storage_format(engine, storage_format)
    if kind == "reclaim":
        agent = make_agent(store, [LLMResponse(text="done")])
        item = await checkpoint(agent, store, engine)
        run_id = item.run_id

        async def invoke(notifier):
            return await agent.reclaim(run_id, tenant_id="tenant", item=item, notifier=notifier)
    else:
        _, run_id, invoke = await submission_case(store, kind)
    before = await store.get_run(run_id)
    notifier = BaseNotifier()
    for hook in ("on_run_started", "on_run_finished", "on_run_failed"):
        setattr(notifier, hook, AsyncMock())
        monkeypatch.setattr(PersistenceRecorder, hook, AsyncMock())
    entered, proceed = asyncio.Event(), asyncio.Event()
    start_trace = runner._start_trace
    failure = RuntimeError("post_trace_setup_failure")

    async def traced(*args, **kwargs):
        result = await start_trace(*args, **kwargs)
        entered.set()
        await proceed.wait()
        return result

    monkeypatch.setattr(_run_observers, "_start_trace", traced)
    monkeypatch.setattr(runner, "_snapshot_mcp_meta", Mock(side_effect=failure))
    task = asyncio.create_task(invoke(notifier=notifier))
    try:
        await wait_for_gate(entered, task=task)
        if cancel:
            await make_agent(store, []).cancel_run(run_id, tenant_id="tenant")
        proceed.set()
        with pytest.raises(RuntimeError) as caught:
            await asyncio.wait_for(task, 10)
        assert caught.value is failure
        expected = RunStatus.CANCELLED if cancel else RunStatus(before.status)
        assert (await store.get_run(run_id)).status == expected.value
        for observer in (notifier, PersistenceRecorder):
            observer.on_run_started.assert_awaited_once()
            if kind == "reclaim" and not cancel:
                observer.on_run_failed.assert_awaited_once()
                observer.on_run_finished.assert_not_awaited()
            else:
                observer.on_run_failed.assert_not_awaited()
                observer.on_run_finished.assert_awaited_once()
                args = observer.on_run_finished.await_args.args
                assert args[0] == run_id and args[1].status == expected
    finally:
        proceed.set()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize("cancel", [False, True])
async def test_aborted_trace_ends_and_exports(store, monkeypatch, cancel):
    _, run_id, invoke = await submission_case(store, "input")
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    notifier = OpenTelemetryNotifier(tracer=provider.get_tracer("resume-test"))
    snapshot = runner._snapshot_mcp_meta

    async def fail_after_trace(*args, **kwargs):
        result = await start_trace(*args, **kwargs)
        if cancel:
            await store.request_cancel(run_id)
        return result

    start_trace = runner._start_trace
    monkeypatch.setattr(_run_observers, "_start_trace", fail_after_trace)
    monkeypatch.setattr(runner, "_snapshot_mcp_meta", Mock(side_effect=RuntimeError("setup")))
    try:
        with pytest.raises(RuntimeError, match="setup"):
            await invoke(notifier=notifier)
        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].end_time is not None
        assert spans[0].attributes["dendrux.run.status"] == (
            "cancelled" if cancel else "waiting_human_input"
        )
        assert run_id not in notifier._run_spans
        if not cancel:
            monkeypatch.setattr(runner, "_snapshot_mcp_meta", snapshot)
            assert (await invoke(notifier=notifier)).status == RunStatus.SUCCESS
            assert (
                len([s for s in exporter.get_finished_spans() if s.name.startswith("invoke_agent")])
                == 2
            )
    finally:
        provider.shutdown()


async def test_missing_prior_status_releases_using_record_status(store, monkeypatch):
    agent, run_id, _ = await submission_case(store, "input")
    claim = await store.claim_paused_run(run_id, expected_status="waiting_human_input")
    release = AsyncMock(wraps=store.release_run_claim)
    monkeypatch.setattr(store, "release_run_claim", release)
    with pytest.raises(ValueError):
        await runner.resume_claimed(
            run_id,
            state_store=store,
            agent=agent,
            provider=agent.provider,
            claim_epoch=claim.claim_epoch,
        )
    release.assert_awaited_once()
    assert release.await_args.kwargs["status"] == "running"
    assert release.await_args.kwargs["claim_epoch"] == claim.claim_epoch


@pytest.mark.parametrize("failure_site", ["before_trace", "trace_link", "recorder_finish"])
async def test_partial_start_and_recorder_failure_balance_observers(
    store, monkeypatch, failure_site
):
    _, run_id, invoke = await submission_case(store, "input")
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    notifier = OpenTelemetryNotifier(tracer=provider.get_tracer("partial-start"))
    recorder_started = AsyncMock()
    recorder_finished = AsyncMock()
    monkeypatch.setattr(PersistenceRecorder, "on_run_started", recorder_started)
    monkeypatch.setattr(PersistenceRecorder, "on_run_finished", recorder_finished)
    failure = RuntimeError("setup")
    if failure_site == "before_trace":
        monkeypatch.setattr(_run_observers, "_start_trace", AsyncMock(side_effect=failure))
    elif failure_site == "trace_link":
        monkeypatch.setattr(store, "set_trace_link", AsyncMock(side_effect=failure))
    else:
        monkeypatch.setattr(runner, "_snapshot_mcp_meta", Mock(side_effect=failure))
        recorder_finished.side_effect = RuntimeError("recorder_failure")
    try:
        with pytest.raises(RuntimeError) as caught:
            await invoke(notifier=notifier)
        assert caught.value is failure
        recorder_started.assert_awaited_once()
        recorder_finished.assert_awaited_once()
        assert len(exporter.get_finished_spans()) == (0 if failure_site == "before_trace" else 1)
        assert run_id not in notifier._run_spans
    finally:
        provider.shutdown()
