"""Failure persistence keeps each entry's ownership and observer contract."""

from dataclasses import replace
from unittest.mock import AsyncMock

import pytest
from sqlalchemy.ext.asyncio import create_async_engine

from dendrux import Agent, tool
from dendrux._meta_keys import ERROR_INFO_META_KEY
from dendrux.db._sqlite_sync import create_sqlite_schema
from dendrux.errors import ProviderTimeoutError, RunFencedError
from dendrux.llm.mock import MockLLM
from dendrux.loops.base import BaseNotifier
from dendrux.loops.react import ReActLoop
from dendrux.runtime import runner
from dendrux.runtime.persistence import PersistenceRecorder
from dendrux.store import SQLiteStateStore
from dendrux.types import LLMResponse, RunEventType, RunStatus, ToolCall, ToolResult


@pytest.fixture
async def store():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    await create_sqlite_schema(engine)
    try:
        yield SQLiteStateStore(engine)
    finally:
        await engine.dispose()


@tool(target="client")
async def client_read() -> str:
    """Read from the client."""
    raise AssertionError("client tool must not execute here")


def observer():
    notifier = BaseNotifier()
    notifier.on_run_started = AsyncMock()
    notifier.on_run_failed = AsyncMock()
    notifier.on_run_finished = AsyncMock()
    return notifier


async def entry_case(store, entry):
    provider = MockLLM(
        [
            LLMResponse(text="prior")
            if entry == "retry"
            else LLMResponse(tool_calls=[ToolCall(name="client_read")]),
            LLMResponse(text="done"),
        ]
    )
    agent = Agent(
        name="failure-save",
        prompt="p",
        tools=[client_read],
        provider=provider,
        state_store=store,
        strict_tenancy=False,
        loop=ReActLoop(),
    )
    prior = await agent.run("go") if entry in {"retry", "resume", "resume_stream"} else None

    def invoke(notifier):
        if entry == "run":
            return agent.run("go", notifier=notifier)
        if entry == "retry":
            return agent.retry(prior.run_id, notifier=notifier)
        if entry == "run_stream":
            return agent.stream("go", notifier=notifier)
        call = prior.pending.tool_calls[0]
        results = [ToolResult(name=call.name, call_id=call.id, payload='"ok"')]
        method = agent.resume_stream if entry == "resume_stream" else agent.resume
        return method(prior.run_id, tool_results=results, notifier=notifier)

    return agent, invoke


@pytest.mark.parametrize("entry", ["run", "retry", "run_stream", "resume", "resume_stream"])
@pytest.mark.parametrize("outcome", ["won", "lost", "raises", "fenced_save"])
async def test_failure_after_start(store, monkeypatch, caplog, entry, outcome):
    agent, invoke = await entry_case(store, entry)
    notifier = observer()
    original = ProviderTimeoutError(
        provider="fixture", message="private", timeout_reason="call_deadline"
    )
    finalize = store.finalize_run
    run_ids = []

    async def fail_loop(**kwargs):
        run_ids.append(kwargs["run_id"])
        if outcome == "lost":
            assert await finalize(
                kwargs["run_id"],
                status="success",
                answer="winner",
                expected_current_status="running",
            )
        raise original

    async def fail_stream(**kwargs):
        await fail_loop(**kwargs)
        yield  # pragma: no cover

    # Cut after startup, beyond provider-level persistence, to isolate the runner's save.
    monkeypatch.setattr(agent._loop, "run", fail_loop)
    monkeypatch.setattr(agent._loop, "run_stream", fail_stream)
    secondary = (
        RunFencedError("fenced") if outcome == "fenced_save" else RuntimeError("save_failed")
    )
    save = AsyncMock(
        side_effect=secondary if outcome in {"raises", "fenced_save"} else store.finalize_with_event
    )
    monkeypatch.setattr(store, "finalize_with_event", save)
    recorder_failed = AsyncMock()
    monkeypatch.setattr(PersistenceRecorder, "on_run_failed", recorder_failed)
    safe = AsyncMock(wraps=runner._emit_event_safe)
    monkeypatch.setattr(runner, "_emit_event_safe", safe)
    if entry.endswith("stream"):
        async with invoke(notifier) as stream:
            events = [event async for event in stream]
        assert events[-1].type == RunEventType.RUN_ERROR
        assert events[-1].run_result.status == RunStatus.ERROR
        assert events[-1].run_result.error_info.category == "timeout"
    else:
        with pytest.raises(ProviderTimeoutError) as caught:
            await invoke(notifier)
        assert caught.value is original
    assert len(run_ids) == 1
    run_id = run_ids[0]
    notifier.on_run_started.assert_awaited_once()
    notifier.on_run_failed.assert_awaited_once()
    recorder_failed.assert_awaited_once()
    assert notifier.on_run_failed.await_args.args[-1] is original
    notifier.on_run_finished.assert_not_awaited()
    save.assert_awaited_once()
    assert save.await_args.kwargs["allowed_current_statuses"] == ["running"]
    assert save.await_args.kwargs["status"] == "error"
    assert save.await_args.kwargs["total_usage"] is None
    assert save.await_args.kwargs["event_data"]["error_info"]["category"] == "timeout"
    stored = [e for e in await store.get_run_events(run_id) if e.event_type == "run.error"]
    assert len(stored) == int(outcome == "won")
    assert safe.await_count == 0  # Lifecycle evidence is in the atomic finalize.
    row = await store.get_run(run_id)
    if outcome == "won":
        assert row.meta[ERROR_INFO_META_KEY]["category"] == "timeout"
    assert (
        row.status
        == {"won": "error", "lost": "success", "raises": "running", "fenced_save": "running"}[
            outcome
        ]
    )
    if outcome in {"raises", "fenced_save"}:
        assert "Failed to persist ERROR status" in caplog.text


async def test_existing_stream_failure_skips_hooks(store, monkeypatch):
    agent, _ = await entry_case(store, "run")
    original = await agent.run("go", idempotency_key="same")
    notifier = observer()
    failed = AsyncMock()
    monkeypatch.setattr(PersistenceRecorder, "on_run_failed", failed)
    async with agent.stream("go", idempotency_key="same", notifier=notifier) as stream:
        events = [event async for event in stream]
    assert events[-1].type == RunEventType.RUN_ERROR
    notifier.on_run_started.assert_not_awaited()
    notifier.on_run_failed.assert_not_awaited()
    failed.assert_not_awaited()
    assert (await store.get_run(original.run_id)).status == "waiting_client_tool"
    assert not any(e.event_type == "run.error" for e in await store.get_run_events(original.run_id))


async def test_retry_seed_uses_requested_id_when_source_id_is_normalized(store, monkeypatch):
    agent, invoke = await entry_case(store, "retry")
    source_lookup = runner._retry_source
    requested_ids = []

    async def normalized_source(store, run_id, tenant_id):
        source, explicit = await source_lookup(store, run_id, tenant_id)
        requested_ids.append(run_id)
        return replace(source, id="normalized-source"), explicit

    history_lookup = runner._retry_history

    async def normalized_history(store, source, *args):
        return await history_lookup(store, replace(source, id=requested_ids[0]), *args)

    monkeypatch.setattr(runner, "_retry_history", normalized_history)
    monkeypatch.setattr(runner, "_retry_source", normalized_source)
    prefix = AsyncMock(wraps=runner.record_run_prefix)
    images = AsyncMock(return_value=None)
    monkeypatch.setattr(runner, "record_run_prefix", prefix)
    monkeypatch.setattr(runner, "_start_image_view", images)
    await invoke(observer())
    assert prefix.await_args.kwargs["seed"] == {"retry_of": requested_ids[0]}
    assert images.await_args.kwargs["sources"][-1][0] == requested_ids[0]


@pytest.mark.parametrize("won", [False, True])
async def test_failure_operation_returns_cas_result(monkeypatch, won):
    from types import SimpleNamespace

    store = SimpleNamespace(finalize_run=AsyncMock(return_value=won))
    emit = AsyncMock()
    monkeypatch.setattr(runner, "_emit_event", emit)
    assert await runner._persist_run_failure(store, "run", RuntimeError("failure")) is won
    assert emit.await_count == int(won)


@pytest.mark.parametrize("safe", [False, True])
async def test_failure_event_write_policy(monkeypatch, safe):
    from types import SimpleNamespace

    store = SimpleNamespace(finalize_run=AsyncMock(return_value=True))
    original = RuntimeError("event_failed")
    monkeypatch.setattr(runner, "_emit_event", AsyncMock(side_effect=original))
    if safe:
        await runner._persist_run_failure(store, "run", RuntimeError("failure"), safe_event=True)
    else:
        with pytest.raises(RuntimeError) as caught:
            await runner._persist_run_failure(store, "run", RuntimeError("failure"))
        assert caught.value is original


@pytest.mark.parametrize("entry", ["resume", "resume_stream"])
async def test_claim_failure_releases_before_failure_persistence(store, monkeypatch, entry):
    agent, invoke = await entry_case(store, entry)
    notifier = observer()
    original = RuntimeError("governance_report_failed")
    monkeypatch.setattr(runner, "report_governance_change", AsyncMock(side_effect=original))
    release = AsyncMock(wraps=runner._release_claim)
    monkeypatch.setattr(runner, "_release_claim", release)
    finalize = AsyncMock(wraps=store.finalize_with_event)
    monkeypatch.setattr(store, "finalize_run", finalize)
    monkeypatch.setattr(store, "finalize_with_event", finalize)
    if entry == "resume_stream":
        async with invoke(notifier) as stream:
            events = [event async for event in stream]
        assert events[-1].type == RunEventType.RUN_ERROR
    else:
        with pytest.raises(RuntimeError) as caught:
            await invoke(notifier)
        assert caught.value is original
    release.assert_awaited_once()
    finalize.assert_not_awaited()
    row = (await store.list_runs())[0]
    assert row.status == "waiting_client_tool"
    assert not any(e.event_type == "run.error" for e in await store.get_run_events(row.id))
    notifier.on_run_started.assert_not_awaited()
    assert notifier.on_run_failed.await_count == int(entry == "resume_stream")
