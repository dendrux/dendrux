"""Callback delivery, failure projection and independent ownership at run startup."""

import asyncio
import logging
import pickle
from datetime import UTC, datetime, timedelta
from functools import wraps
from uuid import uuid4

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from sqlalchemy import update
from sqlalchemy.ext.asyncio import create_async_engine

from dendrux import Agent, aheartbeat, cancel_run, errors, sweep, tool
from dendrux._lifecycle import TERMINAL_EVENT_TYPES
from dendrux.db.models import AgentRun
from dendrux.llm.mock import MockLLM
from dendrux.loops.base import BaseNotifier
from dendrux.notifiers.composite import CompositeNotifier
from dendrux.notifiers.otel import OpenTelemetryNotifier
from dendrux.runtime import runner
from dendrux.runtime.context import get_delegation_context
from dendrux.runtime.stores._factory import store_for_engine
from dendrux.runtime.submit import governance_snapshot
from dendrux.types import LLMResponse, RunStatus, ToolCall
from tests._helpers.gates import wait_for_gate
from tests.integration.test_on_started import make_agent, retry_source


async def invoke(agent, source, callback, **options):
    kwargs = dict(tenant_id=None, on_started=callback, **options)
    return await agent.retry(source.run_id, **kwargs) if source else await agent.run("go", **kwargs)


@pytest.mark.parametrize("entry", ["run", "retry"])
@pytest.mark.parametrize("operation", ["nested_root", "heartbeat"])
async def test_callback_starts_independent_root_and_heartbeat(store, monkeypatch, entry, operation):
    source = await retry_source(store, entry)
    outer, _ = make_agent(store)
    inner, _ = make_agent(store)
    inner_ids, heartbeats = [], []
    touch = store.touch_progress

    @wraps(touch)
    async def track(run_id, **kwargs):
        heartbeats.append(run_id)
        return await touch(run_id, **kwargs)

    monkeypatch.setattr(store, "touch_progress", track)

    async def start(run_id):
        if operation == "heartbeat":
            await aheartbeat()
            assert run_id not in heartbeats
            return
        result = await invoke(inner, None, inner_ids.append)
        row = await store.get_run(result.run_id)
        assert row.parent_run_id is None and row.delegation_level == 0
        assert inner_ids == [result.run_id]

    result = await invoke(outer, source, start)
    assert result.status == RunStatus.SUCCESS and get_delegation_context() is None


@pytest.fixture
def callback_spans():
    tracer = TracerProvider()
    exporter = InMemorySpanExporter()
    tracer.add_span_processor(SimpleSpanProcessor(exporter))
    try:
        yield OpenTelemetryNotifier(tracer.get_tracer("f7")), exporter
    finally:
        tracer.shutdown()


def assert_callback_log(caplog, run_id):
    records = [r for r in caplog.records if getattr(r, "code", None) == "on_started_failed"]
    assert [(r.run_id, r.callback_outcome) for r in records] == [(run_id, "raised")]


def assert_safe_sinks(sentinel, caplog, exporter, *surfaces):
    spans = exporter.get_finished_spans()
    assert spans
    for surface in (
        *surfaces,
        caplog.text,
        [record.__dict__ for record in caplog.records],
        [span.to_json() for span in spans],
    ):
        assert sentinel not in repr(surface)


@pytest.mark.parametrize("entry", ["run", "retry"])
@pytest.mark.parametrize("asynchronous", [True, False])
async def test_failure_is_atomic_and_safe_on_every_sink(
    store, entry, asynchronous, monkeypatch, caplog, callback_spans
):
    source = await retry_source(store, entry)
    agent, provider = make_agent(store)
    observed, ids = [], []
    sentinel = f"callback-private-{uuid4().hex}"
    cause = ValueError(sentinel)
    original = RuntimeError(sentinel + "-outer")
    finalize = store.finalize_with_event
    transitions = []

    @wraps(finalize)
    async def atomic(run_id, **kwargs):
        transitions.append(kwargs)
        return await finalize(run_id, **kwargs)

    monkeypatch.setattr(store, "finalize_with_event", atomic)

    class Observer(BaseNotifier):
        async def on_run_failed(self, run_id, error, **kwargs):
            observed.append(error)

    def fail(run_id):
        ids.append(run_id)
        logging.getLogger(__name__).debug("Entering the failing callback")
        raise original from cause

    async def fail_async(run_id):
        await asyncio.sleep(0)
        fail(run_id)

    otel, exporter = callback_spans
    notifier = CompositeNotifier([otel, Observer()])
    caplog.set_level(logging.DEBUG)
    with pytest.raises(Exception) as caught:
        await invoke(agent, source, fail_async if asynchronous else fail, notifier=notifier)
    (run_id,) = ids
    row = await store.get_run(run_id)
    events = await store.get_run_events(run_id)
    assert "Entering the failing callback" in caplog.messages
    assert_safe_sinks(
        sentinel, caplog, exporter, str(caught.value), caught.value, row, events, observed
    )
    assert isinstance(caught.value, runner.OnStartedError)
    assert caught.value.run_id == run_id and caught.value.__cause__ is original
    assert caught.value.code == "on_started_failed"
    assert observed[0].__cause__ is observed[0].__context__ is None
    assert row.status == "error" and row.error == str(caught.value)
    assert row.failure_reason == "on_started_failed"
    (failed,) = [event for event in events if event.event_type == "run.error"]
    assert failed.data["code"] == row.failure_reason and failed.data["error"] == row.error
    (write,) = transitions
    assert write["allowed_current_statuses"] == ["running"] and write["claim_epoch"] == 0
    assert write["status"] == "error" and write["event_type"] == "run.error"
    assert_callback_log(caplog, run_id)
    assert provider.calls_made == 0 and run_id not in agent._task_manager._owned
    assert get_delegation_context() is None


@pytest.mark.parametrize("engine", ["sqlite_file", "postgres"], indirect=True)
@pytest.mark.parametrize("when", ["inside", "at_failure_write"])
async def test_callback_error_observes_cancel_at_control_read(
    store, engine, monkeypatch, when, caplog, callback_spans
):
    caplog.set_level(logging.DEBUG)
    otel, exporter = callback_spans
    sentinel = f"callback-private-{uuid4().hex}"
    agent, provider = make_agent(store)
    ready, release = asyncio.Event(), asyncio.Event()
    ids = []
    final = store.finalize_with_event

    @wraps(final)
    async def finalize(run_id, **kwargs):
        if kwargs["status"] == "error" and when == "at_failure_write":
            ready.set()
            await release.wait()
        return await final(run_id, **kwargs)

    monkeypatch.setattr(store, "finalize_with_event", finalize)

    async def fail(run_id):
        ids.append(run_id)
        if when == "inside":
            ready.set()
            await release.wait()
        raise ValueError(sentinel) from RuntimeError(sentinel + "-cause")

    task = asyncio.create_task(invoke(agent, None, fail, notifier=otel))
    try:
        await wait_for_gate(ready, task=task)
        if when == "at_failure_write":
            assert_callback_log(caplog, ids[0])
        await cancel_run(ids[0], tenant_id=None, state_store=store)
        pending = await store.get_run(ids[0])
        assert pending.status == "running" and pending.cancel_requested
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
    row = await store.get_run(ids[0])
    if when == "inside":
        result = await task
        assert result.status == RunStatus.CANCELLED and row.error is None
        response = result
        terminal = "run.cancelled"
    else:
        with pytest.raises(runner.OnStartedError) as caught:
            await task
        assert caught.value.run_id == ids[0] and row.error == str(caught.value)
        response = caught.value
        terminal = "run.error"
    assert row.status == terminal.removeprefix("run.")
    events = await store.get_run_events(ids[0])
    assert [e.event_type for e in events if e.event_type in TERMINAL_EVENT_TYPES] == [terminal]
    assert events[-1].event_type == terminal
    assert provider.calls_made == 0 and ids[0] not in agent._task_manager._owned
    assert_safe_sinks(sentinel, caplog, exporter, row, events, response, str(response))
    assert_callback_log(caplog, ids[0])
    if when == "at_failure_write":
        assert row.cancel_requested is False


@pytest.mark.parametrize("engine", ["postgres"], indirect=True)
@pytest.mark.parametrize("entry", ["run", "retry"])
@pytest.mark.parametrize("outcome", ["return", "raise"])
@pytest.mark.parametrize("policy", ["error", "reclaim"])
async def test_stale_callback_yields_to_postgres_winner(
    store, engine, entry, outcome, policy, caplog, callback_spans
):
    sentinel = f"callback-private-{uuid4().hex}"
    otel, exporter = callback_spans
    caplog.set_level(logging.DEBUG)
    source = await retry_source(store, entry)
    effects = []

    @tool()
    async def work() -> str:
        """Record a tool invocation."""
        effects.append(1)
        return "ok"

    provider = MockLLM([LLMResponse(tool_calls=[ToolCall("work")]), LLMResponse(text="done")])
    agent = Agent(prompt="go", provider=provider, state_store=store, tools=[work])
    other_engine = create_async_engine(engine.url)
    other = store_for_engine(other_engine)
    ids = []
    entered, release = asyncio.Event(), asyncio.Event()

    async def callback(run_id):
        ids.append(run_id)
        entered.set()
        await release.wait()
        if outcome == "raise":
            raise ValueError(sentinel)

    task = asyncio.create_task(invoke(agent, source, callback, notifier=otel))
    try:
        await wait_for_gate(entered, task=task)
        (run_id,) = ids
        async with other_engine.begin() as conn:
            await conn.execute(
                update(AgentRun)
                .where(AgentRun.id == run_id)
                .values(last_progress_at=datetime.now(UTC) - timedelta(minutes=10))
            )
        page = await sweep(state_store=other, stale_running=timedelta(minutes=1), on_stale=policy)
        if policy == "reclaim":
            (item,) = page.reclaimable
            assert item.run_id == run_id
            claim = await other.reclaim_run(
                run_id,
                tenant_id=None,
                agent_name=agent.name,
                expected_claim_epoch=item.claim_epoch,
                governance=governance_snapshot(agent),
                governance_policy="raise",
                skill_pins={},
                loop_type="react",
            )
            assert claim.claim_epoch > 0
        else:
            assert [item.run_id for item in page.stale_running] == [run_id]
        winner = await other.get_run(run_id)
        winner_events = await other.get_run_events(run_id)
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        await other_engine.dispose()
    result = await task
    assert result.status.value == winner.status and provider.calls_made == 0 and effects == []
    assert await store.get_run(run_id) == winner
    assert await store.get_run_events(run_id) == winner_events
    assert run_id not in agent._task_manager._owned
    assert_safe_sinks(sentinel, caplog, exporter, result, str(result), winner, winner_events)
    if outcome == "raise":
        assert_callback_log(caplog, run_id)
    assert any(
        getattr(r, "code", None) == "run_fenced" and r.run_id == run_id for r in caplog.records
    )


def test_stream_reports_unsupported_start_callback():
    agent, provider = make_agent()
    with pytest.raises(ValueError, match=r"on_started.*only supported on run\(\)/retry\(\)"):
        agent.stream("go", on_started=lambda run_id: None)
    assert provider.calls_made == 0


async def test_atomic_failure_event_write_rolls_back_without_generic_retry(store, monkeypatch):
    from dendrux.errors import PersistenceError
    from dendrux.runtime import state

    agent, provider = make_agent(store)
    insert = state._insert_event_locked
    ids, writes = [], []

    @wraps(insert)
    async def fail_once(session, run_id, event_type, *args, **kwargs):
        if event_type == "run.error":
            writes.append(run_id)
            if len(writes) == 1:
                raise PersistenceError()
        return await insert(session, run_id, event_type, *args, **kwargs)

    def fail(run_id):
        ids.append(run_id)
        raise ValueError("callback failed")

    monkeypatch.setattr(state, "_insert_event_locked", fail_once)
    with pytest.raises(PersistenceError):
        await invoke(agent, None, fail)
    assert writes == ids
    row = await store.get_run(ids[0])
    assert row.status == "running" and row.error is None
    assert all(e.event_type != "run.error" for e in await store.get_run_events(ids[0]))
    assert ids[0] not in agent._task_manager._owned and provider.calls_made == 0


async def test_failure_logs_registered_callback_code(store, caplog):
    from dendrux import _codes

    agent, _ = make_agent(store)

    def fail(run_id):
        raise ValueError("private")

    with pytest.raises(runner.OnStartedError) as caught:
        await invoke(agent, None, fail)
    code = caught.value.code
    assert code in _codes.REGISTRY
    assert any(
        getattr(r, "code", None) == code and r.run_id == caught.value.run_id for r in caplog.records
    )


@pytest.mark.parametrize("writers", ["atomic", "both_legacy_writers", "finalize_run_only"])
async def test_callback_failure_on_older_store(store, monkeypatch, writers):
    agent, provider = make_agent(store)
    writes = []
    final = store.finalize_with_event if writers == "atomic" else store.finalize_run

    async def legacy_atomic(
        run_id, *, status, allowed_current_statuses, error, claim_epoch, event_type, event_data
    ):
        writes.append((run_id, status, allowed_current_statuses, claim_epoch))
        return await final(
            run_id,
            status=status,
            allowed_current_statuses=allowed_current_statuses,
            error=error,
            claim_epoch=claim_epoch,
            event_type=event_type,
            event_data=event_data,
        )

    async def legacy_finalize(run_id, *, status, expected_current_status, error, claim_epoch):
        writes.append((run_id, status, [expected_current_status], claim_epoch))
        return await final(
            run_id,
            status=status,
            expected_current_status=expected_current_status,
            error=error,
            claim_epoch=claim_epoch,
        )

    monkeypatch.setattr(
        store, "finalize_with_event", legacy_atomic if writers == "atomic" else None
    )
    if writers != "atomic":
        monkeypatch.setattr(store, "finalize_run", legacy_finalize)
    if writers == "finalize_run_only":
        monkeypatch.setattr(store, "finalize_run_if_status_in", None)
    original = ValueError("private")

    def fail(run_id):
        raise original

    with pytest.raises(runner.OnStartedError) as caught:
        await invoke(agent, None, fail)
    error = caught.value
    assert error.__cause__ is original
    assert writes == [(error.run_id, "error", ["running"], 0)]
    row = await store.get_run(error.run_id)
    assert row.status == "error" and row.error == error.safe_message and row.failure_reason is None
    events = await store.get_run_events(error.run_id)
    assert [e for e in events if e.event_type in TERMINAL_EVENT_TYPES] == [events[-1]]
    assert events[-1].event_type == "run.error"
    assert events[-1].data == {"error": error.safe_message, "code": error.code}
    assert provider.calls_made == 0 and error.run_id not in agent._task_manager._owned


def test_on_started_error_public_contract():
    cls = runner.OnStartedError
    assert RuntimeError in cls.__bases__ and not issubclass(cls, errors.RunControlError)
    assert cls is errors.OnStartedError and cls.__module__ == errors.__name__
    assert cls.__name__ in errors.__all__
    error = pickle.loads(pickle.dumps(cls("root-id")))
    assert type(error) is cls and error.run_id == "root-id"
    assert error.code == "on_started_failed"
    assert str(error) == error.safe_message == "The on_started callback failed."
