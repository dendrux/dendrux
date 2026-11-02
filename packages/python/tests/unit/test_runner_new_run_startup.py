"""Owned creation, cached returns, and partial startup keep their lifecycle semantics."""

from unittest.mock import AsyncMock, Mock

import pytest
from sqlalchemy.ext.asyncio import create_async_engine

from dendrux import Agent
from dendrux.db._sqlite_sync import create_sqlite_schema
from dendrux.llm.mock import MockLLM
from dendrux.loops.base import BaseNotifier
from dendrux.runtime import runner
from dendrux.runtime.persistence import PersistenceRecorder
from dendrux.store import SQLiteStateStore
from dendrux.types import LLMResponse, RunEventType, RunStatus


@pytest.fixture
async def store():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    await create_sqlite_schema(engine)
    try:
        yield SQLiteStateStore(engine)
    finally:
        await engine.dispose()


def observer():
    notifier = BaseNotifier()
    notifier.on_run_started = AsyncMock()
    notifier.on_run_finished = AsyncMock()
    notifier.on_run_failed = AsyncMock()
    return notifier


async def invoke(entry, store, notifier, source=None, **kwargs):
    provider = MockLLM([LLMResponse(text="done")])
    agent = Agent(name="startup", prompt="prompt")
    common = dict(
        agent=agent,
        provider=provider,
        state_store=store,
        tenant_id=None,
        extra_notifier=notifier,
        **kwargs,
    )
    if entry == "retry":
        return await runner.retry(source, **common)
    if entry == "run":
        return await runner.run(user_input="hello", **common)
    async with runner.run_stream(user_input="hello", **common) as stream:
        events = [event async for event in stream]
    return events


@pytest.mark.parametrize("entry", ["run", "stream"])
async def test_cached_creation_never_starts(store, monkeypatch, entry):
    original = await invoke("run", store, observer(), idempotency_key="same")
    events_before = await store.get_run_events(original.run_id)
    notifier = observer()
    constructor = Mock(side_effect=AssertionError("cached run constructed a recorder"))
    monkeypatch.setattr(PersistenceRecorder, "__init__", constructor)
    result = await invoke(entry, store, notifier, idempotency_key="same")
    if entry == "stream":
        assert [event.type for event in result] == [RunEventType.RUN_COMPLETED]
        result = result[0].run_result
    assert result.run_id == original.run_id and result.answer == "done"
    assert await store.get_run_events(original.run_id) == events_before
    constructor.assert_not_called()
    notifier.on_run_started.assert_not_awaited()
    notifier.on_run_failed.assert_not_awaited()
    notifier.on_run_finished.assert_not_awaited()


@pytest.mark.parametrize("entry", ["run", "stream", "retry"])
@pytest.mark.parametrize("site", ["create", "recorder", "event", "recorder_hook"])
async def test_partial_startup_failure(store, monkeypatch, entry, site):
    source = await invoke("run", store, observer()) if entry == "retry" else None
    notifier = observer()
    failure = RuntimeError("startup_cut")
    created = []
    create = store.create_run

    async def creating(*args, **kwargs):
        result = await create(*args, **kwargs)
        created.append(result.run_id)
        if site == "create":
            # The write may have committed, but no owned row was returned.
            raise failure
        return result

    monkeypatch.setattr(store, "create_run", creating)
    finalize = AsyncMock(wraps=store.finalize_run)
    monkeypatch.setattr(store, "finalize_run", finalize)
    recorder_failed = AsyncMock()
    monkeypatch.setattr(PersistenceRecorder, "on_run_failed", recorder_failed)
    if site == "recorder":
        monkeypatch.setattr(PersistenceRecorder, "__init__", Mock(side_effect=failure))
    if site == "event":
        save = store.save_run_event

        async def writing(*args, **kwargs):
            if kwargs["event_type"] == "run.started":
                raise failure
            return await save(*args, **kwargs)

        monkeypatch.setattr(store, "save_run_event", writing)
    if site == "recorder_hook":
        monkeypatch.setattr(PersistenceRecorder, "on_run_started", AsyncMock(side_effect=failure))
    if entry == "stream":
        events = await invoke(entry, store, notifier)
        assert [event.type for event in events] == [RunEventType.RUN_ERROR]
        assert events[0].error == "startup_cut"
    else:
        with pytest.raises(RuntimeError) as caught:
            await invoke(entry, store, notifier, source.run_id if source else None)
        assert caught.value is failure
    assert len(created) == 1
    row = await store.get_run(created[0])
    events = await store.get_run_events(created[0])
    names = [event.event_type for event in events]
    if site == "create":
        finalize.assert_not_awaited()
        assert row.status == "running" and names == []
    else:
        finalize.assert_awaited_once()
        assert finalize.await_args.kwargs["expected_current_status"] == "running"
        assert finalize.await_args.kwargs["total_usage"] is None
        assert row.status == RunStatus.ERROR.value
        assert names == (["run.started", "run.error"] if site == "recorder_hook" else ["run.error"])
    notifier.on_run_started.assert_not_awaited()
    notifier.on_run_finished.assert_not_awaited()
    assert notifier.on_run_failed.await_count == int(entry == "stream" or site == "recorder_hook")
    assert recorder_failed.await_count == int(
        site == "recorder_hook" or (entry == "stream" and site == "event")
    )


@pytest.mark.parametrize("entry", ["run", "stream", "retry"])
@pytest.mark.parametrize("storage_format", [1, 2])
async def test_startup_transaction_bytes(store, entry, storage_format):
    import json
    from pathlib import Path

    from sqlalchemy import event

    from tests._fixtures.storage import pin_storage_format

    await pin_storage_format(store._engine, storage_format)
    source = await invoke("run", store, observer()) if entry == "retry" else None
    statements = []
    capturing = True

    def begin(conn):
        if capturing:
            statements.append("BEGIN")

    def commit(conn):
        if capturing:
            statements.append("COMMIT")

    def rollback(conn):
        if capturing:
            statements.append("ROLLBACK")

    def execute(conn, cursor, statement, parameters, context, executemany):
        if capturing:
            statements.append(" ".join(statement.split()))

    notifier = observer()

    async def started(*args, **kwargs):
        nonlocal capturing
        capturing = False

    notifier.on_run_started.side_effect = started
    engine = store._engine.sync_engine
    listeners = [
        ("begin", begin),
        ("commit", commit),
        ("rollback", rollback),
        ("before_cursor_execute", execute),
    ]
    for name, listener in listeners:
        event.listen(engine, name, listener)
    try:
        await invoke(entry, store, notifier, source.run_id if source else None)
    finally:
        for name, listener in listeners:
            event.remove(engine, name, listener)
    notifier.on_run_started.assert_awaited_once()
    wire = json.dumps(statements, indent=2) + "\n"
    expected = Path(__file__).with_name("fixtures") / f"startup_sql_{entry}_{storage_format}.json"
    assert wire == expected.read_text()


@pytest.mark.parametrize("entry", ["run", "stream", "retry"])
async def test_startup_order_and_delegation_context(store, monkeypatch, entry):
    from dendrux.runtime.context import (
        DelegationContext,
        get_delegation_context,
        reset_delegation_context,
        set_delegation_context,
    )

    source = await invoke("run", store, observer()) if entry == "retry" else None
    seen = []
    child_id = None

    def note(stage):
        seen.append((stage, get_delegation_context().run_id))

    constructor = PersistenceRecorder.__init__

    def construct(self, *args, **kwargs):
        nonlocal child_id
        child_id = args[1]
        note("recorder")
        constructor(self, *args, **kwargs)

    emit = runner._emit_event

    async def event(*args, **kwargs):
        if args[2] == "run.started":
            note("event")
        await emit(*args, **kwargs)

    hook = PersistenceRecorder.on_run_started

    async def recorded(self, *args, **kwargs):
        note("record_hook")
        await hook(self, *args, **kwargs)

    async def notified(*args, **kwargs):
        note("notify_hook")

    monkeypatch.setattr(PersistenceRecorder, "__init__", construct)
    monkeypatch.setattr(PersistenceRecorder, "on_run_started", recorded)
    monkeypatch.setattr(runner, "_emit_event", event)
    notifier = observer()
    notifier.on_run_started.side_effect = notified
    token = set_delegation_context(
        DelegationContext("parent", tenant_id=None, tenant_explicit=True)
    )
    try:
        await invoke(entry, store, notifier, source.run_id if source else None)
        assert get_delegation_context().run_id == "parent"
    finally:
        reset_delegation_context(token)
    start_context = child_id if entry == "stream" else "parent"
    assert seen == [
        ("recorder", "parent"),
        ("event", start_context),
        ("record_hook", start_context),
        ("notify_hook", start_context),
    ]
