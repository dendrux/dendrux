"""Raw connect failures must reach every existing store retry scope."""

import asyncio
import socket
import traceback
from unittest.mock import AsyncMock

import pytest
from sqlalchemy.ext.asyncio import create_async_engine

from dendrux.errors import PersistenceError
from dendrux.runtime import durability
from dendrux.store import PostgresStateStore
from dendrux.types import generate_ulid


def writes(s, r):
    return {
        "create_run": lambda: s.create_run(generate_ulid(), "a"),
        "finalize_run": lambda: s.finalize_run(r, status="success"),
        "finalize_run(claim_epoch)": lambda: s.finalize_run(r, status="success", claim_epoch=1),
        "finalize_run_if_status_in": lambda: s.finalize_run_if_status_in(
            r, status="success", allowed_current_statuses=["running"]
        ),
        "finalize_with_event": lambda: s.finalize_with_event(
            r, status="success", allowed_current_statuses=["running"], event_type="run.completed"
        ),
        "try_pause_run": lambda: s.try_pause_run(r, status="waiting_client_tool", pause_data={}),
        "pause_with_event": lambda: s.pause_with_event(
            r, status="waiting_client_tool", pause_data={}
        ),
        "pause_run": lambda: s.pause_run(r, status="waiting_client_tool", pause_data={}),
        "claim_paused_run": lambda: s.claim_paused_run(r, expected_status="waiting_client_tool"),
        "submit_and_claim": lambda: s.submit_and_claim(
            r, expected_status="waiting_client_tool", submitted_data={}
        ),
        "release_run_claim": lambda: s.release_run_claim(r, status="error", claim_epoch=1),
        "request_cancel": lambda: s.request_cancel(r),
        "mark_cancel_requested": lambda: s.mark_cancel_requested([r], tenant_id=None),
        "request_pause": lambda: s.request_pause(r),
        "update_run_meta": lambda: s.update_run_meta(r, {"a": 1}),
        "touch_progress": lambda: s.touch_progress(r, claim_epoch=1),
        "add_note": lambda: s.add_note(r, note_id=generate_ulid(), text="x", tenant_id=None),
        "mark_notes_delivered": lambda: s.mark_notes_delivered(r, ["x"], iteration=0),
        "reclaim_run": lambda: s.reclaim_run(
            r,
            tenant_id=None,
            agent_name="a",
            expected_claim_epoch=None,
            governance={},
            governance_policy="warn",
            skill_pins={},
            loop_type="react",
        ),
        "finalize_with_event(claim_epoch)": lambda: s.finalize_with_event(
            r,
            status="success",
            allowed_current_statuses=["running"],
            event_type="run.completed",
            claim_epoch=1,
        ),
        "finalize_run_if_status_in(claim_epoch)": lambda: s.finalize_run_if_status_in(
            r, status="success", allowed_current_statuses=["running"], claim_epoch=1
        ),
        "try_pause_run(claim_epoch)": lambda: s.try_pause_run(
            r, status="waiting_client_tool", pause_data={}, claim_epoch=1
        ),
        "pause_with_event(claim_epoch)": lambda: s.pause_with_event(
            r, status="waiting_client_tool", pause_data={}, claim_epoch=1
        ),
        "pause_run(claim_epoch)": lambda: s.pause_run(
            r, status="waiting_client_tool", pause_data={}, claim_epoch=1
        ),
        "update_run_meta(claim_epoch)": lambda: s.update_run_meta(r, {"a": 1}, claim_epoch=1),
        "add_note(claim_epoch)": lambda: s.add_note(
            r, note_id=generate_ulid(), text="x", tenant_id=None, claim_epoch=1
        ),
        "mark_notes_delivered(claim_epoch)": lambda: s.mark_notes_delivered(
            r, ["x"], iteration=0, claim_epoch=1
        ),
    }


@pytest.mark.parametrize("name", list(writes(None, "")))
@pytest.mark.parametrize("host", ["127.0.0.1", "localhost"])
async def test_direct_store_write_has_all_refused_backoffs(
    monkeypatch, unused_tcp_port, name, host
):
    backoff = AsyncMock()
    monkeypatch.setattr(durability, "_backoff", backoff)
    engine = create_async_engine(f"postgresql+asyncpg://u:p@{host}:{unused_tcp_port}/refused_test")
    store = PostgresStateStore(engine)
    try:
        with pytest.raises(PersistenceError) as caught:
            await writes(store, generate_ulid())[name]()
        assert caught.value.retryable
        assert [call.args[0] for call in backoff.await_args_list] == [0, 1, 2]
        assert caught.value.__context__ is None and caught.value.__cause__ is None
    finally:
        await store.close()
        await engine.dispose()


@pytest.mark.parametrize(
    "code,backoffs,retryable", [("53300", 3, True), ("57P03", 3, True), ("28P01", 0, False)]
)
async def test_raw_postgres_connect_state(monkeypatch, code, backoffs, retryable, caplog):
    import asyncpg

    raw = asyncpg.PostgresError.new({"C": code, "M": "PRIVATE_CONNECT_SENTINEL"})
    connect = AsyncMock(side_effect=raw)
    delay = AsyncMock()
    monkeypatch.setattr(asyncpg, "connect", connect)
    monkeypatch.setattr(durability, "_backoff", delay)
    engine = create_async_engine("postgresql+asyncpg://u:p@127.0.0.1:1/connect_test")
    store = PostgresStateStore(engine)
    try:
        with pytest.raises(PersistenceError) as caught:
            await store.create_run(generate_ulid(), "connect")
        assert caught.value.retryable is retryable
        assert delay.await_count == backoffs
        assert connect.await_count == backoffs + 1
        assert caught.value.__context__ is None and caught.value.__cause__ is None
        assert "PRIVATE_CONNECT_SENTINEL" not in "".join(traceback.format_exception(caught.value))
        assert "PRIVATE_CONNECT_SENTINEL" not in repr([vars(record) for record in caplog.records])
    finally:
        await store.close()
        await engine.dispose()


@pytest.mark.parametrize(
    "fault", [ValueError(), OSError(), TimeoutError(), asyncio.CancelledError()]
)
async def test_non_driver_failure_does_not_enter_database_retry(monkeypatch, fault):
    delay = AsyncMock()
    monkeypatch.setattr(durability, "_backoff", delay)
    attempt = AsyncMock(side_effect=fault)
    with pytest.raises(type(fault)) as caught:
        await durability.retry_transient_db(attempt, label="hook", run_id=generate_ulid())
    assert caught.value is fault
    assert attempt.await_count == 1
    delay.assert_not_called()


def test_dns_and_bare_timeouts_are_not_availability():
    for fault in (socket.gaierror(), TimeoutError()):
        assert not durability.is_transient_db_error(fault)
