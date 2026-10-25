"""Database failures do not expose statements or request parameters."""

import json
import logging

import pytest
from sqlalchemy import text
from sqlalchemy.exc import OperationalError, StatementError

from dendrux.db._engine_options import engine_options
from dendrux.runtime.durability import retry_transient_db
from dendrux.runtime.runner import _error_payload, _safe_error

PRIVATE = "PRIVATE_SQL_BIND_SENTINEL"


@pytest.mark.parametrize("supplied", [None, {"hide_parameters": False}, {"echo": True}])
def test_engine_always_hides_bound_parameters(supplied):
    assert engine_options("sqlite+aiosqlite://", supplied)["hide_parameters"] is True


@pytest.mark.parametrize("kind", [StatementError, OperationalError])
async def test_failed_write_retries_have_safe_logs_and_caller_error(caplog, kind):
    async def attempt():
        raise kind(PRIVATE, PRIVATE, {"email": PRIVATE}, RuntimeError(PRIVATE))

    with caplog.at_level(logging.WARNING), pytest.raises(Exception) as caught:
        await retry_transient_db(attempt, label="fixture", run_id="safe")
    assert PRIVATE not in str(caught.value)
    assert getattr(caught.value, "code", None) == "persistence_error"
    assert PRIVATE not in caplog.text


def test_raw_database_failure_has_a_safe_run_message():
    error = StatementError(PRIVATE, PRIVATE, {"email": PRIVATE}, RuntimeError(PRIVATE))
    assert PRIVATE not in _safe_error(error)
    payload = _error_payload(error)
    assert PRIVATE not in json.dumps(payload)
    assert payload["code"] == "persistence_error"


async def test_store_failure_returns_a_safe_typed_error(store):
    with pytest.raises(Exception) as caught:
        await store.save_trace("absent", "user", "ok", order_index=0)
    assert getattr(caught.value, "code", None) == "persistence_error"
    assert "INSERT" not in str(caught.value)


async def test_metadata_filter_ignores_unrelated_legacy_nul(store, engine):
    await store.create_run("legacy", "a", meta={"thread_id": "keep"})
    # Plain JSON in older installations permits escaped NUL values.
    async with engine.begin() as conn:
        await conn.execute(
            text("UPDATE agent_runs SET meta = :meta WHERE id = :id"),
            {"meta": json.dumps({"thread_id": "keep", "poison": "\x00"}), "id": "legacy"},
        )
    expected = 0 if engine.dialect.name == "postgresql" else 1
    assert await store.count_runs(metadata_filter={"thread_id": "keep"}) == expected
    rows = await store.list_runs(metadata_filter={"thread_id": "keep"})
    assert [row.id for row in rows] == (["legacy"] if expected else [])


async def test_post_claim_database_failure_is_safe_on_every_run_surface(store, monkeypatch, caplog):
    from dendrux.types import ToolResult
    from tests.integration.test_tenant_run_control import _pause

    paused = await _pause("client", store, "A")
    raw = await store.get_pause_state(paused.run_id)
    call = raw["pending_tool_calls"][0]

    async def fail(*args, **kwargs):
        raise StatementError(PRIVATE, PRIVATE, {"email": PRIVATE}, RuntimeError(PRIVATE))

    monkeypatch.setattr(store, "save_tool_call", fail)
    try:
        with caplog.at_level(logging.WARNING), pytest.raises(Exception) as caught:
            await paused.agent.submit_tool_results(
                paused.run_id,
                [ToolResult(call["name"], call["id"], '"ok"')],
                tenant_id="A",
                expected_pause_id=raw["pause_id"],
            )
        assert PRIVATE not in str(caught.value)
        assert getattr(caught.value, "code", None) == "persistence_error"
        record = await store.get_run(paused.run_id)
        assert record.status == "error"
        assert "persistence_error" in record.error
        assert PRIVATE not in record.error
        events = await store.get_run_events(paused.run_id)
        assert PRIVATE not in json.dumps([event.data for event in events])
        fetched = await paused.agent.get_result(paused.run_id, tenant_id="A")
        assert fetched.error == record.error
        assert PRIVATE not in caplog.text
    finally:
        await paused.agent.close()
