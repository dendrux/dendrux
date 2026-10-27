"""Store writes reject text PostgreSQL cannot persist before changing state."""

import json

import pytest
from sqlalchemy import text

from dendrux.types import InputValidationError


@pytest.mark.parametrize(
    "method,kwargs",
    [
        ("save_trace", dict(role="user", content="x\x00", order_index=0)),
        (
            "save_tool_call",
            dict(
                tool_call_id="c",
                provider_tool_call_id=None,
                tool_name="t",
                target="client",
                params={"x": "\x00"},
                result_payload='"ok"',
                success=True,
                duration_ms=0,
                iteration_index=0,
            ),
        ),
        ("pause_run", dict(status="waiting_client_tool", pause_data={"x": "\x00"})),
        ("try_pause_run", dict(status="waiting_client_tool", pause_data={"x": "\x00"})),
        ("request_pause", dict(reason="x\x00")),
        ("update_run_meta", dict(patch={"x": "\x00"})),
        ("finalize_run", dict(status="success", answer="x\x00")),
        ("add_note", dict(note_id="n", text="x\x00")),
        ("submit_and_claim", dict(expected_status="running", submitted_data={"x": "\x00"})),
        ("save_run_event", dict(event_type="fixture", data={"x": "\x00"})),
    ],
)
async def test_invalid_write_values_leave_the_run_unchanged(store, method, kwargs):
    await store.create_run("r", "a")
    with pytest.raises(InputValidationError):
        await getattr(store, method)("r", **kwargs)
    assert (await store.get_run("r")).status == "running"
    assert (await store.get_run_control("r")).claim_epoch == 0
    assert not await store.get_traces("r")
    assert not await store.get_tool_calls("r")


async def test_legacy_pause_projection_does_not_cast_history_to_jsonb(store, engine):
    await store.create_run("r", "a")
    async with engine.begin() as conn:
        await conn.execute(
            text("UPDATE agent_runs SET pause_data = :data WHERE id = :id"),
            {"data": json.dumps({"pause_id": "p", "history": [{"text": "\x00"}]}), "id": "r"},
        )
    summary = await store.get_pause_summary("r")
    assert summary.pause["pause_id"] == "p"


@pytest.mark.parametrize("value", ["x\x00", "x\ud800"])
@pytest.mark.parametrize("method", ["list_runs", "count_runs"])
async def test_filter_text_is_validated_before_database_work(store, method, value):
    with pytest.raises(InputValidationError):
        await getattr(store, method)(metadata_filter={"thread_id": value})


@pytest.mark.parametrize("value", ["x\x00", "x\ud800", "x" * 27])
async def test_get_result_validates_identifiers_before_resolving_store(value, caplog, monkeypatch):
    from dendrux import Agent
    from dendrux.llm.mock import MockLLM

    agent = Agent(prompt="p", provider=MockLLM([]), strict_tenancy=False)

    async def fail():
        raise AssertionError("store must not be resolved for an invalid identifier")

    monkeypatch.setattr(agent, "_resolve_state_store", fail)
    with pytest.raises(ValueError) as caught:
        await agent.get_result(value, tenant_id="A")
    assert str(caught.value) == "Invalid run_id."
    assert any(
        getattr(record, "code", None) == "read_argument_invalid" for record in caplog.records
    )
