"""Owner wire policy mirrors the in-process restore decision, not raw metadata."""

import json
from dataclasses import replace

import pytest
from pydantic import BaseModel
from sqlalchemy import update

from dendrux import Agent, tool
from dendrux.db.models import AgentRun
from dendrux.guardrails import PII
from dendrux.loops.single import SingleCall
from dendrux.types import LLMResponse
from tests._helpers.scripted_provider import ScriptedLLM, tool_call


@pytest.mark.parametrize(
    "restore,receives,expected",
    [
        ("allowlisted", True, "alice@example.com"),
        ("allowlisted", False, "<<EMAIL_ADDRESS_1>>"),
        ("all", False, "alice@example.com"),
    ],
)
async def test_client_wire_matches_server_restore_policy(store, restore, receives, expected):
    @tool(target="client", receives_pii=receives)
    def send(to: str) -> str:
        return "must run in browser"

    a = Agent(
        name="policy",
        prompt="PROMPT_SENTINEL",
        tools=[send],
        guardrails=[PII(restore=restore)],
        strict_tenancy=False,
        state_store=store,
        provider=ScriptedLLM([tool_call("send", {"to": "<<EMAIL_ADDRESS_1>>"}, pid="C")]),
    )
    try:
        paused = await a.run("Email alice@example.com", tenant_id="A")
        fetched = await a.get_result(paused.run_id, tenant_id="A")
        for result in (paused, fetched):
            assert result.pending.tool_calls[0].params == {"to": expected}
            wire = result.to_dict()
            assert wire["pending"]["pending_tool_calls"][0]["params"] == {"to": expected}
            text = json.dumps(wire)
            assert "pii_mapping" not in text and "__dendrux_origins__" not in text
            assert "PROMPT_SENTINEL" not in text
            if not receives and restore == "allowlisted":
                assert "alice@example.com" not in text
    finally:
        await a.close()


async def test_get_result_filters_origins_and_its_own_projection_scope(store):
    await store.create_run("scoped", "scope", tenant_id="A")
    async with store._session_factory() as session:
        await session.execute(
            update(AgentRun)
            .where(AgentRun.id == "scoped")
            .values(
                pii_mapping={
                    "<<EMAIL_1>>": "a@example.test",
                    "__dendrux_origins__": '{"<<EMAIL_1>>":true}',
                },
            )
        )
        await session.commit()
    a = Agent(name="scope", prompt="p", state_store=store, strict_tenancy=False)
    try:
        result = await a.get_result("scoped", tenant_id="A")
        assert result.meta["pii_mapping"] == {"<<EMAIL_1>>": "a@example.test"}
        assert await store._get_result_record("scoped", tenant_id="B") is None
        assert await store._get_result_record("scoped", tenant_id=None) is None
    finally:
        await a.close()


async def test_structured_output_survives_fresh_cached_and_fetched_wire(store):
    class Answer(BaseModel):
        revenue: int

    llm = ScriptedLLM([LLMResponse(text='{"revenue":42}')])
    llm.capabilities = replace(llm.capabilities, supports_structured_output=True)
    a = Agent(
        name="structured-wire",
        prompt="p",
        provider=llm,
        loop=SingleCall(),
        output_type=Answer,
        state_store=store,
        strict_tenancy=False,
    )
    try:
        fresh = await a.run("go", tenant_id="A", idempotency_key="typed")
        cached = await a.run("go", tenant_id="A", idempotency_key="typed")
        fetched = await a.get_result(fresh.run_id, tenant_id="A")
        for result in (fresh, cached, fetched):
            assert isinstance(result.output, Answer)
            assert result.to_dict()["output"] == {"revenue": 42}
    finally:
        await a.close()


@pytest.mark.parametrize(
    "kind,expected", [("client", 1), ("paused", 1), ("approval", 3), ("input", 3)]
)
async def test_pending_attachment_read_cost(store, engine, kind, expected, record_property):
    from sqlalchemy import event

    from dendrux.runtime.runner import _attach_persisted_pending
    from dendrux.types import RunResult, RunStatus
    from tests.integration.test_tenant_run_control import _pause

    fixture = await _pause(kind, store, "A")
    record = await store.get_run(fixture.run_id)
    result = RunResult(record.id, RunStatus(record.status))
    statements = []

    def capture(conn, cursor, statement, parameters, context, many):
        if statement.lstrip().upper().startswith("SELECT"):
            statements.append(statement)

    event.listen(engine.sync_engine, "before_cursor_execute", capture)
    try:
        await _attach_persisted_pending(store, result, record)
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", capture)
        await fixture.agent.close()
    assert result.pending is not None
    assert len(statements) == expected
    record_property("pending_attachment_" + kind, len(statements))
