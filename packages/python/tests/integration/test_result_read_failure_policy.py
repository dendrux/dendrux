"""Result projection and custom-store failures use safe database diagnostics."""

import json

import pytest
from sqlalchemy import text
from sqlalchemy.exc import StatementError

from dendrux import Agent
from dendrux.errors import PersistenceError
from dendrux.llm.mock import MockLLM
from dendrux.runtime.runner import _nonprovider_error
from dendrux.types import LLMResponse, RunStatus

PRIVATE = "PRIVATE_CUSTOM_STORE_BIND"


async def test_result_projection_database_fault_is_wrapped(store, engine):
    if engine.dialect.name != "postgresql":
        pytest.skip("real PostgreSQL projection fault")
    await store.create_run("r", "safe")
    agent = Agent(
        name="safe", prompt="p", provider=MockLLM([]), state_store=store, strict_tenancy=False
    )
    async with engine.begin() as conn:
        await conn.execute(
            text("ALTER TABLE agent_runs RENAME COLUMN output_data TO unavailable_output")
        )
    try:
        with pytest.raises(PersistenceError) as caught:
            await agent.get_result("r", tenant_id=None)
        assert str(caught.value) == PersistenceError.safe_message
        assert "SELECT" not in str(caught.value) and "unavailable_output" not in str(caught.value)
    finally:
        async with engine.begin() as conn:
            await conn.execute(
                text("ALTER TABLE agent_runs RENAME COLUMN unavailable_output TO output_data")
            )
        await agent.close()


async def test_custom_store_database_failure_has_safe_durable_and_stream_errors(store, caplog):
    class CustomStore:
        def __getattr__(self, name):
            return getattr(store, name)

        async def get_run_control(self, run_id):
            raise StatementError(PRIVATE, PRIVATE, {"email": PRIVATE}, RuntimeError(PRIVATE))

    custom = CustomStore()
    try:
        await custom.get_run_control("probe")
    except StatementError as exc:
        assert _nonprovider_error(exc) == PersistenceError.safe_message
    agent = Agent(
        name="safe",
        prompt="p",
        provider=MockLLM([LLMResponse(text="done")]),
        state_store=custom,
        strict_tenancy=False,
    )
    try:
        async with agent.stream("go", tenant_id="A") as stream:
            events = [event async for event in stream]
        result = stream.result
        assert result.status == RunStatus.ERROR
        assert result.error == PersistenceError.safe_message
        record = await store.get_run(result.run_id)
        assert record.error == PersistenceError.safe_message
        assert PRIVATE not in json.dumps([event.error for event in events])
        assert PRIVATE not in json.dumps(record.meta)
        assert PRIVATE not in caplog.text
    finally:
        await agent.close()
