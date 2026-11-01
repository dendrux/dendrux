"""Native JSON representation, collation, packet-loss and scalar-filter regressions."""

# ruff: noqa: F401, F811
import pytest
from sqlalchemy import event, select

from dendrux.db.models import ToolCallRecord
from dendrux.errors import PersistenceError, StatementTooLargeError
from dendrux.runtime.durability import retry_transient_db
from dendrux.types import UsageStats
from tests.integration._mysql import mysql_engine, store
from tests.integration.test_sql_store_rx_values import (
    test_metadata_scalar_truth_table,
    test_note_scope_precheck_only_on_postgres,
    test_pause_count_chunks_large_input,
)


@pytest.fixture
def backend_engine(mysql_engine):
    return mysql_engine


@pytest.mark.parametrize("value", [-(2**63) - 1, 2**64])
@pytest.mark.parametrize("operation", ["create", "patch", "tool"])
async def test_mysql_rejects_inexact_json_integers(mysql_engine, store, value, operation):

    if operation != "create":
        await store.create_run("run", "agent")

    async def write():
        if operation == "create":
            return await store.create_run("run", "agent", input_data={"nested": [value]})
        if operation == "patch":
            return await store.update_run_meta("run", {"nested": [value]})
        return await store.save_tool_call(
            "run",
            tool_call_id="call",
            provider_tool_call_id=None,
            tool_name="tool",
            target="server",
            params={},
            result_payload=str(value),
            success=True,
            duration_ms=1,
            iteration_index=0,
        )

    if mysql_engine.dialect.is_mariadb:
        await write()
        if operation == "create":
            assert (await store.get_run("run")).input_data == {"nested": [value]}
    else:
        with pytest.raises(PersistenceError, match="int64.*uint64") as caught:
            await write()
        assert str(value) not in str(caught.value)
        assert caught.value.retryable is False
        if operation == "create":
            assert await store.get_run("run") is None


async def test_longtext_unknown_tool_prefix_is_case_sensitive(store):
    await store.create_run("run", "agent")
    for index, message in enumerate(["Unknown tool: x", "UNKNOWN TOOL: x"]):
        await store.save_tool_call(
            "run",
            tool_call_id=f"call-{index}",
            provider_tool_call_id=None,
            tool_name="x",
            target="server",
            params={},
            result_payload="null",
            success=False,
            duration_ms=1,
            iteration_index=0,
            error_message=message,
        )
    async with store._session_factory() as session:
        matches = (
            await session.scalars(
                select(ToolCallRecord.error_message).where(
                    ToolCallRecord.error_message.like("Unknown tool:%")
                )
            )
        ).all()
    assert matches == ["Unknown tool: x"]


async def test_oversized_connection_drop_is_one_typed_attempt(mysql_engine, store, caplog):
    await store.create_run("run", "agent")
    size = max(20 * 1024 * 1024, store._dialect_facts.max_allowed_packet + 1024)
    attempts = 0

    def capture(conn, cursor, sql, params, context, many):
        nonlocal attempts
        if sql.startswith("INSERT INTO llm_interactions"):
            attempts += 1

    event.listen(mysql_engine.sync_engine, "before_cursor_execute", capture)
    try:

        async def write():
            await store.save_llm_interaction(
                "run",
                iteration_index=0,
                usage=UsageStats(),
                provider="mock",
                model="mock",
                semantic_request={},
                semantic_response={"text": "x" * size},
                provider_request={},
                provider_response={},
            )

        with pytest.raises(StatementTooLargeError) as caught:
            await retry_transient_db(write, label="packet", run_id="run")
        assert caught.value.limit == store._dialect_facts.max_allowed_packet
        assert attempts == 1
        assert "Transient DB failure" not in caplog.text
        assert caught.value.__context__ is None
    finally:
        event.remove(mysql_engine.sync_engine, "before_cursor_execute", capture)


async def test_json_fields_have_independent_validation_budgets(store, monkeypatch):
    from dendrux.db._dialects._mysql_types import CheckedMySQLJSON

    bind = CheckedMySQLJSON.process_bind_param
    fields = []

    def capture(self, value, dialect):
        if value is not None:
            fields.append(self.field)
        return bind(self, value, dialect)

    monkeypatch.setattr(CheckedMySQLJSON, "process_bind_param", capture)
    value = [1] * 270000
    result = await store.create_run(
        "independent", "agent", input_data={"values": value}, meta={"values": value}
    )
    assert result.outcome == "created"
    assert fields.count("input_data") == 1
    assert fields.count("meta") == 1


async def test_reads_do_not_walk_operation_arguments(store, monkeypatch):
    await store.create_run("read", "agent")

    def unexpected(value):
        raise AssertionError("read walked arguments")

    monkeypatch.setattr(store, "_validate_database_values", unexpected, raising=False)
    assert await store.get_run("read") is not None
