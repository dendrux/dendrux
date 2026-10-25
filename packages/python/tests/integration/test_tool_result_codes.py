"""Tool evidence has stable classifications independent of error wording."""

from sqlalchemy import event, select

from dendrux import Agent
from dendrux.db.models import ToolCallRecord
from dendrux.loops.react import ReActLoop
from dendrux.runtime.submit import prepare_resume_results
from dendrux.types import LLMResponse, PauseState, RunStatus, ToolResult
from tests._helpers.scripted_provider import ScriptedLLM, tool_call
from tests.integration.test_tenant_run_control import _pause


async def test_unknown_tool_code_is_persisted(store, engine):
    agent = Agent(
        prompt="p",
        tools=[],
        loop=ReActLoop(),
        state_store=store,
        strict_tenancy=False,
        provider=ScriptedLLM([tool_call("not_available", pid="missing"), LLMResponse(text="done")]),
    )
    try:
        result = await agent.run("go", tenant_id="A")
        assert result.status == RunStatus.SUCCESS
        calls = await store.get_tool_calls(result.run_id)
        assert len(calls) == 1
        async with engine.connect() as conn:
            metadata = (
                await conn.execute(
                    select(ToolCallRecord.meta).where(ToolCallRecord.id == calls[0].id)
                )
            ).scalar_one()
        assert (metadata or {}).get("code") == "unknown_tool"
    finally:
        await agent.close()


async def test_resume_result_validation_reads_only_status(store, engine):
    paused = await _pause("client", store, "A")
    raw = await store.get_pause_state(paused.run_id)
    call = raw["pending_tool_calls"][0]
    statements = []

    def capture(conn, cursor, statement, parameters, context, executemany):
        if statement.lstrip().upper().startswith("SELECT"):
            statements.append(statement)

    event.listen(engine.sync_engine, "before_cursor_execute", capture)
    try:
        await prepare_resume_results(
            paused.agent,
            store,
            paused.run_id,
            PauseState.from_dict(raw),
            [ToolResult(call["name"], call["id"], '"ok"')],
        )
        assert len(statements) == 1
        assert "input_data" not in statements[0] and "agent_runs.meta" not in statements[0]
        assert statements[0].startswith("SELECT agent_runs.status ")
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", capture)
        await paused.agent.close()
