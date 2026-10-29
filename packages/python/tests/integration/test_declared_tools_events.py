"""Only code-declared local tools authorize redacted-reader tool names."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from dendrux import Agent, tool
from dendrux.loops.react import ReActLoop
from dendrux.tools import build_tool_lookups
from dendrux.types import LLMResponse, RunStatus, ToolDef, ToolResult
from tests._helpers.scripted_provider import ScriptedLLM, tool_call


@tool()
async def z_server() -> str:
    return "server"


@tool(target="client")
async def a_client() -> str:
    return "client"


@pytest.mark.parametrize("entry", ["run", "stream", "resume", "resume_stream", "retry"])
async def test_declared_tools_event_once_per_segment_excludes_mcp_and_model_names(
    store, monkeypatch, entry
):
    continuation = entry in ("resume", "resume_stream")
    responses = ([tool_call("a_client", pid="client-call")] if continuation else []) + [
        tool_call("PRIVATE_PROVIDER_NAME", pid="unknown-call"),
        LLMResponse(text="done"),
        LLMResponse(text="retried"),
    ]
    agent = Agent(
        name="declared",
        prompt="p",
        tools=[z_server, a_client],
        provider=ScriptedLLM(responses),
        loop=ReActLoop(),
        state_store=store,
        strict_tenancy=False,
    )
    source = SimpleNamespace(
        name="remote", source=SimpleNamespace(name="remote"), last_error=None, close=AsyncMock()
    )
    agent._tool_sources = [source]
    agent._discovered_tool_defs = [
        ToolDef("remote__PRIVATE_MCP_NAME", "", {}, meta={"source_name": "remote"})
    ]
    remote_name = agent._discovered_tool_defs[0].name
    lookups = build_tool_lookups(
        agent.tools,
        mcp_executors={remote_name: AsyncMock(return_value="unused")},
        mcp_tool_defs=agent._discovered_tool_defs,
    )
    assert remote_name in lookups.defs, "fixture must expose a real merged MCP catalog"
    monkeypatch.setattr(agent, "get_tool_lookups", AsyncMock(return_value=lookups))
    try:
        if entry == "stream":
            async with agent.stream("go", tenant_id="A") as stream:
                async for _ in stream:
                    pass
            result = stream.result
        else:
            result = await agent.run("go", tenant_id="A")
        if continuation:
            raw = await store.get_pause_state(result.run_id)
            call = raw["pending_tool_calls"][0]
            kwargs = dict(
                tenant_id="A",
                expected_pause_id=raw["pause_id"],
                tool_results=[ToolResult(call["name"], call["id"], '"ok"')],
            )
            if entry == "resume":
                result = await agent.resume(result.run_id, **kwargs)
            else:
                async with agent.resume_stream(result.run_id, **kwargs) as stream:
                    async for _ in stream:
                        pass
                result = stream.result
        elif entry == "retry":
            result = await agent.retry(result.run_id, tenant_id="A")
        assert result.status == RunStatus.SUCCESS
        events = await store.get_run_events(result.run_id, event_types=["tools.declared"])
        assert len(events) == (2 if continuation else 1), "one declaration per actual segment"
        assert all(event.data == {"local_tool_names": ["a_client", "z_server"]} for event in events)
    finally:
        agent._tool_sources = []
        await agent.close()


@pytest.mark.parametrize("notify_enabled", [True, False])
async def test_declaration_write_has_constant_database_cost(store, engine, notify_enabled):
    from sqlalchemy import event

    from dendrux.runtime.persistence import PersistenceRecorder
    from dendrux.runtime.runner import _emit_declared_tools

    if not notify_enabled:
        store.notify_channel = None
    await store.create_run("declaration-cost", "declared")
    agent = Agent(prompt="p", tools=[z_server, a_client])
    statements = []

    def capture(conn, cursor, statement, parameters, context, executemany):
        statements.append(statement)

    event.listen(engine.sync_engine, "before_cursor_execute", capture)
    try:
        await _emit_declared_tools(
            agent, PersistenceRecorder(store, "declaration-cost"), None, "declaration-cost"
        )
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", capture)
        await agent.close()
    notify_count = (
        1 if engine.dialect.name == "postgresql" and store.notify_channel is not None else 0
    )
    expected = 2 + notify_count  # fence UPDATE, event INSERT, optional transactional NOTIFY
    assert len(statements) == expected, statements
    assert sum("pg_notify" in sql for sql in statements) == notify_count
    assert statements[0].startswith("UPDATE agent_runs")
    assert sum("INSERT INTO run_events" in sql for sql in statements) == 1
    assert all("react_traces" not in sql and "llm_interactions" not in sql for sql in statements)


async def test_redacted_follow_preserves_declared_local_names_and_hides_mcp_names(
    store, engine, monkeypatch
):
    from dendrux.store import FollowEnd, RunStore, StoredEvent

    remote_name = "remote__PRIVATE_MCP_NAME"
    provider = ScriptedLLM(
        [
            tool_call("z_server", pid="local-server"),
            tool_call(remote_name, pid="remote-server"),
            tool_call("a_client", pid="local-client"),
            LLMResponse(text="done"),
        ]
    )
    agent = Agent(
        name="follow-declared",
        prompt="PRIVATE_PROMPT",
        tools=[z_server, a_client],
        provider=provider,
        loop=ReActLoop(),
        state_store=store,
        strict_tenancy=False,
    )
    agent._tool_sources = [
        SimpleNamespace(
            name="remote", source=SimpleNamespace(name="remote"), last_error=None, close=AsyncMock()
        )
    ]
    agent._discovered_tool_defs = [ToolDef(remote_name, "", {}, meta={"source_name": "remote"})]
    remote = AsyncMock(return_value="PRIVATE_REMOTE_RESULT")
    lookups = build_tool_lookups(
        agent.tools, mcp_executors={remote_name: remote}, mcp_tool_defs=agent._discovered_tool_defs
    )
    monkeypatch.setattr(agent, "get_tool_lookups", AsyncMock(return_value=lookups))
    reader = RunStore.from_engine(engine)
    try:
        paused = await agent.run("go", tenant_id="A")
        assert paused.status == RunStatus.WAITING_CLIENT_TOOL
        raw = await store.get_pause_state(paused.run_id)
        call = raw["pending_tool_calls"][0]
        result = await agent.resume(
            paused.run_id,
            tenant_id="A",
            expected_pause_id=raw["pause_id"],
            tool_results=[ToolResult(call["name"], call["id"], '"PRIVATE_CLIENT_RESULT"')],
        )
        assert result.status == RunStatus.SUCCESS
        remote.assert_awaited_once()
        async with reader.follow(result.run_id, tenant_id="A", include_payloads=False) as feed:
            items = [item async for item in feed]
        assert isinstance(items[-1], FollowEnd) and items[-1].reason == "terminal"
        events = [item for item in items if isinstance(item, StoredEvent)]
        completed_names = [
            item.data["tool_name"] for item in events if item.event_type == "tool.completed"
        ]
        assert completed_names == ["z_server", "<mcp_tool>", "a_client"], (
            "runtime declarations must authorize only the code-owned server/client names"
        )
        historical_pause = next(item for item in events if item.event_type == "run.paused")
        assert historical_pause.data["pending_tool_calls"][0]["name"] == "a_client"
        import json

        shown = json.dumps([item.data for item in events])
        for private in (
            remote_name,
            "PRIVATE_PROMPT",
            "PRIVATE_REMOTE_RESULT",
            "PRIVATE_CLIENT_RESULT",
        ):
            assert private not in shown
    finally:
        await reader.close()
        agent._tool_sources = []
        await agent.close()
