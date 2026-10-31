"""Call-local read reuse on the actual client pause/submission path."""

from contextvars import ContextVar
from functools import wraps

import pytest
from sqlalchemy import event

from dendrux import Agent, tool
from dendrux.llm.mock import MockLLM
from dendrux.types import LLMResponse, RunStatus, ToolCall, ToolResult


@pytest.mark.parametrize("entry", ["submit", "resume", "resume_stream"])
@pytest.mark.parametrize(
    "phase,operation,maximum",
    [
        ("run", "settings", 1),
        ("run", "get_run", 2),
        ("submit", "conversation_traces", 2),
        ("submit", "get_loop_state", 1),
        ("submit", "get_pause_state", 4),
        ("submit", "submit_and_claim", 4),
        ("submit", "get_run", 4),
        ("submit", "get_tool_calls", 0),
    ],
)
async def test_client_pause_read_budget(
    store, engine, monkeypatch, phase, operation, maximum, entry
):
    current_phase = "run"
    current_operation = ContextVar("trim_operation", default="other")
    counts = {}

    def instrument(name, method):
        @wraps(method)
        async def wrapped(*args, **kwargs):
            token = current_operation.set(name)
            try:
                return await method(*args, **kwargs)
            finally:
                current_operation.reset(token)

        return wrapped

    for name in (
        "get_run",
        "conversation_traces",
        "get_loop_state",
        "get_pause_state",
        "submit_and_claim",
        "get_tool_calls",
    ):
        monkeypatch.setattr(store, name, instrument(name, getattr(store, name)))

    def capture(conn, cursor, sql, parameters, context, many):
        name = "settings" if "dendrux_settings" in sql else current_operation.get()
        key = (current_phase, name)
        counts[key] = counts.get(key, 0) + 1

    @tool(target="client")
    def client_read() -> str:
        """Read on the client."""
        return "unused"

    agent = Agent(
        name="trim",
        prompt="p",
        state_store=store,
        tools=[client_read],
        provider=MockLLM(
            [
                LLMResponse(tool_calls=[ToolCall("client_read", id="client-call")]),
                LLMResponse(text="done"),
            ]
        ),
    )
    event.listen(engine.sync_engine, "before_cursor_execute", capture)
    try:
        paused = await agent.run("go", tenant_id=None)
        assert paused.status == RunStatus.WAITING_CLIENT_TOOL
        current_phase = "submit"
        results = [ToolResult("client_read", "client-call", '"ok"')]
        kwargs = dict(tenant_id=None, expected_pause_id=paused.pause_id)
        if entry == "submit":
            done = await agent.submit_tool_results(paused.run_id, results, **kwargs)
        elif entry == "resume":
            done = await agent.resume(paused.run_id, tool_results=results, **kwargs)
        else:
            stream = agent.resume_stream(paused.run_id, tool_results=results, **kwargs)
            async for _ in stream:
                pass
            done = stream.result
        assert done.status == RunStatus.SUCCESS
        assert done.recovery is None
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", capture)
        await agent.close()
    if entry != "submit" and operation == "get_pause_state":
        maximum = 2
    assert counts.get((phase, operation), 0) <= maximum, counts


async def test_format_cache_observes_activation_and_keeps_store_isolation(engine):
    from dendrux.runtime.state import SQLAlchemyStateStore
    from tests._fixtures.storage import pin_storage_format

    await pin_storage_format(engine, 1)
    first = SQLAlchemyStateStore(engine)
    assert await first._creation_storage_format() == 1
    await pin_storage_format(engine, 2)
    assert await first._creation_storage_format() == 2
    statements = []

    def capture(conn, cursor, sql, *args):
        statements.append(sql)

    event.listen(engine.sync_engine, "before_cursor_execute", capture)
    try:
        assert await first._creation_storage_format() == 2
        assert statements == []
        # A second store must discover its database's pin independently.
        assert await SQLAlchemyStateStore(engine)._creation_storage_format() == 2
        assert len(statements) == 1
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", capture)


async def test_submissions_rebuild_fresh_history_on_independent_stores(engine):
    from dendrux.runtime.state import SQLAlchemyStateStore

    @tool(target="client")
    def client_read() -> str:
        """Read on the client."""
        return "unused"

    first_llm = MockLLM(
        [
            LLMResponse(tool_calls=[ToolCall("client_read", id="first")]),
            LLMResponse(text="done"),
        ]
    )
    second_llm = MockLLM([LLMResponse(tool_calls=[ToolCall("client_read", id="second")])])
    agents = [
        Agent(
            name="trim",
            prompt="p",
            state_store=SQLAlchemyStateStore(engine),
            tools=[client_read],
            provider=provider,
        )
        for provider in (first_llm, second_llm)
    ]
    try:
        first = await agents[0].run("go", tenant_id=None)
        second = await agents[1].submit_tool_results(
            first.run_id,
            [ToolResult("client_read", "first", '"one"')],
            expected_pause_id=first.pause_id,
            tenant_id=None,
        )
        assert second.status == RunStatus.WAITING_CLIENT_TOOL
        assert second.pause_id != first.pause_id
        done = await agents[0].submit_tool_results(
            first.run_id,
            [ToolResult("client_read", "second", '"two"')],
            expected_pause_id=second.pause_id,
            tenant_id=None,
        )
        assert done.status == RunStatus.SUCCESS
        records = await agents[0]._state_store.get_traces(first.run_id)
        observations = [t for t in records if t.role == "tool"]
        assert len(observations) == 2
        assert "one" in observations[0].content and "two" in observations[1].content
    finally:
        for agent in agents:
            await agent.close()
