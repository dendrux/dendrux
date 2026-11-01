"""Real writer-hook costs, batch and 50-chunk streaming, enabled and disabled."""

import json

# ruff: noqa: F401, F811
import pytest
from sqlalchemy import event

from dendrux.types import UsageStats
from tests.integration._mysql import mysql_engine, store


@pytest.mark.parametrize("streaming", [False, True])
async def test_mysql_iteration_budget(
    store, mysql_engine, monkeypatch, record_property, tmp_path, streaming
):
    """Count every SQL statement in two consecutive actual nonterminal Agent iterations."""
    notify_enabled = False
    target = 22
    from contextvars import ContextVar
    from functools import wraps

    from dendrux import Agent, tool
    from dendrux.llm.mock import MockLLM
    from dendrux.loops.base import BaseNotifier
    from dendrux.types import LLMResponse, RunStatus, ToolCall

    phase = "startup"
    iteration = 0
    operation = ContextVar("budget_operation", default="unlabelled")
    transactions = []
    active = {}

    def begin(conn):
        transaction = {"phase": phase, "operation": operation.get(), "statements": []}
        transactions.append(transaction)
        active[id(conn)] = transaction

    def capture(conn, cursor, statement, parameters, context, executemany):
        active[id(conn)]["statements"].append(statement)

    def wrap_method(name, method):
        @wraps(method)
        async def wrapped(*args, **kwargs):
            nonlocal phase, iteration
            if name == "get_run_control" and phase in ("startup", "between"):
                iteration += 1
                phase = f"iteration_{iteration}" if iteration <= 2 else "terminal_iteration"
            label = name
            if name == "save_run_event":
                label += ":" + kwargs["event_type"]
            if name == "checkpoint_step":
                trace = kwargs.get("trace")
                label += ":" + (trace.role if trace else "state")
            token = operation.set(label)
            try:
                return await method(*args, **kwargs)
            finally:
                operation.reset(token)

        return wrapped

    for name in (
        "create_run",
        "set_trace_link",
        "save_trace",
        "save_run_event",
        "get_run_control",
        "checkpoint_step",
        "touch_progress",
        "save_llm_interaction",
        "save_usage",
        "start_tool_call",
        "complete_tool_call",
        "finalize_with_event",
        "get_run",
        "update_run_meta",
    ):
        monkeypatch.setattr(store, name, wrap_method(name, getattr(store, name)))

    store.notify_channel = "dendrux_events" if notify_enabled else None

    class Observer(BaseNotifier):
        def trace_link(self, run_id):
            return "00-" + "1" * 32 + "-" + "2" * 16 + "-01"

        async def on_message_appended(self, run_id, message, iteration):
            nonlocal phase
            if message.role.value == "tool":
                phase = "between"

    effects = []

    @tool(idempotent=True)
    async def work() -> str:
        """Perform one bounded server operation."""
        effects.append(phase)
        return "ok"

    usage = UsageStats(
        input_tokens=10, output_tokens=5, total_tokens=15, cost_usd=0.001, cost_source="provider"
    )

    class Model(MockLLM):
        async def complete_stream(self, *args, **kwargs):
            from dendrux.types import StreamEvent, StreamEventType

            response = await self.complete(*args, **kwargs)
            for _ in range(50):
                yield StreamEvent(type=StreamEventType.TEXT_DELTA, text="x")
            yield StreamEvent(type=StreamEventType.DONE, raw=response)

    model = Model(
        [
            LLMResponse(tool_calls=[ToolCall("work")], usage=usage),
            LLMResponse(tool_calls=[ToolCall("work")], usage=usage),
            LLMResponse(text="done", usage=usage),
        ]
    )
    event.listen(mysql_engine.sync_engine, "begin", begin)
    event.listen(mysql_engine.sync_engine, "before_cursor_execute", capture)
    try:
        agent = Agent(
            prompt="p", tools=[work], provider=model, state_store=store, strict_tenancy=False
        )
        if streaming:
            async with agent.stream("go", notifier=Observer()) as stream:
                events = [event async for event in stream]
            result = events[-1].run_result
        else:
            result = await agent.run("go", notifier=Observer())
    finally:
        event.remove(mysql_engine.sync_engine, "begin", begin)
        event.remove(mysql_engine.sync_engine, "before_cursor_execute", capture)
    assert result.status == RunStatus.SUCCESS, result.error
    assert effects == ["iteration_1", "iteration_2"]
    assert model._call_count == 3
    counts = {}
    for transaction in transactions:
        counts.setdefault(transaction["phase"], []).append(
            {"operation": transaction["operation"], "count": len(transaction["statements"])}
        )
    record_property("agent_transaction_statements", json.dumps(counts, sort_keys=True))
    (tmp_path / "agent-statements.json").write_text(json.dumps(transactions, indent=2))
    print(
        f"STREAMING={streaming}; NOTIFIER={notify_enabled}; " + json.dumps(counts, sort_keys=True)
    )
    assert not counts.get("between"), counts
    for name in ("iteration_1", "iteration_2"):
        statements = [s for t in transactions if t["phase"] == name for s in t["statements"]]
        assert sum("pg_notify" in s for s in statements) == (3 if notify_enabled else 0)
        assert (
            sum(t["operation"] == "start_tool_call" for t in transactions if t["phase"] == name)
            == 1
        )
        assert (
            sum(t["operation"] == "complete_tool_call" for t in transactions if t["phase"] == name)
            == 1
        )
        assert len(statements) == target, counts[name]
        assert not any(
            t["operation"] == "touch_progress" for t in transactions if t["phase"] == name
        ), counts[name]


@pytest.mark.parametrize("entry", ["submit", "resume", "resume_stream", "url"])
async def test_mysql_scenario_a_budget(store, mysql_engine, entry, monkeypatch):
    """Scenario A target set before measurement: MySQL/MariaDB <=78."""
    from collections import Counter
    from contextvars import ContextVar
    from functools import wraps

    from dendrux import Agent, tool
    from dendrux.llm.mock import MockLLM
    from dendrux.types import LLMResponse, RunStatus, ToolCall, ToolResult

    target = 74 if mysql_engine.dialect.name == "postgresql" else 78
    operation = ContextVar("scenario_operation", default="outside")
    phase = "run"
    statements = []

    for name in (
        "create_run",
        "save_trace",
        "save_run_event",
        "get_run_control",
        "checkpoint_step",
        "save_llm_interaction",
        "save_usage",
        "start_tool_call",
        "complete_tool_call",
        "finalize_with_event",
        "get_run",
        "update_run_meta",
        "claim_paused_run",
        "submit_and_claim",
        "consume_submission",
        "pause_with_event",
    ):
        original = getattr(store, name)

        def wrap(method, label):
            @wraps(method)
            async def call(*args, **kwargs):
                token = operation.set(label)
                try:
                    return await method(*args, **kwargs)
                finally:
                    operation.reset(token)

            return call

        monkeypatch.setattr(store, name, wrap(original, name))

    @tool(target="client")
    async def client_read() -> str:
        """Read on the client."""
        raise AssertionError("client tool ran on server")

    options = dict(state_store=store)
    if entry == "url":
        options = dict(
            database_url=mysql_engine.url.render_as_string(hide_password=False),
            database_options={
                "isolation_level": "READ COMMITTED",
                "connect_args": {"init_command": "SET time_zone='+00:00'"},
            },
        )
    agent = Agent(
        name="load",
        prompt="p",
        tools=[client_read],
        strict_tenancy=False,
        provider=MockLLM(
            [
                LLMResponse(
                    tool_calls=[ToolCall("client_read", provider_tool_call_id="r00000-0")],
                    model="scripted",
                ),
                LLMResponse(text="done", model="scripted"),
            ]
        ),
        **options,
    )

    def capture(conn, cursor, statement, parameters, context, many):
        statements.append((phase, operation.get(), statement))

    event.listen(mysql_engine.sync_engine, "before_cursor_execute", capture)
    try:
        paused = await agent.run("go r00000")
        assert paused.status == RunStatus.WAITING_CLIENT_TOOL
        phase = "submit"
        call = paused.pending.tool_calls[0]
        result = ToolResult(name=call.name, call_id=call.id, payload='"ok"')
        kwargs = dict(expected_pause_id=paused.pause_id)
        if entry in ("submit", "url"):
            done = await agent.submit_tool_results(paused.run_id, [result], **kwargs)
        elif entry == "resume":
            done = await agent.resume(paused.run_id, tool_results=[result], **kwargs)
        else:
            stream = agent.resume_stream(paused.run_id, tool_results=[result], **kwargs)
            async for _ in stream:
                pass
            done = stream.result
        assert done.status == RunStatus.SUCCESS, done.error
    finally:
        event.remove(mysql_engine.sync_engine, "before_cursor_execute", capture)
        await agent.close()
    if entry != "url":
        print(
            "SCENARIO_A",
            getattr(mysql_engine.dialect, "is_mariadb", False),
            entry,
            dict(Counter((p, op) for p, op, _ in statements)),
        )
        assert len(statements) <= target
        assert not any("SAVEPOINT" in sql.upper() for _, _, sql in statements)
