"""Real writer-hook costs, batch and 50-chunk streaming, enabled and disabled."""

import json

import pytest
from sqlalchemy import event

from dendrux.types import UsageStats


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("notify_enabled", [False, True])
async def test_real_notifier_iteration_statement_budget(
    store, engine, monkeypatch, record_property, tmp_path, streaming, notify_enabled
):
    """Count every SQL statement in two consecutive actual nonterminal Agent iterations."""
    if engine.dialect.name != "postgresql":
        pytest.skip("The frozen 25-statement boundary is PostgreSQL with NOTIFY")
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
    event.listen(engine.sync_engine, "begin", begin)
    event.listen(engine.sync_engine, "before_cursor_execute", capture)
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
        event.remove(engine.sync_engine, "begin", begin)
        event.remove(engine.sync_engine, "before_cursor_execute", capture)
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
        assert len(statements) <= 25, counts[name]
        assert len(statements) <= 22, counts[name]
        assert len(statements) == (21 if notify_enabled else 18), counts[name]
        assert not any(
            t["operation"] == "touch_progress" for t in transactions if t["phase"] == name
        ), counts[name]
