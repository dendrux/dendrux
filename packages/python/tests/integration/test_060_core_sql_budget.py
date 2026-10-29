"""Statement costs of the store transactions consumed by a recovered iteration."""

import json

import pytest
from sqlalchemy import event, text

from dendrux.runtime.state import EventWrite, LoopStatePatch, TraceWrite
from dendrux.types import UsageStats, generate_ulid


async def test_transaction_statement_costs(store, engine, monkeypatch, record_property):
    if engine.dialect.name != "postgresql":
        pytest.skip("The frozen 25-statement boundary is PostgreSQL with NOTIFY")

    async def notify(session, run_id):
        await session.execute(
            text("SELECT pg_notify(:channel, :run_id)"),
            {"channel": "dendrux_events", "run_id": run_id},
        )

    monkeypatch.setattr(store, "_after_event_insert", notify)
    groups = {}
    current = "creation"

    def capture(conn, cursor, statement, parameters, context, executemany):
        groups.setdefault(current, []).append(statement)

    event.listen(engine.sync_engine, "before_cursor_execute", capture)
    run_id = generate_ulid()

    def ev(kind):
        return EventWrite(generate_ulid(), kind, 1, {}, None, None)

    try:
        await store.create_run(run_id, "cost", strategy="react")
        current = "assistant_checkpoint"
        await store.checkpoint_step(
            run_id,
            claim_epoch=0,
            loop_state={"v": 1, "phase": "dispatch", "pending": [{"call_id": "c"}]},
            pii_mapping=None,
            trace=TraceWrite(generate_ulid(), "assistant", "search", 1),
        )
        current = "intent"
        await store.start_tool_call(
            run_id,
            claim_epoch=0,
            row_id=generate_ulid(),
            tool_call_id="c",
            provider_tool_call_id=None,
            tool_name="search",
            target="server",
            params={},
            iteration_index=1,
            idempotent=True,
            event=ev("tool.started"),
        )
        current = "completion"
        await store.complete_tool_call(
            run_id,
            claim_epoch=0,
            tool_call_id="c",
            outcome="completed",
            result_payload='"ok"',
            success=True,
            duration_ms=1,
            error_message=None,
            replayed=False,
            observation=TraceWrite(generate_ulid(), "tool", '"ok"', 2),
            event=ev("tool.completed"),
            loop_state_patch=LoopStatePatch(remove_pending=("c",), set_phase="model_next"),
        )
        current = "interaction_and_usage"
        await store.save_llm_interaction(
            run_id,
            claim_epoch=0,
            interaction_id=generate_ulid(),
            iteration_index=1,
            usage=UsageStats(),
        )
        await store.save_usage(
            run_id, claim_epoch=0, usage_id=generate_ulid(), iteration_index=1, usage=UsageStats()
        )
        current = "llm_completed"
        await store.save_run_event(
            run_id, claim_epoch=0, event_id=generate_ulid(), event_type="llm.completed"
        )
        current = "progress"
        for _ in range(4):
            assert await store.touch_progress(run_id, claim_epoch=0)
        current = "control_reads"
        for _ in range(2):
            await store.get_run_control(run_id)
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", capture)
    counts = {name: len(statements) for name, statements in groups.items()}
    record_property("transaction_statements", json.dumps(counts, sort_keys=True))
    assert counts["assistant_checkpoint"] == 2
    assert counts["intent"] == 4
    assert counts["completion"] == 5
    assert sum(count for name, count in counts.items() if name != "creation") <= 25


@pytest.mark.parametrize("streaming", [False, True])
async def test_agent_iteration_statement_budget(
    store, engine, monkeypatch, record_property, tmp_path, streaming
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

    async def notify(session, run_id):
        assert store.notify_channel == "dendrux_events"
        await session.execute(
            text("SELECT pg_notify(:channel, :run_id)"),
            {"channel": store.notify_channel, "run_id": run_id},
        )

    monkeypatch.setattr(store, "_after_event_insert", notify)

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
    print("AGENT_STATEMENT_COUNTS=" + json.dumps(counts, sort_keys=True))
    assert not counts.get("between"), counts
    for name in ("iteration_1", "iteration_2"):
        statements = [s for t in transactions if t["phase"] == name for s in t["statements"]]
        assert sum("pg_notify" in s for s in statements) == 3
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
        assert not any(
            t["operation"] == "touch_progress" for t in transactions if t["phase"] == name
        ), counts[name]


@pytest.mark.parametrize("operation", ["checkpoint", "intent", "completion", "event"])
async def test_fenced_writes_refresh_progress_in_the_same_transaction(store, engine, operation):
    from datetime import UTC, datetime, timedelta

    from sqlalchemy import update

    from dendrux.db.models import AgentRun

    run_id = generate_ulid()
    await store.create_run(run_id, "progress")
    await store.checkpoint_step(
        run_id,
        claim_epoch=0,
        pii_mapping=None,
        trace=None,
        loop_state={
            "v": 1,
            "phase": "dispatch",
            "pending": [
                {"call_id": "c", "tool": "work", "phase": "dispatchable", "order_index": 1}
            ],
        },
    )

    async def intent():
        await store.start_tool_call(
            run_id,
            claim_epoch=0,
            row_id=generate_ulid(),
            tool_call_id="c",
            provider_tool_call_id=None,
            tool_name="work",
            target="server",
            params=None,
            iteration_index=1,
            idempotent=True,
            event=EventWrite(generate_ulid(), "tool.started", 1, {}, None, None),
        )

    if operation == "completion":
        await intent()
    cutoff = datetime.now(UTC) - timedelta(seconds=5)
    old = cutoff - timedelta(hours=1)
    async with engine.begin() as conn:
        await conn.execute(
            update(AgentRun).where(AgentRun.id == run_id).values(last_progress_at=old)
        )
    if operation == "checkpoint":
        await store.checkpoint_step(
            run_id, claim_epoch=0, loop_state={"v": 1}, pii_mapping=None, trace=None
        )
    elif operation == "intent":
        await intent()
    elif operation == "completion":
        await store.complete_tool_call(
            run_id,
            claim_epoch=0,
            tool_call_id="c",
            outcome="completed",
            result_payload='"ok"',
            success=True,
            duration_ms=1,
            error_message=None,
            replayed=False,
            observation=TraceWrite(generate_ulid(), "tool", '"ok"', 1),
            event=EventWrite(generate_ulid(), "tool.completed", 1, {}, None, None),
            loop_state_patch=LoopStatePatch(remove_pending=("c",), set_phase="model_next"),
        )
    else:
        await store.save_run_event(run_id, claim_epoch=0, event_type="llm.completed")
    assert (await store.get_run(run_id)).last_progress_at > cutoff


@pytest.mark.parametrize("storage_format,epoch", [(1, 0), (2, None)])
async def test_non_execution_writes_keep_explicit_progress_policy(
    store, engine, storage_format, epoch
):
    from datetime import UTC, datetime

    from sqlalchemy import update

    from dendrux.db.models import AgentRun
    from tests._fixtures.storage import pin_storage_format

    await pin_storage_format(engine, storage_format)
    run_id = generate_ulid()
    await store.create_run(run_id, "legacy-progress")
    old = datetime(2020, 1, 1, tzinfo=UTC)
    async with engine.begin() as conn:
        await conn.execute(
            update(AgentRun).where(AgentRun.id == run_id).values(last_progress_at=old)
        )
    await store.save_run_event(run_id, claim_epoch=epoch, event_type="llm.completed")
    assert (await store.get_run(run_id)).last_progress_at == old


@pytest.mark.parametrize("fmt", [1, 2])
@pytest.mark.parametrize("same_run", [False, True])
@pytest.mark.parametrize("same_store", [False, True])
@pytest.mark.parametrize("failed", [False, True])
async def test_only_committed_owner_fences_refresh_local_lease(
    store, engine, monkeypatch, fmt, same_run, same_store, failed
):
    from dendrux.runtime.context import (
        DelegationContext,
        reset_delegation_context,
        set_delegation_context,
    )
    from dendrux.runtime.state import ClaimResult, claimed_store
    from tests._fixtures.storage import pin_storage_format

    await pin_storage_format(engine, fmt)
    run = generate_ulid()
    await store.create_run(run, "lease")
    view = claimed_store(store, run, ClaimResult(0, fmt, None, None))
    ctx = DelegationContext(
        run_id=run if same_run else "other",
        persisted=True,
        state_store=view if same_store else object(),
    )
    if failed:

        async def reject(*args, **kwargs):
            raise RuntimeError("failed fence")

        monkeypatch.setattr(store, "checkpoint_step", reject)
    token = set_delegation_context(ctx)
    try:
        if failed:
            with pytest.raises(RuntimeError, match="failed fence"):
                await view.checkpoint_step(run, loop_state={}, pii_mapping=None, trace=None)
        else:
            await view.checkpoint_step(run, loop_state={}, pii_mapping=None, trace=None)
        assert (ctx._lease_at[0] > 0) == (fmt == 2 and same_run and same_store and not failed)
    finally:
        reset_delegation_context(token)
