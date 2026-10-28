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
