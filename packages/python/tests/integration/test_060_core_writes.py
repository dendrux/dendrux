"""Retry identities and fencing on the SQL boundary, on both supported backends."""

import inspect

import pytest
from sqlalchemy import select, update

from dendrux.db.models import (
    AgentRun,
    LLMInteraction,
    ReactTrace,
    RunEvent,
    TokenUsage,
    ToolCallRecord,
)
from dendrux.errors import PersistenceError, RunFencedError
from dendrux.runtime.state import EventWrite, LoopStatePatch, TraceWrite, claimed_store
from dendrux.types import UsageStats, generate_ulid


def writes():
    return [
        (
            "save_trace",
            ("user", "hello"),
            {"order_index": 0},
            "trace_id",
            ReactTrace,
            {"content": "different"},
        ),
        (
            "save_tool_call",
            (),
            dict(
                tool_call_id="call",
                provider_tool_call_id=None,
                tool_name="search",
                target="server",
                params={"q": "x"},
                result_payload='"ok"',
                success=True,
                duration_ms=1,
                iteration_index=0,
            ),
            "row_id",
            ToolCallRecord,
            {"result_payload": '"changed"'},
        ),
        (
            "save_run_event",
            (),
            {"event_type": "llm.completed", "data": {"input_tokens": 1}},
            "event_id",
            RunEvent,
            {"data": {"input_tokens": 2}},
        ),
        (
            "save_usage",
            (),
            {"iteration_index": 0, "usage": UsageStats(input_tokens=1)},
            "usage_id",
            TokenUsage,
            {"usage": UsageStats(input_tokens=2)},
        ),
        (
            "save_llm_interaction",
            (),
            {"iteration_index": 0, "usage": UsageStats(), "semantic_request": {"messages": []}},
            "interaction_id",
            LLMInteraction,
            {"semantic_request": {"messages": ["changed"]}},
        ),
    ]


@pytest.mark.parametrize("method,args,kwargs,id_key,model,changed", writes())
async def test_a21_a22_insert_retry_identity(
    store, engine, method, args, kwargs, id_key, model, changed
):
    run_id, row_id = generate_ulid(), generate_ulid()
    await store.create_run(run_id, "retry")
    target = getattr(store, method)
    assert id_key in inspect.signature(target).parameters, "caller must own the retry identity"
    first = await target(run_id, *args, **kwargs, **{id_key: row_id})
    second = await target(run_id, *args, **kwargs, **{id_key: row_id})
    assert first == second
    async with engine.connect() as conn:
        ids = (
            (await conn.execute(select(model.id).where(model.agent_run_id == run_id)))
            .scalars()
            .all()
        )
    assert ids == [row_id]
    if method == "save_trace":
        args = ("user", changed["content"])
        changed = {}
    with pytest.raises(PersistenceError) as caught:
        await target(run_id, *args, **{**kwargs, **changed}, **{id_key: row_id})
    assert caught.value.code == "write_conflict"


@pytest.mark.parametrize("method,args,kwargs,id_key,model,changed", writes())
async def test_a3_old_claim_cannot_insert(
    store, engine, method, args, kwargs, id_key, model, changed
):
    run_id = generate_ulid()
    await store.create_run(run_id, "fence")
    old = claimed_store(store, run_id, 0)
    async with engine.begin() as conn:
        await conn.execute(update(AgentRun).where(AgentRun.id == run_id).values(claim_epoch=1))
    with pytest.raises(RunFencedError):
        await getattr(old, method)(run_id, *args, **kwargs)
    async with engine.connect() as conn:
        assert not (await conn.execute(select(model.id).where(model.agent_run_id == run_id))).all()


async def test_a3_touch_progress_sql_bool_claimed_raises(store, engine):
    run_id = generate_ulid()
    await store.create_run(run_id, "fence")
    assert "claim_epoch" in inspect.signature(store.touch_progress).parameters
    assert await store.touch_progress(run_id, claim_epoch=0) is True
    assert await store.touch_progress(run_id, claim_epoch=1) is False
    with pytest.raises(RunFencedError):
        await claimed_store(store, run_id, 1).touch_progress(run_id)


async def test_every_claimed_write_is_fenced(store, engine):
    run_id = generate_ulid()
    await store.create_run(run_id, "fence")
    old = claimed_store(store, run_id, 0)
    async with engine.begin() as conn:
        await conn.execute(update(AgentRun).where(AgentRun.id == run_id).values(claim_epoch=1))
    event = EventWrite(generate_ulid(), "tool.started", 0, {}, None, None)
    observation = TraceWrite(generate_ulid(), "tool", "1", 0)
    calls = [
        ("add_note", (), {"note_id": generate_ulid(), "text": "note"}),
        ("mark_notes_delivered", ([generate_ulid()],), {"iteration": 0}),
        ("update_run_meta", ({"dendrux.test": True},), {}),
        (
            "checkpoint_step",
            (),
            {"loop_state": {"phase": "dispatch"}, "pii_mapping": None, "trace": None},
        ),
        (
            "set_trace_link",
            (),
            {"trace_link": "00-0123456789abcdef0123456789abcdef-0123456789abcdef-01"},
        ),
        (
            "start_tool_call",
            (),
            dict(
                row_id=generate_ulid(),
                tool_call_id="c",
                provider_tool_call_id=None,
                tool_name="search",
                target="server",
                params={},
                iteration_index=0,
                idempotent=True,
                event=event,
            ),
        ),
        (
            "complete_tool_call",
            (),
            dict(
                tool_call_id="c",
                outcome="completed",
                result_payload="1",
                success=True,
                duration_ms=1,
                error_message=None,
                replayed=False,
                observation=observation,
                event=event,
                loop_state_patch=LoopStatePatch(remove_pending=("c",)),
            ),
        ),
        (
            "consume_submission",
            (),
            dict(submission_id=generate_ulid(), traces=[], patch=LoopStatePatch()),
        ),
    ]
    for name, args, kwargs in calls:
        with pytest.raises(RunFencedError):
            await getattr(old, name)(run_id, *args, **kwargs)


async def test_a17_format_setting_read_inside_creation_transaction(store, engine):
    from sqlalchemy import event

    seen = []

    def capture(conn, cursor, statement, parameters, context, executemany):
        seen.append(statement)

    event.listen(engine.sync_engine, "before_cursor_execute", capture)
    try:
        created = await store.create_run(generate_ulid(), "format")
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", capture)
    assert created.storage_format == 2
    assert len([s for s in seen if "dendrux_settings" in s]) == 1


async def test_foreign_key_failure_is_not_a_retry_content_conflict(store):
    with pytest.raises(PersistenceError) as caught:
        await store.save_trace(
            generate_ulid(), "user", "ok", order_index=0, trace_id=generate_ulid()
        )
    assert caught.value.code == "persistence_error"


async def test_a22_different_identity_at_same_natural_key_conflicts(store):
    run_id = generate_ulid()
    await store.create_run(run_id, "natural")
    await store.save_trace(run_id, "user", "ok", order_index=0, trace_id=generate_ulid())
    with pytest.raises(PersistenceError) as caught:
        await store.save_trace(run_id, "user", "ok", order_index=0, trace_id=generate_ulid())
    assert caught.value.code == "write_conflict"
