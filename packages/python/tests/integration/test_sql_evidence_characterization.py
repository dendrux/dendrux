"""Tool evidence rolls back if its event boundary fails after inserting the event."""

import pytest
from sqlalchemy import text

from dendrux.errors import PersistenceError
from dendrux.media import BlobEdge
from dendrux.runtime._blob_catalog import BlobServices
from dendrux.runtime.stores.sql import base as sql_base
from dendrux.testing.store_kit.contracts._support import (
    complete_call_kwargs,
    start_call_kwargs,
    tool_call_kwargs,
    trace_write,
)
from dendrux.types import generate_ulid
from tests.unit.test_blob_catalog import make_png, store_png


@pytest.mark.parametrize("operation", ["start", "complete"])
async def test_event_failure_rolls_back_tool_evidence_and_checkpoint(store, monkeypatch, operation):
    run_id = generate_ulid()
    await store.create_run(run_id, "evidence-boundary")
    pending = {"v": 1, "phase": "dispatch", "pending": [{"call_id": "a"}]}
    await store.checkpoint_step(
        run_id, claim_epoch=0, loop_state=pending, pii_mapping=None, trace=None
    )
    if operation == "complete":
        await store.start_tool_call(run_id, claim_epoch=0, **start_call_kwargs("a"))
    original = sql_base._write_event
    calls = []

    async def fail_after_event(session, event_run_id, event):
        calls.append((event_run_id, event.event_type))
        await original(session, event_run_id, event)
        raise ValueError("forced event boundary")

    monkeypatch.setattr(sql_base, "_write_event", fail_after_event)
    with pytest.raises(ValueError, match="forced event boundary"):
        if operation == "start":
            await store.start_tool_call(run_id, claim_epoch=0, **start_call_kwargs("a"))
        else:
            await store.complete_tool_call(run_id, claim_epoch=0, **complete_call_kwargs("a", 1))
    assert calls == [(run_id, "tool.started" if operation == "start" else "tool.completed")]
    assert await store.get_loop_state(run_id) == pending
    assert await store.get_traces(run_id) == []
    rows = await store.get_tool_calls(run_id)
    assert [(row.tool_call_id, row.outcome) for row in rows] == (
        [] if operation == "start" else [("a", "started")]
    )
    events = await store.get_run_events(run_id)
    assert [event.event_type for event in events] == (
        [] if operation == "start" else ["tool.started"]
    )


async def test_tool_call_digest_refusal_and_blob_edge_publish_together(store, engine):
    run_id = generate_ulid()
    await store.create_run(run_id, "tool-evidence")
    kwargs = tool_call_kwargs(tool_call_id="saved", row_id=generate_ulid())
    with pytest.raises(PersistenceError) as caught:
        await store.save_tool_call(run_id, **kwargs, content_digest="different")
    assert caught.value.code == "write_conflict"
    assert await store.get_tool_calls(run_id) == []
    services = BlobServices.for_state_store(store)
    saved = await store_png(services, "-", make_png(214))
    await store.save_tool_call(
        run_id,
        **kwargs,
        blob_refs=[BlobEdge("-", saved.ref.sha256, "tool_call", "saved", 0)],
        blob_refs_scope="-",
    )
    [row] = await store.get_tool_calls(run_id)
    assert (row.tool_call_id, row.outcome, row.success) == ("saved", "completed", True)
    async with engine.connect() as connection:
        count = await connection.scalar(text("SELECT count(*) FROM dendrux_blob_refs"))
    assert count == 1


async def test_completion_retains_raw_payload_and_counts_its_named_tool(store):
    run_id = generate_ulid()
    await store.create_run(run_id, "raw-tool-result")
    await store.checkpoint_step(
        run_id,
        claim_epoch=0,
        loop_state={"v": 1, "phase": "dispatch", "pending": [{"call_id": "a"}]},
        pii_mapping=None,
        trace=None,
    )
    await store.start_tool_call(run_id, claim_epoch=0, **start_call_kwargs("a"))
    await store.complete_tool_call(
        run_id,
        claim_epoch=0,
        **complete_call_kwargs(
            "a",
            1,
            result_payload="not-json",
            observation=trace_write(
                "tool", "not-json", 1, meta={"call_id": "a", "tool_name": "named_tool"}
            ),
        ),
    )
    [row] = await store.get_tool_calls(run_id)
    assert row.result == {"raw": "not-json"}
    state = await store.get_loop_state(run_id)
    assert state["pending"] == []
    assert state["tool_counts"] == {"named_tool": 1}
    assert [event.event_type for event in await store.get_run_events(run_id)] == [
        "tool.started",
        "tool.completed",
    ]
