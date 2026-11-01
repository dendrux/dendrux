"""Adapters may opt out of SQL JSON patch expressions without losing atomicity."""

import asyncio

import pytest

from dendrux.runtime.state import TraceWrite
from dendrux.types import generate_ulid


async def test_none_patch_hook_preserves_rewrite_and_finalizes(store, monkeypatch):
    run_id, trace_id = generate_ulid(), generate_ulid()
    await store.create_run(run_id, "agent", meta={"customer": "kept"})
    await store.checkpoint_step(
        run_id,
        claim_epoch=0,
        loop_state={"v": 1, "phase": "dispatch", "pending": [], "kept": {"nested": True}},
        pii_mapping=None,
        trace=None,
    )
    monkeypatch.setattr(store, "json_patch_expr", lambda *args: None)
    await store.save_trace(
        run_id,
        "system",
        '[{"role":"user","content":"rewritten"}]',
        order_index=1,
        trace_id=trace_id,
        trace_kind="rewrite",
        claim_epoch=0,
    )
    assert (await store.get_run(run_id)).meta == {
        "customer": "kept",
        "dendrux.latest_rewrite": {"trace_id": trace_id, "order_index": 1},
    }
    assert await store.finalize_run_if_status_in(
        run_id,
        status="success",
        allowed_current_statuses=["running"],
        claim_epoch=0,
    )
    assert (await store.get_loop_state(run_id)) == {
        "v": 1,
        "phase": "finalizing",
        "pending": [],
        "kept": {"nested": True},
    }


@pytest.mark.parametrize("engine", ["sqlite_file", "postgres"], indirect=True)
async def test_none_patch_hook_twenty_finalizers_have_one_winner(store, engine, monkeypatch):
    run_id = generate_ulid()
    await store.create_run(run_id, "agent")
    monkeypatch.setattr(store, "json_patch_expr", lambda *args: None)
    results = await asyncio.wait_for(
        asyncio.gather(
            *(
                store.finalize_with_event(
                    run_id,
                    status="success",
                    allowed_current_statuses=["running"],
                    event_type="run.completed",
                    claim_epoch=0,
                    trace=TraceWrite(generate_ulid(), "assistant", "done", 1),
                )
                for _ in range(20)
            ),
            return_exceptions=True,
        ),
        15,
    )
    assert all(not isinstance(result, BaseException) for result in results)
    assert sum(result is not None for result in results) == 1
    assert len(await store.get_run_events(run_id)) == 1
    assert len(await store.get_traces(run_id)) == 1
