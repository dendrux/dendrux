"""Parent admission and notifier/trace-link store seams."""

import pytest
from sqlalchemy import update

from dendrux.db.models import AgentRun
from dendrux.types import generate_ulid


@pytest.mark.parametrize("state", ["cancel_requested", "success", "error", "cancelled"])
async def test_cancelled_or_terminal_parent_refuses_children(store, engine, state):
    parent, child = generate_ulid(), generate_ulid()
    await store.create_run(parent, "parent", tenant_id="t")
    values = {"cancel_requested": True} if state == "cancel_requested" else {"status": state}
    async with engine.begin() as conn:
        await conn.execute(update(AgentRun).where(AgentRun.id == parent).values(**values))
    with pytest.raises(RuntimeError) as caught:
        await store.create_run(child, "child", tenant_id="t", parent_run_id=parent)
    assert type(caught.value).__name__ == "RunCancelledError"
    assert await store.get_run(child) is None


async def test_trace_link_is_fenced_and_roundtrips(store):
    run_id = generate_ulid()
    await store.create_run(run_id, "trace")
    assert hasattr(store, "set_trace_link"), "a run needs a durable trace parent"
    link = "00-0123456789abcdef0123456789abcdef-0123456789abcdef-01"
    await store.set_trace_link(run_id, claim_epoch=0, trace_link=link)
    assert (await store.get_run(run_id)).trace_link == link


async def test_event_hook_runs_inside_every_event_transaction(store, monkeypatch):
    calls = []

    async def after(session, run_id):
        assert session.in_transaction()
        calls.append(run_id)

    # The hook must be dynamically dispatched so subclasses and plugins can supply it.
    monkeypatch.setattr(store, "_after_event_insert", after)
    run_id = generate_ulid()
    await store.create_run(run_id, "notify")
    await store.save_run_event(run_id, event_type="run.started")
    await store.finalize_with_event(
        run_id, status="success", allowed_current_statuses=["running"], event_type="run.completed"
    )
    assert calls == [run_id, run_id]
