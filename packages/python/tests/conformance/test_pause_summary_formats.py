"""The bounded pause projection works on every SQL dialect in the CI store kit."""

import pytest

from dendrux.store import RunStore
from dendrux.types import generate_ulid


@pytest.mark.parametrize("version", [1, 2])
async def test_clarification_projection_on_every_dialect(dendrux_store_factory, version):
    store = await dendrux_store_factory()
    run_id = generate_ulid()
    await store.create_run(run_id, "question", tenant_id="t")
    action = {"type": "clarification", "question": "Which file?", "options": ["one", "two"]}
    data = {"agent_name": "question", "schema_version": version, "pending_tool_calls": []}
    if version == 1:
        data["steps"] = [{"action": action}]
    else:
        data["clarification"] = {k: v for k, v in action.items() if k != "type"}
    await store.pause_run(run_id, status="waiting_human_input", pause_data=data)
    pending = await RunStore(store).get_pending(run_id, tenant_id="t")
    assert pending.question == "Which file?"
    assert pending.options == ["one", "two"]
    assert pending.to_dict()["options"] == ["one", "two"]


async def test_public_trace_slots_on_every_dialect(dendrux_store_factory):
    from sqlalchemy import insert, update

    from dendrux._meta_keys import SEED_PREFIX_META_KEY
    from dendrux.db.models import AgentRun, ReactTrace

    store = await dendrux_store_factory()
    run_id = generate_ulid()
    await store.create_run(run_id, "paging", tenant_id="t")
    rows = [
        (0, "seed", "system", "system", {}),
        (1, "seed", "user", "prefix", {SEED_PREFIX_META_KEY: True}),
        (2, "seed", "user", "input", {}),
        (3, "seed", "user", "visible", {SEED_PREFIX_META_KEY: False}),
        (4, "turn", "assistant", "calls", {}),
        (6, "turn", "tool", "fast", {}),
    ]
    async with store._session_factory() as session:
        await session.execute(
            insert(ReactTrace),
            [
                dict(
                    id=generate_ulid(),
                    agent_run_id=run_id,
                    order_index=i,
                    trace_kind=kind,
                    role=role,
                    content=content,
                    meta=meta,
                )
                for i, kind, role, content, meta in rows
            ],
        )
        await session.execute(
            update(AgentRun)
            .where(AgentRun.id == run_id)
            .values(loop_state={"pending": [{"order_index": 5}]})
        )
        await session.commit()
    page = await store.get_public_traces(run_id, tenant_id="t", limit=50)
    assert [r.content for r in page] == ["input", "visible", "calls"]
    assert await store.get_public_traces(run_id, tenant_id="t", limit=1, offset=3) == []
    assert await store.get_public_traces(run_id, tenant_id="other", limit=1) == []
    async with store._session_factory() as session:
        await session.execute(
            insert(ReactTrace).values(
                id=generate_ulid(),
                agent_run_id=run_id,
                order_index=5,
                trace_kind="turn",
                role="tool",
                content="slow",
                meta={},
            )
        )
        await session.execute(
            update(AgentRun).where(AgentRun.id == run_id).values(loop_state={"pending": []})
        )
        await session.commit()
    page = await store.get_public_traces(run_id, tenant_id="t", limit=2, offset=3)
    assert [r.content for r in page] == ["slow", "fast"]
    retry = await store.get_public_traces(run_id, tenant_id="t", limit=50, retry=True)
    assert [r.content for r in retry] == ["calls", "slow", "fast"]


@pytest.mark.parametrize("status", ["success", "error", "cancelled", "max_iterations"])
async def test_terminal_trace_page_ignores_unfinished_reservations(dendrux_store_factory, status):
    from sqlalchemy import insert, update

    from dendrux.db.models import AgentRun, ReactTrace
    from dendrux.runtime._public_traces import public_traces
    from tests._helpers.public_trace_store import KeysetStore

    store = await dendrux_store_factory()
    run_id = generate_ulid()
    await store.create_run(run_id, "terminal trace", tenant_id="t")
    async with store._session_factory() as session:
        await session.execute(
            insert(ReactTrace).values(
                id=generate_ulid(),
                agent_run_id=run_id,
                order_index=4,
                trace_kind="turn",
                role="tool",
                content="FAST",
                meta={},
            )
        )
        await session.execute(
            update(AgentRun)
            .where(AgentRun.id == run_id)
            .values(
                status=status, loop_state={"pending": [{"order_index": 3}], "trace_order_offset": 5}
            )
        )
        await session.commit()
    for reader in (store, KeysetStore(store)):
        page = await public_traces(reader, run_id, tenant_id="t", limit=1, offset=0)
        assert [(row.content, row.order_index) for row in page] == [("FAST", 0)]
