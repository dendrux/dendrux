"""Public pages and subtree limits on SQLite and the private Postgres matrix."""

import warnings

import pytest
from sqlalchemy import event, insert

from dendrux import errors
from dendrux.db.models import AgentRun, LLMInteraction, ReactTrace, RunEvent, ToolCallRecord
from dendrux.runtime.stores._factory import store_for_engine
from dendrux.store import EventFeed, FollowEnd, RunStore, StoredEvent

ROOT = "00000000000000000000000700"


async def seed(engine, model, rows):
    async with engine.begin() as conn:
        await conn.execute(insert(model), rows)


@pytest.mark.parametrize(
    "method,model,count,index",
    [
        ("get_events", RunEvent, 10000, "sequence_index"),
        ("get_traces", ReactTrace, 5000, "order_index"),
        ("get_llm_calls", LLMInteraction, 2000, "iteration_index"),
        ("get_tool_invocations", ToolCallRecord, 2000, "iteration_index"),
    ],
)
async def test_ten_times_realistic_paging(engine, method, model, count, index):
    state = store_for_engine(engine)
    await seed(
        engine, AgentRun, [dict(id=ROOT, agent_name="reader", tenant_id="mine", storage_format=1)]
    )
    extra = {
        RunEvent: {"event_type": "test"},
        ReactTrace: {"role": "user", "content": "test"},
        LLMInteraction: {"input_tokens": 1, "output_tokens": 1},
        ToolCallRecord: {"tool_name": "test", "target": "server", "success": True},
    }[model]
    rows = [dict(id=f"{i:026d}", agent_run_id=ROOT, **{index: i}, **extra) for i in range(count)]
    if model is ToolCallRecord:
        for i, row in enumerate(rows):
            row["tool_call_id"] = f"tool-{i}"
    await seed(engine, model, rows)
    reader = RunStore(state)
    read = getattr(reader, method)
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        seen = []
        while True:
            kwargs = (
                {"after_sequence_index": seen[-1] if seen else None}
                if model is RunEvent
                else {"offset": len(seen)}
            )
            page = await read(ROOT, tenant_id="mine", limit=347, **kwargs)
            if not page:
                break
            seen.extend(
                getattr(r, index if model in (RunEvent, ReactTrace) else "iteration") for r in page
            )
        assert seen == list(range(count))
        assert await read(ROOT, tenant_id="other", limit=10) == []
    with pytest.warns(FutureWarning):
        assert len(await read(ROOT, tenant_id="mine")) == count
    with pytest.warns(FutureWarning):
        assert len(await read(ROOT, tenant_id="mine", limit=2000)) == 1000


@pytest.mark.parametrize("descendants,refuses", [(9999, False), (10000, True)])
async def test_subtree_ten_times_realistic_and_hostile_boundary(
    engine, caplog, descendants, refuses
):
    state = store_for_engine(engine)
    await state.create_run(ROOT, "root", tenant_id="mine")
    await seed(
        engine,
        AgentRun,
        [
            dict(id=f"c{i:025d}", agent_name="child", parent_run_id=ROOT, tenant_id="mine")
            for i in range(descendants)
        ],
    )
    statements = []
    materialized = []

    def capture(conn, cursor, statement, parameters, context, executemany):
        statements.append((statement, parameters))

    def loaded(session, instance):
        if isinstance(instance, AgentRun):
            materialized.append(instance.id)

    from sqlalchemy.orm import Session

    event.listen(engine.sync_engine, "before_cursor_execute", capture)
    event.listen(Session, "loaded_as_persistent", loaded)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            if refuses:
                with pytest.raises(errors.SubtreeTooLargeError) as caught:
                    await RunStore(state).get_subtree(ROOT, tenant_id="mine", max_nodes=10000)
                assert isinstance(caught.value, ValueError)
                assert (caught.value.run_id, caught.value.max_nodes) == (ROOT, 10000)
                assert len(materialized) == 10001  # root plus capped rows; this is a node bound
                assert len(statements) == 2
                assert "LIMIT" in statements[-1][0]
                assert 10000 in statements[-1][1]
                assert not any("llm_interactions" in sql for sql, _ in statements)
                assert any(
                    getattr(r, "code", None) == "subtree_limit"
                    and r.run_id == ROOT
                    and r.limit == 10000
                    for r in caplog.records
                )
            else:
                tree = await RunStore(state).get_subtree(ROOT, tenant_id="mine", max_nodes=10000)
                assert len(tree.nodes) == 10000
                assert tree.root.run.run_id == ROOT
                assert tree.usage.llm_calls == 0
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", capture)
        event.remove(Session, "loaded_as_persistent", loaded)


async def test_event_feed_oversize_batch_drains_all_pages_without_warning(engine):
    state = store_for_engine(engine)
    await state.create_run(ROOT, "root")
    await seed(
        engine,
        RunEvent,
        [
            dict(id=f"e{i:025d}", agent_run_id=ROOT, sequence_index=i, event_type="test")
            for i in range(2501)
        ],
    )
    await state.finalize_run(ROOT, status="success")
    reader = RunStore(state)
    feed = EventFeed(reader, batch=2000)
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        async with feed.follow(ROOT, tenant_id=None) as follow:
            items = [item async for item in follow]
    assert feed.batch == feed._params.batch == 1000
    assert [i.sequence_index for i in items if isinstance(i, StoredEvent)] == list(range(2501))
    assert items[-1] == FollowEnd("terminal", 2500)
    await reader.close()


async def test_format_two_clamp_keeps_shared_reconstruction_unclamped(engine):
    from dendrux.runtime.replay import conversation_records

    state = store_for_engine(engine)
    await state.create_run(ROOT, "reader", tenant_id="mine")
    await seed(
        engine,
        ReactTrace,
        [
            dict(id=f"t{i:025d}", agent_run_id=ROOT, role="user", content=str(i), order_index=i)
            for i in range(5000)
        ],
    )
    reader = RunStore(state)
    with pytest.warns(FutureWarning):
        page = await reader.get_traces(ROOT, tenant_id="mine", limit=2000, offset=17)
    assert [r.content for r in page] == [str(i) for i in range(17, 1017)]
    with pytest.warns(FutureWarning):
        assert len(await reader.get_traces(ROOT, tenant_id="mine")) == 5000
    for limit in (0, -1, 1, 1000):
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            assert len(await reader.get_traces(ROOT, tenant_id="mine", limit=limit)) == max(
                1, limit
            )
    reconstructed = await conversation_records(
        state, ROOT, tenant_id="mine", scoped=True, limit=2000, offset=17
    )
    assert [r.content for r in reconstructed] == [str(i) for i in range(17, 2017)]


async def test_subtree_bound_excludes_root_in_a_cycle(engine):
    from sqlalchemy import update

    state = store_for_engine(engine)
    await state.create_run(ROOT, "root", tenant_id="mine")
    await state.create_run(
        "00000000000000000000000701", "child", parent_run_id=ROOT, tenant_id="mine"
    )
    async with engine.begin() as conn:
        await conn.execute(
            update(AgentRun)
            .where(AgentRun.id == ROOT)
            .values(parent_run_id="00000000000000000000000701")
        )
    tree = await RunStore(state).get_subtree(ROOT, tenant_id="mine", max_nodes=2)
    assert len(tree.nodes) == 2
    with pytest.warns(FutureWarning):
        assert len((await RunStore(state).get_subtree(ROOT, tenant_id="mine")).nodes) == 2


async def test_live_audit_helpers_keep_full_history(engine, tmp_path, monkeypatch):
    from types import SimpleNamespace

    from tests.live.test_images_live import Ctx

    state = store_for_engine(engine)
    await state.create_run(ROOT, "audit")
    for model, fields in [
        (RunEvent, lambda i: dict(event_type="test", sequence_index=i)),
        (LLMInteraction, lambda i: dict(iteration_index=i, input_tokens=1, output_tokens=1)),
    ]:
        await seed(
            engine,
            model,
            [dict(id=f"{i:026d}", agent_run_id=ROOT, **fields(i)) for i in range(1001)],
        )
    ctx = Ctx(SimpleNamespace(family="test", route=SimpleNamespace(mode="text")), tmp_path, {})
    monkeypatch.setattr(ctx, "store", lambda: RunStore(state))
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        assert len(await ctx.audit(ROOT)) == 1001
        assert [r.sequence_index for r in await ctx.events(ROOT, ["test"])] == list(range(1001))
