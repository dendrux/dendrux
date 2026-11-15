"""Tree reads preserve bounded projections, keysets, ancestry and live usage costs."""

import json
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import event, update
from sqlalchemy.ext.asyncio import AsyncSession

from dendrux.db.models import AgentRun
from dendrux.errors import BudgetConfigurationError
from dendrux.runtime.state import UNSCOPED
from dendrux.runtime.stores.sql.base import SQLStateStore
from dendrux.types import UsageStats
from tests.integration.test_sql_managed_characterization import _transaction_trace

IDS = [f"{i:026d}" for i in range(1, 8)]


@contextmanager
def _read_trace(engine, record_property):
    parameters = []

    def observe(conn, cursor, sql, values, context, many):
        parameters.append(values)

    event.listen(engine.sync_engine, "before_cursor_execute", observe)
    try:
        with _transaction_trace(engine) as trace:
            yield trace
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", observe)
        record_property("parameters", json.dumps(parameters, default=str, sort_keys=True))


async def _seed(store, engine, *, foreign=False):
    root, a, b, a1, a11, other, hidden = IDS
    for run_id, parent in [(root, None), (a, root), (b, root), (a1, a), (a11, a1)]:
        await store.create_run(run_id, "tree", tenant_id="t1", parent_run_id=parent)
    for run_id, inp, out, cost in [(root, 1, 2, 0.1), (a, 2, 3, None), (a11, 4, 5, 0.2)]:
        await store.save_llm_interaction(
            run_id,
            iteration_index=0,
            usage=UsageStats(
                input_tokens=inp,
                output_tokens=out,
                cost_usd=cost,
                cache_read_input_tokens=2,
                cache_creation_input_tokens=3,
                reasoning_tokens=1,
            ),
        )
    if foreign:
        await store.create_run(other, "other", tenant_id="t2")
        await store.create_run(hidden, "hidden", tenant_id="t1")
        async with AsyncSession(engine) as session:
            await session.execute(
                update(AgentRun).where(AgentRun.id == other).values(parent_run_id=root)
            )
            await session.execute(
                update(AgentRun).where(AgentRun.id == hidden).values(parent_run_id=other)
            )
            await session.commit()


def _pin(record_property, statements, boundaries):
    assert statements
    assert all(sql.startswith(("SELECT", "WITH RECURSIVE")) for sql in statements)
    assert all("FOR UPDATE" not in sql and "FOR SHARE" not in sql for sql in statements)
    assert boundaries == ["begin", "rollback"]
    record_property("sql", json.dumps(statements))
    record_property("boundaries", json.dumps(boundaries))


async def test_children_keyset_deduplicates_frontier_and_caps_query(
    store, engine, monkeypatch, record_property
):
    await _seed(store, engine, foreign=True)
    root, a, b, a1, *_ = IDS
    original = store.child_page_query
    plans = []

    def plan(**kwargs):
        plans.append(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(store, "child_page_query", plan)
    with _read_trace(engine, record_property) as (statements, boundaries, connections):
        page = await store.get_children_page([a, root, root], after=b, limit=9000, tenant_id="t1")
    assert page == [a1]
    assert len(statements) == len(connections) == len(plans) == 1
    assert plans[0]["page_size"] == 500
    assert [(p.run_id, p.after) for p in plans[0]["parents"]] == [(root, b), (a, b)]
    assert plans[0]["tenant_scope"] == "t1"
    assert "LIMIT" in statements[0] and "ORDER BY" in statements[0]
    _pin(record_property, statements, boundaries)
    with _read_trace(engine, record_property) as (empty, _, _):
        assert await store.get_children_page([], after=None, limit=1, tenant_id="t1") == []
    assert empty == [] and len(plans) == 1


async def test_recursive_descendant_keyset_is_cycle_safe_and_tenant_pruned(
    store, engine, record_property
):
    await _seed(store, engine, foreign=True)
    root, a, b, a1, a11, *_ = IDS
    async with AsyncSession(engine) as session:
        await session.execute(update(AgentRun).where(AgentRun.id == root).values(parent_run_id=a11))
        await session.commit()
    with _read_trace(engine, record_property) as (first, first_tx, _):
        page = await store.get_descendant_page(root, after=None, limit=2, tenant_id="t1")
    assert page == [a, b]
    assert len(first) == 1
    with _read_trace(engine, record_property) as (second, boundaries, _):
        next_page = await store.get_descendant_page(
            root, after=page[-1], limit=9000, tenant_id="t1"
        )
    assert next_page == [a1, a11]
    assert len(second) == 1
    assert "WITH RECURSIVE descendants" in second[0]
    assert "UNION ALL" not in second[0]
    assert "descendants.id >" in second[0] and "ORDER BY descendants.id" in second[0]
    assert "pause_data" not in second[0] and "loop_state" not in second[0]
    _pin(record_property, first, first_tx)
    record_property("second_sql", json.dumps(second))
    assert boundaries == first_tx


@pytest.mark.parametrize(
    "rows,depth,ancestors,children,complete",
    [
        (3, 10, [IDS[0], IDS[1]], [], True),
        (10, 1, [IDS[1]], [IDS[4]], False),
    ],
)
async def test_delegation_shares_row_budget_and_uses_depth_lookahead(
    store, engine, record_property, rows, depth, ancestors, children, complete
):
    await _seed(store, engine)
    with _read_trace(engine, record_property) as (statements, boundaries, _):
        result = await store.get_delegation_info(IDS[3], max_rows=rows, max_depth=depth)
    assert [r.run_id for r in result.ancestry] == ancestors
    assert [r.run_id for r in result.children] == children
    assert result.ancestry_complete is complete and result.truncated is True
    assert result.parent.run_id == IDS[1] and result.parent.resolved is True
    assert 1 + len(result.ancestry) + result.subtree_summary.descendant_count <= rows
    assert len(statements) == 4
    assert all(
        "pause_data" not in sql and "loop_state" not in sql and "input_data" not in sql
        for sql in statements
    )
    assert all("LIMIT" in sql for sql in statements if "ORDER BY" in sql)
    _pin(record_property, statements, boundaries)


async def test_ancestry_cycle_and_missing_parent_are_incomplete(store, engine, record_property):
    await _seed(store, engine)
    async with AsyncSession(engine) as session:
        await session.execute(
            update(AgentRun).where(AgentRun.id == IDS[0]).values(parent_run_id=IDS[3])
        )
        await session.commit()
    async with AsyncSession(engine) as session:
        row = await session.get(AgentRun, IDS[3])
        with _read_trace(engine, record_property) as (statements, boundaries, _):
            ancestry, complete = await store._walk_ancestry(session, row)
        assert [r.run_id for r in ancestry] == [IDS[0], IDS[1]]
        assert complete is False and len(statements) == 2
        missing, missing_complete = await store._walk_ancestry(
            session, SimpleNamespace(id=IDS[3], parent_run_id="9" * 26)
        )
        assert missing == [] and missing_complete is False
    assert boundaries == []
    assert all("FOR UPDATE" not in sql for sql in statements)
    record_property("sql", json.dumps(statements))


async def test_usage_groups_by_real_chunk_boundary_and_counts_unpriced_calls(
    store, engine, record_property
):
    await _seed(store, engine)
    requested = [IDS[0], IDS[1], IDS[4]] + [f"{i:026d}" for i in range(100, 598)]
    assert len(requested) == 501
    with _read_trace(engine, record_property) as (statements, boundaries, _):
        async with store._session_factory() as session:
            usage = await store._usage_by_run(session, requested)
    assert set(usage) == {IDS[0], IDS[1], IDS[4]}
    assert usage[IDS[0]].input_tokens == 1
    assert usage[IDS[1]].unpriced_calls == 1
    assert usage[IDS[4]].known_cost_usd == pytest.approx(0.2)
    assert len(statements) == 2
    assert all("GROUP BY llm_interactions.agent_run_id" in sql for sql in statements)
    assert all("agent_runs" not in sql for sql in statements)
    _pin(record_property, statements, boundaries)


@pytest.mark.parametrize("operation,count", [("subtree_usage", 5), ("get_subtree", 6)])
async def test_live_subtree_usage_ignores_stale_totals_and_keeps_bfs_cost(
    store, engine, record_property, operation, count
):
    await _seed(store, engine)
    async with AsyncSession(engine) as session:
        await session.execute(update(AgentRun).values(total_input_tokens=999))
        await session.commit()
    with _read_trace(engine, record_property) as (statements, boundaries, _):
        result = await getattr(store, operation)(IDS[0], max_nodes=10)
    usage = result.usage if operation == "get_subtree" else result
    assert (usage.input_tokens, usage.output_tokens, usage.llm_calls) == (7, 10, 3)
    assert (usage.cache_read_tokens, usage.cache_write_tokens, usage.reasoning_tokens) == (6, 9, 3)
    assert usage.unpriced_calls == 1 and usage.known_cost_usd == pytest.approx(0.3)
    if operation == "get_subtree":
        assert [(node.run.id, node.depth) for node in result.nodes] == [
            (IDS[0], 0),
            (IDS[1], 1),
            (IDS[2], 1),
            (IDS[3], 2),
            (IDS[4], 3),
        ]
    else:
        assert all("pause_data" not in sql and "loop_state" not in sql for sql in statements)
    assert len(statements) == count
    _pin(record_property, statements, boundaries)


async def test_budget_limit_keeps_fail_closed_bound_and_original_logger(
    store, engine, record_property, caplog
):
    await _seed(store, engine)
    with (
        _read_trace(engine, record_property) as (statements, boundaries, _),
        pytest.raises(BudgetConfigurationError) as caught,
    ):
        await store.subtree_usage(IDS[0], max_nodes=3)
    assert caught.value.reason == "subtree_limit"
    assert len(statements) == 2 and all("LIMIT" in sql for sql in statements)
    assert all("llm_interactions" not in sql for sql in statements)
    records = [r for r in caplog.records if getattr(r, "code", None) == "subtree_limit"]
    assert len(records) == 1
    assert (records[0].name, records[0].run_id, records[0].limit) == (
        "dendrux.runtime.stores.sql.base",
        IDS[0],
        3,
    )
    _pin(record_property, statements, boundaries)


@pytest.mark.parametrize(
    "operation,kwargs",
    [
        ("get_delegation_info", {"max_rows": True}),
        ("get_delegation_info", {"max_depth": 101}),
        ("subtree_usage", {"max_nodes": True}),
    ],
)
async def test_invalid_bounds_fail_before_opening_a_read_session(
    store, monkeypatch, operation, kwargs
):
    def forbidden():
        raise AssertionError("invalid bound opened a session")

    monkeypatch.setattr(store, "_session_factory", forbidden)
    with pytest.raises(ValueError):
        await getattr(store, operation)(IDS[0], **kwargs)


async def test_limit_logger_direct_pin_without_database(caplog, monkeypatch):
    store = object.__new__(SQLStateStore)
    batch = AsyncMock(return_value=[object()])
    monkeypatch.setattr(store, "_descendant_batch", batch)
    with pytest.raises(BudgetConfigurationError):
        await store._bfs_descendants(None, IDS[0], max_nodes=1)
    batch.assert_awaited_once_with(
        None, IDS[0], [IDS[0]], tenant_id=UNSCOPED, light=False, max_nodes=1, loaded=0
    )
    assert [r.name for r in caplog.records if getattr(r, "code", None) == "subtree_limit"] == [
        "dendrux.runtime.stores.sql.base"
    ]
