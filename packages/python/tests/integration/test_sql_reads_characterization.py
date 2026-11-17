"""SQL reads preserve scoped projections, ordering, bounds and statement budgets."""

import json
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import event, insert, update

from dendrux._meta_keys import LATEST_REWRITE_META_KEY
from dendrux.db.models import AgentRun, LLMInteraction, ReactTrace, RunEvent, ToolCallRecord
from dendrux.runtime._bounds import MAX_PAGE
from dendrux.runtime.state import UNSCOPED
from dendrux.runtime.stores.sql import _sql_reads

RUN = "00000000000000000000000001"
OTHER = "00000000000000000000000002"
STAMP = datetime(2026, 11, 1, tzinfo=UTC)


@contextmanager
def _reads(engine, record_property, count=1):
    statements, parameters = [], []

    def observe(conn, cursor, statement, values, context, many):
        statements.append(" ".join(statement.split()))
        parameters.append(context.compiled_parameters)

    event.listen(engine.sync_engine, "before_cursor_execute", observe)
    try:
        yield statements, parameters
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", observe)
        record_property("sql", json.dumps(statements))
        record_property("parameters", json.dumps(parameters, default=str, sort_keys=True))
    assert len(statements) == count
    assert all(s.startswith("SELECT ") and "FOR UPDATE" not in s for s in statements)


async def _seed(store, engine):
    for run_id, tenant in [(RUN, "t1"), (OTHER, "t2")]:
        await store.create_run(run_id, "reads", tenant_id=tenant)
    async with engine.begin() as conn:
        await conn.execute(
            update(AgentRun).values(
                created_at=STAMP,
                loop_state={"pending": [{"order_index": 4}]},
                pii_mapping={"p": "v"},
                meta={"thread_id": "thread", "keep": 1},
                output_data={"answer": "answer"},
            )
        )
        for index, kind in enumerate(["seed", "turn", "rewrite", "state", "turn"]):
            await conn.execute(
                insert(ReactTrace).values(
                    id=f"{index + 10:026d}",
                    agent_run_id=RUN,
                    order_index=index,
                    role="assistant",
                    trace_kind=kind,
                    content=json.dumps([{"role": "user", "content": "replacement"}])
                    if kind == "rewrite"
                    else f"message-{index}",
                    created_at=STAMP,
                    meta={"keep": 1},
                )
            )
        for model in [ToolCallRecord, LLMInteraction]:
            for index in range(3):
                values = dict(
                    id=f"{index + 30:026d}",
                    agent_run_id=RUN,
                    iteration_index=index // 2,
                    created_at=STAMP + timedelta(seconds=index),
                )
                if model is ToolCallRecord:
                    values.update(
                        tool_call_id=f"{index + 40:026d}",
                        tool_name="tool",
                        meta={"observation": "seen"},
                    )
                else:
                    values.update(model="model", input_tokens=index, output_tokens=2, cost_usd=0.25)
                await conn.execute(insert(model).values(**values))


@pytest.mark.parametrize("tenant", ["t1", "t2", None, UNSCOPED])
@pytest.mark.parametrize(
    "method", ["get_run", "get_run_status", "get_pii_mapping", "_get_result_record"]
)
async def test_run_projections_scope_and_one_statement(
    store, engine, record_property, tenant, method
):
    await _seed(store, engine)
    with _reads(engine, record_property) as (sql, _):
        result = await getattr(store, method)(RUN, tenant_id=tenant)
    assert (result is not None) is (tenant in ("t1", UNSCOPED))
    projection = sql[0].split(" FROM ")[0]
    if method == "get_run_status":
        assert projection == "SELECT agent_runs.status"
    elif method == "get_pii_mapping":
        assert projection == "SELECT agent_runs.pii_mapping"
    elif method == "_get_result_record":
        assert "pause_data" not in projection and "pii_mapping" not in projection
        assert "loop_state" not in projection and "input_data" not in projection
    if tenant == "t1":
        assert "tenant_id" in sql[0].split(" WHERE ")[-1]


@pytest.mark.parametrize("method", ["get_traces", "get_tool_calls", "get_llm_interactions"])
@pytest.mark.parametrize("tenant", ["t1", "t2", None])
async def test_history_scope_offset_and_limit(store, engine, record_property, method, tenant):
    await _seed(store, engine)
    with _reads(engine, record_property) as (sql, params):
        rows = await getattr(store, method)(RUN, tenant_id=tenant, limit=1, offset=1)
    assert len(rows) == (1 if tenant == "t1" else 0)
    if rows:
        assert rows[0].id == (f"{11:026d}" if method == "get_traces" else f"{31:026d}")
    assert "ORDER BY" in sql[0] and "LIMIT" in sql[0] and "OFFSET" in sql[0]
    assert 1 in params[0][0].values()


@pytest.mark.parametrize("method", ["get_tool_calls", "get_llm_interactions"])
async def test_iteration_before_and_clamped_limit(store, engine, record_property, method):
    await _seed(store, engine)
    with _reads(engine, record_property) as (sql, params):
        rows = await getattr(store, method)(
            RUN,
            iteration_index=0,
            before=STAMP + timedelta(seconds=1),
            limit=0,
            offset=-1,
            tenant_id="t1",
        )
    assert [row.id for row in rows] == [f"{30:026d}"]
    assert "iteration_index =" in sql[0] and "created_at <" in sql[0]
    assert 1 in params[0][0].values() and 0 in params[0][0].values()
    with _reads(engine, record_property, 0), pytest.raises(ValueError):
        await getattr(store, method)(RUN, before=datetime(2026, 11, 1))


async def test_trace_keyset_and_authoritative_projections(store, engine, record_property):
    await _seed(store, engine)
    with _reads(engine, record_property) as (sql, params):
        rows = await store.get_traces_after(
            RUN, after_order_index=0, kinds=("turn",), limit=MAX_PAGE + 1, tenant_id="t1"
        )
    assert [row.order_index for row in rows] == [1, 4]
    assert "order_index >" in sql[0] and MAX_PAGE in params[0][0].values()
    with _reads(engine, record_property):
        assert await store.get_trace(RUN, f"{12:026d}", tenant_id="t2") is None
    with _reads(engine, record_property):
        assert (await store.get_trace(RUN, f"{12:026d}", tenant_id="t1")).trace_kind == "rewrite"
    with _reads(engine, record_property) as (sql, _):
        assert await store.get_next_trace_index(RUN) == 5
    assert (
        sql[0].startswith("SELECT react_traces.order_index")
        and "DESC" in sql[0]
        and "LIMIT" in sql[0]
    )
    with _reads(engine, record_property) as (sql, _):
        assert await store.get_loop_state(RUN) == {"pending": [{"order_index": 4}]}
    assert sql[0].startswith("SELECT agent_runs.loop_state")
    with _reads(engine, record_property, 0), pytest.raises(ValueError):
        await store.get_traces_after(RUN, after_order_index=None, limit=0)


@pytest.mark.parametrize("tenant", ["t1", "t2"])
async def test_public_trace_reservation_cutoff_in_one_snapshot(
    store, engine, record_property, tenant
):
    await _seed(store, engine)
    with _reads(engine, record_property) as (sql, _):
        rows = await store.get_public_traces(RUN, tenant_id=tenant, limit=2)
    assert [row.order_index for row in rows] == ([0, 1] if tenant == "t1" else [])
    assert "SELECT CASE WHEN" in sql[0] and "LIMIT" in sql[0]
    with _reads(engine, record_property):
        rows = await store.get_public_traces(RUN, tenant_id=tenant, retry=True)
    assert [row.order_index for row in rows] == ([1] if tenant == "t1" else [])


async def test_rewrite_conversation_cursor_and_owner_constant(
    store, engine, monkeypatch, record_property
):
    await _seed(store, engine)
    monkeypatch.setattr(_sql_reads, "_TRACE_POSITION_STRIDE", 100)
    async with engine.begin() as conn:
        await conn.execute(
            update(AgentRun)
            .where(AgentRun.id == RUN)
            .values(meta={LATEST_REWRITE_META_KEY: {"order_index": 2, "trace_id": f"{12:026d}"}})
        )
    with _reads(engine, record_property, 3):
        page = await store.conversation_traces(RUN, after_position=None, limit=1, tenant_id="t1")
    assert [(row.position, row.content) for row in page.entries] == [(201, "replacement")]
    assert page.next_position == 201
    with _reads(engine, record_property, 3):
        page = await store.conversation_traces(
            RUN, after_position=page.next_position, limit=2, tenant_id="t1"
        )
    assert [row.position for row in page.entries] == [400] and page.next_position is None
    with _reads(engine, record_property):
        page = await store.conversation_traces(
            RUN, after_position=None, include_superseded=True, limit=2, tenant_id="t1"
        )
    assert [row.position for row in page.entries] == [0, 100] and page.next_position == 100
    with _reads(engine, record_property, 2):
        assert not (
            await store.conversation_traces(RUN, after_position=None, tenant_id="t2")
        ).entries


@pytest.mark.parametrize("status", [None, [], ["running"], "running"])
async def test_list_count_identical_filters_and_deferred_payloads(
    store, engine, record_property, status
):
    await _seed(store, engine)
    options = dict(
        tenant_id="t1",
        status=status,
        agent_name="reads",
        started_after=STAMP,
        started_before=STAMP + timedelta(days=1),
        metadata_filter={"thread_id": "thread"},
    )
    with _reads(engine, record_property) as (sql, params):
        rows = await store.list_runs(**options, limit=MAX_PAGE + 1)
    assert len(rows) == (0 if status == [] else 1)
    assert (
        "pause_data" not in sql[0].split(" FROM ")[0]
        and "pii_mapping" not in sql[0].split(" FROM ")[0]
    )
    assert "created_at DESC, agent_runs.id DESC" in sql[0] and MAX_PAGE in params[0][0].values()
    with _reads(engine, record_property):
        assert await store.count_runs(**options) == len(rows)
    with _reads(engine, record_property, 0), pytest.raises(ValueError):
        await store.list_runs(offset=10001)
    with _reads(engine, record_property, 0), pytest.raises(ValueError):
        await store.count_runs(metadata_filter={"nested.key": "v"})


async def test_batch_status_and_pause_count_budgets(store, engine, record_property):
    await _seed(store, engine)
    async with engine.begin() as conn:
        for index, kind in enumerate(["run.paused", "run.paused", "run.resumed"]):
            await conn.execute(
                insert(RunEvent).values(
                    id=f"{index + 50:026d}",
                    agent_run_id=RUN,
                    event_type=kind,
                    sequence_index=index,
                    iteration_index=0,
                    created_at=STAMP,
                )
            )
    with _reads(engine, record_property):
        assert await store.get_run_statuses([RUN, OTHER, RUN]) == {RUN: "running", OTHER: "running"}
    with _reads(engine, record_property, 0):
        assert await store.get_run_statuses([]) == {}
        assert await store.count_pauses_per_run([]) == {}
    ids = [f"{index + 100:026d}" for index in range(501)] + [RUN, RUN]
    with _reads(engine, record_property, 2) as (sql, _):
        assert await store.count_pauses_per_run(ids) == {RUN: 2}
    assert all("GROUP BY run_events.agent_run_id" in statement for statement in sql)


async def test_physical_trace_keyset_refuses_foreign_and_untenanted_scope(
    store, engine, record_property
):
    await _seed(store, engine)
    with _reads(engine, record_property):
        own = await store.get_traces_after(
            RUN, after_order_index=0, kinds=("turn",), tenant_id="t1"
        )
    assert [row.order_index for row in own] == [1, 4]
    for tenant in ("t2", None):
        with _reads(engine, record_property):
            assert (
                await store.get_traces_after(
                    RUN, after_order_index=0, kinds=("turn",), tenant_id=tenant
                )
                == []
            )


async def test_conversation_page_cap_and_strict_continuation(store, engine, record_property):
    await store.create_run(RUN, "reads", tenant_id="t1")
    async with engine.begin() as conn:
        await conn.execute(
            insert(ReactTrace),
            [
                dict(
                    id=f"{index + 100:026d}",
                    agent_run_id=RUN,
                    order_index=index,
                    role="assistant",
                    trace_kind="turn",
                    content=f"message-{index}",
                    created_at=STAMP,
                )
                for index in range(MAX_PAGE + 2)
            ],
        )
    with _reads(engine, record_property):
        first = await store.conversation_traces(
            RUN,
            after_position=None,
            limit=MAX_PAGE + 1,
            include_superseded=True,
            tenant_id="t1",
        )
    assert len(first.entries) == MAX_PAGE
    assert [entry.content for entry in first.entries] == [
        f"message-{index}" for index in range(MAX_PAGE)
    ]
    assert first.next_position == first.entries[-1].position
    with _reads(engine, record_property):
        second = await store.conversation_traces(
            RUN,
            after_position=first.next_position,
            limit=MAX_PAGE + 1,
            include_superseded=True,
            tenant_id="t1",
        )
    assert [entry.content for entry in second.entries] == [
        f"message-{MAX_PAGE}",
        f"message-{MAX_PAGE + 1}",
    ]
    assert second.entries[0].position > first.next_position
    assert second.next_position is None


async def test_rewrite_prefix_is_excluded_by_a_strict_physical_boundary(
    store, engine, record_property
):
    await _seed(store, engine)
    async with engine.begin() as conn:
        await conn.execute(
            update(AgentRun)
            .where(AgentRun.id == RUN)
            .values(meta={LATEST_REWRITE_META_KEY: {"order_index": 2, "trace_id": f"{12:026d}"}})
        )
    with _reads(engine, record_property, 3) as (sql, _):
        page = await store.conversation_traces(RUN, after_position=None, limit=10, tenant_id="t1")
    assert [(entry.position, entry.content) for entry in page.entries] == [
        (2001, "replacement"),
        (4000, "message-4"),
    ]
    assert page.next_position is None
    # Both the continuation cursor and rewrite cutoff are strict index lower bounds.
    assert sql[-1].count("react_traces.order_index > ") == 2
    assert "react_traces.order_index >= " not in sql[-1]


@pytest.mark.parametrize("run_tenant", ["t1", None])
@pytest.mark.parametrize("tenant", ["t1", "t2", None, UNSCOPED])
@pytest.mark.parametrize("status", ["running", "success"])
async def test_public_trace_scope_covers_the_pending_projection(
    store, engine, record_property, run_tenant, tenant, status
):
    await _seed(store, engine)
    async with engine.begin() as conn:
        await conn.execute(
            update(AgentRun)
            .where(AgentRun.id == RUN)
            .values(
                tenant_id=run_tenant,
                status=status,
            )
        )
        await conn.execute(
            update(AgentRun)
            .where(AgentRun.id == OTHER)
            .values(loop_state={"pending": [{"order_index": 0}]})
        )
    with _reads(engine, record_property):
        rows = await store.get_public_traces(RUN, tenant_id=tenant)
    authorized = tenant is UNSCOPED or tenant == run_tenant
    expected = ([0, 1, 4] if status == "success" else [0, 1]) if authorized else []
    assert [row.order_index for row in rows] == expected
    record_property("public_trace_result", json.dumps([row.order_index for row in rows]))


@pytest.mark.parametrize("tenant", ["", False, 123])
async def test_public_trace_invalid_scope_refuses_before_query(
    store, engine, record_property, tenant
):
    with _reads(engine, record_property, 0), pytest.raises(ValueError):
        await store.get_public_traces(RUN, tenant_id=tenant)
