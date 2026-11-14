"""Portable update bounds and persisted retry identity survive SQL ownership moves."""

from decimal import Decimal

import pytest
from sqlalchemy import select
from sqlalchemy.exc import MultipleResultsFound

from dendrux.db.models import AgentRun, TokenUsage
from dendrux.errors import PersistenceError
from dendrux.types import UsageStats, generate_ulid


async def test_usage_retry_compares_the_persisted_decimal_cost(store, engine):
    run_id, usage_id = generate_ulid(), generate_ulid()
    await store.create_run(run_id, "retry-cost")
    for cost in (0.0000014, 0.00000149):
        await store.save_usage(
            run_id,
            iteration_index=0,
            usage=UsageStats(input_tokens=1, cost_usd=cost),
            usage_id=usage_id,
        )
    with pytest.raises(PersistenceError) as caught:
        await store.save_usage(
            run_id,
            iteration_index=0,
            usage=UsageStats(input_tokens=1, cost_usd=0.0000015),
            usage_id=usage_id,
        )
    assert caught.value.code == "write_conflict"
    async with engine.connect() as connection:
        rows = (
            await connection.execute(
                select(TokenUsage.id, TokenUsage.cost_usd).where(TokenUsage.agent_run_id == run_id)
            )
        ).all()
    assert rows == [(usage_id, Decimal("0.000001"))]


async def test_usage_for_an_absent_run_cannot_create_orphan_evidence(store, engine):
    run_id = generate_ulid()
    with pytest.raises(PersistenceError):
        await store.save_usage(run_id, iteration_index=0, usage=UsageStats(input_tokens=1))
    async with engine.connect() as connection:
        rows = (
            await connection.execute(select(TokenUsage.id).where(TokenUsage.agent_run_id == run_id))
        ).all()
    assert rows == []


@pytest.mark.parametrize("case", ["one-empty", "one-key", "callback-key", "many-empty", "many-key"])
async def test_updates_reject_missing_projections_and_primary_key_changes(store, engine, case):
    run_id = generate_ulid()
    await store.create_run(run_id, "immutable-key")
    table = AgentRun.__table__
    values = [(table.c.id, generate_ulid())] if case in {"one-key", "many-key"} else []
    returning = [] if case.endswith("empty") else [table.c.id]

    def change_key(row):
        values.append((table.c.id, generate_ulid()))

    async with store._session_factory() as session:
        with pytest.raises(ValueError, match="primary keys|projection"):
            if case.startswith("many"):
                await store.update_many(
                    session,
                    table=table,
                    predicate=table.c.id == run_id,
                    values=values,
                    returning=returning,
                    page_size=1,
                )
            else:
                await store.update_one(
                    session,
                    table=table,
                    pk=(run_id,),
                    predicate=table.c.id == run_id,
                    values=values,
                    returning=returning,
                    validate_locked=change_key if case == "callback-key" else None,
                )
        await session.rollback()
    async with engine.connect() as connection:
        rows = (await connection.execute(select(AgentRun.id))).scalars().all()
    assert rows == [run_id]


@pytest.mark.parametrize("returning", [False, True])
async def test_bulk_update_adapters_preserve_exact_membership_and_projection(
    store, engine, returning
):
    run_ids = sorted(generate_ulid() for _ in range(3))
    other = generate_ulid()
    for run_id in [*run_ids, other]:
        await store.create_run(run_id, "bounded-write")
    store.supports_update_returning = returning
    table = AgentRun.__table__
    async with store._session_factory() as session:
        result = await store._update_many_rows(
            session,
            table=table,
            predicate=table.c.id.in_(run_ids[:2]),
            values={"iteration_count": 9},
            returning=[table.c.id, table.c.iteration_count],
            page_size=2,
        )
        assert result.first().id in run_ids[:2]
        assert result.first().iteration_count == 9
        with pytest.raises(MultipleResultsFound):
            result.scalar_one_or_none()
        remaining = await store.update_many(
            session,
            table=table,
            predicate=table.c.id.in_(run_ids) & (table.c.iteration_count == 0),
            values=[(table.c.iteration_count, 9)],
            returning=[table.c.id, table.c.iteration_count],
            page_size=1,
        )
        assert [dict(row) for row in remaining] == [{"id": run_ids[2], "iteration_count": 9}]
        await store._write_tx(session, run_ids=run_ids)
    async with engine.connect() as connection:
        rows = (await connection.execute(select(AgentRun.id, AgentRun.iteration_count))).all()
    assert dict(rows) == {**dict.fromkeys(run_ids, 9), other: 0}
