"""0.5.0 T1 real-concurrency tests (spec §1a), on the backend matrix.

N parallel callers per test, asserting exactly one winner, no lost updates and
no orphan rows, for: tenant-scoped idempotency, store-side event index
allocation and the note cap. On Postgres the callers really run in
parallel (separate pooled connections); on SQLite they serialize on the
database lock. Also checks, on Postgres, that the hot queries can use their
indexes (``EXPLAIN`` with sequential scans disabled).
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import pytest
from sqlalchemy import text

from dendrux.errors import NoteLimitExceededError
from dendrux.types import IdempotencyConflictError

if TYPE_CHECKING:
    from dendrux.runtime.state import SQLAlchemyStateStore

pytestmark = pytest.mark.asyncio

N = 24


async def test_idempotency_race_has_one_winner_per_tenant(store: SQLAlchemyStateStore) -> None:
    async def create(i: int, tenant: str) -> Any:
        return await store.create_run(
            f"r{tenant}{i:02d}",
            "agent",
            tenant_id=tenant,
            idempotency_key="shared-key",
            idempotency_fingerprint="fp",
        )

    results = await asyncio.gather(
        *(create(i, t) for i in range(N) for t in ("A", "B")), return_exceptions=True
    )
    assert not [r for r in results if isinstance(r, BaseException)]
    for tenant in ("A", "B"):
        mine = [r for r in results if r.run_id.startswith(f"r{tenant}")]
        created = [r for r in mine if r.outcome == "created"]
        assert len(created) == 1
        assert {r.run_id for r in mine} == {created[0].run_id}
    assert await store.count_runs(tenant_id="A") == 1
    assert await store.count_runs(tenant_id="B") == 1


async def test_idempotency_race_with_different_input_conflicts(
    store: SQLAlchemyStateStore,
) -> None:
    async def create(i: int) -> Any:
        return await store.create_run(
            f"r{i:02d}",
            "agent",
            tenant_id="A",
            idempotency_key="k",
            idempotency_fingerprint=f"fp{i % 2}",
        )

    results = await asyncio.gather(*(create(i) for i in range(N)), return_exceptions=True)
    winners = [r for r in results if not isinstance(r, BaseException) and r.outcome == "created"]
    assert len(winners) == 1
    others = [r for r in results if not isinstance(r, BaseException) and r.outcome != "created"]
    conflicts = [r for r in results if isinstance(r, IdempotencyConflictError)]
    assert len(winners) + len(others) + len(conflicts) == N
    assert all(c.run_id == winners[0].run_id for c in conflicts)
    assert await store.count_runs() == 1


async def test_event_allocation_race_is_contiguous(store: SQLAlchemyStateStore) -> None:
    await store.create_run("r1", "agent")
    seqs = await asyncio.gather(
        *(store.save_run_event("r1", event_type=f"e{i}") for i in range(N * 2))
    )
    assert sorted(seqs) == list(range(N * 2))
    stored = [e.sequence_index for e in await store.get_run_events("r1")]
    assert stored == list(range(N * 2))


async def test_event_allocation_race_with_cas_transitions(store: SQLAlchemyStateStore) -> None:
    """Plain appends and a CAS transition event interleave without duplicate indexes."""
    await store.create_run("r1", "agent")

    async def transition() -> Any:
        return await store.finalize_with_event(
            "r1",
            status="success",
            allowed_current_statuses=["running"],
            event_type="run.completed",
        )

    tasks = [store.save_run_event("r1", event_type=f"e{i}") for i in range(N)]
    tasks.insert(N // 2, transition())
    results = await asyncio.gather(*tasks)
    assert sorted(r for r in results if r is not None) == list(range(N + 1))
    stored = [e.sequence_index for e in await store.get_run_events("r1")]
    assert stored == list(range(N + 1))


async def test_note_cap_race_never_exceeds_the_cap(store: SQLAlchemyStateStore) -> None:
    await store.create_run("r1", "agent", tenant_id="A")
    results = await asyncio.gather(
        *(
            store.add_note("r1", note_id=f"n{i:03d}", text="t", tenant_id="A", max_notes=7)
            for i in range(N * 2)
        ),
        return_exceptions=True,
    )
    assert sum(r is True for r in results) == 7
    assert all(isinstance(r, NoteLimitExceededError) for r in results if r is not True)
    assert len(await store.get_notes("r1")) == 7


async def test_note_cap_race_with_finalize(store: SQLAlchemyStateStore) -> None:
    """A finalize racing note submitters: every accepted note is stored, none after."""
    await store.create_run("r1", "agent")
    half = N // 2
    first = [
        store.add_note("r1", note_id=f"a{i:03d}", text="t", max_notes=100) for i in range(half)
    ]
    second = [
        store.add_note("r1", note_id=f"b{i:03d}", text="t", max_notes=100) for i in range(half)
    ]
    results = await asyncio.gather(
        *first, store.finalize_run("r1", status="success"), *second, return_exceptions=True
    )
    assert not [r for r in results if isinstance(r, BaseException)]
    notes = results[:half] + results[half + 1 :]
    assert len(await store.get_notes("r1")) == sum(r is True for r in notes)
    assert not await store.add_note("r1", note_id="late", text="t", max_notes=100)


async def test_hot_queries_can_use_their_indexes(engine: Any) -> None:
    if engine.dialect.name != "postgresql":
        pytest.skip("EXPLAIN plans are checked on Postgres")
    async with engine.begin() as conn:
        # This is index capability, independent of cost estimates or stale statistics.
        await conn.execute(text("SET LOCAL enable_seqscan = off"))
        await conn.execute(text("SET LOCAL enable_indexscan = on"))
        await conn.execute(text("SET LOCAL enable_bitmapscan = off"))
        await conn.execute(text("SET LOCAL enable_sort = off"))
        indexes = set(
            (
                await conn.execute(
                    text("SELECT indexname FROM pg_indexes WHERE schemaname=current_schema()")
                )
            ).scalars()
        )
        assert {
            "uq_agent_runs_tenant_idempotency_key",
            "ix_agent_runs_tenant_id_created_at",
            "uq_run_events_run_seq",
            "ix_run_inbox_run_pending",
        } <= indexes

        async def plan(sql: str, **params: Any) -> str:
            rows = await conn.execute(text(f"EXPLAIN {sql}"), params)
            return "\n".join(r[0] for r in rows)

        idem = await plan(
            "SELECT id FROM agent_runs WHERE coalesce(agent_runs.tenant_id, '') = :t "
            "AND agent_runs.idempotency_key = :k",
            t="A",
            k="k",
        )
        assert "uq_agent_runs_tenant_idempotency_key" in idem
        listing = await plan(
            "SELECT id FROM agent_runs WHERE tenant_id = :t ORDER BY created_at DESC, id DESC "
            "LIMIT 50",
            t="A",
        )
        assert "ix_agent_runs_tenant_id_created_at" in listing
        alloc = await plan(
            "SELECT coalesce(max(sequence_index), -1) + 1 FROM run_events WHERE agent_run_id = :r",
            r="r1",
        )
        assert "uq_run_events_run_seq" in alloc
        notes = await plan("SELECT count(*) FROM run_inbox WHERE run_id = :r", r="r1")
        assert "ix_run_inbox_run_pending" in notes
