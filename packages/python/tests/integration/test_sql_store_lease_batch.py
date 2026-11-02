"""Lease expiry uses the sweep batch bound before issuing database work."""

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import event, insert

from dendrux.db.models import AgentRun
from dendrux.runtime._bounds import SWEEP_BATCH_MAX
from dendrux.types import generate_ulid


@pytest.mark.parametrize("batch_size", [1001, 100_000])
async def test_lease_batch_refuses_oversized_page_before_sql(store, engine, caplog, batch_size):
    statements = []

    def capture(*args):
        statements.append(args[2])

    event.listen(engine.sync_engine, "before_cursor_execute", capture)
    try:
        with pytest.raises(ValueError, match="batch_size"):
            await store.mark_lease_expired(
                cutoff=datetime.now(UTC),
                batch_size=batch_size,
                max_runs=100_000,
                max_returned=100_000,
                reclaim_limit=3,
                cursor=None,
            )
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", capture)
    assert statements == []
    assert any(getattr(r, "code", None) == "recovery_scan_limit" for r in caplog.records)


async def test_lease_batch_accepts_ten_times_default_batch(store, engine):
    count = 10 * 100
    assert count == SWEEP_BATCH_MAX
    now = datetime.now(UTC)
    async with engine.begin() as conn:
        await conn.execute(
            insert(AgentRun),
            [
                {
                    "id": generate_ulid(),
                    "agent_name": "fresh",
                    "status": "running",
                    "created_at": now,
                    "last_progress_at": now,
                }
                for _ in range(count)
            ],
        )
    result = await store.mark_lease_expired(
        cutoff=now - timedelta(minutes=1),
        batch_size=count,
        max_runs=100_000,
        max_returned=100_000,
        reclaim_limit=3,
        cursor=None,
    )
    assert result.inspected == count
    assert not result.reclaimable and not result.finalised
