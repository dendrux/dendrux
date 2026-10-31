"""One integer sweep limit across the public API and both store categories."""

import datetime as dt
import inspect
from types import SimpleNamespace

import pytest
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.dml import Update
from sqlalchemy.sql.selectable import Select

from dendrux import sweep
from dendrux.db.models import AgentRun
from dendrux.runtime.state import StateStore
from dendrux.runtime.stores.sql.base import SQLStateStore


@pytest.mark.parametrize(
    "entry",
    [
        sweep,
        StateStore.sweep_stale_runs,
        StateStore.sweep_abandoned_runs,
        SQLStateStore.sweep_stale_runs,
        SQLStateStore.sweep_abandoned_runs,
    ],
    ids=["public", "protocol-running", "protocol-waiting", "sql-running", "sql-waiting"],
)
def test_sweep_limit_is_an_integer_with_the_same_default(entry):
    limit = inspect.signature(entry).parameters["max_runs"]
    assert limit.default == 10000, "every category must default to the shared 10k bound"
    assert limit.annotation in (int, "int"), "max_runs must not use a None sentinel"


@pytest.mark.parametrize("entry", ["public", "running", "waiting"])
async def test_sweep_rejects_none_as_a_limit(store, entry):
    with pytest.raises(ValueError, match="max_runs"):
        if entry == "public":
            await sweep(state_store=store, abandoned_waiting=dt.timedelta(days=1), max_runs=None)
        elif entry == "running":
            await store.sweep_stale_runs(dt.timedelta(days=1), max_runs=None)
        else:
            await store.sweep_abandoned_runs(dt.timedelta(days=1), max_runs=None)


@pytest.mark.parametrize("entry", ["public", "store"])
async def test_default_waiting_scan_reaches_beyond_the_old_thousand_row_cap(
    store, engine, monkeypatch, entry
):
    old = dt.datetime.now(dt.UTC) - dt.timedelta(days=2)
    async with engine.begin() as conn:
        for base in range(0, 1500, 500):
            await conn.execute(
                AgentRun.__table__.insert(),
                [
                    {
                        "id": f"waiting-{i:04}",
                        "agent_name": "agent",
                        "status": "paused",
                        "status_changed_at": old,
                    }
                    for i in range(base, base + 500)
                ],
            )
    execute = AsyncSession.execute
    inspected = []

    async def observe(session, statement, *args, **kwargs):
        if isinstance(statement, Update) and "failure_reason" in str(statement):
            # All CAS attempts lose, so the bound must count candidates rather
            # than successful status transitions. Keep real indexed page reads.
            return SimpleNamespace(mappings=lambda: SimpleNamespace(first=lambda: None))
        result = await execute(session, statement, *args, **kwargs)
        if isinstance(statement, Select) and "started" in statement.selected_columns:
            rows = result.all()
            inspected.extend(row.id for row in rows)
            return SimpleNamespace(all=lambda: rows)
        return result

    monkeypatch.setattr(AsyncSession, "execute", observe)
    if entry == "public":
        result = await sweep(state_store=store, abandoned_waiting=dt.timedelta(days=1))
        assert result.abandoned_waiting == []
    else:
        assert await store.sweep_abandoned_runs(dt.timedelta(days=1)) == []
    assert len(inspected) == len(set(inspected)) == 1500, (
        "default waiting inspection must continue beyond 1000 candidates even with no CAS winners"
    )
