"""One tenant's unreadable run cannot poison another tenant's batch."""

import asyncio
import logging

import pytest

from dendrux.store import RunStore, _follow
from tests.unit.test_follow_coordinator import BatchedState


@pytest.mark.parametrize("phase", ["events", "status"])
async def test_bad_tenant_isolated_with_shared_read_budget(phase, caplog):
    class Poisoned(BatchedState):
        async def get_run_statuses(self, run_ids):
            if phase == "status" and "private-bad" in run_ids:
                self.status_reads.append(tuple(run_ids))
                raise ValueError("private-tenant driver payload")
            return await super().get_run_statuses(run_ids)

    state = Poisoned()
    reader = RunStore(state)
    c = reader._follow_coordinator()
    if phase == "events":
        state.fail_runs.add("private-bad")
    for i in range(40):
        rid = "private-bad" if i == 0 else f"good-{i}"
        state.create(rid, tenant=f"tenant-{i}")
        state.write(rid)
        run = _follow._WatchedRun(c, rid, None, i)
        run.followers.add(_follow._Follower(True, 100))
        run.events_due = run.status_due = c._now()
        c.runs[rid] = run
    before = c._now()
    with caplog.at_level(logging.WARNING):
        c.revive()
        async with asyncio.timeout(5):
            while not all(
                run.head == 0 if phase == "events" else run.status_due > before
                for rid, run in c.runs.items()
                if rid != "private-bad"
            ):
                await asyncio.sleep(0.01)
    for rid, run in c.runs.items():
        if rid == "private-bad":
            continue
        if phase == "events":
            assert run.head == 0
        else:
            assert run.status_due > before
            assert run.status_fail_delay == 0
    calls = len(state.event_reads) + len(state.status_reads)
    assert calls <= 20 + 20 * (c._now() - before)
    assert "private-tenant" not in caplog.text
    assert "private-bad" not in caplog.text
    await reader.close()
