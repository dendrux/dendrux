"""State primitives used by the budget: running totals on pause, failure_reason on finalize.

SQLite + Postgres via the ``store`` matrix fixture.
"""

from __future__ import annotations

import pytest

from dendrux.types import UsageStats

pytestmark = pytest.mark.asyncio


class TestStatePrimitives:
    async def test_pause_primitives_write_running_totals(self, store) -> None:
        usage = UsageStats(
            input_tokens=7,
            output_tokens=3,
            total_tokens=10,
            cost_usd=0.5,
            cache_read_input_tokens=2,
            reasoning_tokens=1,
        )
        for run_id, pause in (
            ("p1", store.pause_run),
            ("p2", store.try_pause_run),
            ("p3", store.pause_with_event),
        ):
            await store.create_run(run_id, "A")
            await pause(run_id, status="paused", pause_data={"x": 1}, total_usage=usage)
            record = await store.get_run(run_id)
            assert record.status == "paused"
            assert (record.total_input_tokens, record.total_output_tokens) == (7, 3)
            assert record.total_cost_usd == pytest.approx(0.5)
            assert record.total_cache_read_tokens == 2
            assert record.total_reasoning_tokens == 1

    async def test_finalize_primitives_write_failure_reason(self, store) -> None:
        await store.create_run("f1", "A")
        assert await store.finalize_run_if_status_in(
            "f1",
            status="cancelled",
            allowed_current_statuses=["running"],
            failure_reason="budget_exceeded",
        )
        await store.create_run("f2", "A")
        seq = await store.finalize_with_event(
            "f2",
            status="cancelled",
            allowed_current_statuses=["running"],
            event_type="run.cancelled",
            event_data={"reason": "budget_exceeded"},
            failure_reason="budget_exceeded",
        )
        assert seq is not None
        for run_id in ("f1", "f2"):
            assert (await store.get_run(run_id)).failure_reason == "budget_exceeded"
