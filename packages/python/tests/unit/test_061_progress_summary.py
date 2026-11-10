"""A zero streak retains recovery provenance and the bounded replay summary."""

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from dendrux.runtime.runner import _recovery_info


@pytest.mark.parametrize("best_effort", [False, True])
async def test_061_zero_streak_keeps_recovery_summary(best_effort):
    stamp = datetime(2026, 11, 1, tzinfo=UTC)
    record = SimpleNamespace(
        id="00000000000000000000000001",
        tenant_id="tenant",
        reclaim_attempts=0,
        last_reclaimed_at=stamp,
    )
    ids = [str(i) for i in range(100)]
    store = SimpleNamespace(get_replayed_tool_calls=AsyncMock(return_value=(ids, True)))
    result = await _recovery_info(store, record, best_effort=best_effort)
    assert result is not None
    assert result.reclaim_attempts == 0 and result.last_reclaimed_at == stamp
    assert result.replayed_tool_calls == ids and result.replayed_truncated
    store.get_replayed_tool_calls.assert_awaited_once_with(
        "00000000000000000000000001", limit=100, tenant_id="tenant"
    )


@pytest.mark.parametrize("best_effort", [False, True])
async def test_061_zero_streak_summary_retains_read_failure_policy(best_effort):
    record = SimpleNamespace(
        id="00000000000000000000000001",
        tenant_id="tenant",
        reclaim_attempts=0,
        last_reclaimed_at=datetime.now(UTC),
    )
    store = SimpleNamespace(
        get_replayed_tool_calls=AsyncMock(side_effect=RuntimeError("read failed"))
    )
    if best_effort:
        assert await _recovery_info(store, record) is None
        store.get_replayed_tool_calls.assert_awaited_once()
    else:
        with pytest.raises(RuntimeError, match="read failed"):
            await _recovery_info(store, record, best_effort=False)


@pytest.mark.parametrize(
    "record", [None, SimpleNamespace(reclaim_attempts=0, last_reclaimed_at=None)]
)
async def test_061_never_reclaimed_run_has_no_summary(record):
    store = SimpleNamespace(get_replayed_tool_calls=AsyncMock())
    assert await _recovery_info(store, record) is None
    store.get_replayed_tool_calls.assert_not_awaited()
