"""Optional projections cannot hide a known durable outcome."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from dendrux.errors import PersistenceError
from dendrux.runtime import runner
from dendrux.types import RunResult, RunStatus, generate_ulid


@pytest.mark.parametrize("projection", ["recovery", "pending"])
@pytest.mark.parametrize("cancel", [False, True])
async def test_result_projection_failure_is_isolated(monkeypatch, caplog, projection, cancel):
    run_id = generate_ulid()
    record = SimpleNamespace(id=run_id, tenant_id=None, reclaim_attempts=1, last_reclaimed_at=None)
    pending = SimpleNamespace(pause_id=generate_ulid())
    store = SimpleNamespace(get_replayed_tool_calls=AsyncMock(return_value=(["call"], False)))
    read_pending = AsyncMock(return_value=pending)
    monkeypatch.setattr("dendrux.runtime._pending._PendingReader.get_pending", read_pending)
    failure = asyncio.CancelledError() if cancel else PersistenceError()
    if projection == "recovery":
        store.get_replayed_tool_calls.side_effect = failure
    else:
        read_pending.side_effect = failure
    result = RunResult(
        run_id=run_id, status=RunStatus.WAITING_CLIENT_TOOL, pause_id="stale", pending=pending
    )
    if cancel:
        with pytest.raises(asyncio.CancelledError):
            await runner._attach_persisted_pending(store, result, record)
        return
    assert await runner._attach_persisted_pending(store, result, record) is result
    if projection == "pending":
        assert result.pending is None and result.pause_id is None
        assert result.recovery.replayed_tool_calls == ["call"]
    else:
        assert result.recovery is None
        assert result.pending is pending and result.pause_id == pending.pause_id
    assert any(
        getattr(r, "code", None) == "result_read_failed" and getattr(r, "run_id", None) == run_id
        for r in caplog.records
    )


@pytest.mark.parametrize("cancel", [False, True])
async def test_final_recovery_record_read_is_best_effort(monkeypatch, caplog, cancel):
    run_id = generate_ulid()
    store = SimpleNamespace(
        get_run=AsyncMock(side_effect=asyncio.CancelledError() if cancel else PersistenceError())
    )
    monkeypatch.setattr(
        "dendrux.runtime._capabilities.capabilities", lambda _: SimpleNamespace(recovery=True)
    )
    if cancel:
        with pytest.raises(asyncio.CancelledError):
            await runner._final_run_recovery(store, run_id, [])
    else:
        assert await runner._final_run_recovery(store, run_id, []) is None
        assert any(getattr(r, "code", None) == "result_read_failed" for r in caplog.records)


async def test_committed_pause_refresh_preserves_cancellation():
    store = SimpleNamespace(get_run=AsyncMock(side_effect=asyncio.CancelledError()))
    result = RunResult(run_id=generate_ulid(), status=RunStatus.WAITING_CLIENT_TOOL)
    with pytest.raises(asyncio.CancelledError):
        await runner._committed_pause_result(store, result, AsyncMock())


async def test_concurrent_pause_winner_read_failure_remains_required():
    store = SimpleNamespace(get_run=AsyncMock(return_value=SimpleNamespace(status="cancelled")))
    result = RunResult(run_id=generate_ulid(), status=RunStatus.WAITING_CLIENT_TOOL)
    persisted = AsyncMock(side_effect=PersistenceError())
    with pytest.raises(PersistenceError):
        await runner._committed_pause_result(store, result, persisted)
    persisted.assert_awaited_once_with(RunStatus.CANCELLED)
