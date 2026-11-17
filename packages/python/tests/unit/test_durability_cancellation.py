"""Driver-converted cancellation must never enter the DB retry policy."""

import asyncio
import traceback
from unittest.mock import AsyncMock

import pytest
from sqlalchemy.exc import OperationalError

from dendrux.runtime import durability
from dendrux.runtime.stores._errors import DbError


class ClosedConnectionError(ValueError):
    __module__ = "aiosqlite"


@pytest.mark.parametrize("fault_type", [ClosedConnectionError, OperationalError])
@pytest.mark.parametrize("chain", ["context", "cause", "current-task-only"])
async def test_twice_cancelled_write_is_not_retried(monkeypatch, fault_type, chain):
    backoff = AsyncMock()
    monkeypatch.setattr(durability, "_backoff", backoff)
    attempts = 0

    async def attempt():
        nonlocal attempts
        attempts += 1
        if attempts > 1:
            return "unexpected replay"
        task = asyncio.current_task()
        task.cancel()
        task.cancel()
        try:
            await asyncio.sleep(0)
        except asyncio.CancelledError as cancellation:
            assert task.cancelling() == 2
            fault = (
                OperationalError(None, None, Exception("PRIVATE_DRIVER_SENTINEL"))
                if fault_type is OperationalError
                else ClosedConnectionError("PRIVATE_DRIVER_SENTINEL")
            )
            if chain == "context":
                raise fault  # noqa: B904 - reproduce a driver using implicit context
            if chain == "cause":
                raise fault from cancellation
        # A driver can discard the exception chain while retaining task cancellation.
        raise fault

    task = asyncio.create_task(
        durability.retry_transient_db(
            attempt,
            label="cancelled-write",
            run_id="run",
            classify=lambda _: DbError("connection_lost"),
            _raw_driver_errors=True,
        )
    )
    with pytest.raises(asyncio.CancelledError):
        await task
    assert attempts == 1
    backoff.assert_not_called()


@pytest.mark.parametrize("link", ["__context__", "__cause__", "orig"])
async def test_chained_cancellation_without_pending_task_cancel_is_propagated(monkeypatch, link):
    cancellation = asyncio.CancelledError()
    fault = OperationalError(None, None, Exception("PRIVATE_DRIVER_SENTINEL"))
    setattr(fault.orig if link != "orig" else fault, link, cancellation)
    # The cancellation walk must tolerate cycles and inspect context even when orig exists.
    fault.__context__ = fault
    attempt = AsyncMock(side_effect=fault)
    backoff = AsyncMock()
    monkeypatch.setattr(durability, "_backoff", backoff)
    with pytest.raises(asyncio.CancelledError) as caught:
        await durability.retry_transient_db(attempt, label="converted", run_id="run")
    assert caught.value is cancellation
    assert "PRIVATE_DRIVER_SENTINEL" not in "".join(traceback.format_exception(caught.value))
    assert attempt.await_count == 1
    backoff.assert_not_called()


async def test_genuine_transient_failure_is_still_retried(monkeypatch):
    fault = OperationalError(None, None, Exception("temporary"))
    attempt = AsyncMock(side_effect=[fault, "written"])
    backoff = AsyncMock()
    monkeypatch.setattr(durability, "_backoff", backoff)
    assert await durability.retry_transient_db(attempt, label="write", run_id="run") == "written"
    assert attempt.await_count == 2
    backoff.assert_awaited_once()
