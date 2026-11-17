"""Driver-converted cancellation must never enter the DB retry policy."""

import asyncio
from contextlib import suppress
from unittest.mock import AsyncMock

import pytest
from sqlalchemy.exc import OperationalError

from dendrux.runtime import durability
from dendrux.runtime.stores._errors import DbError
from dendrux.types import generate_ulid


class ClosedConnectionError(ValueError):
    __module__ = "aiosqlite"


@pytest.mark.parametrize("prior_cancel", [False, True])
@pytest.mark.parametrize("fault_type", [ClosedConnectionError, OperationalError])
@pytest.mark.parametrize("chain", ["context", "cause", "current-task-only"])
async def test_twice_cancelled_write_is_not_retried(
    monkeypatch, caplog, fault_type, chain, prior_cancel
):
    run_id = generate_ulid()
    backoff = AsyncMock()
    monkeypatch.setattr(durability, "_backoff", backoff)
    attempts = 0

    async def attempt():
        nonlocal attempts
        attempts += 1
        if attempts > 1:
            return "unexpected replay"
        task = asyncio.current_task()
        before = task.cancelling()
        task.cancel()
        task.cancel()
        try:
            await asyncio.sleep(0)
        except asyncio.CancelledError as cancellation:
            assert task.cancelling() == before + 2
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

    async def worker():
        if prior_cancel:
            task = asyncio.current_task()
            task.cancel()
            # Retain the old count; the later two requests belong to this operation.
            with suppress(asyncio.CancelledError):
                await asyncio.sleep(0)
        return await durability.retry_transient_db(
            attempt,
            label="cancelled-write",
            run_id=run_id,
            classify=lambda _: DbError("connection_lost"),
            _raw_driver_errors=True,
        )

    task = asyncio.create_task(worker())
    with pytest.raises(asyncio.CancelledError) as caught:
        await task
    assert caught.value.__context__ is None
    assert caught.value.__cause__ is None
    assert [(record.code, record.run_id) for record in caplog.records] == [
        ("db_cancel_converted", run_id)
    ]
    assert "PRIVATE_DRIVER_SENTINEL" not in repr([vars(record) for record in caplog.records])
    assert attempts == 1
    backoff.assert_not_called()


@pytest.mark.parametrize("link", ["__context__", "__cause__", "orig"])
async def test_chain_without_a_new_cancel_request_does_not_cancel_the_caller(monkeypatch, link):
    cancellation = asyncio.CancelledError()
    fault = OperationalError(None, None, Exception("PRIVATE_DRIVER_SENTINEL"))
    setattr(fault.orig if link != "orig" else fault, link, cancellation)
    # The cancellation walk must tolerate cycles and inspect context even when orig exists.
    fault.__context__ = fault
    attempt = AsyncMock(side_effect=[fault, "written"])
    backoff = AsyncMock()
    monkeypatch.setattr(durability, "_backoff", backoff)
    assert (
        await durability.retry_transient_db(attempt, label="converted", run_id="run") == "written"
    )
    assert asyncio.current_task().cancelling() == 0
    assert attempt.await_count == 2
    backoff.assert_awaited_once()


async def test_genuine_transient_failure_is_still_retried(monkeypatch):
    fault = OperationalError(None, None, Exception("temporary"))
    attempt = AsyncMock(side_effect=[fault, "written"])
    backoff = AsyncMock()
    monkeypatch.setattr(durability, "_backoff", backoff)
    assert await durability.retry_transient_db(attempt, label="write", run_id="run") == "written"
    assert attempt.await_count == 2
    backoff.assert_awaited_once()


@pytest.mark.parametrize("timeout_kind", ["timeout", "wait_for"])
@pytest.mark.parametrize("stale_cancel", [False, True])
async def test_expired_timeout_is_retried_without_cancelling_caller(
    monkeypatch, timeout_kind, stale_cancel
):
    backoff = AsyncMock()
    monkeypatch.setattr(durability, "_backoff", backoff)

    async def operation():
        task = asyncio.current_task()
        if stale_cancel:
            task.cancel()
            # Simulate a previously handled cancellation without uncancel().
            with suppress(asyncio.CancelledError):
                await asyncio.sleep(0)
        before = task.cancelling()
        try:
            if timeout_kind == "timeout":
                async with asyncio.timeout(0):
                    await asyncio.sleep(0)
            else:
                await asyncio.wait_for(asyncio.Event().wait(), timeout=0)
        except TimeoutError as timeout:
            fault = OperationalError(None, None, timeout)
        assert task.cancelling() == before
        attempt = AsyncMock(side_effect=[fault, "written"])
        result = await durability.retry_transient_db(attempt, label="timeout", run_id="run")
        assert result == "written"
        assert attempt.await_count == 2
        assert task.cancelling() == before

    await asyncio.create_task(operation())
    backoff.assert_awaited_once()


async def test_nested_retry_keeps_operation_baseline_and_uses_retry_run_id(monkeypatch, caplog):
    run_id = generate_ulid()
    attempt = AsyncMock(
        side_effect=OperationalError(None, None, Exception("PRIVATE_DRIVER_SENTINEL"))
    )
    backoff = AsyncMock()
    monkeypatch.setattr(durability, "_backoff", backoff)

    @durability._safe_database_call
    async def operation(store):
        asyncio.current_task().cancel()
        with suppress(asyncio.CancelledError):
            await asyncio.sleep(0)
        return await durability.retry_transient_db(attempt, label="nested", run_id=run_id)

    with pytest.raises(asyncio.CancelledError) as caught:
        await asyncio.create_task(operation(object()))
    assert caught.value.__context__ is None and caught.value.__cause__ is None
    assert [(record.code, record.run_id) for record in caplog.records] == [
        ("db_cancel_converted", run_id)
    ]
    assert attempt.await_count == 1
    backoff.assert_not_called()
