"""Cancelled store errors unwind through session close without explicit rollback."""

import asyncio
import sqlite3
from unittest.mock import AsyncMock

import pytest
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.ext.asyncio import AsyncSession

from dendrux.errors import PersistenceError
from dendrux.runtime.durability import _CommitNotAppliedError
from dendrux.types import generate_ulid


@pytest.mark.parametrize("site", ["commit", "create", "claim", "submit"])
@pytest.mark.parametrize("cancelling", [False, True])
async def test_converted_cancellation_skips_error_handler_rollback(
    store, monkeypatch, site, cancelling
):
    run_id = generate_ulid()
    if site != "create":
        await store.create_run(run_id, "cancel-test")
    if site in {"claim", "submit"}:
        await store.pause_run(run_id, status="waiting_client_tool", pause_data={"ready": True})

    cancellation = asyncio.CancelledError()
    fault = (
        IntegrityError(None, None, sqlite3.IntegrityError("PRIVATE_DRIVER_SENTINEL"))
        if site == "create"
        else OperationalError(None, None, Exception("PRIVATE_DRIVER_SENTINEL"))
    )
    original_rollback = AsyncSession.rollback

    async def checked_rollback(session):
        if cancelling:
            raise AssertionError("unsafe explicit rollback was entered")
        await original_rollback(session)

    rollback = AsyncMock(side_effect=checked_rollback)
    close = AsyncSession.close
    closed = []

    async def track_close(session):
        await close(session)
        closed.append(session)

    def fail(*args, **kwargs):
        fault.__context__ = cancellation
        raise fault

    execute = AsyncSession.execute

    async def cancel_current():
        nonlocal cancellation
        task = asyncio.current_task()
        if cancelling and task.cancelling() == 0:
            task.cancel()
            task.cancel()
            try:
                await asyncio.sleep(0)
            except asyncio.CancelledError as delivered:
                cancellation = delivered

    async def deliver_cancel(session, *args, **kwargs):
        result = await execute(session, *args, **kwargs)
        await cancel_current()
        return result

    monkeypatch.setattr(AsyncSession, "execute", deliver_cancel)

    async def rollback_session(session):
        await rollback(session)

    monkeypatch.setattr(AsyncSession, "rollback", rollback_session)
    monkeypatch.setattr(AsyncSession, "close", track_close)
    if site == "create":

        async def fail_write(session):
            await cancel_current()
            fail()

        monkeypatch.setattr(store, "_write_tx", fail_write)
    elif site == "commit":
        monkeypatch.setattr(AsyncSession, "commit", AsyncMock(side_effect=fail))

    async def operation():
        if site == "create":
            await store.create_run(run_id, "cancel-test")
        elif site == "commit":
            await store.finalize_run(run_id, status="success")
        elif site == "claim":
            await store.claim_paused_run(
                run_id, expected_status="waiting_client_tool", validate=fail
            )
        else:
            await store.submit_and_claim(
                run_id, expected_status="waiting_client_tool", submitted_data={}, validate=fail
            )

    task = asyncio.create_task(operation())
    with pytest.raises(asyncio.CancelledError if cancelling else PersistenceError):
        await task
    if cancelling:
        rollback.assert_not_called()
    else:
        assert rollback.await_count >= 1
    assert closed

    # A failed claim must be undone by close on the real database, too.
    if site in {"claim", "submit"}:
        run = await store.get_run(run_id)
        assert run.status == "waiting_client_tool"
        assert run.claim_epoch == 0


@pytest.mark.parametrize("site", ["claim", "submit"])
async def test_non_cancelled_validation_error_rolls_back_claim(store, site):
    run_id = generate_ulid()
    await store.create_run(run_id, "validation-test")
    await store.pause_run(run_id, status="waiting_client_tool", pause_data={"ready": True})
    failure = ValueError("invalid submission")

    def validate(_):
        raise failure

    with pytest.raises(ValueError) as caught:
        if site == "claim":
            await store.claim_paused_run(
                run_id, expected_status="waiting_client_tool", validate=validate
            )
        else:
            await store.submit_and_claim(
                run_id, expected_status="waiting_client_tool", submitted_data={}, validate=validate
            )
    assert caught.value is failure
    run = await store.get_run(run_id)
    assert run.status == "waiting_client_tool"
    assert run.claim_epoch == 0


async def test_non_cancelled_creation_integrity_error_rolls_back(store, monkeypatch):
    run_id = generate_ulid()
    fault = IntegrityError(None, None, sqlite3.IntegrityError("PRIVATE_DRIVER_SENTINEL"))
    rollback = AsyncSession.rollback
    rolled_back = []

    async def track_rollback(session):
        await rollback(session)
        rolled_back.append(session)

    async def fail_after_insert(session):
        await session.flush()
        raise fault

    with monkeypatch.context() as patch:
        patch.setattr(AsyncSession, "rollback", track_rollback)
        patch.setattr(store, "_write_tx", fail_after_insert)
        with pytest.raises(PersistenceError) as caught:
            await store.create_run(run_id, "integrity-test")
    assert not caught.value.retryable
    assert len(rolled_back) == 1
    assert await store.get_run(run_id) is None


@pytest.mark.parametrize("with_idempotency", [False, True])
async def test_non_cancelled_creation_reproves_identity_after_false_absent_readback(
    store, monkeypatch, with_idempotency
):
    run_id = generate_ulid()
    write = store._write_tx
    calls = []

    async def false_absent(session):
        calls.append(session)
        await write(session)
        raise _CommitNotAppliedError()

    with monkeypatch.context() as patch:
        patch.setattr(store, "_write_tx", false_absent)
        result = await store.create_run(
            run_id,
            "identity-test",
            idempotency_key="key" if with_idempotency else None,
            idempotency_fingerprint="input" if with_idempotency else None,
        )
    assert result.run_id == run_id
    assert result.outcome == "created"
    assert len(calls) == 2
    assert (await store.get_run(run_id)).agent_name == "identity-test"
