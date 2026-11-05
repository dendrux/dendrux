"""Raw transport errors must cross the same COMMIT proof boundary as DBAPI errors."""

import asyncio
import errno
import logging
import traceback
from unittest.mock import AsyncMock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from dendrux.errors import PersistenceError, StoreOutcomeUnknown
from dendrux.runtime.durability import retry_transient_db
from dendrux.runtime.stores.sql._commit import _CommitReadback
from dendrux.types import generate_ulid

SECRET = "PRIVATE_COMMIT_SENTINEL"
FAULTS = [
    lambda: TimeoutError(SECRET),
    lambda: ConnectionRefusedError(errno.ECONNREFUSED, SECRET),
    lambda: ConnectionResetError(errno.ECONNRESET, SECRET),
    lambda: OSError(errno.EACCES, SECRET),
    lambda: OSError(SECRET),
]


@pytest.mark.parametrize(
    "fault", FAULTS, ids=["timeout", "refused", "reset", "os-other", "os-aggregate"]
)
@pytest.mark.parametrize("phase", ["commit", "rollback", "prepare", "lock", "proof"])
async def test_landed_write_is_proven_or_unknown_never_replayed(
    store, monkeypatch, caplog, fault, phase
):
    caplog.set_level(logging.DEBUG, logger="dendrux")
    run_id = generate_ulid()
    await store.create_run(run_id, "raw-commit")
    commit = AsyncSession.commit
    commits = 0

    async def lost_ack(session):
        nonlocal commits
        commits += 1
        await commit(session)
        raise fault()

    async def attempt():
        await store.save_run_event(run_id, event_type="tick")

    with monkeypatch.context() as patch:
        patch.setattr(AsyncSession, "commit", lost_ack)
        targets = {
            "rollback": (AsyncSession, "rollback"),
            "prepare": (store, "_prepare_commit_readback"),
            "lock": (_CommitReadback, "_lock_touched"),
            "proof": (_CommitReadback, "_check_proof"),
        }
        if phase in targets:
            target, name = targets[phase]
            patch.setattr(target, name, AsyncMock(side_effect=fault()))
        if phase == "commit":
            await retry_transient_db(attempt, label="raw-commit", run_id=run_id)
        else:
            with pytest.raises(StoreOutcomeUnknown) as caught:
                await retry_transient_db(attempt, label="raw-commit", run_id=run_id)
            error = caught.value
            assert not error.retryable
            assert error.__context__ is None and error.__cause__ is None
            assert SECRET not in "".join(traceback.format_exception(error))
            assert SECRET not in repr(vars(error))
    events = await store.get_run_events(run_id)
    assert commits == 1
    assert len(events) == 1
    assert SECRET not in repr(events)
    assert SECRET not in repr([vars(record) for record in caplog.records])


@pytest.mark.parametrize("phase", ["rollback", "prepare", "lock", "proof"])
async def test_raw_server_rejection_during_readback_is_unknown(store, monkeypatch, phase):
    import asyncpg

    run_id = generate_ulid()
    await store.create_run(run_id, "readback-unavailable")
    commit = AsyncSession.commit
    calls = 0

    async def lost_ack(session):
        nonlocal calls
        calls += 1
        await commit(session)
        raise ConnectionResetError(errno.ECONNRESET, SECRET)

    targets = {
        "rollback": (AsyncSession, "rollback"),
        "prepare": (store, "_prepare_commit_readback"),
        "lock": (_CommitReadback, "_lock_touched"),
        "proof": (_CommitReadback, "_check_proof"),
    }
    with monkeypatch.context() as patch:
        patch.setattr(AsyncSession, "commit", lost_ack)
        target, name = targets[phase]
        patch.setattr(target, name, AsyncMock(side_effect=asyncpg.CannotConnectNowError(SECRET)))
        with pytest.raises(StoreOutcomeUnknown):
            await retry_transient_db(
                lambda: store.save_run_event(run_id, event_type="tick"),
                label="readback-unavailable",
                run_id=run_id,
            )
    assert calls == 1
    assert len(await store.get_run_events(run_id)) == 1


@pytest.mark.parametrize("fault", [ValueError("logical"), asyncio.CancelledError()])
async def test_commit_programming_error_and_cancellation_propagate(store, monkeypatch, fault):
    run_id = generate_ulid()
    await store.create_run(run_id, "commit-propagation")
    with monkeypatch.context() as patch:
        commit = AsyncMock(side_effect=fault)
        patch.setattr(AsyncSession, "commit", commit)
        with pytest.raises(type(fault)) as caught:
            await store.save_run_event(run_id, event_type="tick")
        assert caught.value is fault
        assert commit.await_count == 1
    assert await store.get_run_events(run_id) == []


@pytest.mark.parametrize("phase", ["rollback", "prepare", "lock", "proof"])
@pytest.mark.parametrize("fault", [ValueError("logical"), asyncio.CancelledError()])
async def test_readback_non_database_failure_is_not_normalized(store, monkeypatch, phase, fault):
    run_id = generate_ulid()
    await store.create_run(run_id, "readback-cancel")
    commit = AsyncSession.commit

    async def lost_ack(session):
        await commit(session)
        raise ConnectionResetError(errno.ECONNRESET, SECRET)

    targets = {
        "rollback": (AsyncSession, "rollback"),
        "prepare": (store, "_prepare_commit_readback"),
        "lock": (_CommitReadback, "_lock_touched"),
        "proof": (_CommitReadback, "_check_proof"),
    }
    with monkeypatch.context() as patch:
        patch.setattr(AsyncSession, "commit", lost_ack)
        target, name = targets[phase]
        patch.setattr(target, name, AsyncMock(side_effect=fault))
        with pytest.raises(type(fault)) as caught:
            await store.save_run_event(run_id, event_type="tick")
        assert caught.value is fault
        assert caught.value.__context__ is None
        assert caught.value.__cause__ is None
        assert SECRET not in "".join(traceback.format_exception(caught.value))
    assert len(await store.get_run_events(run_id)) == 1


@pytest.mark.parametrize("stable_id", [False, True])
async def test_absent_raw_timeout_retries_only_a_stable_write(store, monkeypatch, stable_id):
    run_id = generate_ulid()
    await store.create_run(run_id, "absent-commit")
    commit = AsyncSession.commit
    calls = 0

    async def fail_before_commit(session):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise TimeoutError(SECRET)
        await commit(session)

    event_id = generate_ulid() if stable_id else None
    with monkeypatch.context() as patch:
        patch.setattr(AsyncSession, "commit", fail_before_commit)
        operation = retry_transient_db(
            lambda: store.save_run_event(run_id, event_type="tick", event_id=event_id),
            label="absent-commit",
            run_id=run_id,
        )
        if stable_id:
            await operation
        else:
            with pytest.raises(StoreOutcomeUnknown):
                await operation
    assert calls == (2 if stable_id else 1)
    assert len(await store.get_run_events(run_id)) == (1 if stable_id else 0)


async def test_precommit_driver_timeout_is_not_retried(store, monkeypatch):
    class DriverTimeoutError(TimeoutError):
        __module__ = "asyncpg"

    run_id = generate_ulid()
    await store.create_run(run_id, "precommit-timeout")
    execute = AsyncSession.execute
    calls = 0

    async def fail_once(session, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise DriverTimeoutError(SECRET)
        return await execute(session, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(AsyncSession, "execute", fail_once)
        with pytest.raises(PersistenceError) as caught:
            await retry_transient_db(
                lambda: store.save_run_event(run_id, event_type="tick"),
                label="precommit-timeout",
                run_id=run_id,
            )
        assert not caught.value.retryable
    assert calls == 1
    assert await store.get_run_events(run_id) == []
