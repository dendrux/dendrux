"""Creation keeps seed atomicity, storage capability checks and fail-closed readback."""

from dataclasses import replace
from unittest.mock import AsyncMock

import pytest
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from dendrux.errors import (
    PersistenceError,
    RunCancelledError,
    RunNotFoundError,
    StoreOutcomeUnknown,
)
from dendrux.runtime.durability import _CommitNotAppliedError
from dendrux.runtime.stores.sql._commit import _CommitReadback
from dendrux.testing.store_kit.contracts._support import trace_write
from dendrux.types import IdempotencyConflictError, generate_ulid


async def test_seed_failure_rolls_back_creation_and_allows_the_same_identity_retry(
    store, monkeypatch
):
    run_id = generate_ulid()
    seeds = [trace_write("system", "system", 0, "seed"), trace_write("user", "user", 1, "seed")]
    original = store._insert_trace
    calls = []

    async def fail_after_seed(session, seed_run_id, trace):
        calls.append(trace.trace_id)
        await original(session, seed_run_id, trace)
        if len(calls) == 2:
            raise ValueError("forced seed boundary")

    with monkeypatch.context() as patch:
        patch.setattr(store, "_insert_trace", fail_after_seed)
        with pytest.raises(ValueError, match="forced seed boundary"):
            await store.create_run(
                run_id,
                "seed-atomicity",
                tenant_id="t",
                idempotency_key="same-key",
                idempotency_fingerprint="same-input",
                seed_traces=seeds,
            )
    assert calls == [trace.trace_id for trace in seeds]
    assert await store.get_run(run_id) is None
    assert await store.get_traces(run_id) == []
    assert await store._check_idempotency("same-key", "same-input", tenant_id="t") is None
    result = await store.create_run(
        run_id,
        "seed-atomicity",
        tenant_id="t",
        idempotency_key="same-key",
        idempotency_fingerprint="same-input",
        seed_traces=seeds,
    )
    assert result.outcome == "created"
    assert [(trace.role, trace.content) for trace in await store.get_traces(run_id)] == [
        ("system", "system"),
        ("user", "user"),
    ]


async def test_storage_capability_refusal_precedes_the_cached_format_two_path(store, monkeypatch):
    assert await store._creation_storage_format() == 2
    assert store._format2_active is True
    monkeypatch.setattr(store, "capabilities", replace(store.capabilities, checkpointing=False))

    def forbidden_session():
        raise AssertionError("a capability refusal must not open a session")

    monkeypatch.setattr(store, "_session_factory", forbidden_session)
    assert await store._creation_storage_format() == 1


async def test_creation_with_unprovable_readback_never_reports_success(store, monkeypatch):
    run_id = generate_ulid()
    commit = AsyncSession.commit

    async def lost_ack(session):
        await commit(session)
        raise OperationalError("commit-fault", {}, ConnectionError())

    absent = AsyncMock(side_effect=_CommitNotAppliedError())
    unavailable = AsyncMock(side_effect=SQLAlchemyError("forced readback boundary"))
    with monkeypatch.context() as patch:
        patch.setattr(AsyncSession, "commit", lost_ack)
        patch.setattr(_CommitReadback, "_prove", absent)
        patch.setattr(store, "_prepare_commit_readback", unavailable)
        with pytest.raises(StoreOutcomeUnknown):
            await store.create_run(run_id, "unknown-creation")
    absent.assert_awaited_once()
    unavailable.assert_awaited_once()
    assert await store.get_run_status(run_id) == "running"


@pytest.mark.parametrize("parent_state", ["missing", "cancelled"])
async def test_creation_refuses_an_unavailable_parent_without_leaving_a_child(store, parent_state):
    parent, child = generate_ulid(), generate_ulid()
    expected = RunNotFoundError
    if parent_state == "cancelled":
        await store.create_run(parent, "parent")
        await store.request_cancel(parent)
        expected = RunCancelledError
    with pytest.raises(expected):
        await store.create_run(child, "child", parent_run_id=parent)
    assert await store.get_run(child) is None
    assert await store.get_traces(child) == []


async def test_late_idempotency_conflict_retains_the_original_winner(store, monkeypatch):
    winner, contender = generate_ulid(), generate_ulid()
    await store.create_run(
        winner, "winner", tenant_id="t", idempotency_key="key", idempotency_fingerprint="first"
    )
    original = store._check_idempotency
    calls = []

    async def miss_once(*args, **kwargs):
        calls.append(1)
        return None if len(calls) == 1 else await original(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(store, "_check_idempotency", miss_once)
        with pytest.raises(IdempotencyConflictError) as caught:
            await store.create_run(
                contender,
                "contender",
                tenant_id="t",
                idempotency_key="key",
                idempotency_fingerprint="different",
            )
    assert caught.value.run_id == winner
    assert len(calls) == 2
    assert await store.get_run(contender) is None
    assert (await store.get_run(winner)).agent_name == "winner"


async def test_non_identity_integrity_failure_is_not_reported_as_deduplication(store):
    run_id = generate_ulid()
    with pytest.raises(PersistenceError):
        await store.create_run(run_id, "retry-lineage", retry_of_run_id=generate_ulid())
    assert await store.get_run(run_id) is None
    result = await store.create_run(run_id, "retry-lineage")
    assert result.outcome == "created"
