"""Fenced transitions can prove a committed write without an event row."""

from contextvars import ContextVar
from functools import wraps

import pytest
from sqlalchemy import update
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncSession

from dendrux import Agent
from dendrux.db.models import AgentRun
from dendrux.errors import StoreOutcomeUnknown
from dendrux.llm.mock import MockLLM
from dendrux.types import LLMResponse, RunStatus, generate_ulid
from tests._fixtures.storage import pin_storage_format
from tests._helpers.db_errors import LostConnectionError


@pytest.mark.parametrize("streaming", [False, True])
async def test_format_one_committed_finalize_lost_ack_returns_success(
    engine, store, monkeypatch, streaming
):
    await pin_storage_format(engine, 1)
    active = ContextVar("format_one_finalize", default=False)
    finalize = store.finalize_run
    epochs = []

    @wraps(finalize)
    async def marked(*args, **kwargs):
        epochs.append(kwargs.get("claim_epoch"))
        token = active.set(True)
        try:
            return await finalize(*args, **kwargs)
        finally:
            active.reset(token)

    monkeypatch.setattr(store, "finalize_run", marked)
    commit = AsyncSession.commit
    cuts = 0

    async def commit_then_drop(session):
        nonlocal cuts
        await commit(session)
        if active.get() and cuts == 0:
            cuts += 1
            # Format 1 completion has no caller-unique evidence row.
            assert not session.info.get("dendrux.fresh_evidence")
            raise OperationalError(None, None, LostConnectionError())

    monkeypatch.setattr(AsyncSession, "commit", commit_then_drop)
    agent = Agent(prompt="p", provider=MockLLM([LLMResponse(text="done")]), state_store=store)
    try:
        if streaming:
            async with agent.stream("go", tenant_id=None) as stream:
                events = [event async for event in stream]
            result = events[-1].run_result
        else:
            result = await agent.run("go", tenant_id=None)
        assert cuts == 1
        assert result.status == RunStatus.SUCCESS
        assert epochs == [0], "epoch zero is an explicit caller identity"
        persisted = await store.get_run(result.run_id)
        assert persisted.storage_format == 1
        assert persisted.status == "success"
        assert persisted.claim_epoch == 0
    finally:
        await agent.close()


@pytest.mark.parametrize(
    "epoch,advance_epoch,proven",
    [(None, False, False), (0, False, True), (0, True, False)],
    ids=["no-epoch-unknown", "same-epoch-proven", "changed-epoch-unknown"],
)
async def test_committed_finalize_requires_the_same_caller_epoch(
    engine, store, monkeypatch, epoch, advance_epoch, proven
):
    run_id = generate_ulid()
    await store.create_run(run_id, "agent")
    commit = AsyncSession.commit
    calls = 0

    async def commit_then_drop(session):
        nonlocal calls
        calls += 1
        await commit(session)
        if advance_epoch:
            async with engine.begin() as conn:
                await conn.execute(
                    update(AgentRun).where(AgentRun.id == run_id).values(claim_epoch=1)
                )
        raise OperationalError(None, None, LostConnectionError())

    monkeypatch.setattr(AsyncSession, "commit", commit_then_drop)

    async def finalize():
        return await store.finalize_run(
            run_id, status="success", expected_current_status="running", claim_epoch=epoch
        )

    if proven:
        assert await finalize() is True
    else:
        with pytest.raises(StoreOutcomeUnknown):
            await finalize()
    assert calls == 1
    persisted = await store.get_run(run_id)
    assert persisted.status == "success"
    assert persisted.claim_epoch == (1 if advance_epoch else 0)
