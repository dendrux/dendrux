"""Review regressions at the SQL store transaction boundary."""

from contextvars import ContextVar
from functools import wraps

import pytest
from sqlalchemy import select
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncSession

from dendrux import Agent, tool
from dendrux.db.models import AgentRun
from dendrux.llm.mock import MockLLM
from dendrux.runtime import durability
from dendrux.runtime.state import EventWrite
from dendrux.types import LLMResponse, RunStatus, ToolCall, generate_ulid
from tests._helpers.db_errors import LostConnectionError


@pytest.mark.parametrize("operation", ["pause_with_event", "finalize_with_event"])
@pytest.mark.parametrize("streaming", [False, True])
async def test_dropped_transition_commit_retries_without_a_cas_miss(
    store, monkeypatch, operation, streaming
):
    active = ContextVar("commit_cut", default=False)
    method = getattr(store, operation)

    @wraps(method)
    async def marked(*args, **kwargs):
        token = active.set(True)
        try:
            return await method(*args, **kwargs)
        finally:
            active.reset(token)

    monkeypatch.setattr(store, operation, marked)
    commit = AsyncSession.commit
    cuts = 0

    async def drop_once(session):
        nonlocal cuts
        if active.get() and cuts == 0:
            cuts += 1
            raise OperationalError(None, None, LostConnectionError())
        await commit(session)

    monkeypatch.setattr(AsyncSession, "commit", drop_once)
    monkeypatch.setattr(durability, "_BASE_BACKOFF_S", 0)

    @tool(target="client")
    def client_read() -> str:
        """Read client-owned data."""
        raise AssertionError("client execution")

    response = (
        LLMResponse(tool_calls=[ToolCall("client_read", id="c1")])
        if operation == "pause_with_event"
        else LLMResponse(text="done")
    )
    agent = Agent(prompt="p", provider=MockLLM([response]), tools=[client_read], state_store=store)
    try:
        if streaming:
            async with agent.stream("go", tenant_id=None) as stream:
                events = [event async for event in stream]
            result = events[-1].run_result
        else:
            result = await agent.run("go", tenant_id=None)
        expected = (
            RunStatus.WAITING_CLIENT_TOOL if operation == "pause_with_event" else RunStatus.SUCCESS
        )
        assert cuts == 1
        assert result.status == expected
        assert (await store.get_run(result.run_id)).status == expected.value
        assert not (await store.get_run(result.run_id)).cancel_requested
    finally:
        await agent.close()


async def test_matching_finalizer_without_fresh_identity_is_unknown(store, monkeypatch):
    run_id = generate_ulid()
    await store.create_run(run_id, "agent")
    commit = AsyncSession.commit

    async def commit_then_drop(session):
        await commit(session)
        raise OperationalError(None, None, LostConnectionError())

    monkeypatch.setattr(AsyncSession, "commit", commit_then_drop)
    with pytest.raises(durability.StoreOutcomeUnknown):
        await store.finalize_run_if_status_in(
            run_id, status="success", allowed_current_statuses=["running"]
        )


@pytest.mark.parametrize("operation", ["submit_and_claim", "release_run_claim"])
async def test_ownership_ack_refused_even_with_fresh_matching_evidence(
    store, monkeypatch, operation
):
    from dendrux.db.models import RunEvent
    from dendrux.runtime.state import _write_event

    run_id = generate_ulid()
    await store.create_run(run_id, "agent")
    await store.pause_run(run_id, status="waiting_client_tool", pause_data={"pause_id": "p"})
    if operation == "release_run_claim":
        claim = await store.claim_paused_run(run_id, expected_status="waiting_client_tool")
    write = store._write_tx

    async def with_evidence(session, **kwargs):
        await _write_event(
            session, run_id, EventWrite(generate_ulid(), "ownership.proof", 0, {}, None, None)
        )
        # This makes the fixture fail if the refusal rule is removed: all
        # evidence readbacks can match, even though epoch ownership is ambiguous.
        assert any(
            table.name == RunEvent.__tablename__
            for table, _, _ in session.info["dendrux.commit_proofs"]
        )
        await write(session, **kwargs)

    monkeypatch.setattr(store, "_write_tx", with_evidence)
    commit = AsyncSession.commit
    calls = 0

    async def commit_then_drop(session):
        nonlocal calls
        calls += 1
        await commit(session)
        raise OperationalError(None, None, LostConnectionError())

    monkeypatch.setattr(AsyncSession, "commit", commit_then_drop)
    with pytest.raises(durability.StoreOutcomeUnknown):
        if operation == "submit_and_claim":
            await store.submit_and_claim(
                run_id,
                expected_status="waiting_client_tool",
                submitted_data={},
                submission={
                    "submission_id": generate_ulid(),
                    "kind": "tool_results",
                    "pause_id": "p",
                    "call_ids": [],
                },
            )
        else:
            await store.release_run_claim(
                run_id, status="waiting_client_tool", claim_epoch=claim.claim_epoch
            )
    assert calls == 1


@pytest.mark.parametrize(
    "actual,query", [([1, 2], [1]), ([1, 2], 1), (1, True), (2, 2.0), (True, 1)]
)
async def test_metadata_filter_preserves_backend_matching(store, engine, actual, query):
    from dendrux.errors import PersistenceError

    run_id = generate_ulid()
    await store.create_run(run_id, "agent", meta={"key": actual})
    if engine.dialect.name == "sqlite" and isinstance(query, list):
        # b2938693 binds the comparison value directly; SQLite rejects a list.
        with pytest.raises(PersistenceError):
            await store.count_runs(metadata_filter={"key": query})
        return
    expected = (
        (actual == query)
        if engine.dialect.name == "sqlite"
        else not isinstance(actual, bool) and not isinstance(query, bool)
    )
    if engine.dialect.name == "postgresql" and isinstance(actual, list):
        expected = isinstance(query, list)
    assert await store.count_runs(metadata_filter={"key": query}) == int(expected)


@pytest.mark.parametrize("engine", ["sqlite_file", "postgres"], indirect=True)
async def test_two_instances_merge_unfenced_metadata_without_losing_keys(
    engine, store, monkeypatch
):
    import asyncio

    from sqlalchemy.sql import Select, Update

    from dendrux.runtime.stores._factory import store_for_engine

    run_id = generate_ulid()
    await store.create_run(run_id, "agent", meta={"initial": True})
    peer = store_for_engine(engine)
    original = AsyncSession.execute
    barrier = asyncio.Barrier(2)
    writers = set()

    async def synchronize(session, stmt, *args, **kwargs):
        actor = asyncio.current_task()
        if isinstance(stmt, Update) or str(stmt).lstrip().startswith("UPDATE "):
            writers.add(actor)
        result = await original(session, stmt, *args, **kwargs)
        if (
            engine.dialect.name == "sqlite"
            and isinstance(stmt, Select)
            and actor not in writers
            and "agent_runs" in str(stmt)
        ):
            await asyncio.wait_for(barrier.wait(), 10)
        return result

    monkeypatch.setattr(AsyncSession, "execute", synchronize)
    await asyncio.wait_for(
        asyncio.gather(
            store.update_run_meta(run_id, {"a": 1}), peer.update_run_meta(run_id, {"b": 2})
        ),
        15,
    )
    async with engine.connect() as conn:
        actual = await conn.scalar(select(AgentRun.meta).where(AgentRun.id == run_id))
    assert actual == {"initial": True, "a": 1, "b": 2}


@pytest.mark.parametrize("operation", ["pause_run", "consume_submission"])
async def test_proven_noncommit_raises_for_an_outer_retry(store, monkeypatch, operation):
    from dendrux.runtime.state import LoopStatePatch

    run_id, submission_id = generate_ulid(), generate_ulid()
    await store.create_run(run_id, "agent")
    await store.checkpoint_step(
        run_id,
        claim_epoch=0,
        pii_mapping=None,
        trace=None,
        loop_state={"v": 1, "phase": "dispatch", "submission": {"submission_id": submission_id}},
    )
    commit = AsyncSession.commit
    calls = 0

    async def drop_once(session):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OperationalError(None, None, LostConnectionError())
        await commit(session)

    monkeypatch.setattr(AsyncSession, "commit", drop_once)
    monkeypatch.setattr(durability, "_BASE_BACKOFF_S", 0)

    async def attempt():
        if operation == "pause_run":
            await store.pause_run(
                run_id, status="waiting_client_tool", pause_data={"pause_id": "p"}
            )
        else:
            # The recorder discards consume_submission's bool; retry must be
            # signalled by an exception even when no trace evidence is written.
            await store.consume_submission(
                run_id,
                claim_epoch=0,
                submission_id=submission_id,
                traces=[],
                patch=LoopStatePatch(set_phase="model_next"),
            )

    await durability.retry_critical(attempt, label="test", run_id=run_id)
    assert calls == 2
    if operation == "pause_run":
        assert (await store.get_run(run_id)).status == "waiting_client_tool"
    else:
        assert (await store.get_loop_state(run_id))["phase"] == "model_next"


@pytest.mark.parametrize("site", ["mark", "sweep"])
async def test_recovery_statements_order_readers_before_cancel_clear(
    store, engine, monkeypatch, site
):
    from datetime import timedelta

    from sqlalchemy import update
    from sqlalchemy.dialects import mysql
    from sqlalchemy.sql.dml import Update

    from tests.integration.test_060_core_recovery import mark, stale

    run_id = (await stale(store, engine))[0]
    async with engine.begin() as conn:
        await conn.execute(
            update(AgentRun).where(AgentRun.id == run_id).values(cancel_requested=True)
        )
    statements = []
    execute = AsyncSession.execute

    async def capture(session, stmt, *args, **kwargs):
        if isinstance(stmt, Update):
            sql = str(stmt.compile(dialect=mysql.dialect()))
            if "status=CASE" in sql and "cancel_requested=" in sql:
                statements.append(sql)
        return await execute(session, stmt, *args, **kwargs)

    monkeypatch.setattr(AsyncSession, "execute", capture)
    if site == "mark":
        await mark(store, finalise_unrecoverable=True)
    else:
        await store.sweep_stale_runs(
            timedelta(minutes=1),
            batch_size=10,
            max_runs=100,
        )
    assert statements, "capture the actual recovery UPDATE, not a reconstructed expression"
    for sql in statements:
        assert sql.index("status_changed_at=CASE") < sql.index("status=CASE")
        assert sql.index("status=CASE") < sql.index("cancel_requested=")
