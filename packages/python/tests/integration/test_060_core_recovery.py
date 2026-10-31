"""Recovery marking, keyset scans, tenancy and competing reclaimers."""

import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import update

from dendrux.db.models import AgentRun
from dendrux.errors import RunNotFoundError
from dendrux.types import generate_ulid


async def stale(store, engine, *, tenant="tenant", count=1):
    ids = []
    for _ in range(count):
        run_id = generate_ulid()
        await store.create_run(run_id, "recovery", strategy="react", tenant_id=tenant)
        ids.append(run_id)
    async with engine.begin() as conn:
        await conn.execute(
            update(AgentRun)
            .where(AgentRun.id.in_(ids))
            .values(
                last_progress_at=datetime.now(UTC) - timedelta(minutes=10),
                loop_state={"loop": "react"},
            )
        )
    return ids


async def mark(store, **kwargs):
    assert hasattr(store, "mark_lease_expired"), "recovery needs an atomic lease-expiry mark"
    return await store.mark_lease_expired(
        cutoff=datetime.now(UTC) - timedelta(minutes=1),
        **{
            **dict(batch_size=10, max_runs=100, max_returned=100, reclaim_limit=3, cursor=None),
            **kwargs,
        },
    )


async def reclaim(store, item, tenant="tenant"):
    return await store.reclaim_run(
        item.run_id,
        tenant_id=tenant,
        agent_name="recovery",
        expected_claim_epoch=item.claim_epoch,
        governance={},
        governance_policy="raise",
        skill_pins={},
        loop_type="react",
    )


async def race(store, engine, operations):
    """Prewarm every connection, then release checked-out actors together."""
    if engine.dialect.name != "postgresql":
        return await asyncio.gather(*(op(store) for op in operations), return_exceptions=True)
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

    from dendrux.runtime.stores._factory import store_for_engine

    pooled = create_async_engine(engine.url, pool_size=len(operations), max_overflow=0)
    warm = await asyncio.gather(*(pooled.connect() for _ in operations))
    await asyncio.gather(*(conn.close() for conn in warm))
    barrier = asyncio.Barrier(len(operations))
    entered = set()

    class RacingSession(AsyncSession):
        async def execute(self, statement, *args, **kwargs):
            actor = asyncio.current_task()
            first = actor not in entered
            if first:
                entered.add(actor)
                await self.connection()
                await asyncio.wait_for(barrier.wait(), 10)
            result = await super().execute(statement, *args, **kwargs)
            if first:
                # Give all unlocked readers a chance to observe the old epoch.
                # Correct FOR UPDATE readers remain blocked until the winner commits.
                await asyncio.sleep(0.03)
            return result

    contender = store_for_engine(pooled)
    contender._session_factory = async_sessionmaker(
        pooled, class_=RacingSession, expire_on_commit=False
    )
    try:
        return await asyncio.wait_for(
            asyncio.gather(*(op(contender) for op in operations), return_exceptions=True), 20
        )
    finally:
        await pooled.dispose()


async def test_a2_twenty_reclaimers_one_winner(store, engine):
    await stale(store, engine)
    item = (await mark(store)).reclaimable[0]
    results = await race(store, engine, [lambda s: reclaim(s, item) for _ in range(20)])
    winners = [r for r in results if not isinstance(r, BaseException) and r is not False]
    assert len(winners) == 1
    losers = [r for r in results if isinstance(r, BaseException)]
    assert len(losers) == 19 and all(type(r).__name__ == "RunNotReclaimableError" for r in losers)
    row = await store.get_run(item.run_id)
    assert row.claim_epoch == 2 and row.reclaim_attempts == 1
    assert row.lease_expired_at is None and row.last_reclaimed_at is not None
    assert [e.event_type for e in await store.get_run_events(item.run_id)] == [
        "run.lease_expired",
        "run.reclaimed",
    ]


async def test_a13_a14_two_markers_rediscover_without_duplicate_events(store, engine):
    ids = await stale(store, engine)
    first, second = await race(store, engine, [mark, mark])
    assert first.reclaimable[0].claim_epoch == second.reclaimable[0].claim_epoch == 1
    again = await store.get_lease_expired(limit=10, cursor=None)
    assert [r.run_id for r in again] == ids
    assert len(await store.get_run_events(ids[0])) == 1


async def test_a15_cursor_covers_backlog_never_wraps(store, engine):
    ids = await stale(store, engine, count=7)
    seen, cursor = [], None
    for _ in range(5):
        page = await mark(store, max_runs=3, max_returned=2, cursor=cursor)
        seen.extend(r.run_id for r in page.reclaimable)
        cursor = page.cursor
        if cursor is None:
            break
    assert seen == ids and len(set(seen)) == 7 and cursor is None
    assert not (await mark(store, cursor=ids[-1])).reclaimable


async def test_a12_foreign_and_missing_refused_without_mutation(store, engine):
    await stale(store, engine)
    item = (await mark(store)).reclaimable[0]
    before = await store.get_run(item.run_id)
    with pytest.raises(RunNotFoundError):
        await reclaim(store, item, tenant="foreign")
    after = await store.get_run(item.run_id)
    assert before == after
    from dataclasses import replace

    with pytest.raises(RunNotFoundError):
        await reclaim(store, replace(item, run_id=generate_ulid()))


async def test_a6_limit_mark_finalizes_exhausted_run(store, engine):
    await stale(store, engine)
    item = (await mark(store, reclaim_limit=1)).reclaimable[0]
    await reclaim(store, item)
    async with engine.begin() as conn:
        await conn.execute(
            update(AgentRun)
            .where(AgentRun.id == item.run_id)
            .values(
                last_progress_at=datetime.now(UTC) - timedelta(minutes=10),
                loop_state={"loop": "react"},
            )
        )
    result = await mark(store, reclaim_limit=1)
    assert not result.reclaimable
    assert result.finalised[0].failure_reason == "reclaim_limit"
    assert (await store.get_run(item.run_id)).status == "error"


async def test_fresh_scan_has_no_per_run_read(store, engine):
    from sqlalchemy import event

    for _ in range(20):
        await store.create_run(generate_ulid(), "fresh", strategy="react")
    statements = []

    def capture(conn, cursor, statement, parameters, context, executemany):
        statements.append(statement)

    event.listen(engine.sync_engine, "before_cursor_execute", capture)
    try:
        result = await mark(store, batch_size=100)
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", capture)
    assert result.inspected == 20 and not result.reclaimable
    assert len(statements) <= 2, "fresh runs are inspected in a page, without N+1 row reads"


async def test_recovery_reads_exclude_unneeded_payload_columns(store, engine):
    from sqlalchemy import event

    await stale(store, engine)
    item = (await mark(store)).reclaimable[0]
    statements = []

    def capture(conn, cursor, statement, parameters, context, executemany):
        statements.append(statement)

    event.listen(engine.sync_engine, "before_cursor_execute", capture)
    try:
        await store.get_lease_expired(limit=10, cursor=None)
        await reclaim(store, item)
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", capture)
    for sql in statements:
        if sql.lstrip().upper().startswith("SELECT") or " RETURNING " in sql:
            assert all(
                name not in sql
                for name in ("input_data", "output_data", "pii_mapping", "pause_data")
            ), "recovery must project execution metadata without loading conversation payloads"


async def test_new_mark_persists_the_current_reclaim_limit(store, engine):
    await stale(store, engine)
    first = (await mark(store, reclaim_limit=1)).reclaimable[0]
    await reclaim(store, first)
    async with engine.begin() as conn:
        await conn.execute(
            update(AgentRun)
            .where(AgentRun.id == first.run_id)
            .values(
                last_progress_at=datetime.now(UTC) - timedelta(minutes=10),
                loop_state={"loop": "react"},
            )
        )
    second = await mark(store, reclaim_limit=2)
    assert len(second.reclaimable) == 1 and not second.finalised
    assert second.reclaimable[0].reclaim_limit == 2


async def test_marker_rediscovery_preserves_every_projected_field(store, engine):
    """Repeated CAS losers and marked-row scans must use the same named projection."""
    cursor = None
    for _ in range(10):
        ids = await stale(store, engine)
        outcomes = await race(
            store,
            engine,
            [
                lambda s, cursor=cursor: mark(s, cursor=cursor),
                lambda s, cursor=cursor: mark(s, cursor=cursor),
            ],
        )
        expected = next(
            r for r in await store.get_lease_expired(limit=100, cursor=cursor) if r.run_id == ids[0]
        )
        assert all(not isinstance(result, BaseException) for result in outcomes)
        for outcome in outcomes:
            assert outcome.reclaimable == [expected]
        assert expected.tenant_id == "tenant" and expected.agent_name == "recovery"
        assert expected.claim_epoch == 1 and expected.reclaim_attempts == 0
        assert expected.reclaim_limit == 3
        cursor = ids[0]


async def test_reclaim_returning_preserves_distinct_epoch_and_format(store, engine):
    ids = await stale(store, engine)
    async with engine.begin() as conn:
        await conn.execute(
            update(AgentRun).where(AgentRun.id == ids[0]).values(claim_epoch=7, reclaim_attempts=1)
        )
    item = (await mark(store)).reclaimable[0]
    assert item.claim_epoch == 8
    claimed = await reclaim(store, item)
    row = await store.get_run(ids[0])
    assert claimed.claim_epoch == row.claim_epoch == 9
    assert claimed.storage_format == row.storage_format == 2
    assert claimed.loop_state == await store.get_loop_state(ids[0]) == {"loop": "react"}
    assert claimed.submission is None and row.reclaim_attempts == 2


@pytest.mark.parametrize("poison", [None, {}, {"v": 1, "loop": "unknown"}])
async def test_mark_skips_poisoned_loop_and_advances_cursor(store, engine, caplog, poison):
    ids = await stale(store, engine, count=2)
    async with engine.begin() as conn:
        await conn.execute(update(AgentRun).where(AgentRun.id == ids[0]).values(loop_state=poison))
    result = await mark(store)
    assert result.inspected == 2
    assert [item.run_id for item in result.reclaimable] == [ids[1]]
    assert not result.finalised and not result.legacy
    row = await store.get_run(ids[0])
    assert row.status == "running" and row.claim_epoch == 0
    assert row.lease_expired_at is None
    assert not await store.get_run_events(ids[0])
    assert any(
        getattr(r, "code", None) == "loop_unknown" and getattr(r, "run_id", None) == ids[0]
        for r in caplog.records
    )
    page = await mark(store, max_runs=1)
    assert page.inspected == 1 and page.cursor == ids[0]
    assert not page.reclaimable
    next_page = await mark(store, cursor=page.cursor)
    assert [item.run_id for item in next_page.reclaimable] == [ids[1]]


@pytest.mark.parametrize("loop_type", ["react", "single_call"])
@pytest.mark.parametrize("streaming", [False, True])
async def test_builtin_loop_marker_is_present_at_create_and_every_checkpoint(
    store, monkeypatch, loop_type, streaming
):
    from functools import wraps

    from dendrux import Agent, SingleCall
    from dendrux.llm.mock import MockLLM
    from dendrux.types import LLMResponse

    create = store.create_run
    checkpoint = store.checkpoint_step
    initial = []
    boundaries = []

    @wraps(create)
    async def inspect_create(*args, **kwargs):
        result = await create(*args, **kwargs)
        initial.append(await store.get_loop_state(result.run_id))
        return result

    @wraps(checkpoint)
    async def inspect_checkpoint(*args, **kwargs):
        boundaries.append(kwargs["loop_state"].get("loop"))
        return await checkpoint(*args, **kwargs)

    monkeypatch.setattr(store, "create_run", inspect_create)
    monkeypatch.setattr(store, "checkpoint_step", inspect_checkpoint)
    agent = Agent(
        prompt="p",
        provider=MockLLM([LLMResponse(text="done")]),
        state_store=store,
        strict_tenancy=False,
        loop=SingleCall() if loop_type == "single_call" else None,
    )
    if streaming:
        stream = agent.stream("go")
        async for _ in stream:
            pass
        result = stream.result
    else:
        result = await agent.run("go")
    assert result.status.value == "success"
    assert initial == [{"v": 1, "loop": loop_type}]
    assert boundaries and set(boundaries) == {loop_type}
    assert (await store.get_run(result.run_id)).strategy == "NativeToolCalling"


@pytest.mark.parametrize("kind", ["react", "single_call", "custom", "named_like_builtin"])
@pytest.mark.parametrize("entry", ["run", "stream", "retry"])
async def test_creation_marker_uses_loop_type(store, monkeypatch, kind, entry):
    from functools import wraps

    from dendrux import Agent, SingleCall
    from dendrux.llm.mock import MockLLM
    from dendrux.loops.base import Loop
    from dendrux.loops.react import ReActLoop
    from dendrux.types import LLMResponse, RunResult, RunStatus

    class Custom(Loop):
        async def run(self, agent, user_input, *, run_id=None, **kwargs):
            return RunResult(run_id=run_id, status=RunStatus.SUCCESS, answer="done")

    loop = {
        "react": type("DerivedReact", (ReActLoop,), {}),
        "single_call": type("DerivedSingle", (SingleCall,), {}),
        "custom": Custom,
        "named_like_builtin": type("ReActLoop", (Custom,), {}),
    }[kind]()
    expected = {"v": 1, "loop": kind} if kind in {"react", "single_call"} else None
    inspected = []
    create = store.create_run

    @wraps(create)
    async def inspect(*args, **kwargs):
        result = await create(*args, **kwargs)
        # Creation committed; the loop has not executed or checkpointed yet.
        assert (await store.get_run(result.run_id)).meta["dendrux.loop"] == type(loop).__name__
        assert await store.get_loop_state(result.run_id) == expected
        inspected.append(result.run_id)
        return result

    monkeypatch.setattr(store, "create_run", inspect)
    agent = Agent(
        prompt="p",
        provider=MockLLM([LLMResponse(text="done"), LLMResponse(text="again")]),
        loop=loop,
        state_store=store,
        strict_tenancy=False,
    )
    if entry == "stream":
        async with agent.stream("go") as stream:
            events = [event async for event in stream]
        result = events[-1].run_result
    else:
        result = await agent.run("go")
        if entry == "retry" and kind == "single_call":
            with pytest.raises(ValueError, match="SingleCall runs cannot be retried"):
                await agent.retry(result.run_id)
        elif entry == "retry":
            result = await agent.retry(result.run_id)
    assert result.status == RunStatus.SUCCESS
    assert len(inspected) == (2 if entry == "retry" and kind != "single_call" else 1)


@pytest.mark.parametrize("fmt", [1, 2])
async def test_parallel_creation_retains_loop_class_metadata(store, engine, fmt):
    from functools import wraps

    from dendrux import Agent, SingleCall
    from dendrux.llm.mock import MockLLM
    from dendrux.loops.react import ReActLoop
    from dendrux.types import LLMResponse
    from tests._fixtures.storage import pin_storage_format

    await pin_storage_format(engine, fmt)
    create = store.create_run
    entrants = 0
    ready = asyncio.Event()
    initial = {}

    @wraps(create)
    async def together(*args, **kwargs):
        nonlocal entrants
        entrants += 1
        if entrants == 8:
            ready.set()
        await asyncio.wait_for(ready.wait(), 10)
        result = await create(*args, **kwargs)
        initial[result.run_id] = await store.get_loop_state(result.run_id)
        return result

    store.create_run = together

    async def execute(index):
        base = SingleCall if index % 2 else ReActLoop
        loop = type(f"DistinctLoop{index}", (base,), {})()
        agent = Agent(
            prompt="p",
            loop=loop,
            state_store=store,
            strict_tenancy=False,
            provider=MockLLM([LLMResponse(text="done")]),
        )
        if index % 3:
            result = await agent.run("go")
        else:
            async with agent.stream("go") as stream:
                events = [event async for event in stream]
            result = events[-1].run_result
        assert result.status.value == "success"
        assert (await store.get_run(result.run_id)).meta["dendrux.loop"] == type(loop).__name__
        assert initial[result.run_id] == (
            {"v": 1, "loop": "single_call" if index % 2 else "react"} if fmt == 2 else None
        )

    await asyncio.gather(*(execute(i) for i in range(8)))


@pytest.mark.parametrize("failure", ["cancel", "error"])
async def test_creation_scope_does_not_outlive_a_failed_call(store, monkeypatch, failure):
    from functools import wraps

    from dendrux import Agent
    from dendrux.llm.mock import MockLLM
    from dendrux.loops.react import ReActLoop

    create = store.create_run
    attempted = []

    @wraps(create)
    async def failed(run_id, *args, **kwargs):
        attempted.append(run_id)
        if failure == "cancel":
            raise asyncio.CancelledError
        raise RuntimeError("creation failed")

    monkeypatch.setattr(store, "create_run", failed)
    agent = Agent(
        prompt="p",
        provider=MockLLM([]),
        state_store=store,
        strict_tenancy=False,
        loop=type("DerivedReact", (ReActLoop,), {})(),
    )
    error = asyncio.CancelledError if failure == "cancel" else RuntimeError
    with pytest.raises(error):
        await agent.run("go")
    assert len(attempted) == 1
    await create(attempted[0], "direct", meta={"dendrux.loop": "SingleCall"})
    assert await store.get_loop_state(attempted[0]) == {"v": 1, "loop": "single_call"}
