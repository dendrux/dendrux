"""Readback waits for an independently committing PostgreSQL writer after a raw error."""

import asyncio

import pytest
from sqlalchemy import insert, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from dendrux.db.models import AgentRun, RunEvent
from dendrux.runtime.durability import retry_transient_db
from dendrux.types import generate_ulid


@pytest.mark.parametrize("stable_id", [False, True])
@pytest.mark.parametrize("fault_type", [OSError, TimeoutError])
async def test_raw_failure_while_commit_is_in_flight(
    engine, store, monkeypatch, stable_id, fault_type
):
    if engine.dialect.name != "postgresql":
        pytest.skip("independent committing writer requires PostgreSQL")
    run_id = generate_ulid()
    await store.create_run(run_id, "inflight")
    async with engine.begin() as connection:
        await connection.execute(
            text(
                "CREATE FUNCTION rf5_gate_commit() RETURNS trigger LANGUAGE plpgsql AS $$ "
                "BEGIN PERFORM pg_advisory_xact_lock(1399335); RETURN NULL; END $$"
            )
        )
        await connection.execute(
            text(
                "CREATE CONSTRAINT TRIGGER rf5_gate AFTER INSERT ON run_events "
                "DEFERRABLE INITIALLY DEFERRED FOR EACH ROW "
                "EXECUTE FUNCTION rf5_gate_commit()"
            )
        )
    gate = await engine.connect()
    await gate.execute(text("SELECT pg_advisory_lock(1399335)"))
    held = AsyncSession(engine)
    commit = AsyncSession.commit
    committing = None
    pending = asyncio.Event()
    calls = 0

    async def lost_ack(session):
        nonlocal calls, committing
        calls += 1
        row = (
            (
                await session.execute(
                    select(RunEvent.__table__).where(RunEvent.agent_run_id == run_id)
                )
            )
            .mappings()
            .one()
        )
        await session.rollback()
        # Transfer the exact write to an independent connection, like a server
        # transaction whose client connection was lost. Its actual COMMIT blocks
        # in a deferred trigger until the observer releases the advisory gate.
        await held.execute(select(AgentRun.id).where(AgentRun.id == run_id).with_for_update())
        await held.execute(insert(RunEvent).values(**dict(row)))
        committing = asyncio.create_task(commit(held))
        async with asyncio.timeout(5):
            while True:
                async with engine.connect() as observer:
                    waiting = await observer.scalar(
                        text(
                            "SELECT count(*) FROM pg_stat_activity "
                            "WHERE datname=current_database() AND wait_event='advisory' "
                            "AND upper(query) LIKE 'COMMIT%'"
                        )
                    )
                if waiting:
                    break
                await asyncio.sleep(0.01)
        pending.set()
        raise fault_type("PRIVATE_INFLIGHT_SENTINEL")

    event_id = generate_ulid() if stable_id else None
    task = None
    try:
        with monkeypatch.context() as patch:
            patch.setattr(AsyncSession, "commit", lost_ack)
            task = asyncio.create_task(
                retry_transient_db(
                    lambda: store.save_run_event(run_id, event_type="tick", event_id=event_id),
                    label="inflight",
                    run_id=run_id,
                )
            )
            await asyncio.wait_for(pending.wait(), 5)
            assert committing is not None
            # Observation window only: the advisory gate holds COMMIT, so a slow machine
            # can only miss a regression here, never fail a correct build.
            await asyncio.sleep(0.05)
            assert not committing.done()
            assert not task.done(), "readback must wait for the server's unfinished COMMIT"
            await gate.execute(text("SELECT pg_advisory_unlock(1399335)"))
            await asyncio.wait_for(committing, 5)
            await asyncio.wait_for(task, 5)
        assert calls == 1
        assert len(await store.get_run_events(run_id)) == 1
    finally:
        await gate.execute(text("SELECT pg_advisory_unlock_all()"))
        await gate.close()
        for active in (task, committing):
            if active is not None:
                if not active.done():
                    active.cancel()
                await asyncio.gather(active, return_exceptions=True)
        await held.close()
        async with engine.begin() as connection:
            await connection.execute(text("DROP TRIGGER rf5_gate ON run_events"))
            await connection.execute(text("DROP FUNCTION rf5_gate_commit()"))
