"""Blocked provenance survives independent native store transactions."""

import asyncio
import multiprocessing
import os

import pytest
from sqlalchemy import event, insert, select, text, update

from dendrux.db.models import AgentRun, RunEvent
from dendrux.errors import PersistenceError, RunFencedError, RunNotFoundError
from dendrux.runtime._blocked_content import _blocked_run_ids, _blocked_write
from dendrux.runtime.state import UNSCOPED
from dendrux.types import generate_ulid
from tests._fixtures.storage import pin_storage_format

KEY = "dendrux.blocked"
PLACEHOLDER = "[blocked: 1 secret]"


async def _paused(store, engine, fmt):
    await pin_storage_format(engine, fmt)
    run_id = generate_ulid()
    await store.create_run(run_id, "blocked", tenant_id="tenant")
    await store.pause_run(
        run_id,
        status="waiting_human_input",
        pause_data={"pause_id": "pause", "messages": []},
    )
    return run_id


async def _submit(store, run_id):
    return await store.submit_and_claim(
        run_id,
        expected_status="waiting_human_input",
        submitted_data={"submitted_user_input": PLACEHOLDER},
        submission={
            "submission_id": "submission",
            "kind": "input",
            "pause_id": "pause",
            "call_ids": [],
        },
    )


@pytest.mark.parametrize("fmt", [1, 2])
async def test_061_block_mark_survives_submission_crash(store, engine, fmt):
    run_id = await _paused(store, engine, fmt)
    with _blocked_write(store, run_id):
        assert await _submit(store, run_id)
    # Reopen through a different store, before any terminal write.
    from dendrux.runtime.stores._factory import store_for_engine

    reopened = store_for_engine(engine)
    row = await reopened.get_run(run_id)
    assert row.status == "running"
    assert (row.meta or {}).get(KEY) is True
    await reopened.update_run_meta(run_id, {KEY: False, "stale": "metadata"})
    await reopened.finalize_run(run_id, status="error", error="Stopped")
    assert (await reopened.get_run(run_id)).meta[KEY] is True


@pytest.mark.parametrize("value", [True, False, None, "true"])
async def test_caller_metadata_cannot_set_block_mark(store, value):
    run_id = generate_ulid()
    await store.create_run(run_id, "blocked", meta={KEY: value, "keep": 1})
    assert KEY not in (await store.get_run(run_id)).meta
    await store.update_run_meta(run_id, {KEY: True, "keep": 2})
    assert (await store.get_run(run_id)).meta == {"keep": 2}


@pytest.mark.parametrize("fmt", [1, 2])
async def test_initial_mark_and_event_mark(store, engine, fmt):
    await pin_storage_format(engine, fmt)
    initial, event_run, clean = [generate_ulid() for _ in range(3)]
    with _blocked_write(store, initial):
        await store.create_run(initial, "blocked", input_data={"input": PLACEHOLDER})
        await store.create_run(clean, "blocked", input_data={"input": PLACEHOLDER})
    await store.create_run(event_run, "blocked")
    await store.save_run_event(event_run, event_type="guardrail.blocked", data={"count": 1})
    assert await _blocked_run_ids(store, [initial, event_run, clean], tenant_id=None) == {
        initial,
        event_run,
    }
    assert not (await store.get_run(clean)).meta


@pytest.mark.parametrize("fmt", [1, 2])
@pytest.mark.parametrize("failure", ["validate", "before_commit", "cancel", "fenced"])
async def test_failed_blocked_write_rolls_back(store, engine, fmt, failure, monkeypatch):
    run_id = await _paused(store, engine, fmt)
    if failure in ("before_commit", "cancel"):
        original = store._mark_blocked_writes

        async def fail(session):
            await original(session)
            raise asyncio.CancelledError() if failure == "cancel" else RuntimeError("cut")

        monkeypatch.setattr(store, "_mark_blocked_writes", fail)
    error = asyncio.CancelledError if failure == "cancel" else RuntimeError
    if failure == "fenced":
        error = RunFencedError
    with pytest.raises(error), _blocked_write(store, run_id):
        if failure == "validate":

            def reject(pause):
                raise RuntimeError("cut")

            await store.submit_and_claim(
                run_id, expected_status="waiting_human_input", submitted_data={}, validate=reject
            )
        elif failure == "fenced":
            await store.save_trace(
                run_id, role="user", content=PLACEHOLDER, order_index=0, claim_epoch=999
            )
        else:
            await _submit(store, run_id)
    row = await store.get_run(run_id)
    assert row.status == "waiting_human_input"
    assert not (row.meta or {}).get(KEY)
    assert PLACEHOLDER not in repr(await store.get_pause_state(run_id))


async def test_verdict_lookup_is_scoped_batched_and_uses_all_legacy_events(store, engine):
    ids = [generate_ulid() for _ in range(5)]
    for run_id in ids:
        await store.create_run(run_id, "blocked", tenant_id="tenant")
    # Existing rows may have any one provenance source, independently.
    async with engine.begin() as conn:
        await conn.execute(update(AgentRun).where(AgentRun.id == ids[0]).values(meta={KEY: True}))
        await conn.execute(
            update(AgentRun).where(AgentRun.id == ids[1]).values(failure_reason="guardrail_blocked")
        )
        await conn.execute(update(AgentRun).where(AgentRun.id == ids[4]).values(meta={KEY: "true"}))
        await conn.execute(
            insert(RunEvent),
            [
                {
                    "id": generate_ulid(),
                    "agent_run_id": ids[2],
                    "sequence_index": i,
                    "event_type": "guardrail.blocked" if i == 11000 else "run.started",
                    "data": {},
                }
                for i in range(11001)
            ],
        )
    statements = []

    def capture(conn, cursor, statement, parameters, context, many):
        statements.append(statement)

    event.listen(engine.sync_engine, "before_cursor_execute", capture)
    try:
        assert await _blocked_run_ids(store, ids, tenant_id="tenant") == set(ids[:3])
        assert len(statements) == 1
        assert "EXISTS" in statements[0]
        assert "run_events.data" not in statements[0]
        statements.clear()
        assert await _blocked_run_ids(store, ids * 201, tenant_id="tenant") == set(ids[:3])
        assert len(statements) == 3
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", capture)
    for scope in (None, "other"):
        with pytest.raises(RunNotFoundError):
            await _blocked_run_ids(store, ids, tenant_id=scope)
    for scope in (UNSCOPED, "", 0):
        with pytest.raises(ValueError):
            await _blocked_run_ids(store, ids, tenant_id=scope)
    with pytest.raises(RunNotFoundError):
        await _blocked_run_ids(store, [generate_ulid()], tenant_id="tenant")
    assert await _blocked_run_ids(store, [], tenant_id="tenant") == set()


async def test_failed_lookup_is_safe_and_custom_store_refuses_before_write(store, engine):
    from sqlalchemy.exc import OperationalError

    def fail(conn, cursor, statement, parameters, context, many):
        raise OperationalError("SELECT", {}, Exception("synthetic-private-text"))

    event.listen(engine.sync_engine, "before_cursor_execute", fail)
    try:
        with pytest.raises(PersistenceError) as caught:
            await _blocked_run_ids(store, [generate_ulid()], tenant_id="tenant")
        assert "synthetic-private-text" not in str(caught.value)
        assert caught.value.__context__ is None
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", fail)
    with pytest.raises(PersistenceError), _blocked_write(object(), "run"):
        pytest.fail("Unsupported store accepted a blocked write")


@pytest.mark.parametrize("fmt", [1, 2])
async def test_mark_statement_cost(store, engine, fmt):
    counts = []
    for blocked in (False, True):
        run_id = await _paused(store, engine, fmt)
        statements = []

        def capture(conn, cursor, statement, parameters, context, many, statements=statements):
            statements.append(statement)

        event.listen(engine.sync_engine, "before_cursor_execute", capture)
        try:
            if blocked:
                with _blocked_write(store, run_id):
                    await _submit(store, run_id)
            else:
                await _submit(store, run_id)
        finally:
            event.remove(engine.sync_engine, "before_cursor_execute", capture)
        counts.append(len(statements))
    assert counts == [4, 5]


@pytest.mark.parametrize("path", ["create", "trace", "event"])
async def test_block_mark_lost_commit_ack(store, monkeypatch, path):
    from sqlalchemy.ext.asyncio import AsyncSession

    run_id = generate_ulid()
    if path != "create":
        await store.create_run(run_id, "blocked")
    original = AsyncSession.commit

    async def lost_ack(session):
        await original(session)
        raise OSError("lost acknowledgement")

    with monkeypatch.context() as patch, _blocked_write(store, run_id):
        patch.setattr(AsyncSession, "commit", lost_ack)
        if path == "create":
            await store.create_run(run_id, "blocked", input_data={"input": PLACEHOLDER})
        elif path == "trace":
            await store.save_trace(run_id, role="user", content=PLACEHOLDER, order_index=0)
        else:
            await store.save_run_event(run_id, event_type="guardrail.blocked", data={"count": 1})
    assert (await store.get_run(run_id)).meta[KEY] is True


@pytest.mark.parametrize("path", ["event", "create"])
async def test_block_mark_is_required_by_commit_readback(store, engine, monkeypatch, path):
    from sqlalchemy.ext.asyncio import AsyncSession

    from dendrux.errors import StoreOutcomeUnknown

    run_id = generate_ulid()
    if path == "event":
        await store.create_run(run_id, "blocked")
    original = AsyncSession.commit

    async def lost_mark(session):
        await original(session)
        # Simulate evidence left by an older writer, without its aggregate mark.
        async with engine.begin() as conn:
            await conn.execute(update(AgentRun).where(AgentRun.id == run_id).values(meta={}))
        raise OSError("lost acknowledgement")

    with (
        monkeypatch.context() as patch,
        pytest.raises(StoreOutcomeUnknown),
        _blocked_write(store, run_id),
    ):
        patch.setattr(AsyncSession, "commit", lost_mark)
        if path == "event":
            await store.save_run_event(run_id, event_type="guardrail.blocked", data={"count": 1})
        else:
            await store.create_run(run_id, "blocked", input_data={"input": PLACEHOLDER})


@pytest.mark.parametrize(
    "path", ["model", "server", "server_fallback", "client", "final", "final_cas", "final_event"]
)
async def test_native_evidence_marks_in_same_commit(store, engine, monkeypatch, path):
    from tests.integration.test_061_progress_store import _operation

    run_id, write = await _operation(store, engine, path, monkeypatch)
    with _blocked_write(store, run_id):
        await write()
    assert (await store.get_run(run_id)).meta[KEY] is True


async def test_block_mark_survives_recovery(store, engine):
    from dendrux.runtime.submit import governance_snapshot
    from tests._helpers.recovery_checkpoint import checkpoint
    from tests.integration.test_060_reclaim_agent import make_agent

    agent = make_agent(store)
    try:
        item = await checkpoint(agent, store, engine)
        await store.save_run_event(item.run_id, event_type="guardrail.blocked", data={"count": 1})
        claim = await store.reclaim_run(
            item.run_id,
            tenant_id="tenant",
            agent_name=agent.name,
            expected_claim_epoch=item.claim_epoch,
            governance=governance_snapshot(agent),
            governance_policy="raise",
            skill_pins={},
            loop_type="react",
        )
        assert claim
        assert (await store.get_run(item.run_id)).meta[KEY] is True
    finally:
        await agent.close()


def _race_worker(url, run_id, marker, locked, racing, release, done):
    async def run():
        from sqlalchemy.ext.asyncio import create_async_engine

        from dendrux.runtime.stores._factory import store_for_engine

        engine = create_async_engine(
            url,
            connect_args={
                "server_settings": {
                    "application_name": "dendrux-r0h-marker" if marker else "dendrux-r0h-racer"
                }
            },
        )
        store = store_for_engine(engine)
        if marker:
            original = store._write_tx

            async def commit(session, **kwargs):
                locked.set()
                assert await asyncio.to_thread(release.wait, 20)
                await original(session, **kwargs)

            store._write_tx = commit
            with _blocked_write(store, run_id):
                assert await _submit(store, run_id)
            # End the process immediately after COMMIT, before terminalization.
            os._exit(0)
        else:
            assert await asyncio.to_thread(locked.wait, 15)
            racing.set()
            assert await _submit(store, run_id) is False
            await store.update_run_meta(run_id, {KEY: False, "stale": True})
            done.set()
        await engine.dispose()

    asyncio.run(run())


@pytest.mark.parametrize("fmt", [1, 2])
async def test_two_process_mark_submit_race(store, engine, fmt):
    if engine.dialect.name != "postgresql":
        pytest.skip("Postgres process race")
    run_id = await _paused(store, engine, fmt)
    ctx = multiprocessing.get_context("spawn")
    locked, racing, release, done = [ctx.Event() for _ in range(4)]
    processes = [
        ctx.Process(
            target=_race_worker,
            args=(
                engine.url.render_as_string(hide_password=False),
                run_id,
                marker,
                locked,
                racing,
                release,
                done,
            ),
        )
        for marker in (True, False)
    ]
    try:
        for process in processes:
            process.start()
        assert await asyncio.to_thread(racing.wait, 15)
        async with asyncio.timeout(15):
            while True:
                async with engine.connect() as conn:
                    waiting = await conn.scalar(
                        text(
                            "SELECT count(*) FROM pg_stat_activity "
                            "WHERE datname = current_database() "
                            "AND application_name = 'dendrux-r0h-racer' "
                            "AND wait_event_type = 'Lock'"
                        )
                    )
                if waiting:
                    break
                await asyncio.sleep(0.05)
        release.set()
        for process in processes:
            await asyncio.to_thread(process.join, 25)
            assert process.exitcode == 0
        assert done.is_set()
        row = await store.get_run(run_id)
        assert row.status == "running"
        assert row.meta == {KEY: True, "stale": True}
        assert await _blocked_run_ids(store, [run_id], tenant_id="tenant") == {run_id}
        async with engine.connect() as conn:
            assert await conn.scalar(select(AgentRun.claim_epoch).where(AgentRun.id == run_id)) == 1
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                await asyncio.to_thread(process.join, 5)
