"""Recovery streak resets share the durable progress transaction."""

from datetime import UTC, datetime

import pytest
from sqlalchemy import event, update

from dendrux.db.models import AgentRun
from dendrux.runtime.persistence import message_trace
from dendrux.runtime.state import LoopStatePatch
from dendrux.testing.store_kit.contracts._support import complete_call_kwargs
from dendrux.types import Message, Role, ToolCall
from tests._helpers.recovery_checkpoint import checkpoint
from tests.integration.test_060_reclaim_agent import make_agent

STAMP = datetime(2026, 11, 1, tzinfo=UTC)
PATHS = ["model", "server", "server_fallback", "client", "final", "final_cas", "final_event"]


async def _set_streak(engine, run_id, count=2):
    async with engine.begin() as conn:
        await conn.execute(
            update(AgentRun)
            .where(AgentRun.id == run_id)
            .values(
                reclaim_attempts=count,
                last_reclaimed_at=STAMP,
            )
        )


async def _operation(store, engine, path, monkeypatch):
    agent = make_agent(store)
    tool_path = path in ("server", "server_fallback", "client")
    call = ToolCall("work")
    item = await checkpoint(
        agent,
        store,
        engine,
        calls=[call] if tool_path else [],
        phases=["client" if path == "client" else "dispatchable"] if tool_path else [],
        started=path in ("server", "server_fallback"),
        idempotent=True,
    )
    run_id = item.run_id
    state = await store.get_loop_state(run_id)
    trace = message_trace(Message(Role.ASSISTANT, "new model step"), 10)
    if path == "model":
        state["iteration"] += 1

        async def write():
            await store.checkpoint_step(
                run_id,
                claim_epoch=item.claim_epoch,
                loop_state=state,
                pii_mapping=None,
                trace=trace,
            )
    elif path.startswith("server"):
        if path == "server_fallback":
            original = store.json_patch_expr
            from dendrux.runtime.stores.sql._types import PatchLoopState

            def fallback(column, patch):
                return None if isinstance(patch, PatchLoopState) else original(column, patch)

            monkeypatch.setattr(store, "json_patch_expr", fallback)
        kwargs = complete_call_kwargs(call.id, 3)

        async def write():
            await store.complete_tool_call(run_id, claim_epoch=item.claim_epoch, **kwargs)
    elif path == "client":
        state["submission"] = {"submission_id": "submitted", "kind": "tool_results"}
        async with engine.begin() as conn:
            await conn.execute(
                update(AgentRun).where(AgentRun.id == run_id).values(loop_state=state)
            )
        trace = message_trace(Message(Role.TOOL, "client result", call_id=call.id, name="work"), 3)

        async def write():
            return await store.consume_submission(
                run_id,
                claim_epoch=item.claim_epoch,
                submission_id="submitted",
                traces=[trace],
                patch=LoopStatePatch(remove_pending=(call.id,), set_phase="model_next"),
            )
    else:
        kwargs = dict(status="success", answer="done", claim_epoch=item.claim_epoch)
        if path == "final":

            async def write():
                return await store.finalize_run(run_id, expected_current_status="running", **kwargs)
        elif path == "final_cas":

            async def write():
                return await store.finalize_run_if_status_in(
                    run_id, allowed_current_statuses=["running"], **kwargs
                )
        else:

            async def write():
                return await store.finalize_with_event(
                    run_id,
                    allowed_current_statuses=["running"],
                    event_type="run.completed",
                    trace=trace,
                    **kwargs,
                )

    await _set_streak(engine, run_id)
    return run_id, write


@pytest.mark.parametrize("path", PATHS)
async def test_061_new_progress_resets_streak_atomically(store, engine, monkeypatch, path):
    run_id, write = await _operation(store, engine, path, monkeypatch)
    statements, commits = [], []

    def capture(conn, cursor, statement, *args):
        statements.append(statement)

    def commit(conn):
        commits.append(1)

    event.listen(engine.sync_engine, "before_cursor_execute", capture)
    event.listen(engine.sync_engine, "commit", commit)
    try:
        await write()
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", capture)
        event.remove(engine.sync_engine, "commit", commit)
    print("progress_cost", engine.dialect.name, path, len(statements), len(commits))
    row = await store.get_run(run_id)
    assert row.reclaim_attempts == 0
    assert row.last_reclaimed_at == STAMP
    assert len(commits) == 1
    expected = {
        "model": 2,
        "server": 6 if engine.dialect.name == "sqlite" else 4,
        "server_fallback": 6,
        "client": 4,
        "final": 1,
        "final_cas": 1,
        "final_event": 3,
    }
    assert len(statements) == expected[path]


@pytest.mark.parametrize("path", PATHS)
async def test_061_duplicate_progress_does_not_reset_a_later_streak(
    store, engine, monkeypatch, path
):
    run_id, write = await _operation(store, engine, path, monkeypatch)
    await write()
    await _set_streak(engine, run_id)
    await write()
    assert (await store.get_run(run_id)).reclaim_attempts == 2


@pytest.mark.parametrize("path", PATHS)
async def test_061_progress_failure_rolls_back_reset(store, engine, monkeypatch, path):
    run_id, write = await _operation(store, engine, path, monkeypatch)

    async def fail(session):
        raise RuntimeError("injected before commit")

    with monkeypatch.context() as patch:
        patch.setattr(store, "_write_tx", fail)
        with pytest.raises(RuntimeError, match="injected before commit"):
            await write()
    assert (await store.get_run(run_id)).reclaim_attempts == 2
    await write()
    assert (await store.get_run(run_id)).reclaim_attempts == 0


@pytest.mark.parametrize("path", PATHS)
async def test_061_progress_lost_commit_ack_reads_back_atomically(store, engine, monkeypatch, path):
    from sqlalchemy.ext.asyncio import AsyncSession

    run_id, write = await _operation(store, engine, path, monkeypatch)
    original = AsyncSession.commit
    commits = []

    async def lost_ack(session):
        await original(session)
        commits.append(1)
        if len(commits) == 1:
            raise OSError("injected lost acknowledgement")

    with monkeypatch.context() as patch:
        patch.setattr(AsyncSession, "commit", lost_ack)
        await write()
    assert len(commits) == 1
    row = await store.get_run(run_id)
    assert row.reclaim_attempts == 0 and row.last_reclaimed_at == STAMP


@pytest.mark.parametrize("path", PATHS)
async def test_061_stale_owner_cannot_reset_streak(store, engine, monkeypatch, path):
    from dendrux.errors import RunFencedError

    run_id, write = await _operation(store, engine, path, monkeypatch)
    async with engine.begin() as conn:
        await conn.execute(update(AgentRun).where(AgentRun.id == run_id).values(claim_epoch=100))
    try:
        result = await write()
    except RunFencedError:
        pass
    else:
        assert result in (None, False)
    assert (await store.get_run(run_id)).reclaim_attempts == 2


async def test_061_progress_freshness_lookup_plan(store, engine):
    from sqlalchemy import text

    from dendrux.runtime.stores.sql.base import _checkpoint_progress

    agent = make_agent(store)
    item = await checkpoint(agent, store, engine)
    trace = message_trace(Message(Role.ASSISTANT, "new"), 10)
    statement = (
        update(AgentRun)
        .where(AgentRun.id == item.run_id, AgentRun.claim_epoch == item.claim_epoch)
        .values(**_checkpoint_progress(trace))
    )
    compiled = str(
        statement.compile(
            dialect=engine.sync_engine.dialect, compile_kwargs={"literal_binds": True}
        )
    )
    prefix = "EXPLAIN QUERY PLAN " if engine.dialect.name == "sqlite" else "EXPLAIN "
    async with engine.connect() as conn:
        plan = (await conn.execute(text(prefix + compiled))).all()
    print("freshness_plan", engine.dialect.name, plan)
    assert plan


@pytest.mark.parametrize("status", ["success", "error", "cancelled"])
@pytest.mark.parametrize(
    "api", ["finalize_run", "finalize_run_if_status_in", "finalize_with_event"]
)
async def test_061_finalization_preserves_nonprogress_and_output_fields(store, engine, status, api):
    from dendrux.types import UsageStats

    agent = make_agent(store)
    item = await checkpoint(agent, store, engine)
    await _set_streak(engine, item.run_id)
    kwargs = dict(
        status=status,
        answer="answer" if status == "success" else None,
        claim_epoch=item.claim_epoch,
        error="failure" if status == "error" else None,
        total_usage=UsageStats(input_tokens=2, output_tokens=3, total_tokens=5),
        iteration_count=4,
        pii_mapping={"placeholder": "synthetic"},
        failure_reason="stale_running" if status == "error" else None,
    )
    if api == "finalize_run":
        kwargs["expected_current_status"] = "running"
    else:
        kwargs["allowed_current_statuses"] = ["running"]
    if api == "finalize_with_event":
        kwargs["event_type"] = (
            "run.completed"
            if status == "success"
            else "run.error"
            if status == "error"
            else "run.cancelled"
        )
    await getattr(store, api)(item.run_id, **kwargs)
    row = await store.get_run(item.run_id)
    assert row.status == status
    assert row.reclaim_attempts == (0 if status == "success" else 2)
    assert row.last_reclaimed_at == STAMP
    assert row.iteration_count == 4
    assert await store.get_pii_mapping(item.run_id) == {"placeholder": "synthetic"}
    assert row.error == ("failure" if status == "error" else None)
