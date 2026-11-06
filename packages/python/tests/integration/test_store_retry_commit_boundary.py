"""Every lifecycle retry scope must stop at a landed COMMIT with a lost acknowledgement."""

from datetime import UTC, datetime

import pytest
from sqlalchemy import insert, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from dendrux.db.models import AgentRun, RunEvent
from dendrux.errors import StoreOutcomeUnknown
from dendrux.runtime.durability import _database_operation
from dendrux.types import generate_ulid
from tests.unit.test_store_raw_retries import writes


@pytest.mark.parametrize("name", list(writes(None, "")))
@pytest.mark.parametrize("fault_type", [OSError, TimeoutError])
async def test_each_lifecycle_scope_commits_only_once(store, engine, monkeypatch, name, fault_type):
    run_id = generate_ulid()
    if name != "create_run":
        await store.create_run(run_id, "a")
        values = {"claim_epoch": 1, "storage_format": 1}
        if name in {"claim_paused_run", "submit_and_claim"}:
            values.update(status="waiting_client_tool", pause_data={})
        if name == "reclaim_run":
            values.update(
                storage_format=2,
                lease_expired_at=datetime.now(UTC),
                reclaim_limit=3,
                loop_state={"loop": "react"},
            )
        async with engine.begin() as connection:
            await connection.execute(update(AgentRun).where(AgentRun.id == run_id).values(**values))
    commit = AsyncSession.commit
    commits = 0
    targets = []

    async def lost_ack(session):
        nonlocal commits
        commits += 1
        operation = _database_operation.get()
        target = operation[2].get("run_id", run_id)
        targets.append(target)
        # One audit probe in this actual transaction makes even update-only
        # methods observably non-idempotent if the retry boundary replays them.
        await session.execute(
            insert(RunEvent).values(
                id=generate_ulid(),
                agent_run_id=target,
                event_type="commit.probe",
                sequence_index=900 + commits,
            )
        )
        await commit(session)
        raise fault_type("PRIVATE_LANDED_COMMIT_SENTINEL")

    with monkeypatch.context() as patch:
        patch.setattr(AsyncSession, "commit", lost_ack)
        try:
            await writes(store, run_id)[name]()
        except StoreOutcomeUnknown as error:
            assert not error.retryable
            assert error.__context__ is None and error.__cause__ is None
    assert commits == 1
    async with engine.connect() as connection:
        rows = (
            await connection.execute(
                select(RunEvent.id).where(
                    RunEvent.agent_run_id == targets[0], RunEvent.event_type == "commit.probe"
                )
            )
        ).all()
    assert len(rows) == 1
