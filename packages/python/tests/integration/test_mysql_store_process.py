"""Two processes and twenty database connections repeat every native race wave."""

# ruff: noqa: F401, F811
import asyncio
import multiprocessing

from sqlalchemy import func, select

from dendrux.db.models import AgentRun, RunEvent, RunInbox
from dendrux.types import generate_ulid
from tests.integration._mysql import mysql_engine, process_contender, store
from tests.integration.test_060_core_recovery import mark, stale


async def test_twenty_callers_in_two_processes(store, mysql_engine):
    jobs = []
    claim_id = generate_ulid()
    await store.create_run(claim_id, "agent")
    await store.pause_run(claim_id, status="waiting_client_tool", pause_data={"pause_id": "p"})
    jobs.append(dict(operation="claim", run_id=claim_id))
    await stale(store, mysql_engine)
    item = (await mark(store)).reclaimable[0]
    jobs.append(dict(operation="reclaim", run_id=item.run_id, epoch=item.claim_epoch))
    for operation in ("finalize", "cancel", "note", "events", "collision"):
        run_id = generate_ulid()
        await store.create_run(run_id, "agent", tenant_id="tenant")
        jobs.append(
            dict(
                operation=operation,
                run_id=run_id,
                **({"event_id": generate_ulid()} if operation == "collision" else {}),
            )
        )
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(3)
    queue = context.Queue()
    processes = [
        context.Process(
            target=process_contender,
            args=(
                mysql_engine.url.render_as_string(hide_password=False),
                jobs,
                barrier,
                queue,
                actor,
            ),
        )
        for actor in range(2)
    ]
    try:
        for process in processes:
            process.start()
        for wave, job in enumerate(jobs):
            await asyncio.wait_for(asyncio.to_thread(barrier.wait, 30), 35)
            responses = [
                await asyncio.wait_for(asyncio.to_thread(queue.get, True, 30), 35) for _ in range(2)
            ]
            assert all(response[0] == wave for response in responses), responses
            values = [value for _, _, rows in responses for value in rows]
            assert len(values) == 20
            operation, run_id = job["operation"], job["run_id"]
            winners = [v for v in values if v["kind"] == "won"]
            if operation == "events":
                assert sorted(v["sequence"] for v in winners) == list(range(20)), values
                stored = await store.get_run_events(run_id)
                assert {row.data["index"]: row.sequence_index for row in stored} == {
                    value["index"]: value["sequence"] for value in winners
                }
            else:
                assert len(winners) == 1, values
            if operation in {"claim", "reclaim"}:
                row = await store.get_run(run_id)
                assert winners[0]["epoch"] == row.claim_epoch
                if operation == "reclaim":
                    assert row.reclaim_attempts == 1
                    assert sum(v["kind"] == "RunNotReclaimableError" for v in values) == 19
                else:
                    assert row.claim_epoch == 1
                    assert sum(v["kind"] == "lost" for v in values) == 19
            elif operation in {"finalize", "cancel"}:
                assert (await store.get_run(run_id)).status == (
                    "cancelled" if operation == "cancel" else "success"
                )
                assert len(await store.get_run_events(run_id)) == 1
                assert sum(v["kind"] == "lost" for v in values) == 19
            elif operation == "note":
                assert sum(v["kind"] == "NoteLimitExceededError" for v in values) == 19
                async with store._session_factory() as session:
                    assert (
                        await session.scalar(
                            select(func.count())
                            .select_from(RunInbox)
                            .where(RunInbox.run_id == run_id)
                        )
                        == 1
                    )
            else:
                if operation == "collision":
                    assert sum(v.get("code") == "write_conflict" for v in values) == 19
                assert len(await store.get_run_events(run_id)) == (
                    20 if operation == "events" else 1
                )
            print(
                "PROCESS_RACE",
                mysql_engine.dialect.is_mariadb,
                operation,
                "callers=20",
                "winners=" + str(len(winners)),
            )
        async with store._session_factory() as session:
            assert (
                await session.scalar(
                    select(func.count())
                    .select_from(RunEvent)
                    .outerjoin(AgentRun, AgentRun.id == RunEvent.agent_run_id)
                    .where(AgentRun.id.is_(None))
                )
                == 0
            )
    finally:
        for process in processes:
            if process.pid is not None:
                await asyncio.to_thread(process.join, 5)
                if process.is_alive():
                    process.terminate()
                    await asyncio.to_thread(process.join, 5)
                if process.is_alive():
                    process.kill()
                    await asyncio.to_thread(process.join, 5)
        queue.close()
        queue.join_thread()
    assert all(process.exitcode == 0 for process in processes)
