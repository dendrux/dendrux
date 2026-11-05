"""Concurrent finalizers publish status and lifecycle evidence together."""

import asyncio

import pytest

from dendrux.runtime import runner
from dendrux.types import RunResult, RunStatus, generate_ulid
from tests._fixtures.storage import pin_storage_format

pytestmark = pytest.mark.parametrize("engine", ["postgres"], indirect=True)


@pytest.mark.parametrize("kind", ["failure", "cancellation", "success"])
async def test_finalizer_visibility_and_one_winner(store, engine, monkeypatch, kind):
    if kind == "success":
        await pin_storage_format(engine, 1)
    run_id = generate_ulid()
    await store.create_run(run_id, "atomic-terminal")
    expected = {
        "failure": "run.error",
        "cancellation": "run.cancelled",
        "success": "run.completed",
    }[kind]
    observed = []
    for name in ("finalize_run", "finalize_with_event"):
        original = getattr(store, name)

        async def observe(*args, _original=original, **kwargs):
            result = await _original(*args, **kwargs)
            rows = await store.get_run_events(run_id, limit=100)
            observed.append([row.event_type for row in rows])
            return result

        monkeypatch.setattr(store, name, observe)

    async def finish():
        if kind == "success":
            result = await runner._persist_loop_outcome(
                state_store=store,
                run_id=run_id,
                result=RunResult(run_id=run_id, status=RunStatus.SUCCESS, answer="done"),
            )
            return result.meta["_finalize_won"]
        if kind == "failure":
            return await runner._persist_run_failure(store, run_id, RuntimeError("safe"))
        won, _ = await runner._persist_run_cancellation(
            store, run_id, {"reason": "task_cancelled"}, on_won=lambda: None
        )
        return won

    winners = await asyncio.gather(*(finish() for _ in range(8)))
    assert sum(winners) == 1
    assert all(events == [expected] for events in observed)
    assert [row.event_type for row in await store.get_run_events(run_id, limit=100)] == [expected]
