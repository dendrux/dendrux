"""All storage formats complete atomically and preserve best-effort observations."""

from unittest.mock import AsyncMock

import pytest
from sqlalchemy import text

from dendrux.errors import RunFencedError
from dendrux.runtime import _run_events, runner
from dendrux.runtime.state import _ClaimedStateStore
from dendrux.types import RunResult, RunStatus, generate_ulid
from tests._fixtures.storage import pin_storage_format
from tests.unit.test_runner_failure_save import store as store


@pytest.mark.parametrize("status", [RunStatus.SUCCESS, RunStatus.MAX_ITERATIONS, RunStatus.ERROR])
@pytest.mark.parametrize("storage_format", [1, 2])
async def test_terminal_result_uses_atomic_transition(store, monkeypatch, status, storage_format):
    await pin_storage_format(store._engine, storage_format)
    await store.create_run("r", "test")
    store.storage_format = storage_format
    atomic = AsyncMock(wraps=store.finalize_with_event)
    split = AsyncMock(wraps=store.finalize_run)
    monkeypatch.setattr(store, "finalize_with_event", atomic)
    monkeypatch.setattr(store, "finalize_run", split)
    await runner._persist_loop_outcome(
        state_store=store,
        run_id="r",
        result=RunResult(run_id="r", status=status, answer="done"),
    )
    assert atomic.await_count == 1
    split.assert_not_awaited()
    if storage_format == 1:
        assert "trace" not in atomic.call_args.kwargs
        assert "event" not in atomic.call_args.kwargs
    row = await store.get_run("r")
    assert row.status == status.value and row.answer == "done"
    assert len(await store.get_run_events("r")) == 1


async def test_failed_placeholder_observation_still_persists_answer(store, monkeypatch):
    await store.create_run("r", "test")
    emit = runner._emit_event

    async def fail_observation(state, run_id, event_type, *args, **kwargs):
        if event_type == "guardrail.unmapped_placeholder":
            raise RuntimeError("evidence unavailable")
        return await emit(state, run_id, event_type, *args, **kwargs)

    monkeypatch.setattr(_run_events, "_emit_event", fail_observation)
    result = await runner._persist_loop_outcome(
        state_store=store,
        run_id="r",
        result=RunResult(
            run_id="r",
            status=RunStatus.SUCCESS,
            answer="<<PERSON_99>>",
            meta={"pii_mapping": {"<<PERSON_1>>": "example"}},
        ),
    )
    assert result.status == RunStatus.SUCCESS
    row = await store.get_run("r")
    assert row.status == "success" and row.answer == "<<PERSON_99>>"
    assert [e.event_type for e in await store.get_run_events("r")] == ["run.completed"]


class _NoReasonAtomic:
    """A custom store from before ``finalize_with_event`` took ``failure_reason``."""

    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, name):
        return getattr(self._inner, name)

    async def finalize_with_event(
        self,
        run_id,
        *,
        status,
        allowed_current_statuses,
        event_type,
        event_data=None,
        answer=None,
        error=None,
        iteration_count=None,
        total_usage=None,
        pii_mapping=None,
    ):
        return await self._inner.finalize_with_event(
            run_id,
            status=status,
            allowed_current_statuses=allowed_current_statuses,
            event_type=event_type,
            event_data=event_data,
            answer=answer,
            error=error,
            iteration_count=iteration_count,
            total_usage=total_usage,
            pii_mapping=pii_mapping,
        )


async def test_guardrail_block_finishes_on_an_older_atomic_store(store):
    await store.create_run("r", "test")
    result = await runner._persist_loop_outcome(
        state_store=_NoReasonAtomic(store),
        run_id="r",
        result=RunResult(
            run_id="r",
            status=RunStatus.ERROR,
            error="blocked",
            meta={"failure_reason": "guardrail_blocked"},
        ),
    )
    assert result.status == RunStatus.ERROR
    assert (await store.get_run("r")).status == "error"
    events = await store.get_run_events("r")
    assert [event.event_type for event in events] == ["run.error"]
    assert events[0].data["failure_reason"] == "guardrail_blocked"  # Nowhere else to keep it.


async def test_guardrail_block_reason_stays_on_the_row_when_the_store_takes_it(store):
    await store.create_run("r", "test")
    await runner._persist_loop_outcome(
        state_store=store,
        run_id="r",
        result=RunResult(
            run_id="r",
            status=RunStatus.ERROR,
            error="blocked",
            meta={"failure_reason": "guardrail_blocked"},
        ),
    )
    assert (await store.get_run("r")).failure_reason == "guardrail_blocked"
    assert "failure_reason" not in (await store.get_run_events("r"))[0].data


@pytest.mark.parametrize("storage_format", [1, 2])
async def test_stale_owner_placeholder_observation_stops_with_a_fence(store, storage_format):
    await pin_storage_format(store._engine, storage_format)
    run_id = generate_ulid()
    await store.create_run(run_id, "test")
    row = await store.get_run(run_id)
    async with store._engine.begin() as connection:
        await connection.execute(
            text("update agent_runs set claim_epoch = :epoch where id = :id"),
            {"epoch": row.claim_epoch + 1, "id": run_id},
        )
    with pytest.raises(RunFencedError):
        await runner._persist_loop_outcome(
            state_store=_ClaimedStateStore(store, run_id, row.claim_epoch),
            run_id=run_id,
            result=RunResult(
                run_id=run_id,
                status=RunStatus.SUCCESS,
                answer="<<PERSON_99>>",
                meta={"pii_mapping": {"<<PERSON_1>>": "example"}},
            ),
        )
    assert (await store.get_run(run_id)).status == "running"
    assert await store.get_run_events(run_id) == []
