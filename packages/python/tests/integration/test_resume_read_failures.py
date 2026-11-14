"""Read failures preserve unconsumed submissions and committed outcomes."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from dendrux.errors import PersistenceError, StoreOutcomeUnknown
from dendrux.runtime import _run_events, _run_resume_replay, runner
from dendrux.types import RunStatus
from tests._fixtures.storage import pin_storage_format
from tests._helpers.gates import wait_for_gate
from tests._helpers.recovery_checkpoint import expire
from tests._helpers.runtime_paths import ProbeProvider, path_case
from tests._helpers.scripted_provider import tool_call

ENTRIES = ["tool_results", "input", "approve", "resume", "resume_stream", "recovery"]


async def invoke_failure(case, entry):
    if entry == "resume_stream":
        assert (await case.invoke()).error_info.category == "persistence_error"
    else:
        with pytest.raises(PersistenceError):
            await case.invoke()


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("storage_format", [1, 2])
async def test_setup_read_failure_releases_exact_claim(
    store, engine, monkeypatch, entry, storage_format
):
    if entry == "recovery" and storage_format == 1:
        pytest.skip("reclaim requires storage format 2")
    await pin_storage_format(engine, storage_format)
    provider = ProbeProvider()
    case = await path_case(store, engine, entry, provider)
    before = await store.get_run(case.run_id)
    release = AsyncMock(wraps=store.release_run_claim)
    with monkeypatch.context() as patcher:
        patcher.setattr(store, "release_run_claim", release)
        patcher.setattr(store, "get_next_trace_index", AsyncMock(side_effect=PersistenceError()))
        await invoke_failure(case, entry)
    row = await store.get_run(case.run_id)
    assert row.status == before.status
    assert provider.calls == 0
    release.assert_awaited_once()
    assert release.call_args.kwargs["claim_epoch"] == row.claim_epoch
    assert row.claim_epoch > before.claim_epoch
    if entry == "recovery":
        item = await expire(store, engine, case.run_id)
        result = await case.agent.reclaim(case.run_id, tenant_id="tenant", item=item)
    else:
        assert await store.get_pause_state(case.run_id) is not None
        result = await case.invoke()
    assert result.status == RunStatus.SUCCESS
    assert provider.calls == 1


@pytest.mark.parametrize("entry", ENTRIES)
async def test_stale_setup_release_cannot_change_new_claim(
    store, engine, monkeypatch, entry, *, release_available=True
):
    if engine.dialect.name not in {"postgresql", "mysql"}:
        pytest.skip("independent connection race requires row locking")
    case = await path_case(store, engine, entry, ProbeProvider())
    prior = await store.get_run(case.run_id)
    entered, proceed = asyncio.Event(), asyncio.Event()

    async def fail_read(*args, **kwargs):
        entered.set()
        await proceed.wait()
        raise PersistenceError()

    with monkeypatch.context() as patcher:
        patcher.setattr(store, "get_next_trace_index", fail_read)
        task = asyncio.create_task(invoke_failure(case, entry))
        try:
            await wait_for_gate(entered, task=task)
            old = await store.get_run(case.run_id)
            if entry == "recovery":
                item = await expire(store, engine, case.run_id)
                from dendrux.runtime.submit import governance_snapshot

                winner = await store.reclaim_run(
                    case.run_id,
                    tenant_id="tenant",
                    agent_name=case.agent.name,
                    expected_claim_epoch=item.claim_epoch,
                    governance=governance_snapshot(case.agent),
                    governance_policy="raise",
                    skill_pins={},
                    loop_type="react",
                )
            else:
                assert await store.release_run_claim(
                    case.run_id, status=prior.status, claim_epoch=old.claim_epoch
                )
                barrier = asyncio.Barrier(3)

                async def compete():
                    await barrier.wait()
                    return await store.claim_paused_run(case.run_id, expected_status=prior.status)

                claims = await asyncio.gather(*(compete() for _ in range(3)))
                winners = [c for c in claims if c]
                assert len(winners) == 1
                winner = winners[0]
            assert winner.claim_epoch > old.claim_epoch
            if not release_available:
                patcher.setattr(store, "release_run_claim", None)
            proceed.set()
            await asyncio.wait_for(task, 10)
            row = await store.get_run(case.run_id)
            assert row.status == "running"
            assert row.claim_epoch == winner.claim_epoch
            assert not any(
                e.event_type == "run.error" for e in await store.get_run_events(case.run_id)
            )
        finally:
            proceed.set()
            await asyncio.gather(task, return_exceptions=True)


async def test_reclaim_returns_committed_success_when_recovery_read_fails(
    store, engine, monkeypatch, caplog
):
    provider = ProbeProvider()
    case = await path_case(store, engine, "recovery", provider)
    monkeypatch.setattr(store, "get_replayed_tool_calls", AsyncMock(side_effect=PersistenceError()))
    result = await case.invoke()
    assert result.status == RunStatus.SUCCESS
    assert result.answer == "done"
    assert result.recovery is None
    assert (await store.get_run(case.run_id)).status == "success"
    assert provider.calls == 1
    assert any(
        getattr(r, "run_id", None) == case.run_id
        and getattr(r, "code", None) == "result_read_failed"
        for r in caplog.records
    )


@pytest.mark.parametrize("entry", ENTRIES)
async def test_unknown_write_outcome_is_not_released(store, engine, monkeypatch, entry):
    case = await path_case(store, engine, entry, ProbeProvider())
    release = AsyncMock(wraps=store.release_run_claim)
    monkeypatch.setattr(store, "release_run_claim", release)
    monkeypatch.setattr(store, "get_next_trace_index", AsyncMock(side_effect=StoreOutcomeUnknown()))
    await invoke_failure(case, entry)
    release.assert_not_awaited()
    assert (await store.get_run(case.run_id)).status == "error"


@pytest.mark.parametrize("entry", ENTRIES)
async def test_cancel_request_wins_setup_release(store, engine, monkeypatch, entry):
    if engine.dialect.name not in {"postgresql", "mysql"}:
        pytest.skip("independent connection race requires row locking")
    case = await path_case(store, engine, entry, ProbeProvider())
    entered, proceed = asyncio.Event(), asyncio.Event()

    async def fail_read(*args, **kwargs):
        entered.set()
        await proceed.wait()
        raise PersistenceError()

    monkeypatch.setattr(store, "get_next_trace_index", fail_read)
    task = asyncio.create_task(invoke_failure(case, entry))
    try:
        await wait_for_gate(entered, task=task)
        await store.request_cancel(case.run_id)
        proceed.set()
        await asyncio.wait_for(task, 10)
        assert (await store.get_run(case.run_id)).status == "cancelled"
        assert await store.get_pause_state(case.run_id) is None
        events = [e.event_type for e in await store.get_run_events(case.run_id)]
        assert events.count("run.cancelled") == 1
        assert "run.resume_aborted" not in events and "run.error" not in events
    finally:
        proceed.set()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize("entry", ["tool_results", "input", "resume", "resume_stream"])
@pytest.mark.parametrize("storage_format", [1, 2])
async def test_consumed_history_is_never_released(
    store, engine, monkeypatch, entry, storage_format
):
    await pin_storage_format(engine, storage_format)
    case = await path_case(store, engine, entry, ProbeProvider())
    replay = runner._replay_resume_injected_history

    async def consumed_then_fail(*args, **kwargs):
        await replay(*args, **kwargs)
        raise PersistenceError()

    release = AsyncMock(wraps=store.release_run_claim)
    monkeypatch.setattr(store, "release_run_claim", release)
    monkeypatch.setattr(_run_resume_replay, "_replay_resume_injected_history", consumed_then_fail)
    await invoke_failure(case, entry)
    release.assert_not_awaited()
    assert (await store.get_run(case.run_id)).status == "error"


@pytest.mark.parametrize("storage_format", [1, 2])
async def test_stream_approval_work_is_never_released(store, engine, monkeypatch, storage_format):
    await pin_storage_format(engine, storage_format)
    case = await path_case(store, engine, "approve", ProbeProvider())
    raw = await store.get_pause_state(case.run_id)
    execute = runner._resume_approval_batch

    async def executed_then_fail(*args, **kwargs):
        await execute(*args, **kwargs)
        raise PersistenceError()

    release = AsyncMock(wraps=store.release_run_claim)
    monkeypatch.setattr(store, "release_run_claim", release)
    monkeypatch.setattr(runner, "_resume_approval_batch", executed_then_fail)
    async with case.agent.resume_stream(
        case.run_id, tenant_id="tenant", expected_pause_id=raw["pause_id"]
    ) as stream:
        async for _ in stream:
            pass
    assert stream.result.status == RunStatus.ERROR
    release.assert_not_awaited()
    assert (await store.get_run(case.run_id)).status == "error"
    assert len(await store.get_tool_calls(case.run_id)) == 1


@pytest.mark.parametrize("entry", ["run", "run_stream", "retry", *ENTRIES])
@pytest.mark.parametrize("read", ["pending", "pause_refresh"])
async def test_committed_pause_survives_projection_failure(
    store, engine, monkeypatch, caplog, entry, read
):
    provider = ProbeProvider()
    case = await path_case(store, engine, entry, provider)
    name, params = (
        ("refund", {"order_id": 8})
        if entry == "approve"
        else ("read_range", {"sheet": "S2"})
        if entry in {"tool_results", "input"}
        else ("client_read", {})
    )
    provider._responses = [tool_call(name, params, pid="next")]
    committed = False
    pause = runner._pause_with_event
    get_run = store.get_run

    async def commit_pause(*args, **kwargs):
        nonlocal committed
        won = await pause(*args, **kwargs)
        committed = bool(won)
        return won

    async def read_record(*args, **kwargs):
        if committed:
            raise PersistenceError()
        return await get_run(*args, **kwargs)

    with monkeypatch.context() as patcher:
        patcher.setattr(_run_events, "_pause_with_event", commit_pause)
        if read == "pending":
            patcher.setattr(
                "dendrux.runtime._pending._PendingReader.get_pending",
                AsyncMock(side_effect=PersistenceError()),
            )
        else:
            patcher.setattr(store, "get_run", read_record)
        result = await case.invoke()
    assert committed
    expected = RunStatus.WAITING_APPROVAL if entry == "approve" else RunStatus.WAITING_CLIENT_TOOL
    assert result.status == expected
    assert result.pending is None and result.pause_id is None
    assert (await store.get_run(result.run_id)).status == expected.value
    assert any(
        getattr(r, "code", None) == "result_read_failed"
        and getattr(r, "run_id", None) == result.run_id
        for r in caplog.records
    )
