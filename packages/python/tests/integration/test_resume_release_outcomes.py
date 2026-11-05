"""Continuation cleanup distinguishes unsupported, failed and completed releases."""

from dataclasses import replace
from functools import wraps
from unittest.mock import AsyncMock

import pytest

from dendrux.errors import PersistenceError
from dendrux.loops.base import BaseNotifier
from dendrux.runtime import runner
from dendrux.runtime.persistence import PersistenceRecorder
from dendrux.types import RunEventType, RunStatus
from tests._fixtures.storage import pin_storage_format
from tests._helpers.runtime_paths import ProbeProvider, path_case

ENTRIES = [
    "tool_results",
    "input",
    "approve",
    "reject",
    "decisions",
    "submit_resume",
    "resume",
    "resume_stream",
    "recovery",
]


def legacy_claims(store, monkeypatch):
    for name in ("submit_and_claim", "claim_paused_run"):
        original = getattr(store, name)

        def wrap(method):
            @wraps(method)
            async def claim(*args, **kwargs):
                return bool(await method(*args, **kwargs))

            return claim

        monkeypatch.setattr(store, name, wrap(original))
    control = store.get_run_control

    async def legacy_control(*args, **kwargs):
        return replace(await control(*args, **kwargs), claim_epoch=None)

    monkeypatch.setattr(store, "get_run_control", legacy_control)


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("unsupported", ["method", "epoch"])
async def test_unavailable_release_handles_setup_failure(
    store, engine, monkeypatch, entry, unsupported
):
    if unsupported == "epoch" and entry == "recovery":
        pytest.skip("reclaim requires the epoch-bearing recovery contract")
    if unsupported == "epoch":
        await pin_storage_format(engine, 1)
    case = await path_case(store, engine, entry, ProbeProvider())
    release = AsyncMock(wraps=store.release_run_claim)
    if unsupported == "method":
        monkeypatch.setattr(store, "release_run_claim", None)
    else:
        legacy_claims(store, monkeypatch)
        monkeypatch.setattr(store, "release_run_claim", release)
    monkeypatch.setattr(store, "get_next_trace_index", AsyncMock(side_effect=PersistenceError()))
    if entry == "resume_stream":
        assert (await case.invoke()).status == RunStatus.ERROR
    else:
        with pytest.raises(PersistenceError):
            await case.invoke()
    expected = "running" if entry == "recovery" else "error"
    assert (await store.get_run(case.run_id)).status == expected
    if unsupported == "epoch":
        release.assert_not_awaited()


async def test_get_result_keeps_recovery_read_strict(store, engine, monkeypatch):
    case = await path_case(store, engine, "recovery", ProbeProvider())
    done = await case.invoke()
    assert done.recovery is not None
    monkeypatch.setattr(store, "get_replayed_tool_calls", AsyncMock(side_effect=PersistenceError()))
    with pytest.raises(PersistenceError):
        await case.agent.get_result(case.run_id, tenant_id="tenant")
    cancelled = await case.agent.cancel_run(case.run_id, tenant_id="tenant")
    assert cancelled.status == RunStatus.SUCCESS and cancelled.recovery is None


@pytest.mark.parametrize("entry", ENTRIES[:6])
@pytest.mark.parametrize("unsupported", ["method", "epoch"])
async def test_submit_governance_without_release_finalizes(
    store, engine, monkeypatch, entry, unsupported
):
    from dendrux.runtime import submit

    if unsupported == "epoch":
        await pin_storage_format(engine, 1)
    case = await path_case(store, engine, entry, ProbeProvider())
    if unsupported == "method":
        monkeypatch.setattr(store, "release_run_claim", None)
    else:
        legacy_claims(store, monkeypatch)
    monkeypatch.setattr(
        submit, "report_governance_change", AsyncMock(side_effect=PersistenceError())
    )
    with pytest.raises(PersistenceError):
        await case.invoke()
    assert (await store.get_run(case.run_id)).status == "error"


@pytest.mark.parametrize(
    "entry", ["tool_results", "input", "approve", "resume", "resume_stream", "recovery"]
)
async def test_unavailable_release_cannot_finalize_new_claim(store, engine, monkeypatch, entry):
    from tests.integration.test_resume_read_failures import (
        test_stale_setup_release_cannot_change_new_claim,
    )

    await test_stale_setup_release_cannot_change_new_claim(
        store, engine, monkeypatch, entry, release_available=False
    )


@pytest.mark.parametrize("legacy_epoch", [False, True])
async def test_legacy_claim_postvalidation_failure_without_release(
    store, engine, monkeypatch, legacy_epoch
):
    from dendrux.runtime.submit import claim_paused_validated

    await pin_storage_format(engine, 1)
    case = await path_case(store, engine, "resume", ProbeProvider())
    original = store.claim_paused_run
    read = store.get_pause_state

    async def legacy_claim(run_id, *, expected_status):
        claimed = await original(run_id, expected_status=expected_status)
        return bool(claimed) if legacy_epoch else claimed

    async def post_claim_read(*args, **kwargs):
        if (await store.get_run(case.run_id)).status == "running":
            raise PersistenceError()
        return await read(*args, **kwargs)

    monkeypatch.setattr(store, "claim_paused_run", legacy_claim)
    monkeypatch.setattr(store, "release_run_claim", None)
    monkeypatch.setattr(store, "get_pause_state", post_claim_read)
    with pytest.raises(PersistenceError):
        await claim_paused_validated(
            store, case.run_id, expected_status="waiting_client_tool", validate=lambda raw: None
        )
    assert (await store.get_run(case.run_id)).status == "error"


@pytest.mark.parametrize(
    "entry", ["tool_results", "input", "approve", "recovery", "resume", "resume_stream"]
)
async def test_unavailable_release_during_claimed_prelude(store, engine, monkeypatch, entry):
    case = await path_case(store, engine, entry, ProbeProvider())
    monkeypatch.setattr(store, "release_run_claim", None)
    read = store.get_pause_state

    async def owned_read(*args, **kwargs):
        row = await store.get_run(case.run_id)
        if row.status == "running":
            raise PersistenceError()
        return await read(*args, **kwargs)

    if entry in {"resume", "resume_stream"}:
        monkeypatch.setattr(
            runner, "report_governance_change", AsyncMock(side_effect=PersistenceError())
        )
    else:
        monkeypatch.setattr(store, "get_pause_state", owned_read)
    if entry == "resume_stream":
        await case.invoke()
    else:
        with pytest.raises(PersistenceError):
            await case.invoke()
    expected = "running" if entry == "recovery" else "error"
    assert (await store.get_run(case.run_id)).status == expected


@pytest.mark.parametrize("release_fails", [False, True])
async def test_stream_governance_cleanup_reports_its_row(store, engine, monkeypatch, release_fails):
    case = await path_case(store, engine, "resume_stream", ProbeProvider())
    release = (
        AsyncMock(side_effect=PersistenceError())
        if release_fails
        else AsyncMock(wraps=store.release_run_claim)
    )
    monkeypatch.setattr(store, "release_run_claim", release)
    monkeypatch.setattr(
        runner, "report_governance_change", AsyncMock(side_effect=PersistenceError())
    )
    result = await case.invoke()
    expected = "running" if release_fails else "waiting_client_tool"
    assert result.status.value == expected
    assert (await store.get_run(case.run_id)).status == expected
    release.assert_awaited_once()


@pytest.mark.parametrize("entry", ["resume", "resume_stream"])
@pytest.mark.parametrize("release_result", ["success", "failure", "missing", "lost"])
async def test_release_observers_report_outcome(store, engine, monkeypatch, entry, release_result):
    case = await path_case(store, engine, entry, ProbeProvider())
    notifier = BaseNotifier()
    for hook in ("on_run_started", "on_run_finished", "on_run_failed"):
        setattr(notifier, hook, AsyncMock())
        monkeypatch.setattr(PersistenceRecorder, hook, AsyncMock())
    trace = runner._start_trace

    async def traced(_notifier, *args, **kwargs):
        return await trace(notifier, *args, **kwargs)

    monkeypatch.setattr(runner, "_start_trace", traced)
    monkeypatch.setattr(
        runner, "_replay_resume_injected_history", AsyncMock(side_effect=PersistenceError())
    )
    if release_result == "failure":
        monkeypatch.setattr(store, "release_run_claim", AsyncMock(side_effect=PersistenceError()))
    elif release_result == "lost":
        monkeypatch.setattr(store, "release_run_claim", AsyncMock(return_value=False))
    elif release_result == "missing":
        monkeypatch.setattr(store, "release_run_claim", None)
    expected = {
        "success": "waiting_client_tool",
        "failure": "running",
        "lost": "running",
        "missing": "error",
    }[release_result]
    if entry == "resume_stream":
        result = await case.invoke()
        assert case.events[-1].type == RunEventType.RUN_ERROR
        assert result.status.value == expected
        assert result.error_info.category == "persistence_error"
    else:
        with pytest.raises(PersistenceError):
            await case.invoke()
    assert (await store.get_run(case.run_id)).status == expected
    for observer in (notifier, PersistenceRecorder):
        observer.on_run_started.assert_awaited_once()
        if release_result == "success":
            observer.on_run_failed.assert_not_awaited()
            observer.on_run_finished.assert_awaited_once()
            assert observer.on_run_finished.await_args.args[1].status.value == expected
        else:
            observer.on_run_finished.assert_not_awaited()
            observer.on_run_failed.assert_awaited_once()


@pytest.mark.parametrize("read", ["get_next_trace_index", "get_pause_state"])
async def test_reclaim_without_release_can_retry_setup_failure(store, engine, monkeypatch, read):
    from tests._helpers.recovery_checkpoint import expire

    provider = ProbeProvider()
    case = await path_case(store, engine, "recovery", provider)
    before = await store.get_run(case.run_id)
    with monkeypatch.context() as patcher:
        patcher.setattr(store, "release_run_claim", None)
        patcher.setattr(store, read, AsyncMock(side_effect=PersistenceError()))
        with pytest.raises(PersistenceError):
            await case.invoke()
    row = await store.get_run(case.run_id)
    assert row.status == "running" and row.claim_epoch == before.claim_epoch + 1
    assert provider.calls == 0
    assert not any(e.event_type == "run.error" for e in await store.get_run_events(case.run_id))
    item = await expire(store, engine, case.run_id)
    result = await case.agent.reclaim(case.run_id, item=item, tenant_id="tenant")
    assert result.status == RunStatus.SUCCESS and provider.calls == 1


@pytest.mark.parametrize("kind", ["int", "view", "claim"])
async def test_unsupported_cleanup_keeps_original_epoch(store, engine, monkeypatch, kind):
    import asyncio

    from dendrux.runtime.state import claimed_store
    from dendrux.runtime.submit import _release_claim_result

    case = await path_case(store, engine, "resume", ProbeProvider())
    prior = await store.get_run(case.run_id)
    first = await store.claim_paused_run(case.run_id, expected_status=prior.status)
    assert await store.release_run_claim(
        case.run_id, status=prior.status, claim_epoch=first.claim_epoch
    )
    if engine.dialect.name in {"postgresql", "mysql"}:
        barrier = asyncio.Barrier(3)

        async def compete():
            await barrier.wait()
            return await store.claim_paused_run(case.run_id, expected_status=prior.status)

        winners = [claim for claim in await asyncio.gather(*(compete() for _ in range(3))) if claim]
        assert len(winners) == 1
        second = winners[0]
    else:
        second = await store.claim_paused_run(case.run_id, expected_status=prior.status)
    assert second.claim_epoch > first.claim_epoch
    monkeypatch.setattr(store, "release_run_claim", None)
    target = claimed_store(store, case.run_id, first.claim_epoch) if kind == "view" else store
    token = first if kind == "claim" else first.claim_epoch
    await _release_claim_result(target, case.run_id, prior.status, token, PersistenceError())
    row = await store.get_run(case.run_id)
    assert row.status == "running" and row.claim_epoch == second.claim_epoch
    assert not any(e.event_type == "run.error" for e in await store.get_run_events(case.run_id))
