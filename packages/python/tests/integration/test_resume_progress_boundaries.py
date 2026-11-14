"""Durable consumption, dispatch and lifecycle boundaries of resumed streams."""

from functools import wraps
from unittest.mock import AsyncMock

import pytest

from dendrux import tool
from dendrux.errors import PersistenceError
from dendrux.runtime import _reclaim, _run_resume_prep, runner
from dendrux.runtime.persistence import PersistenceRecorder
from dendrux.types import RunStatus, ToolCall
from tests._helpers.runtime_paths import ProbeProvider, path_case
from tests.integration.test_060_reclaim_agent import checkpoint, make_agent
from tests.integration.test_cross_path_invariants import paused_recovery


async def test_paused_stream_provider_failure_is_not_released(store, engine, monkeypatch):
    case = await path_case(store, engine, "submit_resume", ProbeProvider(error=PersistenceError()))
    pause = await store.get_pause_state(case.run_id)
    release = AsyncMock(wraps=store.release_run_claim)
    monkeypatch.setattr(store, "release_run_claim", release)
    async with case.agent.resume_stream(
        case.run_id, tenant_id="tenant", expected_pause_id=pause["pause_id"]
    ) as stream:
        async for _ in stream:
            pass
    release.assert_not_awaited()
    assert (await store.get_run(case.run_id)).status == "error"


@pytest.mark.parametrize("entry", ["resume", "resume_stream"])
async def test_recovery_consumption_failure_is_not_released(store, engine, monkeypatch, entry):
    @tool()
    async def work() -> str:
        """An interrupted non-idempotent call."""
        raise AssertionError("must fail before dispatch")

    agent = make_agent(store, tools=[work])
    item = await checkpoint(
        agent, store, engine, calls=[ToolCall("work")], phases=["dispatchable"], started=True
    )
    paused = await agent.reclaim(item.run_id, tenant_id="tenant", item=item)
    assert paused.status == RunStatus.WAITING_APPROVAL
    consume = PersistenceRecorder.consume_resume
    consumed = []

    async def consumed_then_fail(self, messages):
        assert self._loop_state["submission"] is not None
        await consume(self, messages)
        assert self._loop_state["submission"] is None
        consumed.append(True)
        raise PersistenceError()

    release = AsyncMock(wraps=store.release_run_claim)
    monkeypatch.setattr(store, "release_run_claim", release)
    monkeypatch.setattr(PersistenceRecorder, "consume_resume", consumed_then_fail)
    scope = dict(tenant_id="tenant", expected_pause_id=paused.pause_id)
    if entry == "resume_stream":
        async with agent.resume_stream(item.run_id, **scope) as stream:
            async for _ in stream:
                pass
    else:
        with pytest.raises(PersistenceError):
            await agent.resume(item.run_id, **scope)
    assert consumed == [True]
    release.assert_not_awaited()
    assert (await store.get_run(item.run_id)).status == "error"


async def test_stream_recovery_marks_work_independently_of_submission(store, engine, monkeypatch):
    agent, paused, effects = await paused_recovery(store, engine)
    prepare = runner._prepare_resume

    async def no_submission(*args, **kwargs):
        ctx = await prepare(*args, **kwargs)
        assert ctx.recorder._loop_state.get("submission") is None
        return ctx

    resolve = _reclaim.resolve
    progress_seen = []

    async def dispatched_then_fail(*args, **kwargs):
        original = kwargs["on_work"]

        def started():
            original()
            shared = original.args[0]
            progress_seen.append(shared["progress"].started)

        kwargs["on_work"] = started
        await resolve(*args, **kwargs)
        assert effects == [True]
        raise PersistenceError()

    release = AsyncMock(wraps=store.release_run_claim)
    monkeypatch.setattr(store, "release_run_claim", release)
    monkeypatch.setattr(_run_resume_prep, "_prepare_resume", no_submission)
    monkeypatch.setattr(_reclaim, "resolve", dispatched_then_fail)
    async with agent.resume_stream(
        paused.run_id, tenant_id="tenant", expected_pause_id=paused.pause_id
    ) as stream:
        async for _ in stream:
            pass
    assert progress_seen == [True]
    release.assert_not_awaited()
    assert (await store.get_run(paused.run_id)).status == "error"


@pytest.mark.parametrize("entry", ["resume", "resume_stream"])
async def test_resolved_rejected_call_is_not_repeated_after_release(
    store, engine, monkeypatch, entry
):
    effects = []

    @tool()
    async def work() -> str:
        """A rejected call must never execute."""
        effects.append(True)
        return "done"

    agent = make_agent(store, tools=[work])
    call = ToolCall("work")
    item = await checkpoint(agent, store, engine, calls=[call], phases=["rejected"], started=True)
    reclaim = store.reclaim_run
    rebuild = _reclaim.rebuild_pause

    async def decided_pause(*args, **kwargs):
        pause = await rebuild(*args, **kwargs)
        pause.pending_decisions = {call.id: {"approved": False, "reason": "declined"}}
        return pause

    @wraps(reclaim)
    async def claim_then_pause(*args, **kwargs):
        claim = await reclaim(*args, **kwargs)
        await store.request_pause(item.run_id)
        return claim

    with monkeypatch.context() as patcher:
        patcher.setattr(store, "reclaim_run", claim_then_pause)
        patcher.setattr(_reclaim, "rebuild_pause", decided_pause)
        paused = await agent.reclaim(item.run_id, tenant_id="tenant", item=item)
    assert paused.status == RunStatus.PAUSED
    assert (await store.get_loop_state(item.run_id))["submission"] is None
    assert (await store.get_loop_state(item.run_id))["pending"][0]["phase"] == "rejected"
    resolve = _reclaim.resolve

    async def observed_then_fail(*args, **kwargs):
        await resolve(*args, **kwargs)
        assert not args[3].recorder._loop_state["pending"]
        raise PersistenceError()

    async def invoke():
        scope = dict(tenant_id="tenant", expected_pause_id=paused.pause_id)
        if entry == "resume":
            return await agent.resume(item.run_id, **scope)
        async with agent.resume_stream(item.run_id, **scope) as stream:
            async for _ in stream:
                pass
        return stream.result

    with monkeypatch.context() as patcher:
        patcher.setattr(_reclaim, "resolve", observed_then_fail)
        if entry == "resume":
            with pytest.raises(PersistenceError):
                await invoke()
        else:
            assert (await invoke()).status == RunStatus.PAUSED
    assert (await store.get_run(item.run_id)).status == "paused"
    assert not (await store.get_loop_state(item.run_id))["pending"]
    assert (await invoke()).status == RunStatus.SUCCESS
    traces = [t for t in await store.get_traces(item.run_id) if t.role == "tool"]
    assert len(traces) == 1 and "outcome_unknown" in traces[0].content
    assert effects == []
