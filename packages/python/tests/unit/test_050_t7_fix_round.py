"""Legacy store claims, cleanup cancellation, and delegation identity regressions."""

import asyncio

import pytest

from dendrux import Agent, tool
from dendrux.runtime.context import get_store_identity
from dendrux.runtime.runner import _shielded_cleanup
from dendrux.runtime.state import claimed_store
from dendrux.types import LLMResponse, PauseState, RunStatus, UsageStats
from tests._helpers.scripted_provider import ScriptedLLM, tool_call
from tests.unit.test_resume import RecordingStateStore


@pytest.mark.parametrize("entry", ["approval", "paused", "resume"])
async def test_legacy_claim_signature_supports_submit_and_resume(entry):
    effects = []

    @tool()
    async def effect() -> str:
        effects.append("executed")
        return "ok"

    store = RecordingStateStore()
    provider = ScriptedLLM([tool_call("effect", pid="a"), LLMResponse(text="done")])
    a = Agent(
        name="legacy",
        prompt="p",
        tools=[effect],
        require_approval=["effect"],
        provider=provider,
        state_store=store,
        strict_tenancy=False,
        on_governance_change="warn",
    )
    if entry == "paused":
        await store.create_run("seed", "legacy")
        pause = PauseState(
            agent_name="legacy",
            pending_tool_calls=[],
            history=[],
            steps=[],
            iteration=1,
            trace_order_offset=0,
            usage=UsageStats(),
        )
        await store.pause_run("seed", status="paused", pause_data=pause.to_dict())
        provider._responses = [LLMResponse(text="done")]
        result = await a.submit_resume("seed")
    else:
        paused = await a.run("go")
        result = await (
            a.submit_approval(paused.run_id, approved=True, expected_pause_id=paused.pause_id)
            if entry == "approval"
            else a.resume(paused.run_id, expected_pause_id=paused.pause_id)
        )
    assert result.status == RunStatus.SUCCESS
    assert effects == ([] if entry == "paused" else ["executed"])


async def test_cleanup_drains_then_reraises_original_cancellation():
    entered, release, finished = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def cleanup():
        entered.set()
        await release.wait()
        finished.set()
        return "done"

    task = asyncio.create_task(_shielded_cleanup(cleanup))
    await entered.wait()
    task.cancel("caller cancellation")
    await asyncio.sleep(0)
    assert not task.done()
    task.cancel("second cancellation")
    release.set()
    with pytest.raises(asyncio.CancelledError) as caught:
        await task
    assert caught.value.args == ("caller cancellation",)
    assert finished.is_set() and task.cancelled()


async def test_custom_store_view_keeps_parent_child_link():
    store = RecordingStateStore()
    assert get_store_identity(claimed_store(store, "parent", 0)) == get_store_identity(store)
    child = Agent(
        name="child",
        prompt="p",
        provider=ScriptedLLM([LLMResponse(text="child")]),
        state_store=store,
        strict_tenancy=False,
    )

    @tool()
    async def delegate() -> str:
        return (await child.run("child input")).answer

    parent = Agent(
        name="parent",
        prompt="p",
        tools=[delegate],
        state_store=store,
        provider=ScriptedLLM([tool_call("delegate", pid="a"), LLMResponse(text="parent")]),
        strict_tenancy=False,
    )
    result = await parent.run("go")
    assert result.status == RunStatus.SUCCESS
    child_row = next(row for row in store.created_runs if row["agent_name"] == "child")
    assert child_row["parent_run_id"] == result.run_id


async def test_legacy_claim_signature_is_cached_and_warning_logged_once(monkeypatch, caplog):
    import inspect

    from dendrux.runtime import submit

    submit._claim_validation_support.cache_clear()
    implementation = RecordingStateStore.claim_paused_run
    signature = inspect.signature
    inspections = []

    def tracked(method, *args, **kwargs):
        if method is implementation:
            inspections.append(method)
        return signature(method, *args, **kwargs)

    monkeypatch.setattr(inspect, "signature", tracked)
    store = RecordingStateStore()
    for rid in ["first", "second"]:
        await store.create_run(rid, "legacy")
        await store.pause_run(rid, status="paused", pause_data={"agent_name": "legacy"})
        await submit.claim_or_raise(store, rid, expected_status=RunStatus.PAUSED)
    assert len(inspections) == 1
    assert (
        sum(getattr(record, "code", "") == "legacy_claim_validation" for record in caplog.records)
        == 1
    )


async def test_legacy_claim_validation_rechecks_after_claim():
    from dendrux.runtime.submit import claim_or_raise

    class InterleavedStore(RecordingStateStore):
        async def claim_paused_run(self, run_id, *, expected_status):
            epoch = await super().claim_paused_run(run_id, expected_status=expected_status)
            self._pause_data[run_id]["pause_id"] = "different"
            return epoch

    store = InterleavedStore()
    await store.create_run("r", "legacy")
    await store.pause_run("r", status="paused", pause_data={"pause_id": "observed"})
    from dendrux.errors import PauseStatusMismatchError

    with pytest.raises(PauseStatusMismatchError):
        await claim_or_raise(store, "r", expected_status=RunStatus.PAUSED)


async def test_fenced_recorder_finish_failure_does_not_leave_notifier_open():
    from unittest.mock import AsyncMock

    from dendrux.loops.base import BaseNotifier, BaseRecorder
    from dendrux.runtime.runner import _finish_fenced_lifecycle
    from dendrux.types import RunResult

    class FailedRecorder(BaseRecorder):
        async def on_run_finished(self, run_id, result):
            raise RuntimeError("finish failed")

    notifier = BaseNotifier()
    notifier.on_run_finished = AsyncMock()
    result = RunResult(run_id="r", status=RunStatus.ERROR)
    await _finish_fenced_lifecycle("r", result, FailedRecorder(), notifier)
    notifier.on_run_finished.assert_awaited_once_with("r", result)


async def test_cleanup_propagates_pending_anyio_cancellation_after_shield():
    import anyio

    entered, release, finished = asyncio.Event(), asyncio.Event(), asyncio.Event()
    returned = False

    async def cleanup():
        entered.set()
        await release.wait()
        finished.set()

    with anyio.CancelScope() as scope:

        async def cancel():
            await entered.wait()
            scope.cancel()
            release.set()

        async with anyio.create_task_group() as group:
            group.start_soon(cancel)
            await _shielded_cleanup(cleanup)
            returned = True
    assert finished.is_set() and not returned
