"""Cancellation races at lifecycle awaits, with a second Postgres caller."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from dendrux import tool
from dendrux.loops import _helpers
from dendrux.runtime import runner
from dendrux.runtime.persistence import PersistenceRecorder
from dendrux.types import LLMResponse, RunStatus
from tests._fixtures.storage import pin_storage_format
from tests._helpers.h0_races import Gate, event_counts, running
from tests._helpers.h0_races import peer as peer
from tests._helpers.runtime_paths import PATHS, path_case
from tests._helpers.scripted_provider import ScriptedLLM, tool_call

pytestmark = pytest.mark.parametrize("engine", ["postgres"], indirect=True)
PATH_FORMATS = [(path, fmt) for path in PATHS for fmt in (1, 2) if path != "recovery" or fmt == 2]


class IdentifiedProvider(ScriptedLLM):
    async def complete(self, *args, **kwargs):
        self.run_id = kwargs["run_id"]
        return await super().complete(*args, **kwargs)


@tool(target="client")
async def wait_on_client() -> str:
    """Pause for a client observation."""
    return "unused"


@pytest.mark.parametrize("path,storage_format", PATH_FORMATS)
@pytest.mark.parametrize("cut", ["outcome", "checkpoint", "pause_cas"])
async def test_cancel_committed_before_pause_wins(
    store, engine, peer, monkeypatch, path, cut, storage_format
):
    await pin_storage_format(engine, storage_format)
    provider = IdentifiedProvider([tool_call("wait_on_client", pid="client")])
    case = await path_case(store, engine, path, provider)
    case.agent.tools.append(wait_on_client)
    gate = Gate()
    owner, method = {
        "outcome": (runner, "_persist_loop_outcome"),
        "checkpoint": (PersistenceRecorder, "checkpoint_outcome"),
        "pause_cas": (store, "pause_with_event"),
    }[cut]
    gate.install(monkeypatch, owner, method)
    other, _ = peer
    try:
        async with running(case.invoke(), gate) as task:
            await gate.entered.wait()
            run_id = provider.run_id
            before = await event_counts(other, run_id)
            assert await other.request_cancel(run_id)
            gate.release.set()
            result = await task
            assert result.status == RunStatus.CANCELLED
            row = await other.get_run(run_id)
            assert row.status == "cancelled"
            assert await other.get_pause_state(run_id) is None
            after = await event_counts(other, run_id)
            assert after["run.cancelled"] - before["run.cancelled"] == 1
            assert after["run.paused"] == before["run.paused"]
    finally:
        gate.release.set()
        await case.agent.close()


@pytest.mark.parametrize("path,storage_format", PATH_FORMATS)
@pytest.mark.parametrize(
    "cut,caller",
    [("before_commit", False), ("after_commit", False), ("finish", False), ("finish", True)],
)
async def test_completion_beats_late_durable_cancel(
    store, engine, peer, monkeypatch, path, cut, caller, storage_format
):
    await pin_storage_format(engine, storage_format)
    provider = IdentifiedProvider([LLMResponse(text="completed answer")])
    case = await path_case(store, engine, path, provider)
    gate = Gate()
    if cut == "finish":
        gate.install(monkeypatch, _helpers, "record_run_finished")
        finish_hook = AsyncMock(wraps=_helpers.record_run_finished)
        monkeypatch.setattr(_helpers, "record_run_finished", finish_hook)
    else:
        gate.install(
            monkeypatch,
            store,
            "finalize_with_event",
            after=cut == "after_commit",
        )
    other, _ = peer
    try:
        async with running(case.invoke(), gate) as task:
            await gate.entered.wait()
            assert gate.args is not None
            run_id = provider.run_id
            requested = await other.request_cancel(run_id)
            assert requested is (cut == "before_commit")
            if caller:
                task.cancel("late caller cancellation")
            gate.release.set()
            if caller:
                with pytest.raises(asyncio.CancelledError):
                    await task
            else:
                result = await task
                assert result.status == RunStatus.SUCCESS
            row = await other.get_run(run_id)
            assert row.status == "success" and row.answer == "completed answer"
            counts = await event_counts(other, run_id)
            assert counts["run.completed"] == 1 and counts["run.cancelled"] == 0
    finally:
        gate.release.set()
        await case.agent.close()


@pytest.mark.parametrize(
    "path,storage_format,method",
    [
        (path, fmt, method)
        for path, fmt in PATH_FORMATS
        for method in (
            "finalize_with_event",
            "get_notes",
            "record_run_finished",
            "notify_run_finished",
        )
        if method != "get_notes" or path not in {"run_stream", "resume_stream"}
    ],
)
@pytest.mark.parametrize("after", [False, True], ids=["before_await", "after_await"])
async def test_repeated_caller_cancel_drains_cleanup_once(
    store, engine, peer, monkeypatch, path, method, after, storage_format
):
    await pin_storage_format(engine, storage_format)
    provider = IdentifiedProvider([LLMResponse(text="unused")], gate_at=0)
    case = await path_case(store, engine, path, provider)
    gate = Gate()
    other, _ = peer
    try:
        async with running(case.invoke(), gate) as task:
            await provider.entered.wait()
            run_id = provider.run_id
            before = await event_counts(other, run_id)
            owner = _helpers if method in {"record_run_finished", "notify_run_finished"} else store
            gate.install(monkeypatch, owner, method, after=after)
            if owner is _helpers:
                finish_hook = AsyncMock(wraps=getattr(_helpers, method))
                monkeypatch.setattr(_helpers, method, finish_hook)
            task.cancel("first cancellation")
            await gate.entered.wait()
            assert gate.args is not None
            # A separate caller also requests cancellation while cleanup is suspended.
            await other.request_cancel(run_id)
            task.cancel("cancellation during cleanup")
            await asyncio.sleep(0)
            if owner is store:
                assert not task.done(), "durable cleanup must survive cancellation of its waiter"
            gate.release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert (await other.get_run(run_id)).status == "cancelled"
            counts = await event_counts(other, run_id)
            assert counts["run.cancelled"] - before["run.cancelled"] == 1
            assert counts["run.completed"] == before["run.completed"]
            assert counts["run.error"] == before["run.error"]
            assert provider.calls == 1
            if owner is _helpers:
                finish_hook.assert_awaited_once()
    finally:
        gate.release.set()
        provider.gate.set()
        await case.agent.close()
