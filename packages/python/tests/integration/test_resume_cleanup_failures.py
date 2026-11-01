"""Secondary persistence failures must preserve the original resume exception."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from dendrux import Agent
from dendrux.llm.mock import MockLLM
from dendrux.loops.base import BaseNotifier
from dendrux.runtime.runner import resume_claimed
from dendrux.runtime.submit import serialize_tool_results
from dendrux.types import LLMResponse, ToolCall, ToolResult
from tests.integration.test_skills_v2 import client


async def paused_client_run(store):
    a = Agent(
        strict_tenancy=False,
        prompt="test",
        provider=MockLLM([LLMResponse(tool_calls=[ToolCall(name="client", params={})])]),
        tools=[client],
        state_store=store,
    )
    paused = await a.run("input")
    state = await store.get_pause_state(paused.run_id)
    results = [
        ToolResult(name="client", call_id=state["pending_tool_calls"][0]["id"], payload='"ok"')
    ]
    return a, paused.run_id, results


@pytest.mark.parametrize("path", ["resume", "submit"])
@pytest.mark.parametrize("failure", ["finalize_run", "save_run_event"])
async def test_cancel_preserves_exception_despite_store_failure(store, path, failure, caplog):
    a, run_id, results = await paused_client_run(store)
    entered = asyncio.Event()

    async def preparing(*args, **kwargs):
        entered.set()
        await asyncio.Event().wait()

    secondary = RuntimeError("injected cancellation persistence failure")
    with (
        patch("dendrux.runtime.runner._prepare_resume", preparing),
        patch.object(store, failure, AsyncMock(side_effect=secondary)),
    ):
        task = asyncio.create_task(
            a.resume(run_id, tool_results=results)
            if path == "resume"
            else a.submit_tool_results(run_id, results)
        )
        await asyncio.wait_for(entered.wait(), 5)
        task.cancel("original cancellation")
        with pytest.raises(asyncio.CancelledError) as caught:
            await task
        assert caught.value.args == ("original cancellation",)

    # A failed event write must not prevent an otherwise healthy finalization.
    assert (await store.get_run(run_id)).status == (
        "running" if failure == "finalize_run" else "cancelled"
    )
    assert any(getattr(r, "code", None) == "cancel_persist_failed" for r in caplog.records)
    assert str(secondary) not in caplog.text


@pytest.mark.parametrize("failure", [None, "finalize_run", "save_run_event", "on_run_failed"])
async def test_claimed_resume_preserves_error_and_notifies(store, failure, caplog):
    # This fault injection / pause blob exercises the format-1 write path.
    from tests._fixtures.storage import pin_storage_format

    await pin_storage_format(store._engine, 1)
    a, run_id, results = await paused_client_run(store)
    await store.submit_and_claim(
        run_id,
        expected_status="waiting_client_tool",
        submitted_data={"submitted_tool_results": serialize_tool_results(results)},
    )
    original = ValueError("original continuation failure")
    secondary = RuntimeError("injected claimed-resume cleanup failure")
    notifier = BaseNotifier()
    failed_hook = AsyncMock(side_effect=secondary if failure == "on_run_failed" else None)
    with (
        patch("dendrux.runtime.runner.continuation", AsyncMock(side_effect=original)),
        patch.object(notifier, "on_run_failed", failed_hook),
    ):
        action = resume_claimed(
            run_id,
            state_store=store,
            agent=a,
            provider=a._provider,
            extra_notifier=notifier,
            pre_claim_status="waiting_client_tool",
        )
        with pytest.raises(ValueError) as caught:
            if failure in {None, "on_run_failed"}:
                await action
            else:
                with patch.object(store, failure, AsyncMock(side_effect=secondary)):
                    await action
    assert caught.value is original
    failed_hook.assert_awaited_once_with(run_id, original, iteration=None)
    assert (await store.get_run(run_id)).status == (
        "running" if failure == "finalize_run" else "error"
    )
    if failure is not None:
        assert str(secondary) in caplog.text
