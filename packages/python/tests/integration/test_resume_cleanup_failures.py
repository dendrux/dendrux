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
from tests._helpers.gates import wait_for_gate
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

    # Exercise both atomic failure and the preserved legacy split-event fallback.
    if failure == "save_run_event":
        store.finalize_with_event = None
    else:
        failure = "finalize_with_event"
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
        await wait_for_gate(entered, task=task)
        task.cancel("original cancellation")
        with pytest.raises(asyncio.CancelledError) as caught:
            await task
        assert caught.value.args == ("original cancellation",)

    # A failed event write must not prevent an otherwise healthy finalization.
    assert (await store.get_run(run_id)).status == (
        "running" if failure == "finalize_with_event" else "cancelled"
    )
    assert any(getattr(r, "code", None) == "cancel_persist_failed" for r in caplog.records)
    assert str(secondary) not in caplog.text


@pytest.mark.parametrize(
    "failure", [None, "finalize_run", "finalize_with_event", "save_run_event", "release_run_claim"]
)
async def test_claimed_resume_prelude_preserves_error_and_releases(store, failure, caplog):
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
    failed_hook = AsyncMock()
    forbidden_write = AsyncMock(side_effect=secondary)
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
            if failure is None:
                await action
            else:
                with patch.object(store, failure, forbidden_write):
                    await action
    assert caught.value is original
    failed_hook.assert_not_awaited()
    assert (await store.get_run(run_id)).status == (
        "running" if failure == "release_run_claim" else "waiting_client_tool"
    )
    if failure in {"finalize_run", "finalize_with_event", "save_run_event"}:
        forbidden_write.assert_not_awaited()
    if failure == "release_run_claim":
        assert any(getattr(r, "code", None) == "persistence_error" for r in caplog.records)
    assert str(secondary) not in caplog.text
