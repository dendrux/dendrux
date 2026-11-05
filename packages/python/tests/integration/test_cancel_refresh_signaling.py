"""A committed cancellation request must signal local work even if its response fails."""

import asyncio
from unittest.mock import Mock

import pytest

from dendrux import Agent, tool
from dendrux.errors import PersistenceError
from dendrux.types import LLMResponse, RunStatus
from tests._helpers.scripted_provider import ScriptedLLM, tool_call


@pytest.mark.parametrize("failure", ["refresh", "transition", "frontier"])
async def test_cancel_response_failure_still_interrupts_tool(store, monkeypatch, failure):
    entered, gate, cancelled = asyncio.Event(), asyncio.Event(), asyncio.Event()

    @tool()
    async def blocked() -> str:
        """Wait until interrupted."""
        entered.set()
        try:
            await gate.wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return "done"

    agent = Agent(
        name="cancel-refresh",
        prompt="p",
        state_store=store,
        strict_tenancy=False,
        tools=[blocked],
        provider=ScriptedLLM([tool_call("blocked", pid="b"), LLMResponse(text="done")]),
    )
    task = asyncio.create_task(agent.run("go", tenant_id="tenant"))
    try:
        await asyncio.wait_for(entered.wait(), 10)
        run_id = (await store.list_runs())[0].id
        signal = Mock(wraps=agent._task_manager.cancel)
        requested = []
        method = {
            "refresh": "get_run",
            "transition": "finalize_with_event",
            "frontier": "get_children_page",
        }[failure]
        original = getattr(store, method)
        calls = 0

        async def fail_response(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == (2 if failure == "refresh" else 1):
                requested.append((await store.get_run_control(run_id)).cancel_requested)
                raise PersistenceError()
            return await original(*args, **kwargs)

        with monkeypatch.context() as patcher:
            patcher.setattr(agent._task_manager, "cancel", signal)
            patcher.setattr(store, method, fail_response)
            with pytest.raises(PersistenceError):
                await agent.cancel_run(run_id, tenant_id="tenant", cascade=failure == "frontier")
        if failure != "frontier":
            assert requested == [True]
        signal.assert_called_once_with(run_id)
        await asyncio.wait_for(cancelled.wait(), 10)
        assert (await asyncio.wait_for(task, 10)).status == RunStatus.CANCELLED
    finally:
        gate.set()
        await asyncio.gather(task, return_exceptions=True)


async def test_legacy_cascade_signals_completed_children_when_later_child_fails(store, monkeypatch):
    from types import SimpleNamespace

    agent = Agent(name="cascade", prompt="p", state_store=store, strict_tenancy=False)
    await store.create_run("root", agent.name, tenant_id="tenant")
    monkeypatch.setattr(store, "mark_cancel_requested", None)
    monkeypatch.setattr(agent, "_mark_cancel_subtree", _levels)
    attempted, signalled = [], []

    async def cancel_one(_store, run_id, *, signal):
        attempted.append(run_id)
        if run_id == "child-b":
            raise PersistenceError()
        return "cancel_requested"

    monkeypatch.setattr(agent, "_cancel_one", cancel_one)
    monkeypatch.setattr(agent, "_task_manager", SimpleNamespace(cancel=signalled.append))
    with pytest.raises(PersistenceError):
        await agent.cancel_run("root", tenant_id="tenant", cascade=True)
    assert attempted == ["child-a", "child-b"]
    assert signalled == ["child-a"]


async def _levels(*args, **kwargs):
    return [["root"], ["child-a", "child-b"]], False
