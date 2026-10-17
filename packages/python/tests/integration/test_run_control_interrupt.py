"""In-process interrupt of batch run(), cascade cancel/pause, tenant rules, heartbeat.

cancel_run interrupts the registered run() task wherever it awaits; the
CancelledError can land mid-query and SQLAlchemy then discards that
connection, which an in-memory SQLite database does not survive. Tests that
interrupt a task therefore run on ``sqlite_file`` and Postgres.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import threading
from typing import Any

import pytest
from sqlalchemy import update

import dendrux
from dendrux.agent import Agent
from dendrux.db.models import AgentRun
from dendrux.errors import RunNotFoundError
from dendrux.loops.base import BaseNotifier
from dendrux.runtime.context import get_delegation_context
from dendrux.tool import tool
from dendrux.types import LLMResponse, RunStatus
from tests._helpers.scripted_provider import ScriptedLLM, tool_call

pytestmark = pytest.mark.asyncio

_INTERRUPTIBLE = pytest.mark.parametrize("engine", ["sqlite_file", "postgres"], indirect=True)


async def _only_run_id(store: Any, *, parent: str | None = None) -> str:
    runs = await store.list_runs(parent_run_id=parent) if parent else await store.list_runs()
    if parent is None:
        runs = [r for r in runs if r.parent_run_id is None]
    assert len(runs) == 1
    return str(runs[0].id)


async def _events(store: Any, run_id: str, event_type: str) -> list[Any]:
    return [e for e in await store.get_run_events(run_id) if e.event_type == event_type]


# ---------------------------------------------------------------------------
# run() cancel at every registration phase
# ---------------------------------------------------------------------------


@_INTERRUPTIBLE
class TestRunInterrupt:
    async def test_cancel_before_registration_uses_durable_flag(self, store) -> None:
        llm = ScriptedLLM([LLMResponse(text="unreached")])
        agent = Agent(provider=llm, prompt="Test.", state_store=store)
        original = store.create_run

        async def create_then_cancel(run_id: str, *args: Any, **kwargs: Any) -> Any:
            created = await original(run_id, *args, **kwargs)
            # Row exists, task not yet registered: only the DB flag can land.
            await agent.cancel_run(run_id)
            return created

        store.create_run = create_then_cancel  # type: ignore[method-assign]
        result = await agent.run("go")

        assert result.status == RunStatus.CANCELLED
        assert "interrupted" not in result.meta
        assert llm.calls == 0
        assert (await store.get_run(result.run_id)).status == "cancelled"

    async def test_cancel_during_complete_returns_interrupted_result(self, store) -> None:
        llm = ScriptedLLM([LLMResponse(text="unreached")], gate_at=0)
        agent = Agent(provider=llm, prompt="Test.", state_store=store)
        task = asyncio.create_task(agent.run("go"))
        await asyncio.wait_for(llm.entered.wait(), 5)
        run_id = await _only_run_id(store)

        requested = await agent.cancel_run(run_id)
        assert requested.status == RunStatus.RUNNING
        assert requested.meta["cancel_requested"] is True
        result = await asyncio.wait_for(task, 5)

        assert result.status == RunStatus.CANCELLED
        assert result.meta["interrupted"] is True
        assert result.meta["partial_output"] is False
        assert result.answer is None
        assert llm.cancelled is True
        record = await store.get_run(run_id)
        assert record.status == "cancelled"
        assert record.cancel_requested is False
        cancelled = await _events(store, run_id, "run.cancelled")
        assert len(cancelled) == 1
        assert cancelled[0].data == {
            "reason": "cancel_requested",
            "interrupted": True,
            "partial_output": False,
        }
        assert not agent._task_manager._owned and not agent._task_manager._pending

    async def test_cancel_during_tool(self, store) -> None:
        entered = asyncio.Event()
        state = {"cancelled": False}

        @tool()
        async def slow() -> str:
            entered.set()
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                state["cancelled"] = True
                raise
            return "unreached"

        llm = ScriptedLLM([tool_call("slow", pid="t1"), LLMResponse(text="unreached")])
        agent = Agent(provider=llm, prompt="Test.", tools=[slow], state_store=store)
        task = asyncio.create_task(agent.run("go"))
        await asyncio.wait_for(entered.wait(), 5)
        run_id = await _only_run_id(store)

        await agent.cancel_run(run_id)
        result = await asyncio.wait_for(task, 5)
        assert result.status == RunStatus.CANCELLED
        assert result.meta["interrupted"] is True
        assert state["cancelled"] is True
        assert llm.calls == 1
        assert (await store.get_run(run_id)).status == "cancelled"

    async def test_cancel_after_provider_returned_before_finish(self, store) -> None:
        """The provider already returned the final answer but the run has not
        persisted it yet: the interrupt wins and nothing is recorded as done."""
        gate = asyncio.Event()
        entered = asyncio.Event()

        class Blocking(BaseNotifier):
            async def on_llm_call_completed(self, *args: Any, **kwargs: Any) -> None:
                entered.set()
                await gate.wait()

        llm = ScriptedLLM([LLMResponse(text="final")])
        agent = Agent(provider=llm, prompt="Test.", state_store=store)
        task = asyncio.create_task(agent.run("go", notifier=Blocking()))
        await asyncio.wait_for(entered.wait(), 5)
        run_id = await _only_run_id(store)

        await agent.cancel_run(run_id)
        result = await asyncio.wait_for(task, 5)
        assert result.status == RunStatus.CANCELLED
        assert result.meta["interrupted"] is True
        assert (await store.get_run(run_id)).status == "cancelled"

    async def test_cancel_after_completion_is_terminal_noop(self, store) -> None:
        llm = ScriptedLLM([LLMResponse(text="final")])
        agent = Agent(provider=llm, prompt="Test.", state_store=store)
        result = await agent.run("go")
        assert result.status == RunStatus.SUCCESS
        again = await agent.cancel_run(result.run_id)
        assert again.status == RunStatus.SUCCESS
        assert again.answer == "final"

    async def test_caller_task_cancel_still_raises(self, store) -> None:
        llm = ScriptedLLM([LLMResponse(text="unreached")], gate_at=0)
        agent = Agent(provider=llm, prompt="Test.", state_store=store)
        task = asyncio.create_task(agent.run("go"))
        await asyncio.wait_for(llm.entered.wait(), 5)
        run_id = await _only_run_id(store)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        record = await store.get_run(run_id)
        assert record.status == "cancelled"
        assert (await _events(store, run_id, "run.cancelled"))[0].data == {
            "reason": "task_cancelled"
        }

    async def test_other_agent_instance_does_not_interrupt(self, store) -> None:
        llm = ScriptedLLM([LLMResponse(text="final")], gate_at=0)
        owner = Agent(provider=llm, prompt="Test.", state_store=store)
        other = Agent(provider=ScriptedLLM([]), prompt="Test.", state_store=store)
        task = asyncio.create_task(owner.run("go"))
        await asyncio.wait_for(llm.entered.wait(), 5)
        run_id = await _only_run_id(store)

        await other.cancel_run(run_id)
        await asyncio.sleep(0.05)
        assert not task.done()
        llm.gate.set()
        result = await asyncio.wait_for(task, 5)
        # The current call finished; the durable flag ended the run after it.
        assert llm.cancelled is False
        assert result.status in (RunStatus.SUCCESS, RunStatus.CANCELLED)


# ---------------------------------------------------------------------------
# Cascade + nested delegation
# ---------------------------------------------------------------------------


def _delegation(store: Any, *, child_gate: bool = True) -> dict[str, Any]:
    """Parent agent whose ``delegate`` tool runs a child agent inline."""
    counts = {"child_step": 0}

    @tool()
    async def child_step() -> str:
        counts["child_step"] += 1
        return "child stepped"

    child_llm = ScriptedLLM(
        [tool_call("child_step", pid="c1"), LLMResponse(text="child done")],
        gate_at=0 if child_gate else None,
    )
    child = Agent(
        name="Child", provider=child_llm, prompt="Child.", tools=[child_step], state_store=store
    )

    @tool()
    async def delegate() -> dict[str, Any]:
        res = await child.run("sub task")
        return {"status": res.status.value, "answer": res.answer, "run_id": res.run_id}

    parent_llm = ScriptedLLM([tool_call("delegate", pid="p1"), LLMResponse(text="parent done")])
    parent = Agent(
        name="Parent", provider=parent_llm, prompt="Parent.", tools=[delegate], state_store=store
    )
    return {
        "parent": parent,
        "child": child,
        "parent_llm": parent_llm,
        "child_llm": child_llm,
        "counts": counts,
    }


class TestCascade:
    async def test_nested_cascade_pause_is_flag_propagation_only(self, store) -> None:
        d = _delegation(store)
        task = asyncio.create_task(d["parent"].run("go"))
        await asyncio.wait_for(d["child_llm"].entered.wait(), 5)
        parent_id = await _only_run_id(store)
        child_id = await _only_run_id(store, parent=parent_id)

        outcome = await d["parent"].request_pause(parent_id)  # cascade=True default
        assert outcome.affected == [(child_id, "requested"), (parent_id, "requested")]
        d["child_llm"].gate.set()
        parent_paused = await asyncio.wait_for(task, 5)

        # Child paused at its checkpoint and returned a paused result to the
        # delegating tool; the parent recorded that and paused at its own.
        assert parent_paused.status == RunStatus.PAUSED
        assert (await store.get_run(child_id)).status == "paused"
        delegate_result = (await store.get_tool_calls(parent_id))[0].result
        assert delegate_result["status"] == "paused"
        assert delegate_result["answer"] is None

        child_done = await d["child"].submit_resume(child_id)
        assert child_done.status == RunStatus.SUCCESS
        assert child_done.answer == "child done"
        # Resuming the child never rewrites the parent's recorded tool result.
        assert (await store.get_tool_calls(parent_id))[0].result == delegate_result

        parent_done = await d["parent"].submit_resume(parent_id)
        assert parent_done.status == RunStatus.SUCCESS
        assert d["parent_llm"].contents(1)[-1].count('"paused"') == 1
        assert d["counts"]["child_step"] == 1

    async def test_cascade_cancel_finalizes_paused_descendants_deepest_first(self, store) -> None:
        @tool(target="client")
        async def client_read() -> str:
            return ""

        child_llm = ScriptedLLM([tool_call("client_read", pid="c1")])
        child = Agent(
            name="Child", provider=child_llm, prompt="C.", tools=[client_read], state_store=store
        )

        @tool()
        async def delegate() -> str:
            res = await child.run("sub")
            return res.status.value

        parent_llm = ScriptedLLM(
            [tool_call("delegate", pid="p1"), tool_call("client_read", pid="p2")]
        )
        parent = Agent(
            name="Parent",
            provider=parent_llm,
            prompt="P.",
            tools=[delegate, client_read],
            state_store=store,
        )
        waiting = await parent.run("go")
        assert waiting.status == RunStatus.WAITING_CLIENT_TOOL
        child_id = await _only_run_id(store, parent=waiting.run_id)
        assert (await store.get_run(child_id)).status == "waiting_client_tool"

        result = await parent.cancel_run(waiting.run_id, cascade=True)
        assert result.status == RunStatus.CANCELLED
        assert result.meta["cascade"] == [
            {"run_id": child_id, "outcome": "cancelled"},
            {"run_id": waiting.run_id, "outcome": "cancelled"},
        ]
        for rid in (child_id, waiting.run_id):
            assert (await store.get_run(rid)).status == "cancelled"
            assert len(await _events(store, rid, "run.cancelled")) == 1

    async def test_cancel_without_cascade_leaves_children(self, store) -> None:
        await store.create_run("p", "a")
        await store.create_run("c", "a", parent_run_id="p", delegation_level=1)
        agent = Agent(provider=ScriptedLLM([]), prompt="T.", state_store=store)
        result = await agent.cancel_run("p")
        assert "cascade" not in result.meta
        assert (await store.get_run("c")).cancel_requested is False

    @_INTERRUPTIBLE
    async def test_cascade_cancel_interrupts_nested_running_child_in_process(self, store) -> None:
        d = _delegation(store)
        task = asyncio.create_task(d["parent"].run("go"))
        await asyncio.wait_for(d["child_llm"].entered.wait(), 5)
        parent_id = await _only_run_id(store)
        child_id = await _only_run_id(store, parent=parent_id)

        await d["parent"].cancel_run(parent_id, cascade=True)
        result = await asyncio.wait_for(task, 5)
        assert result.status == RunStatus.CANCELLED
        assert result.meta["interrupted"] is True
        assert d["child_llm"].cancelled is True
        for rid in (child_id, parent_id):
            assert (await store.get_run(rid)).status == "cancelled"
        # The child runs on its own Agent instance, so the parent's interrupt
        # reaches it as a cancellation of the shared task.
        child_evt = (await _events(store, child_id, "run.cancelled"))[0]
        assert child_evt.data == {"reason": "task_cancelled"}
        assert d["counts"]["child_step"] == 0

    @_INTERRUPTIBLE
    async def test_cascade_cancel_same_agent_interrupts_both_once(self, store) -> None:
        """Parent and child on one Agent share a task: one task.cancel(), both
        runs recorded as interrupted, only the parent's run() returns."""
        holder: dict[str, Agent] = {}

        @tool()
        async def delegate() -> str:
            return (await holder["agent"].run("sub")).status.value

        llm = ScriptedLLM([tool_call("delegate", pid="p1"), LLMResponse(text="child")], gate_at=1)
        agent = Agent(provider=llm, prompt="T.", tools=[delegate], state_store=store)
        holder["agent"] = agent
        task = asyncio.create_task(agent.run("go"))
        await asyncio.wait_for(llm.entered.wait(), 5)
        parent_id = await _only_run_id(store)
        child_id = await _only_run_id(store, parent=parent_id)

        await agent.cancel_run(parent_id, cascade=True)
        result = await asyncio.wait_for(task, 5)
        assert result.status == RunStatus.CANCELLED
        assert result.meta["interrupted"] is True
        for rid in (child_id, parent_id):
            evt = (await _events(store, rid, "run.cancelled"))[0]
            assert evt.data["interrupted"] is True
        assert not agent._task_manager._owned and not agent._task_manager._pending
        current = asyncio.current_task()
        assert current is not None and current.cancelling() == 0

    async def test_pause_cascade_respects_tenant(self, store) -> None:
        await store.create_run("p", "a", tenant_id="t1")
        await store.create_run("c1", "a", parent_run_id="p", delegation_level=1, tenant_id="t1")
        await store.create_run("cx", "a", parent_run_id="p", delegation_level=1, tenant_id="t2")
        agent = Agent(provider=ScriptedLLM([]), prompt="T.", state_store=store)

        with pytest.raises(RunNotFoundError):
            await agent.request_pause("p", tenant_id="t2")
        with pytest.raises(RunNotFoundError):
            await agent.request_pause("p", tenant_id=None)
        outcome = await agent.request_pause("p", tenant_id="t1")
        assert dict(outcome.affected) == {"c1": "requested", "p": "requested"}
        assert (await store.get_run("cx")).pause_requested is False

        with pytest.raises(RunNotFoundError):
            await agent.cancel_run("p", tenant_id="t2", cascade=True)
        result = await agent.cancel_run("p", tenant_id="t1", cascade=True)
        assert {c["run_id"] for c in result.meta["cascade"]} == {"c1", "p"}
        assert (await store.get_run("cx")).cancel_requested is False


# ---------------------------------------------------------------------------
# Tenant inheritance
# ---------------------------------------------------------------------------


class TestTenantInheritance:
    async def _parent(self, store: Any, child_tenant: str | None) -> tuple[Agent, ScriptedLLM]:
        child = Agent(
            name="Child",
            provider=ScriptedLLM([LLMResponse(text="c")]),
            prompt="C.",
            state_store=store,
        )

        @tool()
        async def delegate() -> str:
            if child_tenant is None:
                return (await child.run("sub")).status.value
            return (await child.run("sub", tenant_id=child_tenant)).status.value

        llm = ScriptedLLM([tool_call("delegate", pid="p1"), LLMResponse(text="done")])
        return Agent(
            name="Parent", provider=llm, prompt="P.", tools=[delegate], state_store=store
        ), llm

    async def test_child_inherits_parent_tenant(self, store) -> None:
        parent, _ = await self._parent(store, None)
        result = await parent.run("go", tenant_id="acme")
        child_id = await _only_run_id(store, parent=result.run_id)
        assert (await store.get_run(child_id)).tenant_id == "acme"

    async def test_explicit_different_tenant_rejected_before_creation(self, store) -> None:
        parent, llm = await self._parent(store, "other")
        result = await parent.run("go", tenant_id="acme")
        assert result.status == RunStatus.SUCCESS
        assert await store.list_runs(parent_run_id=result.run_id) == []
        tool_msg = llm.contents(1)[-1]
        assert "tenant" in tool_msg

    async def test_same_explicit_tenant_allowed(self, store) -> None:
        parent, _ = await self._parent(store, "acme")
        result = await parent.run("go", tenant_id="acme")
        child_id = await _only_run_id(store, parent=result.run_id)
        assert (await store.get_run(child_id)).tenant_id == "acme"


# ---------------------------------------------------------------------------
# Heartbeat
# ---------------------------------------------------------------------------


async def _backdate(store: Any, run_id: str) -> dt.datetime:
    old = dt.datetime(2000, 1, 1, tzinfo=dt.UTC)
    async with store._session_factory() as session:
        await session.execute(
            update(AgentRun).where(AgentRun.id == run_id).values(last_progress_at=old)
        )
        await session.commit()
    return old


async def _wait_progress_after(store: Any, run_id: str, old: dt.datetime) -> None:
    for _ in range(200):
        record = await store.get_run(run_id)
        if record.last_progress_at and record.last_progress_at > old:
            return
        await asyncio.sleep(0.01)
    raise AssertionError("last_progress_at was not advanced")


class TestHeartbeat:
    async def test_sync_tool_heartbeat_from_worker_thread(self, store) -> None:
        entered, release, beat = threading.Event(), threading.Event(), threading.Event()

        @tool()
        def slow_sync() -> str:
            entered.set()
            release.wait(5)
            dendrux.heartbeat()
            beat.set()
            release.wait(5)
            return "ok"

        llm = ScriptedLLM([tool_call("slow_sync", pid="t1"), LLMResponse(text="done")])
        agent = Agent(provider=llm, prompt="T.", tools=[slow_sync], state_store=store)
        task = asyncio.create_task(agent.run("go"))
        await asyncio.to_thread(entered.wait, 5)
        run_id = await _only_run_id(store)
        old = await _backdate(store, run_id)
        release.set()
        await asyncio.to_thread(beat.wait, 5)
        await _wait_progress_after(store, run_id, old)
        result = await asyncio.wait_for(task, 5)
        assert result.status == RunStatus.SUCCESS

    async def test_async_tool_aheartbeat(self, store) -> None:
        seen: dict[str, Any] = {}

        @tool()
        async def waiter() -> str:
            ctx = get_delegation_context()
            assert ctx is not None
            seen["old"] = await _backdate(store, ctx.run_id)
            await dendrux.aheartbeat()
            seen["after"] = (await store.get_run(ctx.run_id)).last_progress_at
            return "ok"

        llm = ScriptedLLM([tool_call("waiter", pid="t1"), LLMResponse(text="done")])
        agent = Agent(provider=llm, prompt="T.", tools=[waiter], state_store=store)
        await agent.run("go")
        assert seen["after"] > seen["old"]

    async def test_noop_outside_a_run(self) -> None:
        dendrux.heartbeat()
        await dendrux.aheartbeat()

    async def test_tool_start_touches_progress(self, store) -> None:
        entered = asyncio.Event()
        release = asyncio.Event()

        @tool()
        async def slow() -> str:
            entered.set()
            await release.wait()
            return "ok"

        class BackdateBeforeTool(BaseNotifier):
            async def on_message_appended(self, run_id: str, message: Any, iteration: int) -> None:
                if message.role.value == "assistant":
                    await _backdate(store, run_id)

        llm = ScriptedLLM([tool_call("slow", pid="t1"), LLMResponse(text="done")])
        agent = Agent(provider=llm, prompt="T.", tools=[slow], state_store=store)
        task = asyncio.create_task(agent.run("go", notifier=BackdateBeforeTool()))
        await asyncio.wait_for(entered.wait(), 5)
        run_id = await _only_run_id(store)
        record = await store.get_run(run_id)
        assert record.last_progress_at > dt.datetime(2000, 1, 2, tzinfo=dt.UTC)
        release.set()
        await asyncio.wait_for(task, 5)


async def test_submit_resume_requires_persistence() -> None:
    from dendrux.errors import PersistenceNotConfiguredError

    agent = Agent(provider=ScriptedLLM([]), prompt="T.")
    with pytest.raises(PersistenceNotConfiguredError):
        await agent.submit_resume("r")
    with pytest.raises(PersistenceNotConfiguredError):
        await agent.request_pause("r")
    with pytest.raises(PersistenceNotConfiguredError):
        await agent.submit_note("r", "x")
