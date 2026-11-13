"""A lost database-wait cancel stops before the next external effect."""

import asyncio
from contextlib import suppress

import pytest

from dendrux import Agent, SingleCall, tool
from dendrux.loops.react import ReActLoop
from dendrux.types import (
    LLMResponse,
    ProviderCapabilities,
    StreamEvent,
    StreamEventType,
    ToolResult,
)
from tests._fixtures.storage import pin_storage_format
from tests._helpers.gates import wait_for_gate
from tests._helpers.lost_cancel import LostCancel
from tests._helpers.scripted_provider import ScriptedLLM, tool_call


@pytest.fixture(autouse=True)
def enable_guard(monkeypatch):
    from dendrux.runtime import _cancel_guard

    monkeypatch.setattr(_cancel_guard, "_ENABLED", True)


class StreamingProvider(ScriptedLLM):
    capabilities = ProviderCapabilities(
        supports_native_tools=True, supports_tool_call_ids=True, supports_streaming=True
    )

    async def complete_stream(self, messages, tools=None, **kwargs):
        response = await self.complete(messages, tools=tools, **kwargs)
        yield StreamEvent(type=StreamEventType.DONE, raw=response)


async def invoke(agent, stream=False):
    if not stream:
        return await agent.run("go")
    async with agent.stream("go") as events:
        async for _ in events:
            pass
    return events.result


async def assert_cancelled_run(store, run_id):
    assert (await store.get_run(run_id)).status == "cancelled"
    events = await store.get_run_events(run_id)
    assert sum(event.event_type == "run.cancelled" for event in events) == 1
    assert not any(event.event_type == "run.completed" for event in events)


@pytest.mark.parametrize("loop", [ReActLoop, SingleCall])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("source", ["caller", "timeout"])
async def test_lost_cancel_before_provider(
    store, engine, monkeypatch, caplog, loop, stream, source
):
    await pin_storage_format(engine, 2)
    provider = StreamingProvider([LLMResponse(text="done")])
    gate = LostCancel()
    gate.install(monkeypatch, store, "on_llm_call_started", "get_run_control")
    deadline = asyncio.timeout(None)
    async with Agent(
        prompt="test", provider=provider, state_store=store, loop=loop(), strict_tenancy=False
    ) as agent:

        async def run():
            async with deadline:
                return await invoke(agent, stream)

        task = asyncio.create_task(run())
        try:
            await wait_for_gate(gate.entered, task=task)
            if source == "caller":
                task.cancel()
            else:
                deadline.reschedule(asyncio.get_running_loop().time())
            with pytest.raises(asyncio.CancelledError if source == "caller" else TimeoutError):
                await task
            assert gate.dropped
            assert provider.calls == 0
            await assert_cancelled_run(store, gate.run_id)
            logs = [r for r in caplog.records if getattr(r, "code", None) == "run_cancel_recovered"]
            assert len(logs) == 1
            assert logs[0].run_id == gate.run_id
            assert logs[0].getMessage() == "run_cancel_recovered"
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize("storage_format", [1, 2])
async def test_lost_cancel_before_tool(store, engine, monkeypatch, storage_format):
    await pin_storage_format(engine, storage_format)
    effects = []

    @tool(parallel=False)
    async def effect() -> str:
        """Record an external effect."""
        effects.append("called")
        return "ok"

    provider = ScriptedLLM([tool_call("effect", pid="one"), LLMResponse(text="done")])
    gate = LostCancel()
    gate.install(
        monkeypatch,
        store,
        "on_tool_started",
        "start_tool_call" if storage_format == 2 else "touch_progress",
    )
    async with Agent(
        prompt="test", provider=provider, tools=[effect], state_store=store, strict_tenancy=False
    ) as agent:
        task = asyncio.create_task(agent.run("go"))
        try:
            await wait_for_gate(gate.entered, task=task)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert gate.dropped
            assert not effects
            assert provider.calls == 1
            await assert_cancelled_run(store, gate.run_id)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize("swallow", [False, True])
@pytest.mark.parametrize("source", ["caller", "manager"])
async def test_existing_cancellation_semantics(store, engine, monkeypatch, swallow, source):
    await pin_storage_format(engine, 2)
    provider = ScriptedLLM([LLMResponse(text="done")])
    gate = LostCancel(swallow=swallow)
    gate.install(monkeypatch, store, "on_llm_call_started", "get_run_control")
    async with Agent(
        prompt="test", provider=provider, state_store=store, strict_tenancy=False
    ) as agent:
        task = asyncio.create_task(agent.run("go"))
        try:
            await wait_for_gate(gate.entered, task=task)
            if source == "manager":
                await agent.cancel_run(gate.run_id)
                assert (await task).status.value == "cancelled"
                assert task.cancelling() == 0
            else:
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            assert provider.calls == 0
            await assert_cancelled_run(store, gate.run_id)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


async def test_nested_child_interrupt_leaves_parent_running(store):
    holder = {}

    @tool(parallel=False)
    async def delegate() -> str:
        """Run a child on the parent's task."""
        return (await holder["agent"].run("child")).status.value

    provider = ScriptedLLM(
        [tool_call("delegate", pid="one"), LLMResponse(text="child"), LLMResponse(text="parent")],
        gate_at=1,
    )
    async with Agent(
        prompt="test", provider=provider, tools=[delegate], state_store=store, strict_tenancy=False
    ) as agent:
        holder["agent"] = agent
        task = asyncio.create_task(agent.run("parent"))
        try:
            await wait_for_gate(provider.entered, task=task)
            runs = await store.list_runs()
            child = next(run for run in runs if run.parent_run_id is not None)
            await agent.cancel_run(child.id)
            assert (await task).status.value == "success"
            await assert_cancelled_run(store, child.id)
            assert task.cancelling() == 0
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


async def test_unfired_timeout_with_store(store):
    provider = ScriptedLLM([LLMResponse(text="done")])
    async with Agent(
        prompt="test", provider=provider, state_store=store, strict_tenancy=False
    ) as agent:
        async with asyncio.timeout(30):
            result = await agent.run("go")
        assert result.status.value == "success"
        assert provider.calls == 1


async def test_gate_off_does_not_recover_lost_cancel(store, engine, monkeypatch, caplog):
    from dendrux.runtime import _cancel_guard

    monkeypatch.setattr(_cancel_guard, "_ENABLED", False)
    await pin_storage_format(engine, 2)
    gate = LostCancel()
    gate.install(monkeypatch, store, "on_llm_call_started", "get_run_control")
    provider = ScriptedLLM([LLMResponse(text="done")])
    async with Agent(
        prompt="test", provider=provider, state_store=store, strict_tenancy=False
    ) as agent:
        task = asyncio.create_task(agent.run("go"))
        try:
            await wait_for_gate(gate.entered, task=task)
            task.cancel()
            assert (await task).status.value == "success"
            assert gate.dropped
            assert provider.calls == 1
            assert task.cancelling() == 1
            assert not any(
                getattr(r, "code", None) == "run_cancel_recovered" for r in caplog.records
            )
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize("kind", ["taskgroup", "self_cancel"])
async def test_handled_tool_cancellation_does_not_cancel_run(store, caplog, kind):
    counts = []

    @tool(parallel=False)
    async def handled() -> str:
        """Handle an internal failure and return a usable observation."""
        if kind == "taskgroup":

            async def fail():
                await asyncio.sleep(0)
                raise ValueError("child failed")

            try:
                async with asyncio.TaskGroup() as group:
                    group.create_task(fail())
            except ExceptionGroup:
                pass
        else:
            asyncio.current_task().cancel()
            with suppress(asyncio.CancelledError):
                await asyncio.sleep(0)
        counts.append(asyncio.current_task().cancelling())
        return "one lookup failed"

    provider = ScriptedLLM([tool_call("handled", pid="one"), LLMResponse(text="done")])
    async with Agent(
        prompt="test", provider=provider, tools=[handled], state_store=store, strict_tenancy=False
    ) as agent:
        task = asyncio.create_task(agent.run("go"))
        result = await task
        assert result.status.value == "success"
        assert provider.calls == 2
        assert counts
        if kind == "self_cancel":
            assert counts == [1]
        assert (await store.get_run(result.run_id)).status == "success"
        assert not any(
            getattr(record, "code", None) == "run_cancel_recovered" for record in caplog.records
        )


@pytest.mark.parametrize("entry", ["resume", "resume_stream", "submit_tool_results"])
async def test_continuation_recorder_observes_its_run_task(store, engine, monkeypatch, entry):
    await pin_storage_format(engine, 2)

    @tool(target="client")
    async def client() -> str:
        """Read data from the client."""
        return ""

    provider = StreamingProvider([tool_call("client", pid="one"), LLMResponse(text="done")])
    async with Agent(
        prompt="test", provider=provider, tools=[client], state_store=store, strict_tenancy=False
    ) as agent:
        paused = await agent.run("go")
        results = [ToolResult(name="client", call_id=paused.pending.tool_calls[0].id, payload="ok")]
        gate = LostCancel()
        gate.install(monkeypatch, store, "on_llm_call_started", "get_run_control")

        async def resume():
            if entry == "resume_stream":
                async with agent.resume_stream(paused.run_id, tool_results=results) as events:
                    async for _ in events:
                        pass
            elif entry == "submit_tool_results":
                return await agent.submit_tool_results(paused.run_id, results)
            else:
                return await agent.resume(paused.run_id, tool_results=results)

        task = asyncio.create_task(resume())
        try:
            await wait_for_gate(gate.entered, task=task)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert gate.dropped
            assert provider.calls == 1
            await assert_cancelled_run(store, paused.run_id)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


async def test_parent_store_loss_stops_before_same_task_child(store, engine, monkeypatch):
    await pin_storage_format(engine, 2)
    children = []
    holder = {}

    @tool(parallel=False)
    async def delegate() -> str:
        """Start a child on the parent's task."""
        children.append("started")
        return (await holder["agent"].run("child")).answer

    provider = ScriptedLLM(
        [tool_call("delegate", pid="one"), LLMResponse(text="child"), LLMResponse(text="parent")]
    )
    gate = LostCancel()
    gate.install(monkeypatch, store, "on_llm_call_completed", "save_usage")
    async with Agent(
        prompt="test", provider=provider, tools=[delegate], state_store=store, strict_tenancy=False
    ) as agent:
        holder["agent"] = agent
        task = asyncio.create_task(agent.run("go"))
        try:
            await wait_for_gate(gate.entered, task=task)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert not children
            assert provider.calls == 1
            assert len(await store.list_runs()) == 1
            await assert_cancelled_run(store, gate.run_id)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
