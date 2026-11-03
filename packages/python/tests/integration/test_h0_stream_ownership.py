"""Both public streams advance and close providers in the consumer's context."""

import asyncio
from dataclasses import replace

import pytest

from dendrux import tool
from dendrux.llm.mock import MockLLM
from dendrux.loops import ReActLoop, SingleCall
from dendrux.runtime.context import (
    DelegationContext,
    get_delegation_context,
    reset_delegation_context,
    set_delegation_context,
)
from dendrux.skills._session import SkillSession, current, reset_session, set_session
from dendrux.types import (
    LLMResponse,
    RunEventType,
    RunStatus,
    StreamEvent,
    StreamEventType,
    ToolCall,
    generate_ulid,
)
from tests._fixtures.storage import pin_storage_format
from tests._helpers.runtime_paths import client_read, client_result, make_agent
from tests._helpers.scripted_provider import tool_call

pytestmark = pytest.mark.parametrize("engine", ["sqlite"], indirect=True)


@tool()
async def server_echo() -> str:
    """Server tool the ``tool`` ending calls on stream."""
    return "echo"


class ContextProbe(MockLLM):
    def __init__(self, ending):
        super().__init__([])
        self.ending = ending
        self.probes = []
        self.calls = 0

    def record(self, boundary):
        self.probes.append((boundary, asyncio.current_task(), get_delegation_context(), current()))

    async def _tool_turn(self):
        """First call of the ``tool`` ending: a delta, then a server tool call on stream."""
        call = ToolCall(name="server_echo", params={}, provider_tool_call_id="s1")
        try:
            self.record("pull")
            await asyncio.sleep(0)
            self.record("after_await")
            yield StreamEvent(type=StreamEventType.TEXT_DELTA, text="t")
            self.record("pull")
            yield StreamEvent(
                type=StreamEventType.TOOL_USE_END,
                tool_call=call,
                tool_name=call.name,
                tool_call_id="s1",
            )
            self.record("done")
            yield StreamEvent(type=StreamEventType.DONE, raw=LLMResponse(tool_calls=[call]))
        finally:
            self.record("close")

    async def complete_stream(self, *args, **kwargs):
        self.calls += 1
        if self.ending == "tool" and self.calls == 1:
            async for event in self._tool_turn():
                yield event
            return
        try:
            for _ in range(2):
                self.record("pull")
                await asyncio.sleep(0)
                self.record("after_await")
                yield StreamEvent(type=StreamEventType.TEXT_DELTA, text="x")
                if self.ending == "error":
                    raise RuntimeError("probe_failure")
                if self.ending == "cancel":
                    asyncio.current_task().cancel()
                    await asyncio.sleep(0)
            self.record("done")
            yield StreamEvent(type=StreamEventType.DONE, raw=LLMResponse(text="xx"))
        finally:
            self.record("close")


@pytest.mark.parametrize(
    "entry,loop_type", [("stream", ReActLoop), ("stream", SingleCall), ("resume_stream", ReActLoop)]
)
@pytest.mark.parametrize("ending", ["exhaust", "close", "error", "cancel"])
@pytest.mark.parametrize("storage_format", [1, 2])
@pytest.mark.parametrize("caller", ["top_level", "delegated"])
async def test_provider_pull_yield_and_close_preserve_consumer_context(
    store, engine, entry, loop_type, ending, storage_format, caller
):
    await check_consumer_context(store, engine, entry, loop_type, ending, storage_format, caller)


@pytest.mark.parametrize("entry", ["stream", "resume_stream"])
@pytest.mark.parametrize("storage_format", [1, 2])
@pytest.mark.parametrize("caller", ["top_level", "delegated"])
async def test_tool_call_on_stream_preserves_consumer_context(
    store, engine, entry, storage_format, caller
):
    """The stream ends in a server tool call, then a second provider call answers."""
    await check_consumer_context(store, engine, entry, ReActLoop, "tool", storage_format, caller)


async def check_consumer_context(store, engine, entry, loop_type, ending, storage_format, caller):
    await pin_storage_format(engine, storage_format)
    parent_agent = make_agent(store, [LLMResponse(text="parent")])
    parent = (
        await store.create_run(generate_ulid(), parent_agent.name, tenant_id="tenant")
        if caller == "delegated"
        else None
    )
    pause_agent = make_agent(store, [tool_call("client_read", pid="c")], tools=[client_read])
    paused = await pause_agent.run("go", tenant_id="tenant") if entry == "resume_stream" else None
    provider = ContextProbe(ending)
    agent = make_agent(
        store,
        provider,
        tools=[] if loop_type is SingleCall else [client_read, server_echo],
        loop=loop_type(),
    )
    stream = (
        agent.stream("go", tenant_id="tenant")
        if entry == "stream"
        else agent.resume_stream(
            paused.run_id,
            tool_results=[client_result(paused)],
            tenant_id="tenant",
            expected_pause_id=paused.pause_id,
        )
    )
    consumer = asyncio.current_task()
    caller_session = SkillSession.prepare(parent_agent) if parent else None
    caller_context = (
        DelegationContext(
            run_id=parent.run_id,
            tenant_id="tenant",
            persisted=True,
            store_identity=store.store_identity,
            state_store=store,
        )
        if parent
        else None
    )
    delegation_token = set_delegation_context(caller_context)
    session_token = set_session(caller_session)
    events = []

    async def consume():
        nonlocal caller_context, caller_session
        async with stream:
            async for event in stream:
                events.append(event)
                assert get_delegation_context() is caller_context
                assert current() is caller_session
                # Restore the current caller's context, not the first pull's snapshot.
                caller_context = replace(caller_context) if caller_context else None
                caller_session = SkillSession.prepare(parent_agent)
                set_delegation_context(caller_context)
                set_session(caller_session)
                if ending == "close" and event.type == RunEventType.TEXT_DELTA:
                    break

    try:
        if ending == "cancel":
            with pytest.raises(asyncio.CancelledError):
                await consume()
        else:
            await consume()
        assert get_delegation_context() is caller_context
        assert current() is caller_session
        assert provider.probes and provider.probes[-1][0] == "close"
        assert sum(p[0] == "close" for p in provider.probes) == provider.calls
        assert provider.calls == (2 if ending == "tool" else 1)
        for boundary, task, delegation, session in provider.probes:
            assert task is consumer, boundary
            assert delegation.run_id == stream.run_id, boundary
            assert delegation.tenant_id == "tenant", boundary
            assert session.agent is agent and session.run_id == stream.run_id, boundary
        assert any(p[0] == "after_await" for p in provider.probes)
        if ending == "tool":
            assert [e.tool_call.name for e in events if e.type == RunEventType.TOOL_RESULT] == [
                "server_echo"
            ]
        expected = {
            "tool": RunStatus.SUCCESS,
            "exhaust": RunStatus.SUCCESS,
            "close": RunStatus.CANCELLED,
            "error": RunStatus.ERROR,
            "cancel": RunStatus.CANCELLED,
        }[ending]
        row = await store.get_run(stream.run_id)
        assert row.status == expected.value
        if entry == "stream":
            assert row.parent_run_id == (parent.run_id if parent else None)
            assert row.delegation_level == (1 if parent else 0)
        if ending in {"exhaust", "error", "tool"}:
            assert stream.result.status == expected
            assert events[-1].type == (
                RunEventType.RUN_ERROR if ending == "error" else RunEventType.RUN_COMPLETED
            )
    finally:
        await stream.aclose()
        reset_session(session_token)
        reset_delegation_context(delegation_token)
        await agent.close()
        await pause_agent.close()
        await parent_agent.close()
