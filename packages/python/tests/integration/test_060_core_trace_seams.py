"""Trace links cross real Agent entry, resume and nested tool boundaries."""

import asyncio

import pytest

from dendrux import Agent, tool
from dendrux.llm.mock import MockLLM
from dendrux.loops.base import BaseNotifier
from dendrux.runtime.context import get_delegation_context
from dendrux.types import LLMResponse, RunStatus, ToolCall

LINK = "00-" + "1" * 32 + "-" + "2" * 16 + "-01"
SPAN = "00-" + "1" * 32 + "-" + "3" * 16 + "-01"
OTHER = "00-" + "4" * 32 + "-" + "5" * 16 + "-01"


class Observer(BaseNotifier):
    def __init__(self, link=LINK):
        self.link = link
        self.starts = []
        self.tools = []

    async def on_run_started(self, run_id, *, agent_name=None, agent_model=None, trace_link=None):
        self.starts.append((run_id, trace_link))

    def trace_link(self, run_id):
        return self.link

    def current_span_link(self, run_id, tool_call_id):
        self.tools.append(tool_call_id)
        return SPAN


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("persisted", [False, True])
async def test_parallel_tool_slots_and_custom_child_run_links(store, streaming, persisted):
    child_observer = Observer(OTHER)
    calls = [ToolCall("child"), ToolCall("child")]
    links = {calls[0].id: SPAN, calls[1].id: OTHER}

    class Spans(Observer):
        def current_span_link(self, run_id, tool_call_id):
            self.tools.append(tool_call_id)
            return links[tool_call_id]

    observer = Spans()
    contexts = []

    @tool(parallel=True)
    async def child() -> str:
        """Run a child within the current tool span."""
        ctx = get_delegation_context()
        contexts.append((ctx.trace_link, ctx.tool_span_link))
        await asyncio.sleep(0)
        a = Agent(
            prompt="child",
            provider=MockLLM([LLMResponse(text="child")]),
            state_store=store if persisted else None,
            strict_tenancy=False,
        )
        await a.run("go", notifier=child_observer)
        assert get_delegation_context() is ctx
        return "ok"

    class Model(MockLLM):
        async def complete(self, *args, **kwargs):
            ctx = get_delegation_context()
            assert ctx.trace_link == LINK and ctx.tool_span_link is None
            return await super().complete(*args, **kwargs)

    a = Agent(
        prompt="p",
        provider=Model(
            [
                LLMResponse(tool_calls=calls),
                LLMResponse(text="done"),
            ]
        ),
        tools=[child],
        state_store=store if persisted else None,
        strict_tenancy=False,
    )
    if streaming:
        async with a.stream("go", notifier=observer) as stream:
            events = [event async for event in stream]
        result = events[-1].run_result
    else:
        result = await a.run("go", notifier=observer)
    assert result.status == RunStatus.SUCCESS
    assert sorted(contexts) == sorted([(LINK, SPAN), (LINK, OTHER)])
    assert len(observer.tools) == 2
    assert [link for _, link in child_observer.starts] == [LINK, LINK]
    if persisted:
        assert (await store.get_run(result.run_id)).trace_link == LINK
    assert get_delegation_context() is None


@pytest.mark.parametrize("streaming", [False, True])
async def test_resume_reuses_persisted_link_for_direct_approval_tool(store, streaming):
    contexts = []

    @tool()
    async def work() -> str:
        """Observe a resumed tool's links."""
        ctx = get_delegation_context()
        contexts.append((ctx.trace_link, ctx.tool_span_link))
        return "ok"

    a = Agent(
        prompt="p",
        provider=MockLLM([LLMResponse(tool_calls=[ToolCall("work")]), LLMResponse(text="done")]),
        tools=[work],
        require_approval=["work"],
        state_store=store,
        strict_tenancy=False,
    )
    paused = await a.run("go", notifier=Observer())
    observer = Observer(OTHER)
    if streaming:
        async with a.resume_stream(
            paused.run_id, expected_pause_id=paused.pause_id, notifier=observer
        ) as stream:
            events = [event async for event in stream]
        result = events[-1].run_result
    else:
        result = await a.submit_approval(
            paused.run_id, approved=True, expected_pause_id=paused.pause_id, notifier=observer
        )
    assert result.status == RunStatus.SUCCESS
    assert observer.starts == [(paused.run_id, LINK)]
    assert contexts == [(LINK, SPAN)]
    assert (await store.get_run(paused.run_id)).trace_link == LINK


@pytest.mark.parametrize("link", ["bad", "00-" + "0" * 32 + "-" + "2" * 16 + "-01", None])
async def test_invalid_notifier_links_are_discarded(store, link):
    result = await Agent(
        prompt="p",
        provider=MockLLM([LLMResponse(text="done")]),
        state_store=store,
        strict_tenancy=False,
    ).run("go", notifier=Observer(link))
    assert result.status == RunStatus.SUCCESS
    assert (await store.get_run(result.run_id)).trace_link is None


@pytest.mark.parametrize("engine", ["sqlite_file", "postgres"], indirect=True)
async def test_fresh_process_resume_receives_saved_trace_parent(store, engine):
    import json
    import os
    import sys

    @tool()
    async def work() -> str:
        return "ok"

    a = Agent(
        prompt="p",
        provider=MockLLM([LLMResponse(tool_calls=[ToolCall("work")])]),
        tools=[work],
        require_approval=["work"],
        state_store=store,
        strict_tenancy=False,
    )
    paused = await a.run("go", notifier=Observer())
    code = """
import asyncio, json, sys
from sqlalchemy.ext.asyncio import create_async_engine
from dendrux import Agent, tool
from dendrux.llm.mock import MockLLM
from dendrux.loops.base import BaseNotifier
from dendrux.runtime.state import SQLAlchemyStateStore
from dendrux.runtime.context import get_delegation_context
from dendrux.types import LLMResponse
seen = {}
class Observer(BaseNotifier):
    async def on_run_started(self, run_id, *, agent_name=None, agent_model=None, trace_link=None):
        seen['parent'] = trace_link
    def trace_link(self, run_id):
        return '00-' + '4' * 32 + '-' + '5' * 16 + '-01'
@tool()
async def work() -> str:
    seen['tool_run'] = get_delegation_context().trace_link
    return 'ok'
async def main():
    engine = create_async_engine(sys.argv[1])
    store = SQLAlchemyStateStore(engine)
    a = Agent(prompt='p', provider=MockLLM([LLMResponse(text='done')]), tools=[work],
              require_approval=['work'], state_store=store, strict_tenancy=False)
    result = await a.submit_approval(sys.argv[2], approved=True,
                                    expected_pause_id=sys.argv[3], notifier=Observer())
    seen['status'] = result.status.value
    seen['stored'] = (await store.get_run(sys.argv[2])).trace_link
    print(json.dumps(seen))
    await engine.dispose()
asyncio.run(main())
"""
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        code,
        engine.url.render_as_string(hide_password=False),
        paused.run_id,
        paused.pause_id,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env={**os.environ, "PYTHONPATH": os.path.abspath("src")},
    )
    output, error = await asyncio.wait_for(process.communicate(), 15)
    assert process.returncode == 0, error.decode()
    assert json.loads(output) == {
        "parent": LINK,
        "tool_run": LINK,
        "stored": LINK,
        "status": "success",
    }


@pytest.mark.parametrize("single", [False, True])
async def test_retry_starts_its_own_trace(store, single):
    from dendrux.loops import ReActLoop, SingleCall

    a = Agent(
        prompt="p",
        provider=MockLLM([LLMResponse(text="done"), LLMResponse(text="again")]),
        loop=SingleCall() if single else ReActLoop(),
        state_store=store,
        strict_tenancy=False,
    )
    result = await a.run("go", notifier=Observer(), loop=ReActLoop())
    observer = Observer(OTHER)
    retried = await a.retry(result.run_id, notifier=observer)
    assert retried.status == RunStatus.SUCCESS
    assert observer.starts == [(retried.run_id, None)]
    assert (await store.get_run(retried.run_id)).trace_link == OTHER


@pytest.mark.parametrize("fmt", [1, 2])
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("fail_start", [False, True])
async def test_tool_notifier_failures_reach_result_warnings(
    store, engine, fmt, streaming, fail_start
):
    from tests._fixtures.storage import pin_storage_format

    await pin_storage_format(engine, fmt)
    executed = []

    class FailingObserver(Observer):
        async def on_tool_started(self, run_id, call, iteration):
            if fail_start:
                raise RuntimeError("private notifier details")

        async def on_tool_completed(self, run_id, call, result, iteration):
            raise RuntimeError("private notifier details")

    @tool()
    async def work() -> str:
        """Run despite an observer failure."""
        executed.append(get_delegation_context().tool_span_link)
        return "ok"

    class Model(MockLLM):
        async def complete(self, *args, **kwargs):
            assert get_delegation_context().tool_span_link is None
            return await super().complete(*args, **kwargs)

    agent = Agent(
        prompt="p",
        provider=Model([LLMResponse(tool_calls=[ToolCall("work")]), LLMResponse(text="done")]),
        tools=[work],
        state_store=store,
        strict_tenancy=False,
    )
    notifier = FailingObserver()
    if streaming:
        stream = agent.stream("go", notifier=notifier)
        async for _ in stream:
            pass
        result = stream.result
    else:
        result = await agent.run("go", notifier=notifier)
    assert result.status == RunStatus.SUCCESS
    assert executed == [None if fail_start else SPAN]
    expected = ["on_tool_started failed at iteration 1"] if fail_start else []
    expected.append("on_tool_completed failed at iteration 1")
    assert result.meta.get("notifier_warnings") == expected
    assert "private notifier details" not in str(result.meta)
    assert get_delegation_context() is None
