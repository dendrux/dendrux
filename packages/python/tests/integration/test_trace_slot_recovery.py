import asyncio
from functools import wraps

import pytest

from dendrux import Agent, tool
from dendrux.llm.mock import MockLLM
from dendrux.runtime.stores._factory import store_for_engine
from dendrux.store import RunStore
from dendrux.types import LLMResponse, RunStatus, ToolCall, ToolResult
from tests._fixtures.storage import pin_storage_format
from tests._helpers.recovery_checkpoint import expire


async def test_parallel_public_paging_waits_for_missing_slot(store, engine, monkeypatch):
    if engine.dialect.name != "postgresql":
        pytest.skip("independent connections require Postgres")
    await pin_storage_format(engine, 2)
    release = asyncio.Event()
    fast_saved = asyncio.Event()
    identity = []
    original = store.complete_tool_call

    @wraps(original)
    async def complete(run_id, **kwargs):
        await original(run_id, **kwargs)
        if kwargs["tool_call_id"] == "fast":
            identity.append(run_id)
            fast_saved.set()

    monkeypatch.setattr(store, "complete_tool_call", complete)

    @tool(parallel=True)
    async def slow() -> str:
        await release.wait()
        return "SLOW"

    @tool(parallel=True)
    async def fast() -> str:
        return "FAST"

    agent = Agent(
        prompt="p",
        provider=MockLLM(
            [
                LLMResponse(tool_calls=[ToolCall("slow", id="slow"), ToolCall("fast", id="fast")]),
                LLMResponse(text="done"),
            ]
        ),
        tools=[slow, fast],
        state_store=store,
    )
    task = asyncio.create_task(agent.run("go", tenant_id="t"))
    try:
        await fast_saved.wait()
        reader = RunStore(store_for_engine(engine))
        page = await reader.get_traces(identity[0], tenant_id="t", limit=50)
        assert [r.role for r in page] == ["user", "assistant"]
        assert await reader.get_traces(identity[0], tenant_id="t", limit=1, offset=2) == []
    finally:
        release.set()
        result = await task
    following = await reader.get_traces(result.run_id, tenant_id="t", offset=len(page), limit=50)
    assert [r.content for r in following] == ['"SLOW"', '"FAST"', "done"]
    assert [r.order_index for r in page + following] == list(range(5))


def _slot_agent(store, effects, provider):
    @tool(idempotent=True)
    async def first() -> str:
        with open(effects, "a") as output:
            output.write("A\n")
        return "A"

    @tool(idempotent=True)
    async def second() -> str:
        with open(effects, "a") as output:
            output.write("B\n")
        return "B"

    @tool(target="client")
    async def client() -> str:
        return "unused"

    return Agent(prompt="p", provider=provider, tools=[first, second, client], state_store=store)


def _slot_worker(url, boundary, effects, pipe):
    from dataclasses import replace

    from sqlalchemy.ext.asyncio import create_async_engine

    from dendrux.runtime.persistence import PersistenceRecorder
    from dendrux.runtime.stores.sql.base import SQLStateStore

    def halt(run):
        pipe.send(run)
        pipe.recv()
        raise AssertionError("kill handshake returned")

    async def main():
        engine = create_async_engine(url)

        class Store(type(store_for_engine(engine))):
            @wraps(SQLStateStore.checkpoint_step)
            async def checkpoint_step(self, run_id, **kwargs):
                await super().checkpoint_step(run_id, **kwargs)
                trace = kwargs.get("trace")
                if trace and (
                    (boundary == "reserved" and trace.role == "assistant")
                    or (boundary in ("rejected", "late_rejected") and trace.role == "tool")
                    or (boundary == "rewrite" and trace.trace_kind == "rewrite")
                ):
                    halt(run_id)

            @wraps(SQLStateStore.complete_tool_call)
            async def complete_tool_call(self, run_id, **kwargs):
                await super().complete_tool_call(run_id, **kwargs)
                if kwargs["tool_call_id"] == {"first_complete": "a", "second_complete": "b"}.get(
                    boundary
                ):
                    halt(run_id)

        if boundary == "rewrite":
            original = PersistenceRecorder.checkpoint_view

            async def rewrite(self, messages, iteration):
                if self._loop_state.get("pending"):
                    messages = [
                        replace(m, content="edited input") if m.role.value == "user" else m
                        for m in messages
                    ]
                await original(self, messages, iteration)

            PersistenceRecorder.checkpoint_view = rewrite
        provider = MockLLM(
            [
                LLMResponse(
                    tool_calls=[
                        ToolCall("client", id="c"),
                        ToolCall("first", id="a"),
                        ToolCall("second", {"bad": 1}, id="n")
                        if boundary == "late_rejected"
                        else ToolCall("nope", id="n"),
                        ToolCall("second", id="b"),
                    ]
                )
            ]
        )
        await _slot_agent(Store(engine), effects, provider).run("go", tenant_id="t")
        raise AssertionError("crash boundary was not reached")

    try:
        asyncio.run(main())
    except BaseException:
        import traceback

        pipe.send({"error": traceback.format_exc()})
        raise


@pytest.mark.parametrize(
    "boundary",
    ["reserved", "rejected", "late_rejected", "first_complete", "second_complete", "rewrite"],
)
async def test_crashed_slot_boundaries_reclaim_once(store, engine, tmp_path, boundary):
    from tests.integration.test_060_reclaim_crashes import kill_process

    if engine.dialect.name != "postgresql":
        pytest.skip("crash and competing claims require Postgres")
    await pin_storage_format(engine, 2)
    effects = tmp_path / "effects"
    run = await kill_process(
        _slot_worker, (engine.url.render_as_string(hide_password=False), boundary, str(effects))
    )
    before = await RunStore(store).get_traces(run, tenant_id="t", limit=50)
    item = await expire(store, engine, run)
    provider = MockLLM([LLMResponse(text="done")])
    peers = [_slot_agent(store_for_engine(engine), effects, provider) for _ in range(2)]
    results = await asyncio.gather(
        *(peer.reclaim(run, tenant_id="t", item=item) for peer in peers), return_exceptions=True
    )
    winners = [r for r in results if not isinstance(r, BaseException)]
    assert len(winners) == 1, results
    paused = winners[0]
    assert paused.status == RunStatus.WAITING_CLIENT_TOOL
    result = await peers[0].submit_tool_results(
        run,
        [ToolResult(name="client", call_id="c", payload='"ok"')],
        tenant_id="t",
        expected_pause_id=paused.pause_id,
    )
    assert result.status == RunStatus.SUCCESS
    assert sorted(effects.read_text().splitlines()) == ["A", "B"]
    after = await RunStore(store).get_traces(run, tenant_id="t", limit=50)
    assert [(r.content, r.order_index) for r in before] == [
        (r.content, r.order_index) for r in after[: len(before)]
    ]
    assert [r.content for r in after if r.role == "tool"][1:] == ['"A"', '"B"', '"ok"']
    for peer in peers:
        await peer.close()


@pytest.mark.parametrize("crash_after_rewrite", [False, True])
async def test_rewrite_rebases_pending_observations(
    store, engine, monkeypatch, crash_after_rewrite
):
    from dataclasses import replace

    from dendrux.runtime.persistence import PersistenceRecorder
    from dendrux.runtime.replay import rebuild_history

    await pin_storage_format(engine, 2)

    @tool(target="client")
    async def client() -> str:
        return "unused"

    provider = MockLLM(
        [LLMResponse(tool_calls=[ToolCall("client", id="c")]), LLMResponse(text="done")]
    )
    agent = Agent(prompt="p", provider=provider, tools=[client], state_store=store)
    view = PersistenceRecorder.checkpoint_view
    changed = []

    class Crash(BaseException):
        pass

    async def rewrite(self, messages, iteration):
        if self._loop_state.get("pending") and not changed:
            changed.append(True)
            messages = [
                replace(m, content="edited input") if m.role.value == "user" else m
                for m in messages
            ]
            await view(self, messages, iteration)
            if crash_after_rewrite:
                raise Crash()
            return
        await view(self, messages, iteration)

    monkeypatch.setattr(PersistenceRecorder, "checkpoint_view", rewrite)
    if crash_after_rewrite:
        with pytest.raises(Crash):
            await agent.run("go", tenant_id="t")
        run = (await store.list_runs(limit=1))[0]
        item = await expire(store, engine, run.id)
        paused = await agent.reclaim(run.id, tenant_id="t", item=item)
    else:
        paused = await agent.run("go", tenant_id="t")
    state = await store.get_loop_state(paused.run_id)
    traces = await store.get_traces(paused.run_id)
    rewrite_position = max(t.order_index for t in traces if t.trace_kind == "rewrite")
    assert min(p["order_index"] for p in state["pending"]) > rewrite_position
    result = await agent.submit_tool_results(
        paused.run_id,
        [ToolResult(name="client", call_id="c", payload='"ok"')],
        tenant_id="t",
        expected_pause_id=paused.pause_id,
    )
    assert result.status == RunStatus.SUCCESS
    rebuilt = await rebuild_history(store, result.run_id)
    assert [m.content for m in rebuilt.messages if m.role.value == "tool"] == ['"ok"']
    assert [m.content for m in provider.wire_messages[-1] if m.role.value == "tool"] == ['"ok"']
    public = await RunStore(store).get_traces(result.run_id, tenant_id="t", limit=50)
    assert public[0].content == "go"
