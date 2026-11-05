import asyncio
from collections import Counter
from functools import wraps

import pytest

from dendrux import Agent, tool
from dendrux.llm.mock import MockLLM
from dendrux.runtime.stores._factory import store_for_engine
from dendrux.store import RunStore
from dendrux.types import LLMResponse, RunStatus, ToolCall, ToolResult
from tests._fixtures.storage import pin_storage_format
from tests._helpers.recovery_checkpoint import expire


@pytest.mark.parametrize("custom", [False, True])
async def test_parallel_public_paging_waits_for_missing_slot(store, engine, monkeypatch, custom):
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
        from tests._helpers.public_trace_store import KeysetStore

        reader_store = store_for_engine(engine)
        reader = RunStore(KeysetStore(reader_store) if custom else reader_store)
        page = await reader.get_traces(identity[0], tenant_id="t", limit=50)
        assert [r.role for r in page] == ["user", "assistant"]
        assert await reader.get_traces(identity[0], tenant_id="t", limit=1, offset=2) == []
    finally:
        release.set()
        result = await task
    following = await reader.get_traces(result.run_id, tenant_id="t", offset=len(page), limit=50)
    assert [r.content for r in following] == ['"SLOW"', '"FAST"', "done"]
    assert [r.order_index for r in page + following] == list(range(5))


def _slot_agent(store, effects, provider, running=None):
    @tool(idempotent=True)
    async def first() -> str:
        with open(effects, "a") as output:
            output.write("A\n")
        return "A"

    @tool(idempotent=True)
    async def second() -> str:
        with open(effects, "a") as output:
            output.write("B\n")
        if running is not None:  # the effect is out, the completion is never reached
            running.set()
            await asyncio.Event().wait()
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
        running = asyncio.Event() if boundary == "first_complete_second_running" else None

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
                if running is not None and kwargs["tool_call_id"] == "a":
                    await running.wait()
                await super().complete_tool_call(run_id, **kwargs)
                if kwargs["tool_call_id"] == {
                    "first_complete": "a",
                    "first_complete_second_running": "a",
                    "second_complete": "b",
                }.get(boundary):
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
        await _slot_agent(Store(engine), effects, provider, running).run("go", tenant_id="t")
        raise AssertionError("crash boundary was not reached")

    try:
        asyncio.run(main())
    except BaseException:
        import traceback

        pipe.send({"error": traceback.format_exc()})
        raise


async def _server_call_states(engine, run):
    from sqlalchemy import select

    from dendrux.db.models import ToolCallRecord

    async with engine.connect() as conn:
        rows = (
            await conn.execute(
                select(
                    ToolCallRecord.tool_call_id, ToolCallRecord.outcome, ToolCallRecord.replayed_at
                ).where(ToolCallRecord.agent_run_id == run)
            )
        ).all()
    assert len({r[0] for r in rows}) == len(rows), "one row per call"
    return {r[0]: (r[1], r[2]) for r in rows if r[0] in ("a", "b")}


# What is durable for the two idempotent calls when the worker dies. The two calls run
# in parallel, so the other call of a completion boundary may be unstarted, started or
# completed by then; that call is left out and checked against the rows read after the kill.
DURABLE_AT_KILL = {
    "reserved": {},
    "rejected": {},
    "late_rejected": {},
    "first_complete": {"a": "completed"},
    "second_complete": {"b": "completed"},
    "first_complete_second_running": {"a": "completed", "b": "started"},
    "rewrite": {"a": "completed", "b": "completed"},
}


@pytest.mark.parametrize("boundary", list(DURABLE_AT_KILL))
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
    at_kill = {
        call: outcome for call, (outcome, _) in (await _server_call_states(engine, run)).items()
    }
    for call, outcome in DURABLE_AT_KILL[boundary].items():
        assert at_kill.get(call) == outcome, (boundary, at_kill)
    if boundary in ("reserved", "rejected", "late_rejected", "rewrite"):
        assert set(at_kill) == set(DURABLE_AT_KILL[boundary]), (boundary, at_kill)
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
    # An idempotent call whose completion was durable is never run again. One that had only
    # its started intent replays once, and its effect is out once or twice (twice when the
    # worker had already run it). One with no intent runs once.
    runs = Counter(effects.read_text().splitlines())
    started = sorted(call for call, outcome in at_kill.items() if outcome == "started")
    for call, name in (("a", "A"), ("b", "B")):
        assert runs[name] in ((1, 2) if call in started else (1,)), (boundary, at_kill, runs)
    assert set(runs) == {"A", "B"}
    if boundary == "first_complete_second_running":
        assert runs == {"A": 1, "B": 2}
    replays = await store.get_run_events(run, event_types=["tool.replayed"], tenant_id="t")
    assert sorted(e.correlation_id for e in replays) == started
    assert result.recovery.replayed_tool_calls == started
    final = await _server_call_states(engine, run)
    assert {call: outcome for call, (outcome, _) in final.items()} == {
        "a": "completed",
        "b": "completed",
    }
    assert sorted(call for call, (_, replayed_at) in final.items() if replayed_at) == started
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


async def test_public_trace_pending_projection_is_evaluated_once(store, engine):
    from sqlalchemy import event, insert, update

    from dendrux.db.models import AgentRun, ReactTrace
    from dendrux.types import generate_ulid

    if engine.dialect.name != "postgresql":
        pytest.skip("execution-plan assertion requires Postgres")
    run_id = generate_ulid()
    await store.create_run(run_id, "projection", tenant_id="t")
    async with engine.begin() as conn:
        await conn.execute(
            insert(ReactTrace),
            [
                dict(
                    id=generate_ulid(),
                    agent_run_id=run_id,
                    role="tool",
                    content="ok",
                    trace_kind="turn",
                    order_index=i,
                    meta={},
                )
                for i in range(1000)
            ],
        )
        await conn.execute(
            update(AgentRun)
            .where(AgentRun.id == run_id)
            .values(loop_state={"pending": [{"order_index": 2000 + i} for i in range(64)]})
        )
    captured = []

    def capture(conn, cursor, statement, parameters, context, many):
        captured.append((statement, parameters))

    event.listen(engine.sync_engine, "before_cursor_execute", capture)
    try:
        assert len(await store.get_public_traces(run_id, tenant_id="t", limit=1, offset=900)) == 1
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", capture)
    assert len(captured) == 1
    statement, parameters = captured[0]
    async with engine.connect() as conn:
        raw = await conn.get_raw_connection()
        plan = (
            await raw.driver_connection.fetchval(
                "EXPLAIN (ANALYZE, VERBOSE, FORMAT JSON) " + statement, *parameters
            )
        )[0]["Plan"]
    nodes = [plan]
    projections = []
    while nodes:
        node = nodes.pop()
        nodes.extend(node.get("Plans", []))
        if any("pending" in value and "loop_state" in value for value in node.get("Output", [])):
            projections.append(node)
    assert projections
    assert all(node["Actual Loops"] == node["Actual Rows"] == 1 for node in projections)
