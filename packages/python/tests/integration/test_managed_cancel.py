"""Managed root cancellation, interrupted propagation and Postgres contenders."""

from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from typing import Any

import pytest
from sqlalchemy import update
from sqlalchemy.ext.asyncio import create_async_engine

from dendrux import Agent
from dendrux.db.models import AgentRun, TokenUsage, ToolCallRecord
from dendrux.errors import ManagedChildControlError, RunCancelledError, RunControlError
from dendrux.runtime import _managed
from dendrux.runtime.persistence import PersistenceRecorder
from dendrux.runtime.state import claimed_store
from dendrux.runtime.stores._factory import store_for_engine
from dendrux.types import (
    Budget,
    LLMResponse,
    RunStatus,
    ToolCall,
    ToolResult,
    UsageStats,
    generate_ulid,
)
from tests._helpers.recovery_checkpoint import expire
from tests.integration.test_managed_adoption import (
    child_agent,
    crash_before_adoption,
    fresh_root,
    rows,
)
from tests.integration.test_managed_bind import bind_request, managed_parent
from tests.integration.test_managed_routing import ContextLLM, call, read_sheet

TENANT = "t1"


def controller(store: Any, **kwargs: Any) -> Agent:
    return Agent(
        name="lead",
        prompt="Lead.",
        provider=ContextLLM([]),
        state_store=store,
        strict_tenancy=False,
        **kwargs,
    )


async def chain(store: Any) -> tuple[str, str, str]:
    root, call_id = await managed_parent(store)
    next_call = generate_ulid()
    first = bind_request(
        root,
        call_id,
        child_loop_state={
            "v": 1,
            "loop": "react",
            "phase": "dispatch",
            "pending": [
                {
                    "call_id": next_call,
                    "tool": "reviewer",
                    "phase": "dispatchable",
                    "order_index": 3,
                }
            ],
        },
    )
    await store.bind_managed_child(first)
    second = bind_request(first.child_run_id, next_call, root=root, path=("reviewer", "reviewer"))
    await store.bind_managed_child(second)
    await store.pause_run(second.child_run_id, status="paused", pause_data={})
    return root, first.child_run_id, second.child_run_id


@pytest.mark.parametrize("cascade", [None, False, True])
async def test_managed_root_always_cascades_only_explicit_edges(store: Any, cascade: Any) -> None:
    root, child, leaf = await chain(store)
    legacy = generate_ulid()
    await store.create_run(legacy, "legacy", parent_run_id=root, tenant_id=TENANT)
    foreign = generate_ulid()
    await store.create_run(foreign, "foreign", tenant_id="foreign")
    async with store._session_factory() as session:
        await session.execute(
            update(AgentRun).where(AgentRun.id == foreign).values(parent_run_id=root)
        )
        await session.commit()
    result = await controller(store).cancel_run(
        root, **({} if cascade is None else {"cascade": cascade})
    )
    assert result.status == RunStatus.CANCELLED
    assert [(await store.get_run(rid)).status for rid in (child, leaf)] == [
        "cancelled",
        "cancelled",
    ]
    assert (await store.get_run(legacy)).cancel_requested is (cascade is True)
    assert not (await store.get_run(foreign)).cancel_requested
    assert result.meta["cascade_count"] == (4 if cascade is True else 3)


@pytest.mark.parametrize("cascade", [False, True])
async def test_legacy_root_cancel_golden(store: Any, cascade: bool) -> None:
    root, child = generate_ulid(), generate_ulid()
    await store.create_run(root, "lead", tenant_id=TENANT)
    await store.create_run(child, "legacy", tenant_id=TENANT, parent_run_id=root)
    await controller(store).cancel_run(root, cascade=cascade)
    assert (await store.get_run(root)).cancel_requested
    assert (await store.get_run(child)).cancel_requested is cascade


async def test_repeat_terminal_cancel_finishes_truncated_walk(store: Any) -> None:
    root, child, leaf = await chain(store)
    first = await controller(
        store, budget=Budget(max_tokens=10000, max_subtree_nodes=2)
    ).cancel_run(root)
    assert first.meta["cascade_truncated"]
    assert (await store.get_run(root)).status == "cancelled"
    assert (await store.get_run(leaf)).status == "paused"
    again = await controller(store).cancel_run(root)
    assert not again.meta["cascade_truncated"]
    assert (await store.get_run(leaf)).status == "cancelled"
    assert {r["run_id"] for r in again.meta["cascade"]} == {root, child, leaf}


async def test_repeat_cancel_after_interrupted_mark(store: Any, monkeypatch: Any) -> None:
    root, child, leaf = await chain(store)
    original = store.mark_cancel_requested

    async def interrupted(ids: Any, **kwargs: Any) -> int:
        await original(ids, **kwargs)
        raise OSError("interrupted")

    with monkeypatch.context() as patch:
        patch.setattr(store, "mark_cancel_requested", interrupted)
        with pytest.raises(OSError):
            await controller(store).cancel_run(root)
    assert (await store.get_run(root)).cancel_requested
    assert not (await store.get_run(child)).cancel_requested
    await controller(store).cancel_run(root)
    assert [(await store.get_run(r)).status for r in (root, child, leaf)] == ["cancelled"] * 3


@pytest.mark.parametrize("terminal", [False, True])
async def test_descendant_checkpoint_reads_persisted_root_stop(store: Any, terminal: bool) -> None:
    root, child, leaf = await chain(store)
    await store.request_cancel(root)
    if terminal:
        await store.finalize_run(root, status="cancelled")
    assert not (await store.get_run(leaf)).cancel_requested
    assert await _managed.cancellation_requested(store, leaf)
    assert await _managed.cancellation_requested(store, child)


async def test_marked_parent_refuses_bind_without_orphans(store: Any, engine: Any) -> None:
    root, call_id = await managed_parent(store)
    await store.mark_cancel_requested([root], tenant_id=TENANT)
    with pytest.raises(RunCancelledError):
        await store.bind_managed_child(bind_request(root, call_id))
    assert await rows(engine, ToolCallRecord) == []
    assert await rows(engine, AgentRun, AgentRun.parent_run_id == root) == []


async def test_cancelled_root_cannot_be_revived_by_adoption(
    store: Any, engine: Any, monkeypatch: Any
) -> None:
    root, child = await crash_before_adoption(store, engine, monkeypatch)
    await controller(store).cancel_run(root)
    agent, llm, child_llm = fresh_root(store)
    result = await agent.reclaim(root, tenant_id=TENANT)
    assert result.status == RunStatus.CANCELLED
    assert (await store.get_run(child)).status == "success"
    assert llm.calls == child_llm.calls == 0
    assert len(await rows(engine, TokenUsage, TokenUsage.agent_run_id == child)) == 1


async def paused_workflow(store: Any) -> tuple[Agent, str, str, Any]:
    leaf = child_agent(
        ContextLLM([call("read_sheet", {"sheet": "a"}, "s1"), LLMResponse(text="done")]),
        tools=[read_sheet],
    )
    root = Agent(
        name="lead",
        prompt="Lead.",
        provider=ContextLLM([call("reviewer", {"task": "go"}, "c1"), LLMResponse(text="done")]),
        subagents=[leaf],
        state_store=store,
    )
    result = await root.run("go", tenant_id=TENANT)
    return root, result.run_id, result.pending.origin_run_id, result.pending


async def test_direct_managed_child_cancel_stays_refused(store: Any) -> None:
    root, child, _ = await chain(store)
    before = await store.get_run(child)
    with pytest.raises(ManagedChildControlError) as info:
        await controller(store).cancel_run(child, cascade=True, tenant_id=TENANT)
    assert info.value.code == "managed_root_required"
    assert (await store.get_run(child)).status == before.status
    assert not (await store.get_run(root)).cancel_requested


async def test_root_entry_refuses_leaf_submit_after_interrupted_propagation(store: Any) -> None:
    agent, root, child, pending = await paused_workflow(store)
    await store.request_cancel(root)
    result = ToolResult(name="read_sheet", call_id=pending.tool_calls[0].id, payload='"ok"')
    with pytest.raises(RunCancelledError):
        await agent.submit_tool_results(
            root, [result], expected_pause_id=pending.pause_id, tenant_id=TENANT
        )
    assert (await store.get_run(child)).status == "waiting_client_tool"


@pytest.mark.parametrize("admission", ["tool", "provider", "started"])
async def test_effect_admission_stops_on_terminal_root(store: Any, admission: str) -> None:
    root, call_id = await managed_parent(store)
    tool_call = ToolCall(name="worker", params={"task": "t"}, id=generate_ulid())
    request = bind_request(
        root,
        call_id,
        child_loop_state={
            "v": 1,
            "loop": "react",
            "pending": [
                {
                    "call_id": tool_call.id,
                    "tool": "worker",
                    "target": "server",
                    "phase": "dispatchable",
                    "order_index": 3,
                }
            ],
        },
    )
    created = await store.bind_managed_child(request)
    owned = claimed_store(store, request.child_run_id, created.child_claim)
    recorder = PersistenceRecorder(owned, request.child_run_id, model="m")
    await store.finalize_run(root, status="cancelled")
    with pytest.raises(RunCancelledError):
        if admission == "provider":
            await recorder.on_llm_call_started(request.child_run_id, 1)
        elif admission == "started":
            await recorder.on_run_started(request.child_run_id)
        else:
            await recorder.on_tool_started(request.child_run_id, tool_call, 1)


async def test_intermediate_bind_stops_when_root_mark_committed(store: Any, engine: Any) -> None:
    root, call_id = await managed_parent(store)
    next_call = generate_ulid()
    request = bind_request(
        root,
        call_id,
        child_loop_state={
            "v": 1,
            "loop": "react",
            "pending": [{"call_id": next_call, "order_index": 3}],
        },
    )
    await store.bind_managed_child(request)
    await store.request_cancel(root)
    with pytest.raises(RunCancelledError):
        await store.bind_managed_child(
            bind_request(request.child_run_id, next_call, root=root, path=("reviewer", "reviewer"))
        )
    assert len(await rows(engine, AgentRun)) == 2
    assert len(await rows(engine, ToolCallRecord)) == 1


async def test_intermediate_adoption_does_not_resume_under_cancelled_root(store: Any) -> None:
    root, parent, leaf = await chain(store)
    await store.finalize_run(leaf, status="cancelled")
    await store.finalize_run(root, status="cancelled")
    wait = _managed._Waiting(
        child_agent(ContextLLM([])),
        parent,
        child_agent(ContextLLM([])),
        ("reviewer", "reviewer"),
        ToolCall(**(await store.get_pause_state(parent))["pending_tool_calls"][0]),
        leaf,
    )
    # The stop is checked before the report, so no report transform or claim occurs.
    claim = await _managed.claim_adoption(wait, store)
    assert claim is None
    assert (await store.get_run(parent)).status == "paused"


async def parallel_stores(engine: Any, stack: AsyncExitStack, count: int = 8) -> list[Any]:
    if engine.dialect.name != "postgresql":
        pytest.skip("independent pools and real row-lock races require Postgres")
    stores = []
    for _ in range(count):
        other = create_async_engine(engine.url, pool_size=1, max_overflow=0)
        stack.push_async_callback(other.dispose)
        stores.append(store_for_engine(other))
    return stores


@pytest.mark.parametrize("round_", range(3))
async def test_pg_cancel_versus_eight_binders_has_no_orphans(
    store: Any, engine: Any, round_: int
) -> None:
    async with AsyncExitStack() as stack:
        stores = await parallel_stores(engine, stack)
        root, call_id = await managed_parent(store)
        ready = asyncio.Event()

        async def bind(other: Any) -> Any:
            await ready.wait()
            return await other.bind_managed_child(bind_request(root, call_id))

        tasks = [asyncio.create_task(bind(other)) for other in stores]
        tasks.append(asyncio.create_task(controller(stores[0]).cancel_run(root)))
        ready.set()
        results = await asyncio.gather(*tasks, return_exceptions=True)
    for result in results:
        assert not isinstance(result, BaseException) or isinstance(result, RunCancelledError), (
            result
        )
    created = [r for r in results if getattr(r, "outcome", None) == "created"]
    children = await rows(engine, AgentRun, AgentRun.parent_run_id == root)
    intents = await rows(engine, ToolCallRecord, ToolCallRecord.agent_run_id == root)
    assert len(created) == len(children) == len(intents) <= 1
    root_row = await store.get_run(root)
    assert root_row.cancel_requested or root_row.status == "cancelled"
    if children:
        assert intents[0].meta["managed"]["child_run_id"] == children[0].id
        assert children[0].cancel_requested


def submission_root(store: Any) -> Agent:
    return Agent(
        name="lead",
        prompt="Lead.",
        state_store=store,
        provider=ContextLLM([LLMResponse(text="done")]),
        subagents=[child_agent(ContextLLM([LLMResponse(text="done")]), tools=[read_sheet])],
    )


@pytest.mark.parametrize("round_", range(3))
async def test_pg_cancel_versus_eight_leaf_submissions(
    store: Any, engine: Any, round_: int
) -> None:
    async with AsyncExitStack() as stack:
        stores = await parallel_stores(engine, stack)
        _, root, child, pending = await paused_workflow(store)
        agents = [submission_root(other) for other in stores]
        result = ToolResult(name="read_sheet", call_id=pending.tool_calls[0].id, payload='"ok"')
        results = await asyncio.gather(
            *(
                a.submit_tool_results(
                    root, [result], tenant_id=TENANT, expected_pause_id=pending.pause_id
                )
                for a in agents
            ),
            controller(stores[0]).cancel_run(root),
            return_exceptions=True,
        )
    for outcome in results:
        assert not isinstance(outcome, BaseException) or isinstance(outcome, RunControlError), (
            outcome
        )
    row = await store.get_run(child)
    assert row.claim_epoch in {0, 1}
    assert row.status in {"cancelled", "success"}
    assert (await store.get_run(root)).status in {"cancelled", "success"}
    assert len(await rows(engine, AgentRun)) == 2
    assert sum(a.subagents[0].provider.calls for a in agents) <= 1
    assert sum(a.provider.calls for a in agents) <= 1


@pytest.mark.parametrize("round_", range(3))
async def test_pg_cancel_versus_eight_reclaims(store: Any, engine: Any, round_: int) -> None:
    from tests.integration.test_managed_recovery import _dead_child, lead, reviewer

    async with AsyncExitStack() as stack:
        stores = await parallel_stores(engine, stack)
        root, child = await _dead_child(store, engine)
        await expire(store, engine, child.id)
        before = await store.get_run(child.id)
        agents = [
            lead(
                other,
                ContextLLM([LLMResponse(text="done")]),
                reviewer(ContextLLM([LLMResponse(text="done")])),
            )
            for other in stores
        ]
        results = await asyncio.gather(
            *(a.reclaim(root, tenant_id=TENANT) for a in agents),
            controller(stores[0]).cancel_run(root),
            return_exceptions=True,
        )
    for outcome in results:
        assert not isinstance(outcome, BaseException) or isinstance(outcome, RunControlError), (
            outcome
        )
    row = await store.get_run(child.id)
    assert row.claim_epoch - before.claim_epoch <= 1
    assert row.reclaim_attempts <= 1
    assert row.status in {"cancelled", "success"} or (
        row.status == "running" and row.cancel_requested
    )
    assert (await store.get_run(root)).status in {"cancelled", "success"}
    assert len(await rows(engine, AgentRun)) == 2
    assert sum(a.subagents[0].provider.calls for a in agents) <= 1
    assert sum(a.provider.calls for a in agents) <= 1


@pytest.mark.parametrize("round_", range(3))
async def test_pg_cancel_versus_eight_adoptions(
    store: Any, engine: Any, monkeypatch: Any, round_: int
) -> None:
    from dendrux.db.models import ReactTrace, RunEvent

    async with AsyncExitStack() as stack:
        stores = await parallel_stores(engine, stack)
        root, child = await crash_before_adoption(store, engine, monkeypatch)
        agents = [fresh_root(other, LLMResponse(text="done"))[0] for other in stores]
        results = await asyncio.gather(
            *(a.reclaim(root, tenant_id=TENANT) for a in agents),
            controller(stores[0]).cancel_run(root),
            return_exceptions=True,
        )
    for outcome in results:
        assert not isinstance(outcome, BaseException) or isinstance(outcome, RunControlError), (
            outcome
        )
    row = await store.get_run(root)
    assert row.status in {"cancelled", "success"}
    assert row.claim_epoch in {1, 2}
    traces = await rows(
        engine, ReactTrace, ReactTrace.agent_run_id == root, ReactTrace.role == "tool"
    )
    events = await rows(
        engine, RunEvent, RunEvent.agent_run_id == root, RunEvent.event_type == "tool.completed"
    )
    assert len(traces) == len(events) <= 1
    assert len(await rows(engine, AgentRun)) == 2
    assert len(await rows(engine, TokenUsage, TokenUsage.agent_run_id == child)) == 1
    assert sum(a.provider.calls for a in agents) <= 1
    assert sum(a.subagents[0].provider.calls for a in agents) == 0


@pytest.mark.parametrize("entry", ["run", "stream", "resume", "resume_stream"])
async def test_cancelled_root_stops_child_before_tool_admission(
    store: Any, engine: Any, entry: str
) -> None:
    from dendrux import tool

    effects = []

    @tool()
    async def effect() -> str:
        """Record an effect."""
        effects.append("effect")
        return "ok"

    async def cancel_root(index: int, child_id: str) -> None:
        if entry in {"resume", "resume_stream"} and index == 0:
            return
        state = await store.get_loop_state(child_id)
        root_id = state["managed"]["root_run_id"]
        await controller(store, budget=Budget(max_tokens=10000, max_subtree_nodes=1)).cancel_run(
            root_id
        )

    responses = [call("effect", {}, "e1")]
    if entry in {"resume", "resume_stream"}:
        responses.insert(0, call("read_sheet", {"sheet": "a"}, "s1"))
    for response in responses:
        response.usage = UsageStats(input_tokens=10, output_tokens=1, total_tokens=11)
    child_llm = ContextLLM(responses, on_call=cancel_root)
    child = child_agent(child_llm, tools=[effect, read_sheet])
    root = Agent(
        name="lead",
        prompt="Lead.",
        state_store=store,
        subagents=[child],
        provider=ContextLLM([call("reviewer", {"task": "go"}, "c1")]),
    )
    if entry == "stream":
        stream = root.stream("go", tenant_id=TENANT)
        async with stream:
            events = [event async for event in stream]
        result = stream.result
        assert events[-1].type.value == "run_cancelled"
    elif entry in {"resume", "resume_stream"}:
        paused = await root.run("go", tenant_id=TENANT)
        report = ToolResult(
            name="read_sheet", call_id=paused.pending.tool_calls[0].id, payload='"ok"'
        )
        if entry == "resume":
            result = await root.resume(
                paused.run_id,
                tool_results=[report],
                expected_pause_id=paused.pause_id,
                tenant_id=TENANT,
            )
        else:
            stream = root.resume_stream(
                paused.run_id,
                tool_results=[report],
                expected_pause_id=paused.pause_id,
                tenant_id=TENANT,
            )
            async with stream:
                events = [event async for event in stream]
            result = stream.result
            assert events[-1].type.value == "run_cancelled"
    else:
        result = await root.run("go", tenant_id=TENANT)
    assert result.status == RunStatus.CANCELLED
    assert effects == []
    children = await rows(engine, AgentRun, AgentRun.parent_run_id == result.run_id)
    assert len(children) == 1 and children[0].status == "cancelled"
    assert children[0].total_input_tokens > 0
    intents = await rows(
        engine,
        ToolCallRecord,
        ToolCallRecord.agent_run_id == children[0].id,
        ToolCallRecord.tool_name == "effect",
    )
    assert intents == []


async def test_managed_store_without_cancel_projection_refuses_before_creation(
    store: Any, engine: Any, monkeypatch: Any
) -> None:
    from dendrux.errors import ManagedUnsupportedError

    monkeypatch.setattr(store, "_get_cancel_children_page", None)
    root = submission_root(store)
    with pytest.raises(ManagedUnsupportedError):
        await root.run("go", tenant_id=TENANT)
    assert await rows(engine, AgentRun) == []


async def test_filtered_full_page_does_not_hide_a_later_managed_child(
    store: Any, engine: Any
) -> None:
    root, call_id = await managed_parent(store)
    bound = await store.bind_managed_child(bind_request(root, call_id))
    async with engine.begin() as conn:
        await conn.execute(
            AgentRun.__table__.insert(),
            [
                {
                    "id": f"{i:026d}",
                    "agent_name": "legacy",
                    "tenant_id": TENANT,
                    "parent_run_id": root,
                }
                for i in range(501)
            ],
        )
    result = await controller(store).cancel_run(root)
    assert result.meta["cascade_count"] == 2
    assert (await store.get_run(bound.child_run_id)).cancel_requested
    legacy = await rows(engine, AgentRun, AgentRun.agent_name == "legacy")
    assert len(legacy) == 501 and not any(r.cancel_requested for r in legacy)


async def test_already_admitted_effect_finishes_and_provider_checkpoint_stops(
    store: Any, engine: Any
) -> None:
    from dendrux import tool
    from dendrux.runtime.context import get_delegation_context

    effects = []

    @tool()
    async def effect() -> str:
        """Complete an already admitted operation."""
        state = await store.get_loop_state(get_delegation_context().run_id)
        await controller(store, budget=Budget(max_tokens=10000, max_subtree_nodes=1)).cancel_run(
            state["managed"]["root_run_id"]
        )
        effects.append("completed")
        return "ok"

    response = call("effect", {}, "e1")
    response.usage = UsageStats(input_tokens=10, output_tokens=1, total_tokens=11)
    llm = ContextLLM([response, LLMResponse(text="never")])
    root = Agent(
        name="lead",
        prompt="Lead.",
        state_store=store,
        subagents=[child_agent(llm, tools=[effect])],
        provider=ContextLLM([call("reviewer", {"task": "go"}, "c1")]),
    )
    result = await root.run("go", tenant_id=TENANT)
    assert result.status == RunStatus.CANCELLED
    assert effects == ["completed"] and llm.calls == 1
    children = await rows(engine, AgentRun, AgentRun.parent_run_id == result.run_id)
    assert len(children) == 1 and children[0].status == "cancelled"
    assert children[0].total_input_tokens == 10
    intents = await rows(engine, ToolCallRecord, ToolCallRecord.agent_run_id == children[0].id)
    assert len(intents) == 1 and intents[0].outcome == "completed"
    assert len(await rows(engine, TokenUsage, TokenUsage.agent_run_id == children[0].id)) == 1


@pytest.mark.parametrize("read", ["root", "children", "combined_control"])
async def test_cancel_projection_database_errors_are_safe(
    store: Any, engine: Any, read: str, caplog: Any
) -> None:
    from sqlalchemy import event
    from sqlalchemy.exc import OperationalError

    from dendrux.errors import PersistenceError

    root, _, leaf = await chain(store)
    sentinel = "SENTINEL-database-vendor-cancel"

    def fail(conn, cursor, statement, parameters, context, executemany):
        raise OperationalError("query", {}, RuntimeError(sentinel))

    event.listen(engine.sync_engine, "before_cursor_execute", fail)
    try:
        with pytest.raises(PersistenceError) as info:
            if read == "root":
                await store._managed_root_stopped(leaf)
            elif read == "combined_control":
                await store._get_managed_run_control(leaf)
            else:
                await store._get_cancel_children_page(
                    [root], after=None, limit=500, tenant_id=TENANT
                )
        assert sentinel not in str(info.value)
        assert sentinel not in caplog.text
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", fail)


async def test_marked_root_reclaim_refuses_before_descendant_claim(store: Any, engine: Any) -> None:
    from tests.integration.test_managed_recovery import _dead_child, lead, reviewer

    root, child = await _dead_child(store, engine)
    await expire(store, engine, child.id)
    before = await store.get_run(child.id)
    await store.request_cancel(root)
    agent = lead(
        store,
        ContextLLM([LLMResponse(text="done")]),
        reviewer(ContextLLM([LLMResponse(text="done")])),
    )
    with pytest.raises(RunCancelledError):
        await agent.reclaim(root, tenant_id=TENANT)
    after = await store.get_run(child.id)
    assert after.claim_epoch == before.claim_epoch
    assert after.reclaim_attempts == before.reclaim_attempts


async def test_root_checkpoint_result_keeps_completed_usage(store: Any, engine: Any) -> None:
    from dendrux import tool

    effects = []

    @tool()
    async def effect() -> str:
        """Record an effect."""
        effects.append("effect")
        return "ok"

    async def cancel(index: int, run_id: str) -> None:
        await store.request_cancel(run_id)

    response = call("effect", {}, "e1")
    response.usage = UsageStats(input_tokens=10, output_tokens=1, total_tokens=11)
    root = Agent(
        name="lead",
        prompt="Lead.",
        state_store=store,
        tools=[effect],
        provider=ContextLLM([response], on_call=cancel),
        subagents=[child_agent(ContextLLM([]))],
    )
    result = await root.run("go", tenant_id=TENANT)
    assert result.status == RunStatus.CANCELLED
    assert effects == []
    assert "pause_state" not in result.meta
    assert result.usage.input_tokens == 10
    assert (await store.get_run(result.run_id)).total_input_tokens == 10


async def test_format_one_cancel_never_reads_a_checkpoint(
    store: Any, engine: Any, monkeypatch: Any
) -> None:
    from tests._fixtures.storage import pin_storage_format

    await pin_storage_format(engine, 1)
    root = generate_ulid()
    await store.create_run(root, "lead", tenant_id=TENANT)
    await store.pause_run(root, status="paused", pause_data={})

    async def unsupported(run_id: str) -> Any:
        raise NotImplementedError("format one has no checkpoint reader")

    monkeypatch.setattr(store, "get_loop_state", unsupported)
    result = await controller(store).cancel_run(root)
    assert result.status == RunStatus.CANCELLED


@pytest.mark.parametrize("read", ["root", "combined_control"])
async def test_pg_root_stop_reads_at_most_one_root_row(store: Any, engine: Any, read: str) -> None:
    import json

    from sqlalchemy import event

    if engine.dialect.name != "postgresql":
        pytest.skip("actual Postgres root probe work")
    async with engine.begin() as conn:
        await conn.execute(
            AgentRun.__table__.insert(),
            [
                {
                    "id": generate_ulid(),
                    "agent_name": "history",
                    "tenant_id": TENANT,
                    "status": "success",
                }
                for _ in range(10000)
            ],
        )
    root, call_id = await managed_parent(store)
    bound = await store.bind_managed_child(bind_request(root, call_id))
    await store.finalize_run(root, status="cancelled")
    statements = []

    def capture(conn, cursor, statement, parameters, context, executemany):
        statements.append((statement, parameters))

    event.listen(engine.sync_engine, "before_cursor_execute", capture)
    try:
        if read == "root":
            assert await store._managed_root_stopped(bound.child_run_id)
        else:
            assert (await store._get_managed_run_control(bound.child_run_id)).cancel_requested
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", capture)
    assert len(statements) == 1
    sql, parameters = statements[0]
    async with engine.connect() as conn:
        plan = (
            await conn.exec_driver_sql("EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) " + sql, parameters)
        ).scalar_one()
    if isinstance(plan, str):
        plan = json.loads(plan)
    pending = [plan[0]["Plan"]]
    probes = []
    while pending:
        node = pending.pop()
        if node.get("Alias") == "managed_root":
            probes.append(node)
        pending.extend(node.get("Plans", []))
    assert len(probes) == 1
    assert probes[0]["Actual Rows"] + probes[0].get("Rows Removed by Filter", 0) <= 1, probes[0]


@pytest.mark.parametrize("admission", ["provider", "tool"])
async def test_fix1_managed_admission_has_one_control_select(
    store: Any, engine: Any, monkeypatch: Any, admission: str
) -> None:
    from unittest.mock import AsyncMock

    from sqlalchemy import event

    root, call_id = await managed_parent(store)
    bound = await store.bind_managed_child(bind_request(root, call_id))
    owned = claimed_store(store, bound.child_run_id, bound.child_claim)
    recorder = PersistenceRecorder(owned, bound.child_run_id)
    monkeypatch.setattr(recorder, "_write_tool_intent", AsyncMock())
    statements = []

    def capture(conn, cursor, statement, parameters, context, executemany):
        if statement.lstrip().upper().startswith("SELECT"):
            statements.append(statement)

    event.listen(engine.sync_engine, "before_cursor_execute", capture)
    try:
        for _ in range(3):
            if admission == "provider":
                await recorder.on_llm_call_started(bound.child_run_id, 1)
            else:
                await recorder.on_tool_started(
                    bound.child_run_id, ToolCall(name="work", id="w1"), 1
                )
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", capture)
    assert len(statements) == 3, statements


@pytest.mark.parametrize("changed", ["epoch", "status"])
@pytest.mark.parametrize("admission", ["started", "provider", "tool"])
async def test_fix1_combined_control_keeps_claim_fencing(
    store: Any, engine: Any, changed: str, admission: str
) -> None:
    from dendrux.errors import RunFencedError

    root, call_id = await managed_parent(store)
    bound = await store.bind_managed_child(bind_request(root, call_id))
    recorder = PersistenceRecorder(
        claimed_store(store, bound.child_run_id, bound.child_claim), bound.child_run_id
    )
    async with engine.begin() as conn:
        await conn.execute(
            update(AgentRun)
            .where(AgentRun.id == bound.child_run_id)
            .values(**({"claim_epoch": 1} if changed == "epoch" else {"status": "paused"}))
        )
    with pytest.raises(RunFencedError):
        if admission == "started":
            await recorder.on_run_started(bound.child_run_id)
        elif admission == "provider":
            await recorder.on_llm_call_started(bound.child_run_id, 1)
        else:
            await recorder.on_tool_started(bound.child_run_id, ToolCall(name="work", id="w1"), 1)


@pytest.mark.parametrize("stop_at", ["after_bind", "after_provider"])
async def test_fix1_cancel_closes_only_a_started_notifier(
    store: Any, monkeypatch: Any, stop_at: str
) -> None:
    from collections import Counter

    from dendrux import tool
    from dendrux.loops.base import BaseNotifier

    starts, finishes = Counter(), Counter()

    class Observer(BaseNotifier):
        async def on_run_started(self, run_id: str, **kwargs: Any) -> None:
            starts[run_id] += 1

        async def on_run_finished(self, run_id: str, result: Any) -> None:
            finishes[run_id] += 1

    @tool()
    async def effect() -> str:
        """Perform an operation."""
        raise RuntimeError("effect must not run")

    async def cancel(root_id: str) -> None:
        await controller(store, budget=Budget(max_tokens=10000, max_subtree_nodes=1)).cancel_run(
            root_id
        )

    bound_ids = []
    original = store.bind_managed_child

    async def bind(request: Any) -> Any:
        result = await original(request)
        bound_ids.append(result.child_run_id)
        if stop_at == "after_bind":
            await cancel(request.root_run_id)
        return result

    async def provider_cancel(index: int, run_id: str) -> None:
        if stop_at == "after_provider":
            await cancel((await store.get_loop_state(run_id))["managed"]["root_run_id"])

    monkeypatch.setattr(store, "bind_managed_child", bind)
    child = child_agent(
        ContextLLM([call("effect", {}, "e1")], on_call=provider_cancel), tools=[effect]
    )
    root = Agent(
        name="lead",
        prompt="Lead.",
        state_store=store,
        subagents=[child],
        provider=ContextLLM([call("reviewer", {"task": "go"}, "c1")]),
    )
    result = await root.run("go", tenant_id=TENANT, notifier=Observer())
    assert result.status == RunStatus.CANCELLED
    assert len(bound_ids) == 1
    child_id = bound_ids[0]
    expected = 0 if stop_at == "after_bind" else 1
    assert starts[child_id] == expected
    assert finishes[child_id] == expected


@pytest.mark.parametrize("recorder_started", [False, True])
@pytest.mark.parametrize("notifier_started", [False, True])
async def test_fix1_resume_stop_closes_started_sinks(
    store: Any, recorder_started: bool, notifier_started: bool
) -> None:
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from dendrux.errors import ManagedRootStoppedError
    from dendrux.runtime.runner import _stopped_resume_result
    from dendrux.runtime.state import ClaimResult

    root, _ = await managed_parent(store)
    state = await store.get_loop_state(root)
    owned = claimed_store(store, root, ClaimResult(0, 2, state, None))
    recorder = SimpleNamespace(_loop_source=None, _loop_state=state, on_run_finished=AsyncMock())
    notifier = SimpleNamespace(on_run_finished=AsyncMock())
    ctx = SimpleNamespace(
        recorder=recorder,
        notifier=notifier,
        recorder_started=recorder_started,
        notifier_started=notifier_started,
    )
    result = await _stopped_resume_result(ManagedRootStoppedError(root, root), owned, root, ctx)
    assert result.status == RunStatus.CANCELLED
    assert recorder.on_run_finished.await_count == int(recorder_started)
    assert notifier.on_run_finished.await_count == int(notifier_started)
