"""Aggregate root budget checkpoints, root-routed raises and usage identity."""

from typing import Any

import pytest

from dendrux import Agent, tool
from dendrux.store import RunStore
from dendrux.types import Budget, LLMResponse, RunStatus, ToolCall, UsageStats
from tests.integration.test_budget import _CostReportingLLM


@tool()
async def work() -> str:
    """Do work."""
    return "worked"


def response(cost=0.0, target=None):
    return LLMResponse(
        text=None if target else "done",
        tool_calls=[ToolCall(name=target, params={"task": "work"} if target == "child" else {})]
        if target
        else None,
        usage=UsageStats(input_tokens=1, output_tokens=1, total_tokens=2, cost_usd=cost),
    )


def agents(store, root_responses, child_responses, budget=None):
    child_llm = _CostReportingLLM(child_responses)
    template = Agent(
        name="child", description="Worker", prompt="Work", provider=child_llm, tools=[work]
    )
    root_llm = _CostReportingLLM(root_responses)
    root = Agent(
        name="root",
        prompt="Lead",
        provider=root_llm,
        subagents=[template],
        state_store=store,
        budget=budget,
        strict_tenancy=False,
    )
    return root, root_llm, template, child_llm


async def test_child_spends_only_root_remainder_then_root_raise(store: Any):
    root, llm, template, child_llm = agents(
        store,
        [response(9, "child"), response()],
        [response(2, "work"), response()],
        Budget(max_cost_usd=10, on_exceed="pause"),
    )
    result = await root.run("go")
    assert result.status == RunStatus.PAUSED
    assert child_llm.calls == 1 and llm.calls == 1
    assert template.budget is None
    assert (await store.subtree_usage(result.run_id)).known_cost_usd == 11
    pending = await RunStore(store).get_pending(result.run_id, tenant_id=None)
    raised, raised_llm, raised_child, raised_child_llm = agents(
        store, [response()], [response()], Budget(max_cost_usd=20, on_exceed="pause")
    )
    resumed = await raised.submit_resume(
        result.run_id, tenant_id=None, expected_pause_id=pending.pause_id
    )
    assert resumed.status == RunStatus.SUCCESS
    assert raised_llm.calls == 1 and raised_child_llm.calls == 1
    assert raised_child.budget is None
    usage = await store.subtree_usage(result.run_id)
    assert usage.known_cost_usd == 11 and usage.llm_calls == 4


async def test_bind_is_extra_budget_checkpoint(store: Any, caplog):
    root, llm, _, child_llm = agents(
        store,
        [response(10, "child"), response()],
        [response()],
        Budget(max_cost_usd=10, on_exceed="pause"),
    )
    result = await root.run("go")
    assert result.status == RunStatus.PAUSED
    assert child_llm.calls == 0 and llm.calls == 1
    assert any(
        getattr(record, "code", None) == "managed_budget_exceeded"
        and record.run_id == result.run_id
        for record in caplog.records
    )
    assert len((await store.get_subtree(result.run_id)).nodes) == 1
    pending = await RunStore(store).get_pending(result.run_id, tenant_id=None)
    repeated = await root.submit_resume(
        result.run_id, tenant_id=None, expected_pause_id=pending.pause_id
    )
    assert repeated.status == RunStatus.PAUSED and child_llm.calls == 0
    raised, _, _, raised_child_llm = agents(
        store, [response()], [response()], Budget(max_cost_usd=20, on_exceed="pause")
    )
    pending = await RunStore(store).get_pending(result.run_id, tenant_id=None)
    resumed = await raised.submit_resume(
        result.run_id, tenant_id=None, expected_pause_id=pending.pause_id
    )
    assert resumed.status == RunStatus.SUCCESS and raised_child_llm.calls == 1


@pytest.mark.parametrize("failure", ["read", "invalid", "truncated"])
async def test_unknown_shared_usage_pauses_without_local_fallback(
    store: Any, monkeypatch, caplog, failure
):
    root, llm, _, child_llm = agents(
        store,
        [response(1, "child"), response()],
        [response()],
        Budget(max_tokens=100, on_exceed="cancel"),
    )
    original = store.subtree_usage

    async def unknown(run_id, *, max_nodes):
        if failure == "read":
            raise RuntimeError("SENTINEL-shared-read")
        if failure == "truncated":
            from dendrux.errors import BudgetConfigurationError

            raise BudgetConfigurationError(reason="subtree_limit")
        from dendrux.runtime.state import UsageRollup

        return UsageRollup(known_cost_usd=float("nan"))

    monkeypatch.setattr(store, "subtree_usage", unknown)
    result = await root.run("go")
    assert result.status == RunStatus.PAUSED
    assert llm.calls == 0 and child_llm.calls == 0
    saved = await store.get_pause_state(result.run_id)
    assert saved["pause_reason"] == "budget"
    assert saved["pause_message"] == "managed_usage_unknown"
    assert "SENTINEL" not in caplog.text
    assert any(
        getattr(r, "code", None) == "managed_usage_unknown" and r.run_id == result.run_id
        for r in caplog.records
    )
    monkeypatch.setattr(store, "subtree_usage", original)
    pending = await RunStore(store).get_pending(result.run_id, tenant_id=None)
    resumed = await root.submit_resume(
        result.run_id, tenant_id=None, expected_pause_id=pending.pause_id
    )
    assert resumed.status == RunStatus.SUCCESS


async def test_known_unpriced_child_warns_and_adoption_adds_no_charge(store: Any):
    root, _, _, _ = agents(
        store,
        [response(1, "child"), response()],
        [response(None, "work"), response()],
        Budget(max_cost_usd=10, on_exceed="pause"),
    )
    result = await root.run("go")
    assert result.status == RunStatus.SUCCESS
    usage = await store.subtree_usage(result.run_id)
    assert usage.llm_calls == 4 and usage.known_cost_usd == 1 and usage.unpriced_calls == 1
    events = []
    for node in (await store.get_subtree(result.run_id)).nodes:
        events.extend(await store.get_run_events(node.run.id))
    assert any(e.event_type == "budget.unpriced" for e in events)


async def test_siblings_and_restart_share_spend(store: Any, monkeypatch):
    root, _, _, _ = agents(
        store,
        [response(9, "child")],
        [response(2, "work")],
        Budget(max_cost_usd=10, on_exceed="pause", max_subtree_nodes=17),
    )
    reads = []
    original = store.subtree_usage

    async def measured(run_id, *, max_nodes):
        reads.append((run_id, max_nodes))
        return await original(run_id, max_nodes=max_nodes)

    async def forbidden(*args, **kwargs):
        raise AssertionError("Managed meter must not add an own-run usage read")

    monkeypatch.setattr(store, "subtree_usage", measured)
    monkeypatch.setattr(store, "run_usage", forbidden)
    first = await root.run("go")
    pending = await RunStore(store).get_pending(first.run_id, tenant_id=None)
    same, same_llm, _, same_child = agents(
        store, [], [], Budget(max_cost_usd=10, on_exceed="pause", max_subtree_nodes=17)
    )
    again = await same.submit_resume(
        first.run_id, tenant_id=None, expected_pause_id=pending.pause_id
    )
    assert again.status == RunStatus.PAUSED and same_llm.calls == same_child.calls == 0
    second, _, _, _ = agents(
        store,
        [response(0, "child"), response()],
        [response(), response(4, "work")],
        Budget(max_cost_usd=15, on_exceed="pause", max_subtree_nodes=17),
    )
    pending = await RunStore(store).get_pending(first.run_id, tenant_id=None)
    paused = await second.submit_resume(
        first.run_id, tenant_id=None, expected_pause_id=pending.pause_id
    )
    assert paused.status == RunStatus.PAUSED
    tree = await store.get_subtree(first.run_id)
    assert len(tree.nodes) == 3 and tree.usage.known_cost_usd == 15
    assert all(scope == first.run_id and bound == 17 for scope, bound in reads)
    assert all(n.run.agent_name in {"root", "child"} for n in tree.nodes)


@pytest.mark.parametrize("streaming", [False, True])
async def test_final_response_overshoot_completes(store: Any, streaming):
    root, _, _, _ = agents(
        store,
        [response(9, "child"), response()],
        [response(2)],
        Budget(max_cost_usd=10, on_exceed="pause"),
    )
    if streaming:
        async with root.stream("go") as stream:
            events = [e async for e in stream]
        result = stream.result
        assert events[-1].type.value == "run_paused"
    else:
        result = await root.run("go")
    # The child completes its final response; the parent stops at its next checkpoint.
    assert result.status == RunStatus.PAUSED
    tree = await store.get_subtree(result.run_id)
    assert tree.nodes[1].run.status == "success" and tree.usage.known_cost_usd == 11


async def test_standalone_budget_template_still_pauses(store: Any):
    provider = _CostReportingLLM([response(2, "work"), response()])
    template = Agent(
        name="child",
        description="Worker",
        prompt="Work",
        provider=provider,
        tools=[work],
        budget=Budget(max_cost_usd=1, on_exceed="pause"),
        state_store=store,
    )
    result = await template.run("go", tenant_id=None)
    assert result.status == RunStatus.PAUSED and provider.calls == 1


async def test_unknown_usage_pause_write_failure_propagates(store: Any, monkeypatch):
    from dendrux.errors import PersistenceError

    root, _, _, _ = agents(store, [response()], [], Budget(max_tokens=100, on_exceed="pause"))

    async def unknown(*args, **kwargs):
        raise RuntimeError("SENTINEL-usage")

    async def unwritable(*args, **kwargs):
        raise PersistenceError()

    monkeypatch.setattr(store, "subtree_usage", unknown)
    monkeypatch.setattr(store, "pause_with_event", unwritable)
    with pytest.raises(PersistenceError, match="persistence_error"):
        await root.run("go")


@pytest.mark.parametrize("round_", range(3))
async def test_pg_two_sibling_meters_race_last_dollar(store: Any, engine: Any, round_):
    import asyncio
    from contextlib import AsyncExitStack
    from types import SimpleNamespace

    from dendrux.loops._budget import BudgetTracker
    from dendrux.runtime import _managed_budget
    from dendrux.types import generate_ulid
    from tests.integration.test_managed_cancel import parallel_stores

    root = generate_ulid()
    await store.create_run(root, "root")
    await store.save_llm_interaction(
        root, iteration_index=1, usage=response(9).usage, interaction_id=generate_ulid()
    )
    async with AsyncExitStack() as stack:
        stores = await parallel_stores(engine, stack, count=2)
        child_ids = [generate_ulid(), generate_ulid()]
        for child_id in child_ids:
            await store.create_run(child_id, "child", parent_run_id=root)
        ready = asyncio.Barrier(2)

        async def writer(peer, child_id):
            tracker = BudgetTracker(
                Budget(max_cost_usd=10, on_exceed="pause"),
                run_id=child_id,
                usage=UsageStats(),
                recorder=None,
                notifier=None,
                warnings=[],
            )
            recorder = SimpleNamespace(_loop_state={"managed": {"root_run_id": root}}, _store=peer)
            _managed_budget.bind(recorder, tracker)
            assert (await tracker.checkpoint(1))[1] == []
            await ready.wait()
            identity = generate_ulid()
            for _ in range(2):
                await peer.save_llm_interaction(
                    child_id, iteration_index=1, usage=response(1).usage, interaction_id=identity
                )
            await ready.wait()
            action, breaches = await tracker.checkpoint(2)
            assert action == "pause" and breaches[0].used == 11

        await asyncio.gather(
            *(writer(peer, child_id) for peer, child_id in zip(stores, child_ids, strict=True))
        )
    usage = await store.subtree_usage(root)
    assert usage.known_cost_usd == 11 and usage.llm_calls == 3


@pytest.mark.parametrize("depth", [0, 2, 10])
async def test_root_measurement_queries_and_plans(store: Any, engine: Any, depth, capsys):
    import json
    from types import SimpleNamespace

    from sqlalchemy import event

    from dendrux.db.models import AgentRun, LLMInteraction
    from dendrux.loops._budget import BudgetTracker
    from dendrux.runtime import _managed_budget
    from dendrux.types import generate_ulid

    if engine.dialect.name == "postgresql":
        async with engine.begin() as conn:
            history = [{"id": generate_ulid(), "agent_name": "history"} for _ in range(10000)]
            await conn.execute(AgentRun.__table__.insert(), history)
            await conn.execute(
                LLMInteraction.__table__.insert(),
                [
                    {
                        "id": generate_ulid(),
                        "agent_run_id": row["id"],
                        "iteration_index": 1,
                        "input_tokens": 1,
                        "output_tokens": 1,
                        "cost_usd": 1,
                    }
                    for row in history
                ],
            )
            await conn.exec_driver_sql("ANALYZE agent_runs")
            await conn.exec_driver_sql("ANALYZE llm_interactions")
    root = generate_ulid()
    parent = None
    for level in range(depth + 1):
        current = root if not level else generate_ulid()
        await store.create_run(current, "meter", parent_run_id=parent)
        await store.save_llm_interaction(
            current, iteration_index=1, usage=response(1).usage, interaction_id=generate_ulid()
        )
        parent = current
    tracker = BudgetTracker(
        Budget(max_cost_usd=100, on_exceed="pause", max_subtree_nodes=33),
        run_id=parent,
        usage=UsageStats(),
        recorder=None,
        notifier=None,
        warnings=[],
    )
    _managed_budget.bind(
        SimpleNamespace(_loop_state={"managed": {"root_run_id": root}}, _store=store), tracker
    )
    statements = []
    transactions = []

    def capture(conn, cursor, statement, parameters, context, many):
        statements.append((statement, parameters))

    def begun(conn):
        transactions.append("begin")

    def rolled_back(conn):
        transactions.append("rollback")

    event.listen(engine.sync_engine, "before_cursor_execute", capture)
    event.listen(engine.sync_engine, "begin", begun)
    event.listen(engine.sync_engine, "rollback", rolled_back)
    try:
        assert (await tracker.checkpoint(1))[1] == []
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", capture)
        event.remove(engine.sync_engine, "begin", begun)
        event.remove(engine.sync_engine, "rollback", rolled_back)
    assert len(statements) == depth + 2
    assert transactions == ["begin", "rollback"]
    plans = []
    if engine.dialect.name == "postgresql":
        async with engine.connect() as conn:
            for statement, parameters in statements:
                plan = await conn.exec_driver_sql(
                    "EXPLAIN (ANALYZE, BUFFERS) " + statement, parameters
                )
                plans.append({"sql": statement, "plan": [row[0] for row in plan]})
    with capsys.disabled():
        print(
            json.dumps(
                {
                    "backend": engine.dialect.name,
                    "depth": depth,
                    "selects": len(statements),
                    "transactions": transactions,
                    "plans": plans,
                }
            )
        )


async def test_root_policy_change_refuses_before_leaf_effect(store: Any):
    from dendrux.errors import GovernanceMismatchError

    root, _, _, _ = agents(
        store,
        [response(9, "child")],
        [response(2, "work")],
        Budget(max_cost_usd=10, on_exceed="pause"),
    )
    first = await root.run("go")
    changed, _, _, child_llm = agents(
        store, [], [response()], Budget(max_cost_usd=20, on_exceed="cancel")
    )
    pending = await RunStore(store).get_pending(first.run_id, tenant_id=None)
    with pytest.raises(GovernanceMismatchError):
        await changed.submit_resume(
            first.run_id, tenant_id=None, expected_pause_id=pending.pause_id
        )
    assert child_llm.calls == 0


async def test_reclaim_adopts_without_new_provider_charge(store: Any, engine: Any, monkeypatch):
    from dendrux.runtime import _managed
    from tests.integration.test_managed_recovery import ProcessDeathError, fresh_store

    root, _, _, _ = agents(
        store, [response(9, "child")], [response(2)], Budget(max_cost_usd=10, on_exceed="pause")
    )
    original = _managed.adopt

    async def die(*args, **kwargs):
        raise ProcessDeathError

    monkeypatch.setattr(_managed, "adopt", die)
    with pytest.raises(ProcessDeathError):
        await root.run("go")
    from sqlalchemy import select
    from sqlalchemy.ext.asyncio import AsyncSession

    from dendrux.db.models import AgentRun

    async with AsyncSession(engine) as session:
        root_id = await session.scalar(select(AgentRun.id).where(AgentRun.parent_run_id.is_(None)))
    monkeypatch.setattr(_managed, "adopt", original)
    async with fresh_store(engine) as restarted:
        raised, llm, _, child_llm = agents(
            restarted, [response()], [], Budget(max_cost_usd=20, on_exceed="pause")
        )
        result = await raised.reclaim(root_id, tenant_id=None)
        assert result.status == RunStatus.SUCCESS
        assert llm.calls == 1 and child_llm.calls == 0
        assert (await restarted.subtree_usage(root_id)).llm_calls == 3
        assert (await restarted.subtree_usage(root_id)).known_cost_usd == 11
        replay = await raised.reclaim(root_id, tenant_id=None)
        assert replay.status == RunStatus.SUCCESS and llm.calls == 1


async def test_streamed_leaf_budget_pause_and_raised_root_resume(store: Any):
    root, _, _, _ = agents(
        store,
        [response(9, "child")],
        [response(2, "work")],
        Budget(max_cost_usd=10, on_exceed="pause"),
    )
    async with root.stream("go") as stream:
        events = [event async for event in stream]
    assert events[-1].type.value == "run_paused"
    paused = stream.result
    raised, _, template, _ = agents(
        store, [response()], [response()], Budget(max_cost_usd=20, on_exceed="pause")
    )
    pending = await RunStore(store).get_pending(paused.run_id, tenant_id=None)
    async with raised.resume_stream(
        paused.run_id, tenant_id=None, expected_pause_id=pending.pause_id
    ) as resumed:
        events = [event async for event in resumed]
    assert events[-1].type.value == "run_completed"
    assert resumed.result.status == RunStatus.SUCCESS and template.budget is None
    usage = await store.subtree_usage(paused.run_id)
    assert usage.known_cost_usd == 11 and usage.llm_calls == 4


async def test_no_root_budget_has_no_usage_ceiling_or_reads(store: Any, monkeypatch):
    root, _, _, _ = agents(
        store, [response(900, "child"), response()], [response(900, "work"), response()]
    )

    async def forbidden(*args, **kwargs):
        raise AssertionError("No budget means no shared accounting read")

    monkeypatch.setattr(store, "subtree_usage", forbidden)
    result = await root.run("go")
    assert result.status == RunStatus.SUCCESS


async def test_bound_child_shared_read_failure_never_uses_its_empty_local_meter(
    store: Any, monkeypatch
):
    from dendrux.errors import PersistenceError

    root, root_llm, _, child_llm = agents(
        store, [response(9, "child")], [response()], Budget(max_cost_usd=10, on_exceed="pause")
    )
    original_bind, original_read = store.bind_managed_child, store.subtree_usage
    bound = False

    async def bind(request):
        nonlocal bound
        result = await original_bind(request)
        bound = True
        return result

    async def read(run_id, *, max_nodes):
        if bound:
            raise PersistenceError()
        return await original_read(run_id, max_nodes=max_nodes)

    monkeypatch.setattr(store, "bind_managed_child", bind)
    monkeypatch.setattr(store, "subtree_usage", read)
    result = await root.run("go")
    assert result.status == RunStatus.PAUSED
    assert root_llm.calls == 1 and child_llm.calls == 0
    tree = await store.get_subtree(result.run_id)
    assert len(tree.nodes) == 2 and tree.usage.known_cost_usd == 9
    child = tree.nodes[1].run
    pause = await store.get_pause_state(child.id)
    assert child.status == "paused" and pause["pause_message"] == "managed_usage_unknown"


def unknown_response():
    result = response(None)
    result.usage.usage_reported = False
    result.usage.cost_unknown = True
    return result


async def resume_unknown(root, store, result, *, streaming=False):
    pending = await RunStore(store).get_pending(result.run_id, tenant_id=None)
    if streaming:
        async with root.resume_stream(
            result.run_id, tenant_id=None, expected_pause_id=pending.pause_id
        ) as stream:
            _ = [event async for event in stream]
        return stream.result
    return await root.submit_resume(
        result.run_id, tenant_id=None, expected_pause_id=pending.pause_id
    )


@pytest.mark.parametrize("known_unpriced", [False, True])
@pytest.mark.parametrize("streaming", [False, True])
async def test_finished_child_usage_fault_is_adopted_but_known_unpriced_is_not(
    store, known_unpriced, streaming
):
    child_response = response(None) if known_unpriced else unknown_response()
    root, root_llm, _, child_llm = agents(
        store,
        [response(9, "child"), response()],
        [child_response],
        Budget(max_cost_usd=10, on_exceed="pause"),
    )
    if streaming:
        async with root.stream("go") as stream:
            _ = [event async for event in stream]
        result = stream.result
    else:
        result = await root.run("go")
    assert child_llm.calls == 1
    if known_unpriced:
        assert result.status == RunStatus.SUCCESS and root_llm.calls == 2
        assert (await store.subtree_usage(result.run_id)).unpriced_calls == 1
        return
    assert result.status == RunStatus.PAUSED and root_llm.calls == 1
    tree = await store.get_subtree(result.run_id)
    child = tree.nodes[1].run
    assert child.status == "success"
    for run_id in (child.id, result.run_id):
        fault = (await store.get_loop_state(run_id))["managed_usage_unknown"]
        assert fault == {"code": "managed_usage_unknown", "origin_run_id": child.id}
    assert (await store.get_pause_state(result.run_id))["pause_message"] == "managed_usage_unknown"
    raised, raised_llm, _, replay_child = agents(
        store, [response()], [], Budget(max_cost_usd=20, on_exceed="pause")
    )
    resumed = await resume_unknown(raised, store, result, streaming=streaming)
    assert resumed.status == RunStatus.SUCCESS
    assert raised_llm.calls == 1 and replay_child.calls == 0


@pytest.mark.parametrize("cut", ["before_adoption", "after_claim", "after_completion"])
async def test_unknown_child_outcome_survives_adoption_crashes(store, engine, monkeypatch, cut):
    from sqlalchemy import select
    from sqlalchemy.ext.asyncio import AsyncSession

    from dendrux.db.models import AgentRun
    from dendrux.runtime import _managed
    from tests.integration.test_managed_recovery import ProcessDeathError, fresh_store

    root, _, _, _ = agents(
        store,
        [response(9, "child")],
        [unknown_response()],
        Budget(max_cost_usd=10, on_exceed="pause"),
    )
    target = "adopt" if cut == "before_adoption" else "complete_adoption"
    original = getattr(_managed, target)

    async def die(*args, **kwargs):
        if cut == "after_completion":
            await original(*args, **kwargs)
        raise ProcessDeathError

    monkeypatch.setattr(_managed, target, die)
    with pytest.raises(ProcessDeathError):
        await root.run("go")
    monkeypatch.setattr(_managed, target, original)
    async with AsyncSession(engine) as session:
        root_id = await session.scalar(select(AgentRun.id).where(AgentRun.parent_run_id.is_(None)))
    if cut != "before_adoption":
        from tests._helpers.recovery_checkpoint import expire

        await expire(store, engine, root_id)
    async with fresh_store(engine) as restarted:
        recovered, llm, _, child_llm = agents(
            restarted, [response()], [], Budget(max_cost_usd=10, on_exceed="pause")
        )
        paused = await recovered.reclaim(root_id, tenant_id=None)
        assert paused.status == RunStatus.PAUSED and llm.calls == child_llm.calls == 0
        assert (await restarted.get_pause_state(root_id))[
            "pause_message"
        ] == "managed_usage_unknown"
        raised, raised_llm, _, child_llm = agents(
            restarted, [response()], [], Budget(max_cost_usd=20, on_exceed="pause")
        )
        resumed = await resume_unknown(raised, restarted, paused)
        assert resumed.status == RunStatus.SUCCESS
        assert raised_llm.calls == 1 and child_llm.calls == 0
        assert (await restarted.subtree_usage(root_id)).llm_calls == 3


@pytest.mark.parametrize("exhausted", [False, True])
async def test_grandchild_fault_propagates_through_intermediate_outcome(store, exhausted):
    def build(limit, root_responses, child_responses, grand_responses):
        grand = Agent(
            name="grand",
            description="Grandchild",
            prompt="Work",
            provider=_CostReportingLLM(grand_responses),
        )
        child = Agent(
            name="child",
            description="Child",
            prompt="Work",
            provider=_CostReportingLLM(child_responses),
            subagents=[grand],
            max_iterations=1 if exhausted else 10,
        )
        return Agent(
            name="root",
            prompt="Lead",
            provider=_CostReportingLLM(root_responses),
            subagents=[child],
            state_store=store,
            strict_tenancy=False,
            budget=Budget(max_cost_usd=limit, on_exceed="pause"),
        )

    to_grand = LLMResponse(
        tool_calls=[ToolCall(name="grand", params={"task": "work"})], usage=response().usage
    )
    root = build(
        10, [response(9, "child"), response()], [to_grand, response()], [unknown_response()]
    )
    paused = await root.run("go")
    assert paused.status == RunStatus.PAUSED and root.provider.calls == 1
    tree = await store.get_subtree(paused.run_id)
    child, grand = tree.nodes[1].run, tree.nodes[2].run
    assert grand.status == "success"
    active = paused.run_id if exhausted else child.id
    fault = (await store.get_loop_state(active))["managed_usage_unknown"]
    assert fault == {"code": "managed_usage_unknown", "origin_run_id": grand.id}
    raised = build(20, [response()], [response()], [])
    resumed = await resume_unknown(raised, store, paused)
    assert resumed.status == RunStatus.SUCCESS


@pytest.mark.parametrize("fault", ["write", "read"])
async def test_observed_final_child_fault_is_sticky_even_if_parent_read_is_healthy(
    store, monkeypatch, fault
):
    from dendrux.errors import PersistenceError
    from dendrux.runtime.context import get_delegation_context

    root, root_llm, _, child_llm = agents(
        store,
        [response(9, "child"), response()],
        [response()],
        Budget(max_cost_usd=10, on_exceed="pause"),
    )
    original_write, original_read = store.save_llm_interaction, store.subtree_usage

    async def write(*args, **kwargs):
        ctx = get_delegation_context()
        if fault == "write" and ctx and ctx.delegation_level == 1:
            raise PersistenceError()
        return await original_write(*args, **kwargs)

    async def read(*args, **kwargs):
        ctx = get_delegation_context()
        if fault == "read" and child_llm.calls and ctx and ctx.delegation_level == 1:
            raise PersistenceError()
        return await original_read(*args, **kwargs)

    monkeypatch.setattr(store, "save_llm_interaction", write)
    monkeypatch.setattr(store, "subtree_usage", read)
    result = await root.run("go")
    assert result.status == RunStatus.PAUSED and root_llm.calls == 1
    assert (await store.get_pause_state(result.run_id))["pause_message"] == "managed_usage_unknown"


@pytest.mark.parametrize("faulty", [False, True])
async def test_adoption_fault_read_query_cost(store, engine, monkeypatch, capsys, faulty):
    import json

    from sqlalchemy import event

    from dendrux.db.models import AgentRun
    from dendrux.types import generate_ulid

    if engine.dialect.name == "postgresql":
        async with engine.begin() as connection:
            await connection.execute(
                AgentRun.__table__.insert(),
                [{"id": generate_ulid(), "agent_name": "history"} for _ in range(10000)],
            )
    original = store.get_loop_state
    reads = []

    async def read(run_id):
        statements = []

        def capture(connection, cursor, statement, parameters, context, many):
            statements.append((statement, parameters))

        event.listen(engine.sync_engine, "before_cursor_execute", capture)
        try:
            result = await original(run_id)
        finally:
            event.remove(engine.sync_engine, "before_cursor_execute", capture)
        reads.append((run_id, statements))
        return result

    monkeypatch.setattr(store, "get_loop_state", read)
    root, _, _, _ = agents(
        store,
        [response(9, "child"), response()],
        [unknown_response() if faulty else response()],
        Budget(max_cost_usd=10, on_exceed="pause"),
    )
    result = await root.run("go")
    child = (await store.get_subtree(result.run_id)).nodes[1].run
    adoption_reads = [statements for run_id, statements in reads if run_id == child.id]
    assert len(adoption_reads) == 1 and len(adoption_reads[0]) == 1
    plans = []
    if engine.dialect.name == "postgresql":
        async with engine.connect() as connection:
            for statement, parameters in adoption_reads[0]:
                plan = await connection.exec_driver_sql(
                    "EXPLAIN (ANALYZE, BUFFERS) " + statement, parameters
                )
                plans = [row[0] for row in plan]
    with capsys.disabled():
        print(
            json.dumps(
                {
                    "operation": "adoption_fault_read",
                    "backend": engine.dialect.name,
                    "faulty": faulty,
                    "extra_selects": 1,
                    "plan": plans,
                }
            )
        )


async def test_fault_measurement_query_cost(store, engine, monkeypatch, capsys):
    import json
    from collections import Counter

    from sqlalchemy import event

    from dendrux.loops._budget import BudgetTracker

    measurements = []
    original = BudgetTracker._measure

    async def measured(tracker):
        statements = []

        def capture(connection, cursor, statement, parameters, context, many):
            statements.append(statement.split()[0])

        event.listen(engine.sync_engine, "before_cursor_execute", capture)
        try:
            result = await original(tracker)
        finally:
            event.remove(engine.sync_engine, "before_cursor_execute", capture)
        measurements.append(
            {"unknown": tracker._shared_unknown, "statements": dict(Counter(statements))}
        )
        return result

    monkeypatch.setattr(BudgetTracker, "_measure", measured)
    root, _, _, _ = agents(
        store,
        [response(9, "child"), response()],
        [unknown_response()],
        Budget(max_cost_usd=10, on_exceed="pause"),
    )
    result = await root.run("go")
    assert result.status == RunStatus.PAUSED
    assert any(m["unknown"] for m in measurements)
    with capsys.disabled():
        print(
            json.dumps(
                {
                    "operation": "fault_measurements",
                    "backend": engine.dialect.name,
                    "measurements": measurements,
                }
            )
        )


async def test_failed_fault_checkpoint_cannot_let_parent_continue_on_a_partial_total(
    store, engine, monkeypatch, capsys
):
    import json

    from sqlalchemy import event

    from dendrux.errors import PersistenceError
    from dendrux.runtime import _managed_budget
    from dendrux.runtime.context import get_delegation_context

    observations = []
    original_fault = _managed_budget.completed_fault

    async def observed_fault(*args, **kwargs):
        statements = []

        def capture(connection, cursor, statement, parameters, context, many):
            statements.append(statement.split()[0])

        event.listen(engine.sync_engine, "before_cursor_execute", capture)
        try:
            result = await original_fault(*args, **kwargs)
        finally:
            event.remove(engine.sync_engine, "before_cursor_execute", capture)
        observations.append(statements)
        return result

    monkeypatch.setattr(_managed_budget, "completed_fault", observed_fault)
    original = store.checkpoint_step

    async def checkpoint(run_id, **kwargs):
        ctx = get_delegation_context()
        if ctx and ctx.delegation_level == 1 and kwargs["loop_state"].get("managed_usage_unknown"):
            raise PersistenceError()
        return await original(run_id, **kwargs)

    monkeypatch.setattr(store, "checkpoint_step", checkpoint)
    root, llm, _, _ = agents(
        store,
        [response(9, "child"), response()],
        [unknown_response()],
        Budget(max_cost_usd=10, on_exceed="pause"),
    )
    result = await root.run("go")
    assert result.status == RunStatus.PAUSED and llm.calls == 1
    assert (await store.get_pause_state(result.run_id))["pause_message"] == "managed_usage_unknown"
    assert observations == [["SELECT", "SELECT"]]
    with capsys.disabled():
        print(
            json.dumps(
                {
                    "operation": "failed_marker_adoption",
                    "backend": engine.dialect.name,
                    "extra_selects": 2,
                }
            )
        )
    raised, _, _, _ = agents(store, [response()], [], Budget(max_cost_usd=20, on_exceed="pause"))
    assert (await resume_unknown(raised, store, result)).status == RunStatus.SUCCESS
