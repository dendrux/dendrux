"""Atomic managed admission bounds, shared by SQLite and Postgres."""

import asyncio
import json
from collections import Counter
from copy import copy
from dataclasses import asdict, make_dataclass
from functools import partial

import pytest
from sqlalchemy import event, func, insert, select, text, update
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from dendrux.agent import Agent
from dendrux.db.models import AgentRun, ToolCallRecord
from dendrux.errors import GovernanceMismatchError, ManagedStateError
from dendrux.runtime import _managed
from dendrux.runtime.state import MANAGED_HEADER_KEY, ManagedHeader
from dendrux.runtime.stores._factory import store_for_engine
from dendrux.runtime.submit import GOVERNANCE_KEY, governance_snapshot
from dendrux.types import (
    InputValidationError,
    Limits,
    LLMResponse,
    RunStatus,
    ToolResult,
    generate_ulid,
)
from tests.integration.test_managed_bind import bind_request, managed_parent, snapshot
from tests.integration.test_managed_inline import ContextLLM, call, child_agent, root_agent
from tests.integration.test_managed_routing import ContextLLM as ObservedLLM
from tests.integration.test_managed_routing import read_sheet


def legacy_snapshot(agent):
    legacy = copy(agent)
    values = asdict(agent._limits)
    values.pop("max_managed_runs")
    values.pop("max_subagent_definitions")
    old_limits = make_dataclass("LegacyLimits", [(name, int) for name in values])
    legacy._limits = old_limits(**values)
    return governance_snapshot(legacy)


async def consume(stream):
    async with stream:
        async for _ in stream:
            pass
    return stream.result


@pytest.mark.parametrize("override", [None, "max_managed_runs", "max_subagent_definitions"])
@pytest.mark.parametrize("entry", ["resume", "resume_stream"])
async def test_legacy_pause_governance_compatibility(store, engine, override, entry):
    original = Agent(
        name="legacy",
        prompt="Work.",
        provider=ContextLLM([call("read_sheet", {"sheet": "S1"}, "c1")]),
        tools=[read_sheet],
        state_store=store,
        on_governance_change="raise",
    )
    paused = await original.run("work", tenant_id="t1")
    raw = await store.get_pause_state(paused.run_id)
    saved = legacy_snapshot(original)
    raw[GOVERNANCE_KEY] = saved
    record = await store.get_run(paused.run_id)
    async with AsyncSession(engine) as session:
        await session.execute(
            update(AgentRun)
            .where(AgentRun.id == paused.run_id)
            .values(pause_data=raw, meta={**(record.meta or {}), GOVERNANCE_KEY: saved})
        )
        await session.commit()
    limits = Limits(**{override: getattr(Limits(), override) + 1}) if override else Limits()
    provider = ContextLLM([LLMResponse(text="done")])
    upgraded = Agent(
        name="legacy",
        prompt="Work.",
        provider=provider,
        tools=[read_sheet],
        state_store=store_for_engine(engine),
        limits=limits,
        on_governance_change="raise",
    )
    pending = paused.pending.tool_calls[0]
    results = [ToolResult(name=pending.name, call_id=pending.id, payload='"ok"')]
    kwargs = {"tool_results": results, "expected_pause_id": paused.pause_id, "tenant_id": "t1"}
    if entry == "resume_stream":
        result = await consume(upgraded.resume_stream(paused.run_id, **kwargs))
        assert result.status == (RunStatus.ERROR if override else RunStatus.SUCCESS)
    elif override:
        with pytest.raises(GovernanceMismatchError) as error:
            await upgraded.resume(paused.run_id, **kwargs)
        assert error.value.changed == ("limits",)
    else:
        assert (await upgraded.resume(paused.run_id, **kwargs)).status == RunStatus.SUCCESS
    if override:
        assert provider.calls == 0
        assert (await store.get_run(paused.run_id)).claim_epoch == record.claim_epoch
        assert (await store.get_run(paused.run_id)).status == record.status
    else:
        assert provider.calls == 1


@pytest.mark.parametrize("entry", ["run", "stream"])
@pytest.mark.parametrize("continuation", ["resume", "resume_stream"])
async def test_restart_before_handoff_keeps_first_checkpoint_cap(
    store, engine, entry, continuation
):
    checkpoints = []

    async def observe_checkpoint(index, run_id):
        checkpoints.append((await store.get_loop_state(run_id)).get("managed_limits"))

    child_llm = ContextLLM([LLMResponse(text="child done")])
    original = root_agent(
        store,
        ObservedLLM([call("read_sheet", {"sheet": "S1"}, "c1")], on_call=observe_checkpoint),
        child_agent(store, child_llm),
        tools=[read_sheet],
        limits=Limits(max_managed_runs=1),
    )
    paused = await (
        consume(original.stream("work", tenant_id="t1"))
        if entry == "stream"
        else original.run("work", tenant_id="t1")
    )
    assert paused.status == RunStatus.WAITING_CLIENT_TOOL
    assert checkpoints == [{"max_managed_runs": 1}]
    assert (await store.get_loop_state(paused.run_id)).get("managed_limits") == {
        "max_managed_runs": 1
    }
    upgraded = root_agent(
        store_for_engine(engine),
        ContextLLM([call("reviewer", {"task": "work"}, "c2")]),
        child_agent(store, child_llm),
        tools=[read_sheet],
        limits=Limits(max_managed_runs=128),
        on_governance_change="warn",
    )
    pending = paused.pending.tool_calls[0]
    kwargs = {
        "tool_results": [ToolResult(name=pending.name, call_id=pending.id, payload='"ok"')],
        "expected_pause_id": paused.pause_id,
        "tenant_id": "t1",
    }
    if continuation == "resume_stream":
        result = await consume(upgraded.resume_stream(paused.run_id, **kwargs))
        assert result.status == RunStatus.ERROR
    else:
        with pytest.raises(ManagedStateError) as error:
            await upgraded.resume(paused.run_id, **kwargs)
        assert error.value.code == "managed_run_limit"
    assert child_llm.calls == 0
    assert (await store.get_loop_state(paused.run_id))["managed_limits"] == {"max_managed_runs": 1}
    async with AsyncSession(engine) as session:
        assert await session.scalar(select(func.count()).select_from(AgentRun)) == 1
        assert (
            await session.scalar(
                select(func.count())
                .select_from(ToolCallRecord)
                .where(ToolCallRecord.target == "agent")
            )
            == 0
        )


async def saved_chain(store, engine, depth, cap):
    root, first_call = await managed_parent(store)
    ids = [root, *(generate_ulid() for _ in range(depth))]
    calls = [first_call, *(generate_ulid() for _ in range(depth - 1))]
    values, intents = [], []
    for index, run_id in enumerate(ids):
        path = ("reviewer",) * index
        header = (
            ManagedHeader(root, ())
            if index == 0
            else ManagedHeader(root, path, ids[index - 1], calls[index - 1])
        )
        state = {"v": 1, "loop": "react", MANAGED_HEADER_KEY: header.to_dict()}
        if index == 0 and cap is not None:
            state["managed_limits"] = {"max_managed_runs": cap}
        raw = None
        if index < depth:
            request = bind_request(
                run_id, calls[index], root=root, child_id=ids[index + 1], path=(*path, "reviewer")
            )
            raw = request.parent_wait.to_dict()
            intents.append(
                {
                    "id": request.row_id,
                    "agent_run_id": run_id,
                    "tool_call_id": calls[index],
                    "tool_name": "reviewer",
                    "target": "agent",
                    "iteration_index": 1,
                    "outcome": "started",
                    "params": {"task": "work"},
                }
            )
        values.append(
            {
                "id": run_id,
                "agent_name": "lead" if index == 0 else "reviewer",
                "parent_run_id": ids[index - 1] if index else None,
                "tenant_id": "t1",
                "status": "paused" if index < depth else "running",
                "storage_format": 2,
                "claim_epoch": 1,
                "loop_state": state,
                "pause_data": raw,
            }
        )
    async with engine.begin() as connection:
        await connection.execute(
            update(AgentRun)
            .where(AgentRun.id == root)
            .values(**{key: value for key, value in values[0].items() if key != "id"})
        )
        await connection.execute(insert(AgentRun), values[1:])
        await connection.execute(insert(ToolCallRecord), intents)
    return ids


@pytest.mark.parametrize(
    "saved_cap,configured_cap,depth,valid", [(300, 1, 258, True), (2, 300, 2, False)]
)
async def test_saved_path_walk_uses_persisted_run_cap(
    store, engine, saved_cap, configured_cap, depth, valid
):
    ids = await saved_chain(store, engine, depth, saved_cap)
    child = None
    for _ in range(depth):
        child = Agent(
            name="reviewer",
            description="Specialist.",
            prompt="Work.",
            provider=ContextLLM([]),
            subagents=[child] if child else [],
            limits=Limits(max_subagent_definitions=300),
            max_delegation_depth=None,
        )
    root = root_agent(
        store,
        ContextLLM([]),
        child,
        max_delegation_depth=None,
        limits=Limits(max_managed_runs=configured_cap, max_subagent_definitions=301),
    )
    statements = Counter()

    def record(connection, cursor, statement, parameters, context, many):
        statements[statement.split()[0].upper()] += 1

    event.listen(engine.sync_engine, "before_cursor_execute", record)
    try:
        if valid:
            waiting, leaf = await _managed.walk(root, store, ids[0])
            assert len(waiting) == depth
            assert leaf.id == ids[-1]
            assert statements["SELECT"] == 2 + 4 * depth
        else:
            with pytest.raises(ManagedStateError) as error:
                await _managed.walk(root, store, ids[0])
            assert error.value.run_id == ids[0]
            assert statements["SELECT"] == 3 + 4 * (saved_cap - 1)
        assert set(statements) == {"SELECT"}
        print(
            "T5C_WALK_EVIDENCE="
            + json.dumps(
                {
                    "depth": depth,
                    "saved_cap": saved_cap,
                    "configured_cap": configured_cap,
                    "statements": dict(statements),
                }
            )
        )
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", record)


@pytest.mark.parametrize("cap", [0, True, "SENTINEL"])
async def test_saved_path_walk_refuses_invalid_persisted_cap(store, cap):
    root, _ = await managed_parent(store)
    await cap_root(store, root, cap)
    agent = root_agent(store, ContextLLM([]), child_agent(store, ContextLLM([])))
    with pytest.raises(ManagedStateError) as error:
        await _managed.walk(agent, store, root)
    assert error.value.run_id == root
    assert "SENTINEL" not in str(error.value)


async def cap_root(store, root, cap):
    state = await store.get_loop_state(root)
    state["managed_limits"] = {"max_managed_runs": cap}
    await store.checkpoint_step(root, claim_epoch=0, loop_state=state, pii_mapping=None, trace=None)


async def member(store, root, *, name="worker", completed=False):
    child, call = await managed_parent(store)
    state = await store.get_loop_state(child)
    state[MANAGED_HEADER_KEY] = ManagedHeader(root, (name,), root, generate_ulid()).to_dict()
    await store.checkpoint_step(
        child, claim_epoch=0, loop_state=state, pii_mapping=None, trace=None
    )
    async with store._session_factory() as session:
        await session.execute(
            update(AgentRun)
            .where(AgentRun.id == child)
            .values(
                parent_run_id=root,
                status=RunStatus.SUCCESS.value if completed else RunStatus.RUNNING.value,
            )
        )
        await session.commit()
    return child, call


async def test_cap_refusal_writes_nothing(store, engine):
    root, call = await managed_parent(store)
    await cap_root(store, root, 1)
    before = await snapshot(engine, root)
    with pytest.raises(ManagedStateError) as error:
        await store.bind_managed_child(bind_request(root, call))
    assert error.value.code == "managed_run_limit"
    assert error.value.run_id == root
    assert await snapshot(engine, root) == before


async def test_runtime_root_override_persists_before_bind(store, engine):
    child_llm = ContextLLM([LLMResponse(text="child done")])
    child = child_agent(store, child_llm)
    root = root_agent(
        store,
        ContextLLM([call("reviewer", {"task": "work"}, "c1")]),
        child,
        limits=Limits(max_managed_runs=1),
    )
    with pytest.raises(ManagedStateError) as error:
        await root.run("work", tenant_id="t1")
    assert error.value.code == "managed_run_limit"
    assert error.value.run_id is not None
    async with AsyncSession(engine) as session:
        actual_ids = list((await session.execute(select(AgentRun.id))).scalars())
    assert len(actual_ids) == 1
    root_id = actual_ids[0]
    assert (await store.get_run(root_id)).status == RunStatus.ERROR.value
    assert child_llm.served == []
    assert (await store.get_loop_state(root_id))["managed_limits"] == {"max_managed_runs": 1}
    async with AsyncSession(engine) as session:
        assert await session.scalar(select(func.count()).select_from(AgentRun)) == 1
        assert await session.scalar(select(func.count()).select_from(ToolCallRecord)) == 0


async def test_completed_children_count_and_legacy_does_not(store, engine):
    root, call = await managed_parent(store)
    await cap_root(store, root, 2)
    await member(store, root, completed=True)
    with pytest.raises(ManagedStateError):
        await store.bind_managed_child(bind_request(root, call))


async def test_legacy_and_foreign_members_do_not_count(store, engine):
    root, call = await managed_parent(store)
    await cap_root(store, root, 2)
    for tenant in ("t1", "foreign"):
        child = generate_ulid()
        await store.create_run(child, "legacy", tenant_id=tenant, parent_run_id=root)
    result = await store.bind_managed_child(bind_request(root, call))
    assert result.outcome == "created"
    assert await store.get_managed_run_count(root, limit=128, tenant_id="t1") == 2
    assert await store.get_managed_run_count(root, limit=128, tenant_id="foreign") is None


async def test_inconclusive_header_refuses_without_writes(store, engine):
    root, call = await managed_parent(store)
    child, _ = await member(store, root)
    async with AsyncSession(engine) as session:
        await session.execute(
            update(AgentRun)
            .where(AgentRun.id == child)
            .values(loop_state={MANAGED_HEADER_KEY: {"root_run_id": root}})
        )
        await session.commit()
    before = await snapshot(engine, root)
    with pytest.raises(ManagedStateError) as error:
        await store.bind_managed_child(bind_request(root, call))
    assert error.value.code == "managed_run_limit"
    assert await snapshot(engine, root) == before


async def test_legacy_candidate_overload_refuses_without_writes(store, engine):
    root, call = await managed_parent(store)
    async with engine.begin() as connection:
        await connection.execute(
            insert(AgentRun),
            [
                {
                    "id": generate_ulid(),
                    "agent_name": "legacy",
                    "status": "success",
                    "parent_run_id": root,
                    "tenant_id": "t1",
                }
                for _ in range(200)
            ],
        )
    before = await snapshot(engine, root)
    with pytest.raises(ManagedStateError) as error:
        await store.bind_managed_child(bind_request(root, call))
    assert error.value.code == "managed_run_limit"
    assert await snapshot(engine, root) == before


@pytest.mark.parametrize("cap", [0, -1, True, "SENTINEL"])
async def test_persisted_invalid_cap_fails_closed(store, cap):
    root, call = await managed_parent(store)
    await cap_root(store, root, cap)
    with pytest.raises(ManagedStateError) as error:
        await store.bind_managed_child(bind_request(root, call))
    assert error.value.code == "managed_run_limit"
    assert "SENTINEL" not in str(error.value)


async def test_restart_keeps_root_cap(store, engine):
    root, call = await managed_parent(store)
    await cap_root(store, root, 1)
    fresh = store_for_engine(engine)
    with pytest.raises(ManagedStateError) as error:
        await fresh.bind_managed_child(bind_request(root, call))
    assert error.value.code == "managed_run_limit"


@pytest.mark.parametrize("limit", [0, -1, True, "SENTINEL"])
async def test_count_invalid_limit_is_safe(store, limit):
    root, _ = await managed_parent(store)
    with pytest.raises(InputValidationError) as error:
        await store.get_managed_run_count(root, limit=limit, tenant_id="t1")
    assert error.value.code == "invalid_limit"
    assert "SENTINEL" not in str(error.value)


async def test_pg_populated_count_plan_and_query_counts(store, engine, monkeypatch):
    if engine.dialect.name != "postgresql":
        pytest.skip("EXPLAIN ANALYZE requires Postgres")
    async with engine.begin() as connection:
        await connection.execute(
            insert(AgentRun),
            [
                {"id": generate_ulid(), "agent_name": "unrelated", "status": "success"}
                for _ in range(10_000)
            ],
        )
        await connection.execute(text("ANALYZE agent_runs"))
    counts = {}
    for mode in ("before", "after"):
        root, call = await managed_parent(store)
        statements = Counter()

        def record(
            connection, cursor, statement, parameters, context, many, *, statements=statements
        ):
            statements[statement.split()[0].upper()] += 1

        def transaction(connection, *, verb, statements=statements):
            statements[verb] += 1

        async def old_lock(session, request):
            await store.reserve_parent_writer(session, request.parent_run_id)

        async def old_admit(session, request, parent):
            return store._bind_admit(request, parent)

        listeners = [("before_cursor_execute", record)] + [
            (verb.lower(), partial(transaction, verb=verb))
            for verb in ("BEGIN", "COMMIT", "ROLLBACK")
        ]
        for event_name, listener in listeners:
            event.listen(engine.sync_engine, event_name, listener)
        try:
            with monkeypatch.context() as patch:
                if mode == "before":
                    patch.setattr(store, "_lock_managed_root", old_lock)
                    patch.setattr(store, "_admit_managed_limit", old_admit)
                request = bind_request(root, call)
                await store.bind_managed_child(request)
                counts[mode] = dict(statements)
                if mode == "after":
                    statements.clear()
                    assert (await store.bind_managed_child(request)).outcome == "existing"
                    counts["replay"] = dict(statements)
        finally:
            for event_name, listener in listeners:
                event.remove(engine.sync_engine, event_name, listener)
    statement = store._managed_count_query([root], 128, "t1")
    compiled = statement.compile(dialect=engine.dialect, compile_kwargs={"literal_binds": True})
    async with engine.connect() as connection:
        plan = await connection.scalar(
            text("EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) " + str(compiled))
        )
        root_read = select(AgentRun.loop_state).where(AgentRun.id == root)
        root_sql = root_read.compile(dialect=engine.dialect, compile_kwargs={"literal_binds": True})
        root_cap_plan = await connection.scalar(
            text("EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) " + str(root_sql))
        )

    def scans(node):
        own = [node] if node.get("Relation Name") == "agent_runs" else []
        return own + [item for child in node.get("Plans", []) for item in scans(child)]

    for scan in scans(plan[0]["Plan"]):
        assert scan["Actual Rows"] + scan.get("Rows Removed by Filter", 0) <= 1
    root_scan = root_cap_plan[0]["Plan"]
    assert root_scan["Index Name"] == "agent_runs_pkey"
    assert root_scan["Actual Rows"] == 1
    assert counts["after"]["SELECT"] > counts["before"]["SELECT"]
    print(
        "T5C_QUERY_EVIDENCE="
        + json.dumps({"counts": counts, "plan": plan, "root_cap_plan": root_cap_plan}, default=str)
    )


async def test_ten_runs_pass_and_replay_is_free(store):
    root, call = await managed_parent(store)
    for i in range(8):
        await member(store, root, name=f"worker{i}", completed=True)
    request = bind_request(root, call)
    created = await store.bind_managed_child(request)
    replay = await store.bind_managed_child(request)
    assert created.outcome == "created"
    assert replay.outcome == "existing"
    count = getattr(store, "get_managed_run_count", None)
    assert callable(count), "managed count capability missing"
    assert await count(root, limit=128, tenant_id="t1") == 10


async def test_replay_at_full_cap_is_free(store):
    root, call = await managed_parent(store)
    await cap_root(store, root, 2)
    request = bind_request(root, call)
    assert (await store.bind_managed_child(request)).outcome == "created"
    assert (await store.bind_managed_child(request)).outcome == "existing"
    assert await store.get_managed_run_count(root, limit=128, tenant_id="t1") == 2


@pytest.mark.parametrize("round_number", range(3))
async def test_pg_different_parents_race_last_slot(store, engine, round_number):
    if engine.dialect.name != "postgresql":
        pytest.skip("independent pools require Postgres")
    root, _ = await managed_parent(store)
    parents = [await member(store, root, name=f"p{i}") for i in range(8)]
    await cap_root(store, root, 10)
    engines = [create_async_engine(engine.url) for _ in parents]
    requests = [
        bind_request(parent, call, root=root, path=(f"p{i}", "reviewer"))
        for i, (parent, call) in enumerate(parents)
    ]
    before = [await snapshot(engine, parent) for parent, _ in parents]
    try:
        results = await asyncio.gather(
            *(
                store_for_engine(pool).bind_managed_child(request)
                for pool, request in zip(engines, requests, strict=True)
            ),
            return_exceptions=True,
        )
        assert sum(getattr(result, "outcome", None) == "created" for result in results) == 1
        refused = [result for result in results if isinstance(result, ManagedStateError)]
        assert len(refused) == 7
        assert all(error.code == "managed_run_limit" for error in refused)
        for i, result in enumerate(results):
            parent = parents[i][0]
            saved = await snapshot(engine, parent)
            if isinstance(result, ManagedStateError):
                assert saved == before[i]
            else:
                assert saved["epoch"] == 1
                assert len(saved["children"]) == 1
        async with AsyncSession(engine) as session:
            assert await session.scalar(select(func.count()).select_from(AgentRun)) == 10
            assert await session.scalar(select(func.count()).select_from(ToolCallRecord)) == 1
    finally:
        for pool in engines:
            await pool.dispose()
