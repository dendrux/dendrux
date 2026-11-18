"""0.8 fan-out store (T3): bind races, same-root admission, tenants and costs.

Races (spec §16.2 R1, R7, R8) run on Postgres with independent engines and deterministic
schedules: one binder is held right after it takes the root lock, the other is observed blocked
on that lock in ``pg_stat_activity`` before the first is released. SQLite runs the
single-connection cases (discovery, the cut at child k, statement counts).
"""

from __future__ import annotations

import asyncio
import json
from contextlib import AsyncExitStack
from dataclasses import fields, replace
from typing import Any

import pytest
from sqlalchemy import event, func, select, text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from dendrux._lifecycle import StoredEventType
from dendrux.db.models import AgentRun, ReactTrace, RunEvent, ToolCallRecord
from dendrux.errors import ManagedStateError, StoreCapabilityError
from dendrux.runtime._capabilities import fanout_support
from dendrux.runtime._managed_fanout_store import BindManagedChildrenRequest, ManagedChildSpec
from dendrux.runtime.state import MANAGED_HEADER_KEY, EventWrite, LoopStatePatch, ManagedHeader
from dendrux.runtime.stores._factory import store_for_engine
from dendrux.runtime.stores.sql.sqlite import SQLiteStateStore
from dendrux.types import PauseState, ToolCall, UsageStats, generate_ulid
from tests.integration.test_managed_bind import bind_request, managed_parent


class _EncodedWait(PauseState):
    """A parent wait whose stored JSON is given verbatim (the frozen spec §4.1 v4 shape)."""

    encoded: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(self.encoded))


def fanout_wait(bindings: list[tuple[str, str]]) -> PauseState:
    base = PauseState(
        agent_name="lead",
        pending_tool_calls=[ToolCall(name="reviewer", params={}, id=c) for c, _ in bindings],
        history=[],
        steps=[],
        iteration=1,
        trace_order_offset=4,
        usage=UsageStats(input_tokens=3, output_tokens=2, total_tokens=5),
        pending_targets={c: "agent" for c, _ in bindings},
        pause_reason="managed_wait",
        pause_id=generate_ulid(),
        schema_version=2,
        loop_state_ref=True,
    )
    data = base.to_dict()
    data["schema_version"] = 4
    data["managed_fanout"] = {
        "v": 1,
        "n": len(bindings),
        "bindings": [{"tool_call_id": c, "child_run_id": ch} for c, ch in bindings],
    }
    wait = _EncodedWait(**{f.name: getattr(base, f.name) for f in fields(PauseState)})
    wait.encoded = data
    return wait


async def fanout_parent(
    store: Any, n: int, *, tenant: str | None = "t1", cap: int | None = None
) -> tuple[str, list[str]]:
    """A running managed root at claim epoch 0 with ``n`` pending managed calls."""
    root, calls = generate_ulid(), [generate_ulid() for _ in range(n)]
    await store.create_run(root, "lead", tenant_id=tenant, input_data={"input": "go"})
    state: dict[str, Any] = {
        "v": 1,
        "loop": "react",
        "phase": "dispatch",
        "pending": [{"call_id": c, "tool": "reviewer", "target": "agent"} for c in calls],
        MANAGED_HEADER_KEY: ManagedHeader(root_run_id=root, path=()).to_dict(),
    }
    if cap is not None:
        state["managed_limits"] = {"max_managed_runs": cap}
    await store.checkpoint_step(root, claim_epoch=0, loop_state=state, pii_mapping=None, trace=None)
    return root, calls


async def nest(store: Any, child: str, n: int) -> list[str]:
    """Give a bound running child ``n`` pending managed calls of its own."""
    state = await store.get_loop_state(child)
    calls = [generate_ulid() for _ in range(n)]
    state["pending"] = [{"call_id": c, "tool": "reviewer", "target": "agent"} for c in calls]
    await store.checkpoint_step(
        child, claim_epoch=0, loop_state=state, pii_mapping=None, trace=None
    )
    return calls


def batch(
    parent: str,
    calls: list[str],
    *,
    tenant: str | None = "t1",
    root: str | None = None,
    path: tuple[str, ...] = (),
) -> BindManagedChildrenRequest:
    children = tuple(
        ManagedChildSpec(
            tool_call_id=call,
            row_id=generate_ulid(),
            provider_tool_call_id=None,
            tool_name="reviewer",
            params={"task": "SENTINEL-fanout-brief"},
            child_run_id=generate_ulid(),
            child_agent_name="reviewer",
            child_path=(*path, "reviewer"),
            child_input_data={"input": "SENTINEL-fanout-brief"},
            child_loop_state={"v": 1, "loop": "single_call"},
            intent_event=EventWrite(
                generate_ulid(),
                StoredEventType.TOOL_STARTED.value,
                1,
                {"call_id": call},
                call,
                None,
            ),
        )
        for call in calls
    )
    return BindManagedChildrenRequest(
        tenant_id=tenant,
        root_run_id=root or parent,
        parent_run_id=parent,
        parent_claim_epoch=0,
        iteration_index=1,
        parent_wait=fanout_wait([(c.tool_call_id, c.child_run_id) for c in children]),
        parent_patch=LoopStatePatch(set_fields={"iteration": 1}),
        children=children,
        max_parallel=32,
    )


async def counts(engine: Any, parent: str) -> dict[str, int]:
    """Rows a bind writes under ``parent``, read on an independent session."""
    async with AsyncSession(engine) as session:
        children = select(AgentRun.id).where(AgentRun.parent_run_id == parent)
        return {
            "children": await session.scalar(select(func.count()).select_from(children.subquery())),
            "intents": await session.scalar(
                select(func.count()).where(ToolCallRecord.agent_run_id == parent)
            ),
            "events": await session.scalar(
                select(func.count()).where(RunEvent.agent_run_id == parent)
            ),
            "seeds": await session.scalar(
                select(func.count()).where(ReactTrace.agent_run_id.in_(children))
            ),
            "epoch": await session.scalar(
                select(AgentRun.claim_epoch).where(AgentRun.id == parent)
            ),
        }


def _postgres_only(engine: Any) -> None:
    if engine.dialect.name != "postgresql":
        pytest.skip("independent row-lock races require Postgres; SQLite shares a connection")


async def _blocked_on_lock(engine: Any, application_name: str) -> None:
    async def observed() -> None:
        async with engine.connect() as connection:
            while not await connection.scalar(
                text(
                    "SELECT EXISTS (SELECT 1 FROM pg_stat_activity "
                    "WHERE datname = current_database() AND application_name = :name "
                    "AND wait_event_type = 'Lock')"
                ),
                {"name": application_name},
            ):
                await connection.rollback()  # refresh pg_stat_activity's snapshot
                await asyncio.sleep(0.01)

    await asyncio.wait_for(observed(), timeout=10)


async def _pools(stack: AsyncExitStack, engine: Any, names: list[str]) -> list[Any]:
    """One single-connection pool and store per name (``application_name`` = name)."""
    stores = []
    for name in names:
        pool = create_async_engine(
            engine.url,
            pool_size=1,
            max_overflow=0,
            connect_args={"server_settings": {"application_name": name}},
        )
        stack.push_async_callback(pool.dispose)
        stores.append(store_for_engine(pool))
    return stores


def _hold_after_root_lock(
    monkeypatch: Any, store: Any, locked: asyncio.Event, release: asyncio.Event
) -> None:
    original = store._lock_managed_root

    async def held(session: Any, request: Any) -> None:
        await original(session, request)
        locked.set()
        await release.wait()

    monkeypatch.setattr(store, "_lock_managed_root", held)


# ---------------------------------------------------------------------------------------------
# K10 on the real store: the declaration is the only discovery
# ---------------------------------------------------------------------------------------------


class _NoFanoutStore(SQLiteStateStore):
    capabilities = replace(SQLiteStateStore.capabilities, managed_fanout=False)


async def test_sql_store_declares_fanout_and_undeclared_refuses_before_writes(engine):
    store = store_for_engine(engine)
    assert store.capabilities.managed_fanout is True
    assert fanout_support(store).capable is True
    if engine.dialect.name != "sqlite":
        return
    narrowed = _NoFanoutStore(engine)
    assert fanout_support(narrowed).capable is False
    root, calls = await fanout_parent(narrowed, 2)
    with pytest.raises(StoreCapabilityError) as refused:
        await narrowed.bind_managed_children(batch(root, calls))
    assert refused.value.capability == "managed_fanout"
    assert await counts(engine, root) == {
        "children": 0,
        "intents": 0,
        "events": 0,
        "seeds": 0,
        "epoch": 0,
    }
    # N = 1 keeps the unchanged 0.7 bind.
    single, call = await managed_parent(narrowed)
    assert (await narrowed.bind_managed_child(bind_request(single, call))).outcome == "created"


# ---------------------------------------------------------------------------------------------
# K2 on the real store: a failure at child k writes nothing
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("k", [0, 1, 7])
async def test_bind_failure_at_child_k_leaves_no_rows(engine, monkeypatch, k):
    store = store_for_engine(engine)
    root, calls = await fanout_parent(store, 8)
    request = batch(root, calls)
    original, seen = store._bind_child, []

    async def failing(session: Any, call: Any, parent: Any, state: Any) -> None:
        seen.append(call.child_run_id)
        if len(seen) == k + 1:
            raise RuntimeError("cut at child k")
        await original(session, call, parent, state)

    monkeypatch.setattr(store, "_bind_child", failing)
    with pytest.raises(RuntimeError):
        await store.bind_managed_children(request)
    assert await counts(engine, root) == {
        "children": 0,
        "intents": 0,
        "events": 0,
        "seeds": 0,
        "epoch": 0,
    }
    monkeypatch.undo()
    assert (await store.bind_managed_children(request)).outcome == "created"
    assert (await counts(engine, root))["children"] == 8


# ---------------------------------------------------------------------------------------------
# R1: bind contenders (live vs recovery), N = 2, 8, 32
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("first", ["live", "recovery"])
@pytest.mark.parametrize("n", [2, 8, 32])
async def test_pg_bind_contenders_have_one_creator(engine, monkeypatch, n, first):
    _postgres_only(engine)
    store = store_for_engine(engine)
    root, calls = await fanout_parent(store, n)
    live, recovery = batch(root, calls), batch(root, calls)
    names = [f"t3-r1-live-{root}", f"t3-r1-recovery-{root}"]
    locked, release = asyncio.Event(), asyncio.Event()
    async with AsyncExitStack() as stack:
        stores = await _pools(stack, engine, names)
        held, waiter = (0, 1) if first == "live" else (1, 0)
        _hold_after_root_lock(monkeypatch, stores[held], locked, release)
        requests = (live, recovery)
        async with asyncio.timeout(60):
            leader = asyncio.create_task(stores[held].bind_managed_children(requests[held]))
            await locked.wait()
            follower = asyncio.create_task(stores[waiter].bind_managed_children(requests[waiter]))
            await _blocked_on_lock(engine, names[waiter])
            release.set()
            won, lost = await leader, await follower
    assert (won.outcome, lost.outcome) == ("created", "existing")
    assert all(b.child_claim is not None for b in won.children)
    assert all(b.child_claim is None for b in lost.children)
    assert [b.child_run_id for b in lost.children] == [b.child_run_id for b in won.children]
    assert lost.parent_wait_id == won.parent_wait_id
    assert await counts(engine, root) == {
        "children": n,
        "intents": n,
        "events": n,
        "seeds": 0,
        "epoch": 1,
    }
    for child in requests[waiter].children:
        assert await store.get_run(child.child_run_id) is None
    pause = await store.get_pause_state(root)
    assert (pause["schema_version"], pause["managed_fanout"]["n"]) == (4, n)


# ---------------------------------------------------------------------------------------------
# R7: two parents under one root, room for one batch
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("schedule", ["a_first", "b_first", "together"])
async def test_pg_same_root_binders_admit_exactly_one_batch(engine, monkeypatch, schedule):
    _postgres_only(engine)
    store = store_for_engine(engine)
    root, calls = await fanout_parent(store, 2, cap=3 + 4)  # root, A, B and one batch of 4
    top = batch(root, calls)
    await store.bind_managed_children(top)
    parents = [c.child_run_id for c in top.children]
    requests = [batch(p, await nest(store, p, 4), root=root, path=("reviewer",)) for p in parents]
    names = [f"t3-r7-a-{root}", f"t3-r7-b-{root}"]
    async with AsyncExitStack() as stack:
        stores = await _pools(stack, engine, names)
        async with asyncio.timeout(60):
            if schedule == "together":
                results = await asyncio.gather(
                    *(s.bind_managed_children(r) for s, r in zip(stores, requests, strict=True)),
                    return_exceptions=True,
                )
            else:
                held = 0 if schedule == "a_first" else 1
                locked, release = asyncio.Event(), asyncio.Event()
                _hold_after_root_lock(monkeypatch, stores[held], locked, release)
                leader = asyncio.create_task(stores[held].bind_managed_children(requests[held]))
                await locked.wait()
                other = 1 - held
                follower = asyncio.create_task(stores[other].bind_managed_children(requests[other]))
                await _blocked_on_lock(engine, names[other])
                release.set()
                pair = await asyncio.gather(leader, follower, return_exceptions=True)
                results = [pair[0], pair[1]] if held == 0 else [pair[1], pair[0]]
    won = [i for i, r in enumerate(results) if not isinstance(r, BaseException)]
    lost = [i for i, r in enumerate(results) if isinstance(r, BaseException)]
    assert len(won) == 1 and len(lost) == 1, results
    if schedule != "together":
        assert won == [0 if schedule == "a_first" else 1]
    error = results[lost[0]]
    assert isinstance(error, ManagedStateError) and error.code == "managed_run_limit"
    assert (await counts(engine, parents[won[0]]))["children"] == 4
    assert await counts(engine, parents[lost[0]]) == {
        "children": 0,
        "intents": 0,
        "events": 0,
        "seeds": 0,
        "epoch": 0,
    }


# ---------------------------------------------------------------------------------------------
# R8: two tenants fan out at once on one pool
# ---------------------------------------------------------------------------------------------


async def test_pg_two_tenants_fan_out_at_once_without_crossing(engine):
    _postgres_only(engine)
    async with AsyncExitStack() as stack:
        [shared] = await _pools(stack, engine, ["t3-r8-shared"])
        roots = [await fanout_parent(shared, 8, tenant=t, cap=9) for t in ("t1", "t2")]
        requests = [
            batch(root, calls, tenant=t)
            for (root, calls), t in zip(roots, ("t1", "t2"), strict=True)
        ]
        async with asyncio.timeout(60):
            results = await asyncio.gather(*(shared.bind_managed_children(r) for r in requests))
        assert [r.outcome for r in results] == ["created", "created"]
        for request, other in ((requests[0], "t2"), (requests[1], "t1")):
            ids = [c.child_run_id for c in request.children]
            own = await shared.get_managed_children(
                request.parent_run_id, ids, tenant_id=request.tenant_id
            )
            assert [r.run_id for r in own] == ids
            assert {r.tenant_id for r in own} == {request.tenant_id}
            assert (
                await shared.get_managed_children(request.parent_run_id, ids, tenant_id=other) == []
            )
            replay = replace(request, tenant_id=other)
            with pytest.raises(Exception) as hidden:
                await shared.bind_managed_children(replay)
            assert type(hidden.value).__name__ == "RunNotFoundError"


# ---------------------------------------------------------------------------------------------
# Scale evidence: statements per bind and per read
# ---------------------------------------------------------------------------------------------


class _Statements:
    def __init__(self, engine: Any) -> None:
        self.engine = engine.sync_engine
        self.seen: list[str] = []

    def __enter__(self) -> _Statements:
        event.listen(self.engine, "before_cursor_execute", self._record)
        return self

    def __exit__(self, *exc: object) -> None:
        event.remove(self.engine, "before_cursor_execute", self._record)

    def _record(self, conn: Any, cursor: Any, statement: str, *rest: Any) -> None:
        self.seen.append(statement.split(None, 1)[0].upper())


async def _bind_cost(store: Any, engine: Any, n: int) -> tuple[int, dict[str, int]]:
    root, calls = await fanout_parent(store, n)
    request = batch(root, calls)
    with _Statements(engine) as recorded:
        await store.bind_managed_children(request)
    kinds: dict[str, int] = {}
    for kind in recorded.seen:
        kinds[kind] = kinds.get(kind, 0) + 1
    return len(recorded.seen), kinds


async def test_bind_and_status_read_statement_counts(engine, capsys):
    store = store_for_engine(engine)
    small, small_kinds = await _bind_cost(store, engine, 2)
    large, large_kinds = await _bind_cost(store, engine, 8)
    per_child = (large - small) / 6
    # Linear in N with a constant per-child cost: the intent, child, event and seed inserts
    # (and their read-back proofs), never a per-child read of other state.
    assert per_child == int(per_child) and per_child <= 4, (small_kinds, large_kinds)
    assert large_kinds.get("SELECT", 0) == small_kinds.get("SELECT", 0)
    root, calls = await fanout_parent(store, 8)
    request = batch(root, calls)
    await store.bind_managed_children(request)
    ids = [c.child_run_id for c in request.children]
    with _Statements(engine) as recorded:
        rows = await store.get_managed_children(root, ids, tenant_id="t1")
    assert len(rows) == 8
    assert recorded.seen == ["SELECT"], recorded.seen
    with capsys.disabled():
        print(
            f"\n[{engine.dialect.name}] bind N=2: {small} {small_kinds}; N=8: {large} "
            f"{large_kinds}; per child {per_child:g}; status read N=8: {len(recorded.seen)}"
        )


async def _populate(engine: Any, runs: int, *, parent: str | None = None) -> None:
    """History: ``runs`` runs (children of ``parent`` when given), each with one intent row."""
    from sqlalchemy import insert

    ids = [generate_ulid() for _ in range(runs)]
    async with AsyncSession(engine) as session:
        await session.execute(
            insert(AgentRun),
            [
                {
                    "id": i,
                    "agent_name": "history",
                    "status": "success",
                    "tenant_id": "t1",
                    "parent_run_id": parent,
                }
                for i in ids
            ],
        )
        await session.execute(
            insert(ToolCallRecord),
            [
                {
                    "id": generate_ulid(),
                    "agent_run_id": i,
                    "tool_call_id": f"c{n}",
                    "tool_name": "x",
                }
                for n, i in enumerate(ids)
            ],
        )
        await session.commit()
        await session.execute(text("ANALYZE agent_runs"))
        await session.execute(text("ANALYZE tool_calls"))
        await session.commit()


class _Captured(_Statements):
    """Also keeps each statement's text and driver parameters."""

    def __init__(self, engine: Any) -> None:
        super().__init__(engine)
        self.sql: list[tuple[str, Any]] = []

    def _record(self, conn: Any, cursor: Any, statement: str, parameters: Any, *rest: Any) -> None:
        super()._record(conn, cursor, statement, parameters, *rest)
        self.sql.append((statement, parameters))


async def test_pg_new_queries_use_primary_key_and_unique_index(engine, capsys):
    _postgres_only(engine)
    store = store_for_engine(engine)
    await _populate(engine, 20000)
    root, calls = await fanout_parent(store, 8)
    request = batch(root, calls)
    await store.bind_managed_children(request)
    await _populate(engine, 5000, parent=root)  # a parent with a long (legacy) child history
    ids = [c.child_run_id for c in request.children]
    with _Captured(engine) as status:
        await store.get_managed_children(root, ids, tenant_id="t1")
    with _Captured(engine) as replay:
        assert (await store.bind_managed_children(request)).outcome == "existing"
    [status_sql] = status.sql
    readback = next(item for item in replay.sql if "FROM tool_calls" in item[0])
    children = next(
        item for item in replay.sql if "agent_runs.id IN" in item[0] and "FOR UPDATE" not in item[0]
    )
    plans = {}
    async with engine.connect() as connection:
        for name, (sql, parameters) in (
            ("status read", status_sql),
            ("intent read-back", readback),
            ("bound-children read-back", children),
        ):
            result = await connection.exec_driver_sql(f"EXPLAIN {sql}", parameters)
            plans[name] = "\n".join(row[0] for row in result)
    assert "agent_runs_pkey" in plans["status read"], plans["status read"]
    assert "agent_runs_pkey" in plans["bound-children read-back"], plans
    assert "tool_calls" in plans["intent read-back"] and "Index" in plans["intent read-back"], plans
    with capsys.disabled():
        for name, plan in plans.items():
            print(f"\nEXPLAIN {name} (20,000 other runs, 5,000 other children):\n{plan}")
