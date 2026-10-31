"""Frontier cancellation closes admission before scanning each subtree level."""

import asyncio
import json
import time
from contextlib import asynccontextmanager
from functools import wraps

import pytest
from sqlalchemy import event, insert, text, update

from dendrux import Agent
from dendrux.db.models import AgentRun
from dendrux.errors import RunCancelledError
from dendrux.runtime.stores._factory import store_for_engine
from dendrux.types import Budget, generate_ulid

pytestmark = pytest.mark.parametrize("engine", ["sqlite", "sqlite_file", "postgres"], indirect=True)


async def seed(store, links, *, tenant="t", status="paused"):
    ids = {name: generate_ulid() for name, _ in links}
    async with store._session_factory() as session:
        await session.execute(
            insert(AgentRun),
            [
                dict(id=rid, agent_name="cascade", status=status, tenant_id=tenant)
                for rid in ids.values()
            ],
        )
        for name, parent in links:
            if parent is not None:
                await session.execute(
                    update(AgentRun)
                    .where(AgentRun.id == ids[name])
                    .values(parent_run_id=ids[parent])
                )
        await session.commit()
    return ids


def observe_frontiers(store, monkeypatch):
    marks, scans, finalized = [], [], []
    mark, scan, finalize = (
        store.mark_cancel_requested,
        store.get_children_page,
        store.finalize_with_event,
    )

    async def marking(ids, **kwargs):
        marks.append(tuple(ids))
        return await mark(ids, **kwargs)

    async def scanning(ids, **kwargs):
        assert tuple(ids) in marks, "frontier scanned before durable mark"
        for rid in ids:
            row = await store.get_run(rid)
            assert row.cancel_requested or row.status in {
                "success",
                "error",
                "cancelled",
                "max_iterations",
            }
        assert not finalized, "finalization began before marking finished"
        result = await scan(ids, **kwargs)
        scans.append((tuple(ids), kwargs["after"], len(result)))
        assert kwargs["limit"] == 500
        return result

    @wraps(finalize)
    async def finalizing(rid, **kwargs):
        finalized.append(rid)
        return await finalize(rid, **kwargs)

    monkeypatch.setattr(store, "mark_cancel_requested", marking)
    monkeypatch.setattr(store, "get_children_page", scanning)
    monkeypatch.setattr(store, "finalize_with_event", finalizing)
    return marks, scans, finalized


@pytest.mark.parametrize(
    "shape", ["star", "deep", "cycle", "terminal", "terminal_root", "mid_tree", "foreign"]
)
@pytest.mark.parametrize("scope", ["t", None])
async def test_frontiers_mark_before_scan_and_finalize_deepest(store, monkeypatch, shape, scope):
    links = [("r", None), ("c", "r"), ("g", "c"), ("s", "r")]
    if shape == "star":
        links = [("r", None), *((f"c{i}", "r") for i in range(501))]
    elif shape == "deep":
        links = [("r", None), *((f"c{i}", "r" if i == 0 else f"c{i - 1}") for i in range(32))]
    if shape == "foreign":
        links.append(("blocked", "s"))
    ids = await seed(store, links, tenant=scope)
    if shape == "cycle":
        async with store._session_factory() as session:
            await session.execute(
                update(AgentRun).where(AgentRun.id == ids["r"]).values(parent_run_id=ids["g"])
            )
            await session.commit()
    if shape in {"terminal", "terminal_root"}:
        async with store._session_factory() as session:
            await session.execute(
                update(AgentRun)
                .where(AgentRun.id == ids["r" if shape == "terminal_root" else "c"])
                .values(status="success")
            )
            await session.commit()
    if shape == "foreign":
        async with store._session_factory() as session:
            await session.execute(
                update(AgentRun).where(AgentRun.id == ids["s"]).values(tenant_id="foreign")
            )
            await session.commit()
    marks, scans, finalized = observe_frontiers(store, monkeypatch)
    agent = Agent(prompt="p", state_store=store, strict_tenancy=False)
    root = ids["c"] if shape == "mid_tree" else ids["r"]
    # Omitted tenant scope still confines traversal to the actual root's tenant.
    result = await agent.cancel_run(root, cascade=True)
    expected = set(ids.values())
    if shape == "mid_tree":
        expected = {ids["c"], ids["g"]}
    elif shape == "foreign":
        expected.difference_update({ids["s"], ids["blocked"]})
    assert result.meta["cancel_requested"] == (await store.get_run(root)).cancel_requested
    assert marks, "cascade did not mark frontiers"
    assert set(rid for frontier in marks for rid in frontier) == expected
    assert len([rid for frontier in marks for rid in frontier]) == len(expected)
    if shape == "terminal_root":
        assert root not in finalized
        assert result.status.value == "success"
        assert result.meta["cancel_requested"] is False
    else:
        assert finalized[-1] == root
    if shape == "terminal":
        assert ids["c"] not in finalized
        terminal = await store.get_run(ids["c"])
        assert terminal.status == "success"
        assert terminal.cancel_requested is False
        assert finalized.index(ids["g"]) < finalized.index(root)
    elif "g" in ids and shape != "star":
        assert finalized.index(ids["g"]) < finalized.index(ids["c"])
    assert result.meta["cascade_count"] == len(expected)
    assert len({o["run_id"] for o in result.meta["cascade"]}) == len(result.meta["cascade"])
    assert result.meta["cascade_truncated"] == (len(expected) > 1000)
    for rid in set(ids.values()) - expected:
        row = await store.get_run(rid)
        assert row.status == "paused" and not row.cancel_requested
    if shape == "star":
        assert [size for frontier, after, size in scans if frontier == (root,)] == [500, 1]
    await agent.close()


async def test_cascade_bound_marks_retained_frontier_and_reports_truncation(
    store, monkeypatch, caplog
):
    ids = await seed(store, [("r", None), ("c", "r"), ("g", "c"), ("gg", "g")])
    marks, scans, finalized = observe_frontiers(store, monkeypatch)
    agent = Agent(
        prompt="p",
        state_store=store,
        budget=Budget(max_tokens=1, max_subtree_nodes=2),
        strict_tenancy=False,
    )
    result = await agent.cancel_run(ids["r"], cascade=True, tenant_id="t")
    assert result.meta["cascade_truncated"] is True
    assert result.meta["cascade_count"] == 2
    assert set(rid for frontier in marks for rid in frontier) == {ids["r"], ids["c"]}
    assert finalized == [ids["c"], ids["r"]]
    assert (await store.get_run(ids["g"])).status == "paused"
    assert any(
        getattr(r, "code", None) == "cascade_truncated" and r.limit == 2 for r in caplog.records
    )
    await agent.close()


async def test_grandchild_after_parent_scan_is_refused(store, monkeypatch):
    ids = await seed(store, [("r", None), ("c", "r")], status="running")
    original = store.get_children_page
    attempts = []

    async def scanning(frontier, **kwargs):
        page = await original(frontier, **kwargs)
        if ids["c"] in frontier:
            try:
                await store.create_run(
                    generate_ulid(), "late", parent_run_id=ids["c"], tenant_id="t"
                )
            except RunCancelledError:
                attempts.append("refused")
            else:
                attempts.append("admitted")
        return page

    monkeypatch.setattr(store, "get_children_page", scanning)
    agent = Agent(prompt="p", state_store=store, strict_tenancy=False)
    await agent.cancel_run(ids["r"], cascade=True, tenant_id="t")
    assert attempts == ["refused"]
    await agent.close()


@pytest.mark.parametrize("size", [1, 1001])
async def test_frontier_store_statement_budget(store, engine, size, record_property):
    root = generate_ulid()
    await store.create_run(root, "parent", tenant_id="t")
    await store.create_run(generate_ulid(), "child", parent_run_id=root, tenant_id="t")
    frontier = [root, *(generate_ulid() for _ in range(size - 1))]
    statements = []

    def capture(conn, cursor, statement, parameters, context, many):
        statements.append(statement)

    event.listen(engine.sync_engine, "before_cursor_execute", capture)
    try:
        assert await store.mark_cancel_requested(frontier, tenant_id="t") == 1
        assert (
            len(await store.get_children_page(frontier, after=None, limit=500, tenant_id="t")) == 1
        )
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", capture)
    record_property("frontier_statements", json.dumps(statements))
    print(f"{engine.dialect.name} frontier={size}: mark+scan statements={len(statements)}")
    assert len(statements) == 2
    if engine.dialect.name == "postgresql":
        assert "ANY" in statements[0]
        assert "unnest" in statements[1] and "LATERAL" in statements[1]
        assert all("IS NOT DISTINCT FROM" in sql for sql in statements)


async def test_creator_checked_parent_before_mark_is_seen_after_commit(
    store, engine, monkeypatch, record_property
):
    if engine.dialect.name != "postgresql":
        pytest.skip("Postgres FOR SHARE and UPDATE row-lock serialization")
    root, child = generate_ulid(), generate_ulid()
    await store.create_run(root, "parent", tenant_id="t")
    creator = store_for_engine(engine)
    checked, release, marking = asyncio.Event(), asyncio.Event(), asyncio.Event()
    original_factory = creator._session_factory
    original_mark = store.mark_cancel_requested
    elapsed = []

    @asynccontextmanager
    async def held_commit():
        async with original_factory() as session:
            original_commit = session.commit

            async def commit():
                # create_run has read the parent successfully and holds FOR SHARE.
                checked.set()
                await release.wait()
                await original_commit()

            session.commit = commit
            yield session

    async def mark(frontier, **kwargs):
        marking.set()
        started = time.monotonic()
        result = await original_mark(frontier, **kwargs)
        elapsed.append(time.monotonic() - started)
        return result

    monkeypatch.setattr(creator, "_session_factory", held_commit)
    monkeypatch.setattr(store, "mark_cancel_requested", mark)
    agent = Agent(prompt="p", state_store=store, strict_tenancy=False)
    creation = asyncio.create_task(
        creator.create_run(child, "child", parent_run_id=root, tenant_id="t")
    )
    await asyncio.wait_for(checked.wait(), 5)
    cancellation = asyncio.create_task(agent.cancel_run(root, cascade=True, tenant_id="t"))
    try:
        # Prove the database is waiting on a lock, without an elapsed-time threshold.
        for _ in range(500):
            async with engine.connect() as conn:
                waiting = await conn.scalar(
                    text(
                        "SELECT count(*) FROM pg_stat_activity "
                        "WHERE datname = current_database() AND wait_event_type = 'Lock' "
                        "AND query LIKE 'UPDATE agent_runs%'"
                    )
                )
            if waiting:
                break
            await asyncio.sleep(0.01)
        assert waiting, "cascade did not wait for the in-flight creator's parent lock"
        assert not cancellation.done()
        release.set()
        await asyncio.wait_for(creation, 5)
        result = await asyncio.wait_for(cancellation, 5)
        assert marking.is_set(), "cascade did not use the frontier mark"
        assert {o["run_id"] for o in result.meta["cascade"]} == {root, child}
        assert (await store.get_run(child)).cancel_requested
        record_property("lock_wait_seconds", elapsed[0])
        print(f"Postgres creator lock wait: {elapsed[0]:.6f}s")
    finally:
        release.set()
        await asyncio.gather(creation, cancellation, return_exceptions=True)
        await agent.close()


async def test_parallel_cascades_finalize_each_run_once(store, engine):
    if engine.dialect.name != "postgresql":
        pytest.skip("Real independent Postgres transactions")
    ids = await seed(store, [("r", None), ("c", "r"), ("g", "c")])
    agents = [
        Agent(prompt="p", state_store=store_for_engine(engine), strict_tenancy=False)
        for _ in range(8)
    ]
    marks = []
    for agent in agents:
        original = agent._state_store.mark_cancel_requested

        async def mark(frontier, *, tenant_id, original=original):
            marks.append(tuple(frontier))
            return await original(frontier, tenant_id=tenant_id)

        agent._state_store.mark_cancel_requested = mark
    try:
        results = await asyncio.gather(
            *(a.cancel_run(ids["r"], cascade=True, tenant_id="t") for a in agents)
        )
        assert all(r.status.value == "cancelled" for r in results)
        assert marks, "parallel cascades did not use frontier marking"
        for rid in ids.values():
            events = await store.get_run_events(rid)
            assert len([e for e in events if e.event_type == "run.cancelled"]) == 1
            assert (await store.get_run(rid)).status == "cancelled"
    finally:
        for agent in agents:
            await agent.close()


async def test_sqlite_creator_checked_parent_is_seen_after_serialized_commit(
    store, engine, monkeypatch
):
    if engine.dialect.name != "sqlite":
        pytest.skip("Postgres parent-row locking is exercised separately")
    root, child = generate_ulid(), generate_ulid()
    await store.create_run(root, "parent", tenant_id="t")
    creator = store_for_engine(engine)
    original_factory = creator._session_factory
    original_mark = store.mark_cancel_requested
    checked, release, cancel_started = asyncio.Event(), asyncio.Event(), asyncio.Event()
    marks = []

    @asynccontextmanager
    async def held_commit():
        async with original_factory() as session:
            original_commit = session.commit

            async def commit():
                checked.set()
                await release.wait()
                await original_commit()

            session.commit = commit
            yield session

    async def mark(frontier, **kwargs):
        marks.append(tuple(frontier))
        return await original_mark(frontier, **kwargs)

    monkeypatch.setattr(creator, "_session_factory", held_commit)
    monkeypatch.setattr(store, "mark_cancel_requested", mark)
    agent = Agent(prompt="p", state_store=store, strict_tenancy=False)
    creation = asyncio.create_task(
        creator.create_run(child, "child", parent_run_id=root, tenant_id="t")
    )
    await asyncio.wait_for(checked.wait(), 5)

    async def cancel():
        cancel_started.set()
        return await agent.cancel_run(root, cascade=True, tenant_id="t")

    cancellation = asyncio.create_task(cancel())
    try:
        await asyncio.wait_for(cancel_started.wait(), 5)
        await asyncio.sleep(0)
        assert not cancellation.done()
        release.set()
        await asyncio.wait_for(creation, 5)
        result = await asyncio.wait_for(cancellation, 5)
        assert {o["run_id"] for o in result.meta["cascade"]} == {root, child}
        assert (await store.get_run(child)).cancel_requested
        assert marks == [(root,), (child,)]
    finally:
        release.set()
        await asyncio.gather(creation, cancellation, return_exceptions=True)
        await agent.close()
