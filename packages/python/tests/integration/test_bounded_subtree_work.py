"""Large delegation trees have bounded accounting and paged cancellation."""

import pytest
from sqlalchemy import event, insert

from dendrux import Agent
from dendrux.db.models import AgentRun, LLMInteraction
from dendrux.errors import BudgetConfigurationError
from dendrux.loops._budget import BudgetTracker
from dendrux.loops.base import BaseRecorder
from dendrux.types import Budget, UsageStats


async def seed_tree(store):
    async with store._session_factory() as session:
        await session.execute(insert(AgentRun), [dict(id="root", agent_name="a", status="running")])
        await session.execute(
            insert(AgentRun),
            [
                dict(id=f"c{i:025d}", parent_run_id="root", agent_name="a", status="running")
                for i in range(2499)
            ],
        )
        await session.execute(
            insert(LLMInteraction),
            [
                dict(
                    id=f"l{i:025d}",
                    agent_run_id=f"c{i:025d}",
                    iteration_index=0,
                    input_tokens=1,
                    output_tokens=0,
                )
                for i in range(2499)
            ],
        )
        await session.commit()


async def test_budget_rollup_fails_closed_at_traversal_bound(store, engine, caplog):
    await seed_tree(store)
    queries = []

    def capture(conn, cursor, statement, parameters, context, executemany):
        if statement.lstrip().upper().startswith("SELECT"):
            queries.append(statement)

    event.listen(engine.sync_engine, "before_cursor_execute", capture)
    try:
        assert (await store.subtree_usage("root")).input_tokens == 2499
        print(f"{engine.dialect.name}: budget full rollup SELECTs={len(queries)}")
        assert len(queries) <= 13
        assert all("LIMIT" in q for q in queries if "agent_runs" in q)
        queries.clear()
        with pytest.raises(BudgetConfigurationError) as caught:
            await store.subtree_usage("root", max_nodes=1000)
        assert caught.value.reason == "subtree_limit"
        print(f"{engine.dialect.name}: budget capped rollup SELECTs={len(queries)}")
        assert len(queries) <= 3
        assert any(
            getattr(r, "code", None) == "subtree_limit" and getattr(r, "run_id", None) == "root"
            for r in caplog.records
        )
        tracker = BudgetTracker(
            budget=Budget(max_tokens=100, max_subtree_nodes=1000),
            run_id="root",
            usage=UsageStats(),
            recorder=BaseRecorder(),
            notifier=None,
            warnings=[],
            state_store=store,
        )
        with pytest.raises(BudgetConfigurationError):
            await tracker.evaluate(0)
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", capture)


async def test_cascade_signals_pages_before_reading_next_page(store, engine, monkeypatch):
    await seed_tree(store)
    agent = Agent(prompt="p", state_store=store)
    pages = []
    signals = []
    from collections import Counter

    counts = Counter()

    def count_sql(conn, cursor, statement, parameters, context, executemany):
        counts[statement.lstrip().split()[0]] += 1

    event.listen(engine.sync_engine, "before_cursor_execute", count_sql)
    original = getattr(store, "get_descendant_page", None)
    assert original is not None, "cascade requires a bounded descendant read"

    async def page(*args, **kwargs):
        if pages:
            assert all(rid in signals for rid in pages[-1])
        rows = await original(*args, **kwargs)
        assert len(rows) <= 500
        pages.append(rows)
        return rows

    monkeypatch.setattr(store, "get_descendant_page", page)
    monkeypatch.setattr(store, "get_subtree", lambda *a, **kw: pytest.fail("full tree loaded"))
    monkeypatch.setattr(agent._task_manager, "cancel", lambda rid: signals.append(rid))
    try:
        result = await agent.cancel_run("root", cascade=True)
        assert len(signals) == 2500
        assert len(pages) == 5
        print(f"{engine.dialect.name}: cancellation pages={len(pages)} SQL={dict(counts)}")
        assert result.meta["cascade_count"] == 2500
        assert result.meta["cascade_truncated"] is True
        assert len(result.meta["cascade"]) <= 1000
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", count_sql)
        await agent.close()
