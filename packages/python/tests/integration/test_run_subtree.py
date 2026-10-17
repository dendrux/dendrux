"""RunStore.get_subtree, StateStore.subtree_usage, and ``before=`` / ``event_types=`` filters.

Runs across the backend matrix (SQLite always, Postgres when configured).
"""

from __future__ import annotations

import datetime as dt

import pytest
from sqlalchemy import update
from sqlalchemy.ext.asyncio import async_sessionmaker

from dendrux.db.models import LLMInteraction, RunEvent, ToolCallRecord
from dendrux.runtime.state import SQLAlchemyStateStore, UsageRollup
from dendrux.store import RunNotFoundError, RunStore, RunSubtree, RunUsage, SubtreeNode
from dendrux.types import UsageStats


@pytest.fixture
def internal_store(engine):
    return SQLAlchemyStateStore(engine)


@pytest.fixture
def store(engine):
    return RunStore.from_engine(engine)


async def _call(
    s: SQLAlchemyStateStore,
    run_id: str,
    *,
    i: int = 0,
    inp: int = 10,
    out: int = 5,
    cost: float | None = 0.01,
    cache_read: int | None = None,
    cache_write: int | None = None,
    reasoning: int | None = None,
) -> None:
    await s.save_llm_interaction(
        run_id,
        iteration_index=i,
        usage=UsageStats(
            input_tokens=inp,
            output_tokens=out,
            cost_usd=cost,
            cache_read_input_tokens=cache_read,
            cache_creation_input_tokens=cache_write,
            reasoning_tokens=reasoning,
        ),
    )


async def _tree(s: SQLAlchemyStateStore) -> None:
    """root -> (a -> (a1 -> a11), b)"""
    await s.create_run("root", "Root", tenant_id="t1")
    await s.create_run("a", "A", parent_run_id="root", delegation_level=1, tenant_id="t1")
    await s.create_run("b", "B", parent_run_id="root", delegation_level=1, tenant_id="t1")
    await s.create_run("a1", "A1", parent_run_id="a", delegation_level=2, tenant_id="t1")
    await s.create_run("a11", "A11", parent_run_id="a1", delegation_level=3, tenant_id="t1")


class TestGetSubtree:
    async def test_shape_depths_and_rollup(self, store, internal_store) -> None:
        await _tree(internal_store)
        await _call(internal_store, "root", inp=1, out=1, cost=0.5, cache_read=2, reasoning=3)
        await _call(internal_store, "a", inp=10, out=10, cost=0.25, cache_write=4)
        await _call(internal_store, "a", i=1, inp=10, out=10, cost=0.25)
        await _call(internal_store, "a11", inp=100, out=100, cost=1.0, cache_read=8, reasoning=7)

        tree = await store.get_subtree("root")

        assert isinstance(tree, RunSubtree)
        assert tree.root_run_id == "root"
        assert all(isinstance(n, SubtreeNode) for n in tree.nodes)
        assert tree.root.run.run_id == "root"
        by_id = {n.run.run_id: n for n in tree.nodes}
        assert {k: v.depth for k, v in by_id.items()} == {
            "root": 0,
            "a": 1,
            "b": 1,
            "a1": 2,
            "a11": 3,
        }
        assert [n.depth for n in tree.nodes] == sorted(n.depth for n in tree.nodes)

        assert isinstance(tree.usage, RunUsage)
        assert tree.usage.input_tokens == 121
        assert tree.usage.output_tokens == 121
        assert tree.usage.cache_read_tokens == 10
        assert tree.usage.cache_write_tokens == 4
        assert tree.usage.reasoning_tokens == 10
        assert tree.usage.cost_usd == pytest.approx(2.0)
        assert tree.usage.unpriced_calls == 0
        assert tree.usage.llm_calls == 4

        assert by_id["a"].usage.llm_calls == 2
        assert by_id["a"].usage.cost_usd == pytest.approx(0.5)
        assert by_id["b"].usage.llm_calls == 0
        assert by_id["b"].usage.cost_usd == 0.0
        assert sum(n.usage.input_tokens for n in tree.nodes) == tree.usage.input_tokens

    async def test_live_usage_ignores_finalized_totals_and_counts_running(
        self, store, internal_store
    ) -> None:
        await internal_store.create_run("p", "P")
        await internal_store.create_run("c", "C", parent_run_id="p", delegation_level=1)
        await _call(internal_store, "c", inp=50, out=7)
        # Child is still RUNNING: agent_runs.total_* is 0 but live usage counts.
        tree = await store.get_subtree("p")
        child = next(n for n in tree.nodes if n.run.run_id == "c")
        assert child.run.status == "running"
        assert child.run.total_input_tokens == 0
        assert child.usage.input_tokens == 50
        assert tree.usage.input_tokens == 50

        # Finalize with totals that disagree with llm_interactions: the
        # rollup still reflects what was actually recorded per call.
        await internal_store.finalize_run(
            "c", status="success", total_usage=UsageStats(input_tokens=999)
        )
        tree = await store.get_subtree("p")
        child = next(n for n in tree.nodes if n.run.run_id == "c")
        assert child.run.total_input_tokens == 999
        assert child.usage.input_tokens == 50

    async def test_unpriced_calls_make_cost_none(self, store, internal_store) -> None:
        await _tree(internal_store)
        await _call(internal_store, "root", cost=0.5)
        await _call(internal_store, "a", cost=None)
        await _call(internal_store, "a1", cost=None)
        await _call(internal_store, "b", cost=0.25)

        tree = await store.get_subtree("root")
        by_id = {n.run.run_id: n for n in tree.nodes}
        assert tree.usage.cost_usd is None
        assert tree.usage.unpriced_calls == 2
        assert tree.usage.known_cost_usd == pytest.approx(0.75)
        assert by_id["a"].usage.cost_usd is None
        assert by_id["a"].usage.unpriced_calls == 1
        assert by_id["b"].usage.cost_usd == pytest.approx(0.25)

    async def test_zero_cost_is_priced_not_unpriced(self, store, internal_store) -> None:
        await internal_store.create_run("r", "R")
        await _call(internal_store, "r", cost=0.0)
        tree = await store.get_subtree("r")
        assert tree.usage.cost_usd == 0.0
        assert tree.usage.unpriced_calls == 0

    async def test_leaf_and_subtree_of_inner_node(self, store, internal_store) -> None:
        await _tree(internal_store)
        leaf = await store.get_subtree("a11")
        assert [n.run.run_id for n in leaf.nodes] == ["a11"]
        inner = await store.get_subtree("a")
        assert {n.run.run_id: n.depth for n in inner.nodes} == {"a": 0, "a1": 1, "a11": 2}

    async def test_deep_chain(self, store, internal_store) -> None:
        depth = 40
        await internal_store.create_run("d0", "D")
        for i in range(1, depth + 1):
            await internal_store.create_run(
                f"d{i}", "D", parent_run_id=f"d{i - 1}", delegation_level=i
            )
            await _call(internal_store, f"d{i}", inp=1, out=0, cost=0.001)
        tree = await store.get_subtree("d0")
        assert len(tree.nodes) == depth + 1
        assert max(n.depth for n in tree.nodes) == depth
        assert tree.usage.input_tokens == depth
        assert tree.usage.cost_usd == pytest.approx(0.04)

    async def test_wide_tree_over_in_chunk(self, store, internal_store) -> None:
        await internal_store.create_run("w", "W")
        for i in range(520):
            await internal_store.create_run(f"w{i:04d}", "W", parent_run_id="w")
        await _call(internal_store, "w0519", inp=3)
        tree = await store.get_subtree("w")
        assert len(tree.nodes) == 521
        assert tree.usage.input_tokens == 3
        usage = await internal_store.subtree_usage("w")
        assert usage.input_tokens == 3

    async def test_tenant_filter_excludes_other_tenants_and_their_subtrees(
        self, store, internal_store
    ) -> None:
        await internal_store.create_run("root", "Root", tenant_id="t1")
        await internal_store.create_run("mine", "M", parent_run_id="root", tenant_id="t1")
        await internal_store.create_run("theirs", "T", parent_run_id="root", tenant_id="t2")
        await internal_store.create_run("theirs_kid", "TK", parent_run_id="theirs", tenant_id="t1")
        await _call(internal_store, "mine", inp=1)
        await _call(internal_store, "theirs", inp=100)
        await _call(internal_store, "theirs_kid", inp=1000)

        scoped = await store.get_subtree("root", tenant_id="t1")
        assert {n.run.run_id for n in scoped.nodes} == {"root", "mine"}
        assert scoped.usage.input_tokens == 1

        unscoped = await store.get_subtree("root")
        assert {n.run.run_id for n in unscoped.nodes} == {"root", "mine", "theirs", "theirs_kid"}
        assert unscoped.usage.input_tokens == 1101

    async def test_root_of_other_tenant_is_not_found(self, store, internal_store) -> None:
        await internal_store.create_run("root", "Root", tenant_id="t1")
        with pytest.raises(RunNotFoundError):
            await store.get_subtree("root", tenant_id="t2")

    async def test_untenanted_root_not_found_when_tenant_supplied(
        self, store, internal_store
    ) -> None:
        await internal_store.create_run("root", "Root")
        with pytest.raises(RunNotFoundError):
            await store.get_subtree("root", tenant_id="t1")
        assert (await store.get_subtree("root")).root.run.run_id == "root"

    async def test_explicit_none_scopes_to_untenanted_runs(self, store, internal_store) -> None:
        await internal_store.create_run("null-root", "N")
        await internal_store.create_run("null-kid", "N", parent_run_id="null-root")
        await internal_store.create_run("b-kid", "B", parent_run_id="null-root", tenant_id="b")
        await internal_store.create_run("b-grandkid", "B", parent_run_id="b-kid")
        await _call(internal_store, "null-kid", inp=1)
        await _call(internal_store, "b-kid", inp=10)
        await _call(internal_store, "b-grandkid", inp=100)

        scoped = await store.get_subtree("null-root", tenant_id=None)
        assert {n.run.run_id for n in scoped.nodes} == {"null-root", "null-kid"}
        assert scoped.usage.input_tokens == 1

        unscoped = await store.get_subtree("null-root")
        assert len(unscoped.nodes) == 4
        assert unscoped.usage.input_tokens == 111

    async def test_untenanted_root_with_foreign_child_returns_only_root(
        self, store, internal_store
    ) -> None:
        await internal_store.create_run("null-root", "N")
        await internal_store.create_run("b-kid", "B", parent_run_id="null-root", tenant_id="b")
        tree = await store.get_subtree("null-root", tenant_id=None)
        assert [n.run.run_id for n in tree.nodes] == ["null-root"]

    async def test_tenanted_root_with_explicit_none_is_not_found(
        self, store, internal_store
    ) -> None:
        await internal_store.create_run("a-root", "A", tenant_id="a")
        with pytest.raises(RunNotFoundError):
            await store.get_subtree("a-root", tenant_id=None)

    async def test_missing_run_raises(self, store) -> None:
        with pytest.raises(RunNotFoundError):
            await store.get_subtree("nope")

    async def test_delegation_info_still_works(self, internal_store) -> None:
        await _tree(internal_store)
        info = await internal_store.get_delegation_info("root")
        assert info is not None
        assert [c.run_id for c in info.children] == ["a", "b"]
        assert info.subtree_summary.descendant_count == 4
        assert info.subtree_summary.max_depth == 3
        assert info.subtree_summary.direct_child_count == 2


class TestUsageRollupApi:
    async def test_subtree_and_own_usage(self, internal_store) -> None:
        await _tree(internal_store)
        await _call(internal_store, "a", inp=5, out=1, cost=0.1, cache_read=1)
        await _call(internal_store, "a1", inp=7, out=2, cost=None, reasoning=4)
        await _call(internal_store, "b", inp=100)

        sub = await internal_store.subtree_usage("a")
        assert isinstance(sub, UsageRollup)
        assert (sub.input_tokens, sub.output_tokens) == (12, 3)
        assert sub.cache_read_tokens == 1
        assert sub.reasoning_tokens == 4
        assert sub.unpriced_calls == 1
        assert sub.cost_usd is None
        assert sub.known_cost_usd == pytest.approx(0.1)
        assert sub.llm_calls == 2

        own = await internal_store.run_usage("a")
        assert own.input_tokens == 5
        assert own.cost_usd == pytest.approx(0.1)

    async def test_missing_run_is_empty(self, internal_store) -> None:
        assert await internal_store.subtree_usage("nope") == UsageRollup()
        assert await internal_store.run_usage("nope") == UsageRollup()

    async def test_matches_get_subtree(self, store, internal_store) -> None:
        await _tree(internal_store)
        for rid in ("root", "a", "a1", "a11", "b"):
            await _call(internal_store, rid, inp=3, out=2, cost=0.125, reasoning=1)
        tree = await store.get_subtree("root")
        rollup = await internal_store.subtree_usage("root")
        assert tree.usage.input_tokens == rollup.input_tokens
        assert tree.usage.cost_usd == rollup.cost_usd
        assert tree.usage.llm_calls == rollup.llm_calls == 5

    async def test_combine_empty(self) -> None:
        assert UsageRollup.combine([]) == UsageRollup()
        assert UsageRollup().cost_usd == 0.0


class TestBeforeFilters:
    @pytest.fixture
    def session_factory(self, engine):
        return async_sessionmaker(engine, expire_on_commit=False)

    async def _stamp(self, session_factory, model, ids_to_times) -> None:
        async with session_factory() as session:
            for row_id, ts in ids_to_times.items():
                await session.execute(update(model).where(model.id == row_id).values(created_at=ts))
            await session.commit()

    async def test_events_before_and_event_types(
        self, store, internal_store, session_factory
    ) -> None:
        await internal_store.create_run("r", "R")
        for i, et in enumerate(["run.started", "llm.completed", "tool.completed", "llm.completed"]):
            await internal_store.save_run_event("r", event_type=et, sequence_index=i)
        events = await store.get_events("r")
        base = dt.datetime(2026, 1, 1, tzinfo=dt.UTC)
        # Event ids are not exposed publicly; stamp by sequence via a raw read.
        async with session_factory() as session:
            from sqlalchemy import select

            rows = (
                (await session.execute(select(RunEvent).where(RunEvent.agent_run_id == "r")))
                .scalars()
                .all()
            )
            ids = {r.sequence_index: r.id for r in rows}
        await self._stamp(
            session_factory,
            RunEvent,
            {ids[i]: base + dt.timedelta(minutes=i) for i in range(len(events))},
        )

        cutoff = base + dt.timedelta(minutes=2)
        got = await store.get_events("r", before=cutoff)
        assert [e.sequence_index for e in got] == [0, 1]  # exclusive bound

        got = await store.get_events("r", event_types=["llm.completed"])
        assert [e.sequence_index for e in got] == [1, 3]

        got = await store.get_events("r", event_types=["llm.completed"], before=cutoff)
        assert [e.sequence_index for e in got] == [1]

        assert await store.get_events("r", event_types=[]) == []

        # Non-UTC aware datetimes are converted, not misread.
        plus5 = dt.timezone(dt.timedelta(hours=5))
        got = await store.get_events("r", before=cutoff.astimezone(plus5))
        assert [e.sequence_index for e in got] == [0, 1]

        # Composes with the cursor.
        got = await store.get_events("r", after_sequence_index=0, before=cutoff)
        assert [e.sequence_index for e in got] == [1]

    async def test_llm_calls_before(self, store, internal_store, session_factory) -> None:
        await internal_store.create_run("r", "R")
        for i in range(3):
            await _call(internal_store, "r", i=i)
        calls = await store.get_llm_calls("r")
        base = dt.datetime(2026, 1, 1, tzinfo=dt.UTC)
        await self._stamp(
            session_factory,
            LLMInteraction,
            {c.id: base + dt.timedelta(hours=c.iteration) for c in calls},
        )
        got = await store.get_llm_calls("r", before=base + dt.timedelta(hours=2))
        assert [c.iteration for c in got] == [0, 1]
        got = await store.get_llm_calls("r", before=base)
        assert got == []

    async def test_tool_invocations_before(self, store, internal_store, session_factory) -> None:
        await internal_store.create_run("r", "R")
        for i in range(3):
            await internal_store.save_tool_call(
                "r",
                tool_call_id=f"tc_{i}",
                provider_tool_call_id=None,
                tool_name="noop",
                target="server",
                params={},
                result_payload='""',
                success=True,
                duration_ms=1,
                iteration_index=i,
            )
        records = await internal_store.get_tool_calls("r")
        base = dt.datetime(2026, 1, 1, tzinfo=dt.UTC)
        await self._stamp(
            session_factory,
            ToolCallRecord,
            {r.id: base + dt.timedelta(hours=r.iteration_index) for r in records},
        )
        got = await store.get_tool_invocations("r", before=base + dt.timedelta(hours=1, seconds=1))
        assert [t.iteration for t in got] == [0, 1]

    async def test_naive_before_rejected(self, store, internal_store) -> None:
        await internal_store.create_run("r", "R")
        naive = dt.datetime(2026, 1, 1)
        with pytest.raises(ValueError, match="timezone-aware"):
            await store.get_events("r", before=naive)
        with pytest.raises(ValueError, match="timezone-aware"):
            await store.get_llm_calls("r", before=naive)
        with pytest.raises(ValueError, match="timezone-aware"):
            await store.get_tool_invocations("r", before=naive)
