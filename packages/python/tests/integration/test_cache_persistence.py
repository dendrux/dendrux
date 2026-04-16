"""Integration tests for cache token persistence (PR 2).

Verifies that per-call cache fields land in token_usage and llm_interactions,
and that finalize_run rolls them up into agent_runs.total_cache_*.
"""

from __future__ import annotations

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from dendrux.db.models import AgentRun, Base, LLMInteraction, TokenUsage
from dendrux.runtime.state import SQLAlchemyStateStore
from dendrux.types import UsageStats


@pytest.fixture
async def engine():
    """Fresh in-memory SQLite engine with tables."""
    eng = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield eng
    await eng.dispose()


@pytest.fixture
def store(engine):
    return SQLAlchemyStateStore(engine)


@pytest.fixture
def session_factory(engine):
    return sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


# ---------------------------------------------------------------------------
# Per-call persistence
# ---------------------------------------------------------------------------


class TestPerCallCachePersistence:
    """save_usage and save_llm_interaction round-trip cache fields."""

    async def test_save_usage_persists_cache_fields(
        self, store: SQLAlchemyStateStore, session_factory
    ) -> None:
        await store.create_run("r1", "Agent", input_data={"x": 1})
        usage = UsageStats(
            input_tokens=500,
            output_tokens=50,
            total_tokens=550,
            cache_read_input_tokens=1000,
            cache_creation_input_tokens=200,
        )
        await store.save_usage("r1", iteration_index=0, usage=usage)

        async with session_factory() as session:
            row = (
                await session.execute(select(TokenUsage).where(TokenUsage.agent_run_id == "r1"))
            ).scalar_one()
            assert row.cache_read_input_tokens == 1000
            assert row.cache_creation_input_tokens == 200
            assert row.input_tokens == 500

    async def test_save_usage_preserves_none_distinction(
        self, store: SQLAlchemyStateStore, session_factory
    ) -> None:
        """None means provider didn't report. Persists as SQL NULL, not 0."""
        await store.create_run("r2", "Agent", input_data={"x": 1})
        usage = UsageStats(input_tokens=200, output_tokens=30, total_tokens=230)
        await store.save_usage("r2", iteration_index=0, usage=usage)

        async with session_factory() as session:
            row = (
                await session.execute(select(TokenUsage).where(TokenUsage.agent_run_id == "r2"))
            ).scalar_one()
            assert row.cache_read_input_tokens is None
            assert row.cache_creation_input_tokens is None

    async def test_save_llm_interaction_persists_cache_fields(
        self, store: SQLAlchemyStateStore, session_factory
    ) -> None:
        await store.create_run("r3", "Agent", input_data={"x": 1})
        usage = UsageStats(
            input_tokens=500,
            output_tokens=50,
            total_tokens=550,
            cache_read_input_tokens=800,
        )
        await store.save_llm_interaction("r3", iteration_index=0, usage=usage)

        async with session_factory() as session:
            row = (
                await session.execute(
                    select(LLMInteraction).where(LLMInteraction.agent_run_id == "r3")
                )
            ).scalar_one()
            assert row.cache_read_input_tokens == 800
            assert row.cache_creation_input_tokens is None


# ---------------------------------------------------------------------------
# Run-level rollup at finalize_run
# ---------------------------------------------------------------------------


class TestRunRollup:
    """finalize_run sums cache tokens into agent_runs.total_cache_*."""

    async def test_finalize_rolls_up_cache_tokens(
        self, store: SQLAlchemyStateStore, session_factory
    ) -> None:
        await store.create_run("r4", "Agent", input_data={"x": 1})
        total = UsageStats(
            input_tokens=1500,
            output_tokens=150,
            total_tokens=1650,
            cache_read_input_tokens=3000,
            cache_creation_input_tokens=500,
        )
        await store.finalize_run(
            "r4",
            status="success",
            answer="ok",
            iteration_count=3,
            total_usage=total,
        )

        async with session_factory() as session:
            row = (await session.execute(select(AgentRun).where(AgentRun.id == "r4"))).scalar_one()
            assert row.total_cache_read_tokens == 3000
            assert row.total_cache_creation_tokens == 500
            assert row.total_input_tokens == 1500

    async def test_finalize_with_none_cache_writes_zero(
        self, store: SQLAlchemyStateStore, session_factory
    ) -> None:
        """Provider that doesn't report cache fields → rollup writes 0,
        not NULL. Aggregate is non-null per the column definition."""
        await store.create_run("r5", "Agent", input_data={"x": 1})
        total = UsageStats(input_tokens=200, output_tokens=30, total_tokens=230)
        await store.finalize_run(
            "r5",
            status="success",
            answer="ok",
            iteration_count=1,
            total_usage=total,
        )

        async with session_factory() as session:
            row = (await session.execute(select(AgentRun).where(AgentRun.id == "r5"))).scalar_one()
            assert row.total_cache_read_tokens == 0
            assert row.total_cache_creation_tokens == 0


# ---------------------------------------------------------------------------
# In-loop accumulation (UsageStats _accumulate_usage)
# ---------------------------------------------------------------------------


class TestUsageAccumulation:
    """The React-loop running total accumulates cache fields correctly,
    treating None as 0 in summation but preserving 'never reported' state
    if no step ever provides a value."""

    def test_accumulate_sums_cache_read(self) -> None:
        from dendrux.loops.react import _accumulate_usage

        total = UsageStats()
        _accumulate_usage(
            total,
            UsageStats(input_tokens=100, output_tokens=10, cache_read_input_tokens=500),
        )
        _accumulate_usage(
            total,
            UsageStats(input_tokens=80, output_tokens=15, cache_read_input_tokens=700),
        )
        assert total.cache_read_input_tokens == 1200
        assert total.input_tokens == 180

    def test_accumulate_sums_cache_creation(self) -> None:
        from dendrux.loops.react import _accumulate_usage

        total = UsageStats()
        _accumulate_usage(
            total,
            UsageStats(input_tokens=100, output_tokens=10, cache_creation_input_tokens=4096),
        )
        _accumulate_usage(
            total,
            UsageStats(input_tokens=80, output_tokens=15, cache_creation_input_tokens=0),
        )
        assert total.cache_creation_input_tokens == 4096

    def test_accumulate_no_cache_steps_keeps_none(self) -> None:
        """If no step ever reports a cache field, the running total stays
        None — distinguishes 'no cache info' from '0 cache used'."""
        from dendrux.loops.react import _accumulate_usage

        total = UsageStats()
        _accumulate_usage(total, UsageStats(input_tokens=100, output_tokens=10))
        _accumulate_usage(total, UsageStats(input_tokens=80, output_tokens=15))
        assert total.cache_read_input_tokens is None
        assert total.cache_creation_input_tokens is None
