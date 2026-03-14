"""Integration tests for SQLAlchemyStateStore — real SQLite, full CRUD cycle.

Uses in-memory SQLite (sqlite+aiosqlite:///:memory:) so no files are created.
Each test gets a fresh engine and tables via the `engine` fixture.
"""

from __future__ import annotations

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from dendrite.db.models import AgentRun, Base, ReactTrace, TokenUsage, ToolCallRecord
from dendrite.db.session import get_engine, reset_engine
from dendrite.runtime.state import SQLAlchemyStateStore
from dendrite.types import UsageStats

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
async def engine():
    """Create a fresh in-memory SQLite engine with all tables and FK enforcement."""
    from sqlalchemy import event

    eng = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )

    # SQLite needs PRAGMA foreign_keys = ON for CASCADE/SET NULL to work
    @event.listens_for(eng.sync_engine, "connect")
    def _enable_fk(dbapi_conn, _connection_record):
        dbapi_conn.execute("PRAGMA foreign_keys = ON")

    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield eng
    await eng.dispose()


@pytest.fixture
def store(engine):
    """Create a StateStore backed by the test engine."""
    return SQLAlchemyStateStore(engine)


@pytest.fixture
def session_factory(engine):
    """Raw session factory for direct DB assertions."""
    return sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


# ------------------------------------------------------------------
# create_run + get_run
# ------------------------------------------------------------------


class TestCreateAndGetRun:
    async def test_create_and_retrieve(self, store) -> None:
        await store.create_run(
            "run_1",
            "TestAgent",
            input_data={"input": "hello"},
            model="claude-sonnet",
            strategy="NativeToolCalling",
        )

        record = await store.get_run("run_1")
        assert record is not None
        assert record.id == "run_1"
        assert record.agent_name == "TestAgent"
        assert record.status == "running"
        assert record.input_data == {"input": "hello"}
        assert record.model == "claude-sonnet"
        assert record.strategy == "NativeToolCalling"

    async def test_get_nonexistent_returns_none(self, store) -> None:
        assert await store.get_run("does_not_exist") is None

    async def test_create_with_tenant_and_meta(self, store) -> None:
        meta = {"thread_id": "th_1", "user_id": "u_42"}
        await store.create_run(
            "run_2",
            "Agent",
            tenant_id="tenant_abc",
            meta=meta,
        )

        record = await store.get_run("run_2")
        assert record is not None
        assert record.meta == meta

    async def test_create_with_parent_run(self, store) -> None:
        await store.create_run("parent_1", "ParentAgent")
        await store.create_run(
            "child_1",
            "ChildAgent",
            parent_run_id="parent_1",
            delegation_level=1,
        )

        child = await store.get_run("child_1")
        assert child is not None
        assert child.parent_run_id == "parent_1"
        assert child.delegation_level == 1


# ------------------------------------------------------------------
# save_trace + get_traces
# ------------------------------------------------------------------


class TestTraces:
    async def test_save_and_get_ordered(self, store) -> None:
        await store.create_run("run_t", "Agent")

        await store.save_trace("run_t", "user", "hello", order_index=0)
        await store.save_trace("run_t", "assistant", "hi back", order_index=1)
        await store.save_trace(
            "run_t", "tool", '{"result": 42}', order_index=2, meta={"call_id": "c1"}
        )

        traces = await store.get_traces("run_t")
        assert len(traces) == 3
        assert [t.role for t in traces] == ["user", "assistant", "tool"]
        assert [t.order_index for t in traces] == [0, 1, 2]
        assert traces[0].content == "hello"
        assert traces[2].meta == {"call_id": "c1"}

    async def test_traces_returned_in_order(self, store) -> None:
        """Even if inserted out of order, traces come back sorted by order_index."""
        await store.create_run("run_o", "Agent")

        await store.save_trace("run_o", "assistant", "second", order_index=1)
        await store.save_trace("run_o", "user", "first", order_index=0)

        traces = await store.get_traces("run_o")
        assert [t.order_index for t in traces] == [0, 1]

    async def test_no_traces_returns_empty(self, store) -> None:
        await store.create_run("run_empty", "Agent")
        assert await store.get_traces("run_empty") == []


# ------------------------------------------------------------------
# save_tool_call + get_tool_calls
# ------------------------------------------------------------------


class TestToolCalls:
    async def test_save_and_get(self, store) -> None:
        await store.create_run("run_tc", "Agent")

        await store.save_tool_call(
            "run_tc",
            tool_call_id="tc_1",
            provider_tool_call_id="p_1",
            tool_name="add",
            target="server",
            params={"a": 1, "b": 2},
            result_payload='{"result": 3}',
            success=True,
            duration_ms=42,
            iteration_index=1,
        )

        records = await store.get_tool_calls("run_tc")
        assert len(records) == 1
        r = records[0]
        assert r.tool_call_id == "tc_1"
        assert r.provider_tool_call_id == "p_1"
        assert r.tool_name == "add"
        assert r.params == {"a": 1, "b": 2}
        assert r.result == {"result": 3}
        assert r.success is True
        assert r.duration_ms == 42
        assert r.iteration_index == 1

    async def test_failed_tool_call(self, store) -> None:
        await store.create_run("run_f", "Agent")

        await store.save_tool_call(
            "run_f",
            tool_call_id="tc_2",
            provider_tool_call_id=None,
            tool_name="bad",
            target="server",
            params=None,
            result_payload='{"error": "boom"}',
            success=False,
            duration_ms=5,
            iteration_index=1,
            error_message="boom",
        )

        records = await store.get_tool_calls("run_f")
        assert records[0].success is False
        assert records[0].error_message == "boom"
        assert records[0].provider_tool_call_id is None

    async def test_invalid_json_payload_stored_as_raw(self, store) -> None:
        await store.create_run("run_raw", "Agent")

        await store.save_tool_call(
            "run_raw",
            tool_call_id="tc_3",
            provider_tool_call_id=None,
            tool_name="test",
            target="server",
            params=None,
            result_payload="not valid json",
            success=True,
            duration_ms=1,
            iteration_index=0,
        )

        records = await store.get_tool_calls("run_raw")
        assert records[0].result == {"raw": "not valid json"}


# ------------------------------------------------------------------
# save_usage
# ------------------------------------------------------------------


class TestUsage:
    async def test_save_usage(self, store, session_factory) -> None:
        await store.create_run("run_u", "Agent")

        usage = UsageStats(input_tokens=100, output_tokens=50, total_tokens=150)
        await store.save_usage(
            "run_u",
            iteration_index=1,
            usage=usage,
            model="claude-sonnet",
            provider="AnthropicProvider",
        )

        # Verify via raw query
        async with session_factory() as session:
            from sqlalchemy import select

            stmt = select(TokenUsage).where(TokenUsage.agent_run_id == "run_u")
            result = await session.execute(stmt)
            rows = result.scalars().all()

        assert len(rows) == 1
        assert rows[0].input_tokens == 100
        assert rows[0].output_tokens == 50
        assert rows[0].model == "claude-sonnet"
        assert rows[0].provider == "AnthropicProvider"


# ------------------------------------------------------------------
# finalize_run
# ------------------------------------------------------------------


class TestFinalizeRun:
    async def test_finalize_success(self, store) -> None:
        await store.create_run("run_fs", "Agent")

        usage = UsageStats(input_tokens=300, output_tokens=80)
        await store.finalize_run(
            "run_fs",
            status="success",
            answer="42",
            iteration_count=3,
            total_usage=usage,
        )

        record = await store.get_run("run_fs")
        assert record is not None
        assert record.status == "success"
        assert record.answer == "42"
        assert record.output_data == {"answer": "42"}
        assert record.iteration_count == 3
        assert record.total_input_tokens == 300
        assert record.total_output_tokens == 80

    async def test_finalize_error(self, store) -> None:
        await store.create_run("run_fe", "Agent")

        await store.finalize_run(
            "run_fe",
            status="error",
            error="LLM exploded",
        )

        record = await store.get_run("run_fe")
        assert record is not None
        assert record.status == "error"
        assert record.error == "LLM exploded"
        assert record.answer is None

    async def test_finalize_max_iterations(self, store) -> None:
        await store.create_run("run_fm", "Agent")

        await store.finalize_run(
            "run_fm",
            status="max_iterations",
            iteration_count=10,
        )

        record = await store.get_run("run_fm")
        assert record is not None
        assert record.status == "max_iterations"
        assert record.iteration_count == 10


# ------------------------------------------------------------------
# list_runs
# ------------------------------------------------------------------


class TestListRuns:
    async def test_list_runs_ordered_by_created_at_desc(self, store) -> None:
        await store.create_run("run_a", "Agent")
        await store.create_run("run_b", "Agent")
        await store.create_run("run_c", "Agent")

        runs = await store.list_runs()
        # Most recent first
        assert len(runs) == 3
        assert runs[0].id == "run_c"
        assert runs[2].id == "run_a"

    async def test_list_with_limit_and_offset(self, store) -> None:
        for i in range(5):
            await store.create_run(f"run_{i}", "Agent")

        runs = await store.list_runs(limit=2, offset=1)
        assert len(runs) == 2

    async def test_filter_by_tenant(self, store) -> None:
        await store.create_run("run_t1", "Agent", tenant_id="tenant_a")
        await store.create_run("run_t2", "Agent", tenant_id="tenant_b")
        await store.create_run("run_t3", "Agent", tenant_id="tenant_a")

        runs = await store.list_runs(tenant_id="tenant_a")
        assert len(runs) == 2
        assert all(r.id in ("run_t1", "run_t3") for r in runs)

    async def test_filter_by_status(self, store) -> None:
        await store.create_run("run_s1", "Agent")
        await store.create_run("run_s2", "Agent")
        await store.finalize_run("run_s1", status="success")

        runs = await store.list_runs(status="success")
        assert len(runs) == 1
        assert runs[0].id == "run_s1"

    async def test_combined_tenant_and_status_filter(self, store) -> None:
        """T6: Combined filters narrow results correctly."""
        await store.create_run("r1", "Agent", tenant_id="t_a")
        await store.create_run("r2", "Agent", tenant_id="t_a")
        await store.create_run("r3", "Agent", tenant_id="t_b")
        await store.create_run("r4", "Agent", tenant_id="t_a")

        await store.finalize_run("r1", status="success")
        await store.finalize_run("r2", status="error")
        await store.finalize_run("r3", status="success")
        # r4 stays "running"

        runs = await store.list_runs(tenant_id="t_a", status="success")
        assert len(runs) == 1
        assert runs[0].id == "r1"

    async def test_empty_list(self, store) -> None:
        assert await store.list_runs() == []


# ------------------------------------------------------------------
# Model relationships and cascades
# ------------------------------------------------------------------


class TestModelRelationships:
    async def test_delete_run_cascades_to_traces(self, store, session_factory) -> None:
        await store.create_run("run_cas", "Agent")
        await store.save_trace("run_cas", "user", "hello", order_index=0)
        await store.save_trace("run_cas", "assistant", "hi", order_index=1)

        # Delete the run
        async with session_factory() as session:
            from sqlalchemy import delete, select

            await session.execute(delete(AgentRun).where(AgentRun.id == "run_cas"))
            await session.commit()

            # Traces should be gone
            result = await session.execute(
                select(ReactTrace).where(ReactTrace.agent_run_id == "run_cas")
            )
            assert result.scalars().all() == []

    async def test_delete_run_cascades_to_tool_calls(self, store, session_factory) -> None:
        await store.create_run("run_cas2", "Agent")
        await store.save_tool_call(
            "run_cas2",
            tool_call_id="tc_1",
            provider_tool_call_id=None,
            tool_name="test",
            target="server",
            params=None,
            result_payload="{}",
            success=True,
            duration_ms=1,
            iteration_index=0,
        )

        async with session_factory() as session:
            from sqlalchemy import delete, select

            await session.execute(delete(AgentRun).where(AgentRun.id == "run_cas2"))
            await session.commit()

            result = await session.execute(
                select(ToolCallRecord).where(ToolCallRecord.agent_run_id == "run_cas2")
            )
            assert result.scalars().all() == []

    async def test_get_engine_raises_on_url_mismatch(self) -> None:
        """H1: get_engine() must refuse a different URL after first init."""
        await reset_engine()  # clean slate
        try:
            await get_engine("sqlite+aiosqlite:///:memory:")
            with pytest.raises(RuntimeError, match="already initialized"):
                await get_engine("sqlite+aiosqlite:///./other.db")
        finally:
            await reset_engine()

    async def test_get_engine_allows_same_url(self) -> None:
        """get_engine() with the same URL should return the cached engine."""
        await reset_engine()
        try:
            e1 = await get_engine("sqlite+aiosqlite:///:memory:")
            e2 = await get_engine("sqlite+aiosqlite:///:memory:")
            assert e1 is e2
        finally:
            await reset_engine()

    async def test_delete_parent_sets_child_null(self, store, session_factory) -> None:
        await store.create_run("parent_del", "Parent")
        await store.create_run("child_del", "Child", parent_run_id="parent_del")

        async with session_factory() as session:
            from sqlalchemy import delete, select

            await session.execute(delete(AgentRun).where(AgentRun.id == "parent_del"))
            await session.commit()

            result = await session.execute(select(AgentRun).where(AgentRun.id == "child_del"))
            child = result.scalar_one()
            assert child.parent_run_id is None
