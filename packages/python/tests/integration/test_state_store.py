"""Integration tests for SQLAlchemyStateStore — real SQLite, full CRUD cycle.

Uses in-memory SQLite (sqlite+aiosqlite:///:memory:) so no files are created.
Each test gets a fresh engine and tables via the `engine` fixture.
"""

from __future__ import annotations

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from dendrux.db.models import (
    AgentRun,
    Base,
    LLMInteraction,
    ReactTrace,
    RunEvent,
    TokenUsage,
    ToolCallRecord,
)
from dendrux.db.session import get_engine, reset_engine
from dendrux.runtime.state import SQLAlchemyStateStore
from dendrux.types import UsageStats

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

    async def test_filter_by_agent_name(self, store) -> None:
        """Agent-name filtering happens in SQL, not Python."""
        await store.create_run("r1", "Alpha")
        await store.create_run("r2", "Beta")
        await store.create_run("r3", "Alpha")

        runs = await store.list_runs(agent_name="Alpha")
        assert len(runs) == 2
        assert {r.id for r in runs} == {"r1", "r3"}

    async def test_filter_by_parent_run_id(self, store) -> None:
        """Parent-run filtering returns only direct children."""
        await store.create_run("parent", "Coord")
        await store.create_run("child_a", "Worker", parent_run_id="parent")
        await store.create_run("child_b", "Worker", parent_run_id="parent")
        await store.create_run("other", "Worker")

        runs = await store.list_runs(parent_run_id="parent")
        assert {r.id for r in runs} == {"child_a", "child_b"}

    async def test_filter_by_status_list(self, store) -> None:
        """Status accepts a list; matches any of the given values."""
        await store.create_run("r1", "Agent")
        await store.create_run("r2", "Agent")
        await store.create_run("r3", "Agent")
        await store.finalize_run("r1", status="success")
        await store.finalize_run("r2", status="error")
        # r3 stays running

        runs = await store.list_runs(status=["success", "error"])
        assert {r.id for r in runs} == {"r1", "r2"}

    async def test_empty_status_list_matches_none(self, store) -> None:
        """``status=[]`` means 'match no status', not 'match any'."""
        await store.create_run("r1", "Agent")
        await store.create_run("r2", "Agent")

        runs = await store.list_runs(status=[])
        assert runs == []

    async def test_stable_pagination_with_tied_created_at(self, store) -> None:
        """Pagination must not duplicate or skip rows when ``created_at`` ties."""
        import datetime as _dt

        from sqlalchemy import update

        from dendrux.db.models import AgentRun

        # Seed five runs with the same created_at to force ties.
        for i in range(5):
            await store.create_run(f"r_{i}", "Agent")
        same = _dt.datetime(2026, 4, 17, 12, 0, 0)
        async with store._session_factory() as session:
            await session.execute(update(AgentRun).values(created_at=same))
            await session.commit()

        page1 = await store.list_runs(limit=2, offset=0)
        page2 = await store.list_runs(limit=2, offset=2)
        page3 = await store.list_runs(limit=2, offset=4)

        seen = [r.id for r in page1 + page2 + page3]
        assert sorted(seen) == sorted(f"r_{i}" for i in range(5))
        assert len(seen) == len(set(seen)), f"duplicates across pages: {seen}"

    async def test_filter_by_started_after_and_before(self, store) -> None:
        """Time bounds apply inclusive-start / exclusive-end against created_at."""
        import datetime as _dt

        from dendrux.db.models import AgentRun

        await store.create_run("r1", "Agent")
        await store.create_run("r2", "Agent")
        await store.create_run("r3", "Agent")

        # Pin created_at values deterministically so we can slice a window.
        base = _dt.datetime(2026, 4, 1, 12, 0, 0)
        async with store._session_factory() as session:
            from sqlalchemy import update

            await session.execute(
                update(AgentRun).where(AgentRun.id == "r1").values(created_at=base)
            )
            await session.execute(
                update(AgentRun)
                .where(AgentRun.id == "r2")
                .values(created_at=base + _dt.timedelta(hours=1))
            )
            await session.execute(
                update(AgentRun)
                .where(AgentRun.id == "r3")
                .values(created_at=base + _dt.timedelta(hours=2))
            )
            await session.commit()

        window = await store.list_runs(
            started_after=base + _dt.timedelta(minutes=30),
            started_before=base + _dt.timedelta(hours=1, minutes=30),
        )
        assert [r.id for r in window] == ["r2"]


class TestCountRuns:
    async def test_count_matches_list_length(self, store) -> None:
        for i in range(7):
            await store.create_run(f"r_{i}", "Agent")
        assert await store.count_runs() == 7

    async def test_count_with_filters(self, store) -> None:
        await store.create_run("r1", "Alpha")
        await store.create_run("r2", "Alpha")
        await store.create_run("r3", "Beta")
        await store.finalize_run("r1", status="success")

        assert await store.count_runs(agent_name="Alpha") == 2
        assert await store.count_runs(status="success") == 1
        assert await store.count_runs(agent_name="Alpha", status="success") == 1
        assert await store.count_runs(agent_name="Missing") == 0

    async def test_count_ignores_limit_offset(self, store) -> None:
        """count_runs returns the full matching count regardless of any pagination."""
        for i in range(12):
            await store.create_run(f"r_{i}", "Agent")
        # Even with a limited list, the count is the total.
        listed = await store.list_runs(limit=3, offset=0)
        assert len(listed) == 3
        assert await store.count_runs() == 12

    async def test_empty_status_list_returns_zero(self, store) -> None:
        """``status=[]`` matches no rows — count is zero."""
        await store.create_run("r1", "Agent")
        await store.create_run("r2", "Agent")
        assert await store.count_runs(status=[]) == 0

    async def test_count_matches_list_with_filters(self, store) -> None:
        """Parity: same filters produce same row set on list and count."""
        await store.create_run("r1", "Alpha")
        await store.create_run("r2", "Alpha")
        await store.create_run("r3", "Beta")
        await store.finalize_run("r1", status="success")

        for filters in (
            {"agent_name": "Alpha"},
            {"agent_name": "Alpha", "status": "success"},
            {"status": ["success", "error"]},
            {"agent_name": "Missing"},
        ):
            listed = await store.list_runs(limit=100, **filters)
            counted = await store.count_runs(**filters)
            assert counted == len(listed), f"parity broke for {filters}"


class TestCountPausesPerRun:
    async def test_aggregates_per_run(self, store) -> None:
        await store.create_run("r1", "Agent")
        await store.create_run("r2", "Agent")
        await store.create_run("r3", "Agent")

        # r1: 2 pauses, r2: 1 pause, r3: 0 pauses
        await store.save_run_event("r1", event_type="run.paused", sequence_index=0)
        await store.save_run_event("r1", event_type="run.resumed", sequence_index=1)
        await store.save_run_event("r1", event_type="run.paused", sequence_index=2)
        await store.save_run_event("r2", event_type="run.paused", sequence_index=0)
        # Non-pause events on r3 must not show up.
        await store.save_run_event("r3", event_type="llm.completed", sequence_index=0)

        counts = await store.count_pauses_per_run(["r1", "r2", "r3"])
        assert counts == {"r1": 2, "r2": 1}  # r3 absent (0 pauses)

    async def test_empty_input(self, store) -> None:
        assert await store.count_pauses_per_run([]) == {}


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


# ------------------------------------------------------------------
# Pause / Resume (Sprint 3)
# ------------------------------------------------------------------


class TestPauseResume:
    async def test_pause_run_persists_state(self, store) -> None:
        """pause_run stores status + pause_data in the DB."""
        await store.create_run("run_pr", "Agent")

        pause_data = {
            "agent_name": "Agent",
            "pending_tool_calls": [
                {"name": "read", "params": {}, "id": "tc1", "provider_tool_call_id": None}
            ],
            "history": [{"role": "user", "content": "hello"}],
            "steps": [],
            "iteration": 2,
            "trace_order_offset": 3,
            "usage": {
                "input_tokens": 100,
                "output_tokens": 50,
                "total_tokens": 150,
                "cost_usd": None,
            },
        }
        await store.pause_run(
            "run_pr", status="waiting_client_tool", pause_data=pause_data, iteration_count=2
        )

        record = await store.get_run("run_pr")
        assert record is not None
        assert record.status == "waiting_client_tool"
        assert record.iteration_count == 2

    async def test_get_pause_state_roundtrip(self, store) -> None:
        """pause_data survives DB roundtrip via get_pause_state."""
        await store.create_run("run_gps", "Agent")

        pause_data = {
            "agent_name": "TestAgent",
            "pending_tool_calls": [
                {
                    "name": "read",
                    "params": {"sheet": "S1"},
                    "id": "tc1",
                    "provider_tool_call_id": "p1",
                }
            ],
            "history": [{"role": "user", "content": "go"}],
            "steps": [
                {
                    "reasoning": "think",
                    "action": {
                        "type": "tool_call",
                        "name": "read",
                        "params": {},
                        "id": "tc1",
                        "provider_tool_call_id": None,
                    },
                }
            ],
            "iteration": 3,
            "trace_order_offset": 5,
            "usage": {
                "input_tokens": 200,
                "output_tokens": 80,
                "total_tokens": 280,
                "cost_usd": 0.01,
            },
        }
        await store.pause_run("run_gps", status="waiting_client_tool", pause_data=pause_data)

        loaded = await store.get_pause_state("run_gps")
        assert loaded is not None
        assert loaded["agent_name"] == "TestAgent"
        assert loaded["iteration"] == 3
        assert len(loaded["pending_tool_calls"]) == 1
        assert loaded["pending_tool_calls"][0]["name"] == "read"
        assert loaded["usage"]["cost_usd"] == 0.01

    async def test_get_pause_state_returns_none_for_non_paused(self, store) -> None:
        """get_pause_state returns None when no pause_data exists."""
        await store.create_run("run_np", "Agent")
        assert await store.get_pause_state("run_np") is None

    async def test_get_pause_state_returns_none_for_nonexistent_run(self, store) -> None:
        """get_pause_state returns None for a run_id that doesn't exist."""
        assert await store.get_pause_state("nonexistent") is None

    async def test_claim_paused_run_succeeds(self, store) -> None:
        """claim_paused_run transitions WAITING → RUNNING atomically."""
        await store.create_run("run_cl", "Agent")
        await store.pause_run("run_cl", status="waiting_client_tool", pause_data={"x": 1})

        claimed = await store.claim_paused_run("run_cl", expected_status="waiting_client_tool")
        assert claimed is True

        record = await store.get_run("run_cl")
        assert record is not None
        assert record.status == "running"

    async def test_claim_paused_run_fails_wrong_status(self, store) -> None:
        """claim_paused_run returns False if status doesn't match."""
        await store.create_run("run_cf", "Agent")
        await store.pause_run("run_cf", status="waiting_client_tool", pause_data={"x": 1})

        # Try to claim with wrong expected status
        claimed = await store.claim_paused_run("run_cf", expected_status="waiting_human_input")
        assert claimed is False

        # Status unchanged
        record = await store.get_run("run_cf")
        assert record is not None
        assert record.status == "waiting_client_tool"

    async def test_double_claim_fails(self, store) -> None:
        """Second claim on the same run returns False — atomic CAS."""
        await store.create_run("run_dc", "Agent")
        await store.pause_run("run_dc", status="waiting_client_tool", pause_data={"x": 1})

        claimed1 = await store.claim_paused_run("run_dc", expected_status="waiting_client_tool")
        assert claimed1 is True

        # Second claim fails — status is now 'running', not 'waiting_client_tool'
        claimed2 = await store.claim_paused_run("run_dc", expected_status="waiting_client_tool")
        assert claimed2 is False

    async def test_finalize_clears_pause_data(self, store) -> None:
        """finalize_run sets pause_data to None."""
        await store.create_run("run_fc", "Agent")
        await store.pause_run("run_fc", status="waiting_client_tool", pause_data={"x": 1})

        # pause_data exists
        assert await store.get_pause_state("run_fc") is not None

        await store.finalize_run("run_fc", status="success", answer="done")

        # pause_data cleared
        assert await store.get_pause_state("run_fc") is None

        record = await store.get_run("run_fc")
        assert record is not None
        assert record.status == "success"


# ------------------------------------------------------------------
# submit_and_claim (G2: persist-first handoff)
# ------------------------------------------------------------------


class TestSubmitAndClaim:
    """submit_and_claim atomically saves submitted data + claims the run."""

    _PAUSE_DATA = {
        "agent_name": "Agent",
        "pending_tool_calls": [
            {"name": "read", "params": {}, "id": "tc1", "provider_tool_call_id": None}
        ],
        "pending_targets": {"tc1": "client"},
        "history": [{"role": "user", "content": "hello"}],
        "steps": [],
        "iteration": 2,
        "trace_order_offset": 3,
        "usage": {
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
            "cost_usd": None,
        },
    }

    async def test_submit_and_claim_succeeds(self, store) -> None:
        """First caller wins: saves submitted data + transitions to running."""
        await store.create_run("run_sac1", "Agent")
        await store.pause_run(
            "run_sac1", status="waiting_client_tool", pause_data=dict(self._PAUSE_DATA)
        )

        won = await store.submit_and_claim(
            "run_sac1",
            expected_status="waiting_client_tool",
            submitted_data={
                "submitted_tool_results": [
                    {"name": "read", "call_id": "tc1", "payload": '"data"', "success": True}
                ]
            },
        )
        assert won is True

        # Status transitioned to running
        record = await store.get_run("run_sac1")
        assert record is not None
        assert record.status == "running"

        # Submitted data merged into pause_data
        pause = await store.get_pause_state("run_sac1")
        assert pause is not None
        assert "submitted_tool_results" in pause
        assert pause["submitted_tool_results"][0]["call_id"] == "tc1"
        # Original data preserved
        assert pause["agent_name"] == "Agent"

    async def test_second_caller_loses(self, store) -> None:
        """Second submit_and_claim returns False — first writer wins."""
        await store.create_run("run_sac2", "Agent")
        await store.pause_run(
            "run_sac2", status="waiting_client_tool", pause_data=dict(self._PAUSE_DATA)
        )

        won1 = await store.submit_and_claim(
            "run_sac2",
            expected_status="waiting_client_tool",
            submitted_data={"submitted_tool_results": [{"call_id": "tc1"}]},
        )
        assert won1 is True

        won2 = await store.submit_and_claim(
            "run_sac2",
            expected_status="waiting_client_tool",
            submitted_data={"submitted_tool_results": [{"call_id": "tc1"}]},
        )
        assert won2 is False

    async def test_wrong_status_returns_false(self, store) -> None:
        """submit_and_claim fails if run is not in expected status."""
        await store.create_run("run_sac3", "Agent")
        await store.pause_run(
            "run_sac3", status="waiting_client_tool", pause_data=dict(self._PAUSE_DATA)
        )

        won = await store.submit_and_claim(
            "run_sac3",
            expected_status="waiting_human_input",  # wrong status
            submitted_data={"submitted_user_input": "yes"},
        )
        assert won is False

        # Status unchanged
        record = await store.get_run("run_sac3")
        assert record is not None
        assert record.status == "waiting_client_tool"

    async def test_nonexistent_run_returns_false(self, store) -> None:
        """submit_and_claim returns False for a run that doesn't exist."""
        won = await store.submit_and_claim(
            "nonexistent",
            expected_status="waiting_client_tool",
            submitted_data={"submitted_tool_results": []},
        )
        assert won is False

    async def test_no_pause_data_returns_false(self, store) -> None:
        """submit_and_claim returns False when run has no pause_data."""
        await store.create_run("run_sac5", "Agent")
        # Running, no pause_data
        won = await store.submit_and_claim(
            "run_sac5",
            expected_status="running",
            submitted_data={"submitted_tool_results": []},
        )
        assert won is False

    async def test_submit_input(self, store) -> None:
        """submit_and_claim works for clarification input too."""
        await store.create_run("run_sac6", "Agent")
        await store.pause_run(
            "run_sac6", status="waiting_human_input", pause_data=dict(self._PAUSE_DATA)
        )

        won = await store.submit_and_claim(
            "run_sac6",
            expected_status="waiting_human_input",
            submitted_data={"submitted_user_input": "yes, proceed"},
        )
        assert won is True

        pause = await store.get_pause_state("run_sac6")
        assert pause is not None
        assert pause["submitted_user_input"] == "yes, proceed"


# ------------------------------------------------------------------
# Run Events (Dashboard Phase 0)
# ------------------------------------------------------------------


class TestRunEvents:
    async def test_save_and_get_events(self, store) -> None:
        """Events are saved and retrieved in sequence_index order."""
        await store.create_run("run_ev", "Agent")

        await store.save_run_event(
            "run_ev", event_type="run.started", sequence_index=0, data={"agent": "A"}
        )
        await store.save_run_event(
            "run_ev",
            event_type="llm.completed",
            sequence_index=1,
            iteration_index=1,
            data={"tokens": 500},
        )
        await store.save_run_event(
            "run_ev",
            event_type="run.completed",
            sequence_index=2,
            data={"status": "success"},
        )

        events = await store.get_run_events("run_ev")
        assert len(events) == 3
        assert events[0].event_type == "run.started"
        assert events[0].sequence_index == 0
        assert events[1].event_type == "llm.completed"
        assert events[1].iteration_index == 1
        assert events[1].data == {"tokens": 500}
        assert events[2].event_type == "run.completed"

    async def test_events_have_timestamps(self, store) -> None:
        """Each event has a created_at timestamp."""
        await store.create_run("run_ts", "Agent")
        await store.save_run_event("run_ts", event_type="run.started")

        events = await store.get_run_events("run_ts")
        assert len(events) == 1
        assert events[0].created_at is not None

    async def test_correlation_id_links_related_events(self, store) -> None:
        """correlation_id links tool.completed back to the tool_call_id."""
        await store.create_run("run_corr", "Agent")

        await store.save_run_event(
            "run_corr",
            event_type="tool.completed",
            sequence_index=0,
            iteration_index=1,
            correlation_id="tc_abc123",
            data={"tool_name": "lookup", "success": True},
        )

        events = await store.get_run_events("run_corr")
        assert len(events) == 1
        assert events[0].correlation_id == "tc_abc123"

    async def test_pause_resume_events_give_exact_timing(self, store) -> None:
        """Pause and resume events provide durable timestamps for the dashboard."""
        await store.create_run("run_pr_ev", "Agent")

        await store.save_run_event(
            "run_pr_ev",
            event_type="run.paused",
            sequence_index=0,
            data={"pending_tool_calls": [{"name": "read_excel"}]},
        )
        await store.save_run_event(
            "run_pr_ev",
            event_type="run.resumed",
            sequence_index=1,
            data={"resumed_from": "waiting_client_tool"},
        )

        events = await store.get_run_events("run_pr_ev")
        assert len(events) == 2
        pause_event = events[0]
        resume_event = events[1]

        assert pause_event.event_type == "run.paused"
        assert resume_event.event_type == "run.resumed"
        assert pause_event.created_at is not None
        assert resume_event.created_at is not None

    async def test_sequence_index_determines_order(self, store) -> None:
        """Events are returned by sequence_index, not insertion order."""
        await store.create_run("run_seq", "Agent")

        # Insert out of order
        await store.save_run_event("run_seq", event_type="second", sequence_index=1)
        await store.save_run_event("run_seq", event_type="first", sequence_index=0)

        events = await store.get_run_events("run_seq")
        assert events[0].event_type == "first"
        assert events[1].event_type == "second"

    async def test_events_empty_for_nonexistent_run(self, store) -> None:
        events = await store.get_run_events("nonexistent")
        assert events == []

    async def test_delete_run_cascades_to_events(self, store, session_factory) -> None:
        """Events are deleted when the parent run is deleted."""
        await store.create_run("run_cas_ev", "Agent")
        await store.save_run_event("run_cas_ev", event_type="run.started")
        await store.save_run_event("run_cas_ev", event_type="run.completed", sequence_index=1)

        async with session_factory() as session:
            from sqlalchemy import delete, select

            await session.execute(delete(AgentRun).where(AgentRun.id == "run_cas_ev"))
            await session.commit()

            result = await session.execute(
                select(RunEvent).where(RunEvent.agent_run_id == "run_cas_ev")
            )
            assert result.scalars().all() == []


# ------------------------------------------------------------------
# LLM Interactions (Sprint 3.5)
# ------------------------------------------------------------------


class TestLLMInteractions:
    async def test_save_and_get_roundtrip(self, store) -> None:
        """Full roundtrip: save_llm_interaction → get_llm_interactions."""
        await store.create_run("run_llm1", "Agent")

        usage = UsageStats(input_tokens=100, output_tokens=50, total_tokens=150, cost_usd=0.005)
        await store.save_llm_interaction(
            "run_llm1",
            iteration_index=1,
            usage=usage,
            model="claude-sonnet-4-6",
            provider="Anthropic",
            semantic_request={"messages": [{"role": "user", "content": "hello"}]},
            semantic_response={"text": "hi there"},
            provider_request={"model": "claude-sonnet-4-6", "max_tokens": 1024},
            provider_response={"id": "msg_123", "type": "message"},
        )

        records = await store.get_llm_interactions("run_llm1")
        assert len(records) == 1
        r = records[0]
        assert r.iteration_index == 1
        assert r.model == "claude-sonnet-4-6"
        assert r.provider == "Anthropic"
        assert r.input_tokens == 100
        assert r.output_tokens == 50
        assert r.cost_usd == pytest.approx(0.005)
        assert r.semantic_request["messages"][0]["content"] == "hello"
        assert r.semantic_response["text"] == "hi there"
        assert r.provider_request["model"] == "claude-sonnet-4-6"
        assert r.provider_response["id"] == "msg_123"

    async def test_multiple_interactions_ordered(self, store) -> None:
        """Multiple interactions for one run are returned in iteration order."""
        await store.create_run("run_llm2", "Agent")

        for i in [3, 1, 2]:
            await store.save_llm_interaction(
                "run_llm2",
                iteration_index=i,
                usage=UsageStats(),
            )

        records = await store.get_llm_interactions("run_llm2")
        assert [r.iteration_index for r in records] == [1, 2, 3]

    async def test_empty_interactions(self, store) -> None:
        """No interactions returns empty list."""
        await store.create_run("run_llm3", "Agent")
        records = await store.get_llm_interactions("run_llm3")
        assert records == []

    async def test_nullable_payloads(self, store) -> None:
        """All payload fields can be None."""
        await store.create_run("run_llm4", "Agent")
        await store.save_llm_interaction(
            "run_llm4",
            iteration_index=1,
            usage=UsageStats(),
        )

        records = await store.get_llm_interactions("run_llm4")
        r = records[0]
        assert r.semantic_request is None
        assert r.semantic_response is None
        assert r.provider_request is None
        assert r.provider_response is None
        assert r.model is None
        assert r.cost_usd is None

    async def test_cascade_delete(self, store, session_factory) -> None:
        """Deleting a run cascades to llm_interactions."""
        await store.create_run("run_llm_cas", "Agent")
        await store.save_llm_interaction("run_llm_cas", iteration_index=1, usage=UsageStats())

        async with session_factory() as session:
            from sqlalchemy import delete, select

            await session.execute(delete(AgentRun).where(AgentRun.id == "run_llm_cas"))
            await session.commit()

            result = await session.execute(
                select(LLMInteraction).where(LLMInteraction.agent_run_id == "run_llm_cas")
            )
            assert result.scalars().all() == []


# ------------------------------------------------------------------
# Delegation queries (get_delegation_info)
# ------------------------------------------------------------------


class TestDelegationInfo:
    """Integration tests for get_delegation_info against real SQLite."""

    async def _create_finalized_run(
        self,
        store,
        run_id: str,
        agent_name: str,
        *,
        parent_run_id: str | None = None,
        delegation_level: int = 0,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost_usd: float | None = None,
        status: str = "success",
    ) -> None:
        """Helper: create a run and finalize it with tokens/cost."""
        await store.create_run(
            run_id,
            agent_name,
            parent_run_id=parent_run_id,
            delegation_level=delegation_level,
        )
        usage = UsageStats(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
        )
        await store.finalize_run(
            run_id,
            status=status,
            iteration_count=1,
            total_usage=usage,
        )

    async def test_nonexistent_run_returns_none(self, store) -> None:
        result = await store.get_delegation_info("nonexistent")
        assert result is None

    async def test_solo_root_run(self, store) -> None:
        """A root run with no children or parent."""
        await self._create_finalized_run(
            store,
            "root",
            "Orch",
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.01,
        )

        info = await store.get_delegation_info("root")
        assert info is not None
        assert info.parent is None
        assert info.children == []
        assert info.ancestry == []
        assert info.ancestry_complete is True
        ss = info.subtree_summary
        assert ss.direct_child_count == 0
        assert ss.descendant_count == 0
        assert ss.max_depth == 0
        assert ss.subtree_input_tokens == 100
        assert ss.subtree_output_tokens == 50
        assert ss.subtree_cost_usd == pytest.approx(0.01)
        assert ss.unknown_cost_count == 0
        assert ss.status_counts == {"success": 1}

    async def test_parent_child_tree(self, store) -> None:
        """Two-level tree: root → child."""
        await self._create_finalized_run(
            store,
            "root",
            "Orch",
            input_tokens=200,
            output_tokens=100,
            cost_usd=0.02,
        )
        await self._create_finalized_run(
            store,
            "child",
            "Worker",
            parent_run_id="root",
            delegation_level=1,
            input_tokens=300,
            output_tokens=150,
            cost_usd=0.03,
        )

        # Root perspective
        info = await store.get_delegation_info("root")
        assert info is not None
        assert info.parent is None
        assert len(info.children) == 1
        assert info.children[0].run_id == "child"
        ss = info.subtree_summary
        assert ss.direct_child_count == 1
        assert ss.descendant_count == 1
        assert ss.max_depth == 1
        assert ss.subtree_input_tokens == 500
        assert ss.subtree_cost_usd == pytest.approx(0.05)

        # Child perspective
        info = await store.get_delegation_info("child")
        assert info is not None
        assert info.parent is not None
        assert info.parent.run_id == "root"
        assert info.parent.resolved is True
        assert info.parent.agent_name == "Orch"
        assert len(info.ancestry) == 1
        assert info.ancestry[0].run_id == "root"
        assert info.ancestry_complete is True
        assert info.children == []

    async def test_three_level_deep_tree(self, store) -> None:
        """root → mid → leaf — verifies depth, ancestry, and subtree rollup."""
        await self._create_finalized_run(
            store,
            "root",
            "Orch",
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.01,
        )
        await self._create_finalized_run(
            store,
            "mid",
            "Research",
            parent_run_id="root",
            delegation_level=1,
            input_tokens=200,
            output_tokens=100,
            cost_usd=0.02,
        )
        await self._create_finalized_run(
            store,
            "leaf",
            "Fact",
            parent_run_id="mid",
            delegation_level=2,
            input_tokens=50,
            output_tokens=25,
            cost_usd=0.005,
            status="error",
        )

        info = await store.get_delegation_info("root")
        assert info is not None
        ss = info.subtree_summary
        assert ss.descendant_count == 2
        assert ss.max_depth == 2
        assert ss.subtree_input_tokens == 350
        assert ss.status_counts == {"success": 2, "error": 1}

        # Leaf sees full ancestry
        info = await store.get_delegation_info("leaf")
        assert info is not None
        assert len(info.ancestry) == 2
        assert info.ancestry[0].run_id == "root"
        assert info.ancestry[1].run_id == "mid"
        assert info.ancestry_complete is True

    async def test_broken_parent_chain(self, store, session_factory) -> None:
        """Parent run_id points to a missing row."""
        # Create orphan without parent first, then corrupt via raw SQL.
        # Must disable FK temporarily — SQLite enforces FKs on UPDATE too.
        await store.create_run("orphan", "Worker")
        await store.finalize_run("orphan", status="success")

        async with session_factory() as session:
            from sqlalchemy import text

            await session.execute(text("PRAGMA foreign_keys = OFF"))
            await session.execute(
                text(
                    "UPDATE agent_runs SET parent_run_id = 'ghost', "
                    "delegation_level = 1 WHERE id = 'orphan'"
                )
            )
            await session.commit()
            await session.execute(text("PRAGMA foreign_keys = ON"))

        info = await store.get_delegation_info("orphan")
        assert info is not None
        assert info.parent is not None
        assert info.parent.run_id == "ghost"
        assert info.parent.resolved is False
        assert info.ancestry == []
        assert info.ancestry_complete is False

    async def test_mixed_known_unknown_cost(self, store) -> None:
        """Subtree with some known and some NULL costs."""
        await self._create_finalized_run(
            store,
            "root",
            "Orch",
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.01,
        )
        # c1 has no cost
        await self._create_finalized_run(
            store,
            "c1",
            "W1",
            parent_run_id="root",
            delegation_level=1,
            input_tokens=200,
            output_tokens=100,
            cost_usd=None,
        )
        # c2 has cost
        await self._create_finalized_run(
            store,
            "c2",
            "W2",
            parent_run_id="root",
            delegation_level=1,
            input_tokens=300,
            output_tokens=150,
            cost_usd=0.03,
        )

        info = await store.get_delegation_info("root")
        assert info is not None
        ss = info.subtree_summary
        assert ss.subtree_cost_usd == pytest.approx(0.04)  # 0.01 + 0.03
        assert ss.unknown_cost_count == 1
        assert ss.subtree_input_tokens == 600

    async def test_all_unknown_cost(self, store) -> None:
        """When no runs have cost, subtree_cost_usd is None."""
        await self._create_finalized_run(
            store,
            "root",
            "Orch",
            input_tokens=100,
            output_tokens=50,
            cost_usd=None,
        )

        info = await store.get_delegation_info("root")
        assert info is not None
        assert info.subtree_summary.subtree_cost_usd is None
        assert info.subtree_summary.unknown_cost_count == 1

    async def test_cyclic_parent_chain(self, store, session_factory) -> None:
        """A→B→A cycle: ancestry is incomplete, BFS doesn't hang."""
        # Create A and B normally, then corrupt B's parent to point at A
        await store.create_run("a", "A")
        await store.finalize_run("a", status="success")
        await store.create_run("b", "B", parent_run_id="a", delegation_level=1)
        await store.finalize_run("b", status="success")

        # Corrupt: set A's parent_run_id to B (creating a cycle)
        async with session_factory() as session:
            from sqlalchemy import update

            await session.execute(
                update(AgentRun).where(AgentRun.id == "a").values(parent_run_id="b")
            )
            await session.commit()

        # Should not hang — cycle guard stops ancestry walk
        info = await store.get_delegation_info("a")
        assert info is not None
        assert info.ancestry_complete is False

    async def test_wide_fanout(self, store) -> None:
        """Root with 5 children — all discovered by BFS."""
        await self._create_finalized_run(
            store,
            "root",
            "Orch",
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.01,
        )
        for i in range(5):
            await self._create_finalized_run(
                store,
                f"c{i}",
                f"W{i}",
                parent_run_id="root",
                delegation_level=1,
                input_tokens=10,
                output_tokens=5,
                cost_usd=0.001,
            )

        info = await store.get_delegation_info("root")
        assert info is not None
        ss = info.subtree_summary
        assert ss.direct_child_count == 5
        assert ss.descendant_count == 5
        assert ss.max_depth == 1
        assert ss.subtree_input_tokens == 150  # 100 + 5*10
