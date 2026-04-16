"""Integration tests for RunStore — the public read facade.

Exercises the public contract against real SQLite. RunStore sits over
SQLAlchemyStateStore; these tests verify that the public dataclasses,
filters, cursors, and stream semantics are correct and that
``get_pauses`` derives from ``run_events`` only (never ``pause_data``).
"""

from __future__ import annotations

import asyncio

import pytest
from sqlalchemy import event
from sqlalchemy.ext.asyncio import create_async_engine

from dendrux.db.models import Base
from dendrux.runtime.state import SQLAlchemyStateStore
from dendrux.store import (
    LLMCall,
    PausePair,
    RunDetail,
    RunNotFoundError,
    RunStore,
    RunSummary,
    StoredEvent,
    ToolInvocation,
)
from dendrux.types import UsageStats

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
async def engine():
    eng = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )

    @event.listens_for(eng.sync_engine, "connect")
    def _enable_fk(dbapi_conn, _connection_record):
        dbapi_conn.execute("PRAGMA foreign_keys = ON")

    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield eng
    await eng.dispose()


@pytest.fixture
def internal_store(engine):
    return SQLAlchemyStateStore(engine)


@pytest.fixture
def store(engine):
    return RunStore.from_engine(engine)


# ------------------------------------------------------------------
# list_runs
# ------------------------------------------------------------------


class TestListRuns:
    async def test_returns_public_dataclass(self, store, internal_store) -> None:
        await internal_store.create_run("r1", "Alpha", model="m")

        runs = await store.list_runs()
        assert len(runs) == 1
        assert isinstance(runs[0], RunSummary)
        assert runs[0].run_id == "r1"
        assert runs[0].agent_name == "Alpha"

    async def test_pagination(self, store, internal_store) -> None:
        for i in range(5):
            await internal_store.create_run(f"r_{i}", "Agent")
        page1 = await store.list_runs(limit=2, offset=0)
        page2 = await store.list_runs(limit=2, offset=2)
        assert len(page1) == 2
        assert len(page2) == 2
        assert {r.run_id for r in page1}.isdisjoint({r.run_id for r in page2})

    async def test_filter_agent_name(self, store, internal_store) -> None:
        await internal_store.create_run("r1", "Alpha")
        await internal_store.create_run("r2", "Beta")
        await internal_store.create_run("r3", "Alpha")

        runs = await store.list_runs(agent_name="Alpha")
        assert {r.run_id for r in runs} == {"r1", "r3"}

    async def test_filter_parent_run_id(self, store, internal_store) -> None:
        await internal_store.create_run("parent", "Coord")
        await internal_store.create_run("child_a", "Worker", parent_run_id="parent")
        await internal_store.create_run("orphan", "Worker")

        runs = await store.list_runs(parent_run_id="parent")
        assert [r.run_id for r in runs] == ["child_a"]

    async def test_filter_status_list(self, store, internal_store) -> None:
        await internal_store.create_run("r1", "Agent")
        await internal_store.create_run("r2", "Agent")
        await internal_store.create_run("r3", "Agent")
        await internal_store.finalize_run("r1", status="success")
        await internal_store.finalize_run("r2", status="error")

        runs = await store.list_runs(status=["success", "error"])
        assert {r.run_id for r in runs} == {"r1", "r2"}

    async def test_empty(self, store) -> None:
        assert await store.list_runs() == []


# ------------------------------------------------------------------
# count_runs
# ------------------------------------------------------------------


class TestCountRuns:
    async def test_counts_match_list(self, store, internal_store) -> None:
        for i in range(4):
            await internal_store.create_run(f"r_{i}", "Agent")

        assert await store.count_runs() == 4

    async def test_count_with_filters(self, store, internal_store) -> None:
        await internal_store.create_run("r1", "Alpha")
        await internal_store.create_run("r2", "Beta")
        await internal_store.create_run("r3", "Alpha")

        assert await store.count_runs(agent_name="Alpha") == 2
        assert await store.count_runs(agent_name="Beta") == 1
        assert await store.count_runs(agent_name="Missing") == 0


# ------------------------------------------------------------------
# get_run
# ------------------------------------------------------------------


class TestGetRun:
    async def test_existing(self, store, internal_store) -> None:
        await internal_store.create_run("r1", "Analyst", model="claude-sonnet", strategy="react")
        await internal_store.finalize_run("r1", status="success", answer="done")

        detail = await store.get_run("r1")
        assert detail is not None
        assert isinstance(detail, RunDetail)
        assert detail.run_id == "r1"
        assert detail.agent_name == "Analyst"
        assert detail.status == "success"
        assert detail.answer == "done"
        assert detail.model == "claude-sonnet"

    async def test_unknown(self, store) -> None:
        assert await store.get_run("missing") is None


# ------------------------------------------------------------------
# get_events
# ------------------------------------------------------------------


class TestGetEvents:
    async def test_returns_public_dataclass_ordered_by_sequence(
        self, store, internal_store
    ) -> None:
        await internal_store.create_run("r1", "Agent")
        await internal_store.save_run_event("r1", event_type="run.started", sequence_index=0)
        await internal_store.save_run_event("r1", event_type="llm.completed", sequence_index=1)
        await internal_store.save_run_event("r1", event_type="run.completed", sequence_index=2)

        events = await store.get_events("r1")
        assert len(events) == 3
        assert all(isinstance(e, StoredEvent) for e in events)
        assert [e.sequence_index for e in events] == [0, 1, 2]
        assert events[1].event_type == "llm.completed"

    async def test_cursor(self, store, internal_store) -> None:
        await internal_store.create_run("r1", "Agent")
        for i in range(5):
            await internal_store.save_run_event("r1", event_type="tick", sequence_index=i)

        rest = await store.get_events("r1", after_sequence_index=2)
        assert [e.sequence_index for e in rest] == [3, 4]

    async def test_unknown_run_returns_empty(self, store) -> None:
        assert await store.get_events("missing") == []


# ------------------------------------------------------------------
# stream_events
# ------------------------------------------------------------------


class TestStreamEvents:
    async def test_yields_existing_then_new(self, store, internal_store) -> None:
        await internal_store.create_run("r1", "Agent")
        await internal_store.save_run_event("r1", event_type="run.started", sequence_index=0)

        collected: list[StoredEvent] = []

        async def _consume() -> None:
            stream = store.stream_events("r1", poll_interval_s=0.01)
            async for ev in stream:
                collected.append(ev)
                if ev.sequence_index == 1:
                    await stream.aclose()
                    return

        async def _write_later() -> None:
            await asyncio.sleep(0.05)
            await internal_store.save_run_event("r1", event_type="run.completed", sequence_index=1)

        await asyncio.gather(_consume(), _write_later())

        assert [e.sequence_index for e in collected] == [0, 1]

    async def test_aclose_stops_cleanly(self, store, internal_store) -> None:
        await internal_store.create_run("r1", "Agent")

        stream = store.stream_events("r1", poll_interval_s=0.01)
        task = asyncio.create_task(stream.__anext__())
        await asyncio.sleep(0.02)

        # Cancel the in-flight __anext__ before aclose — an async generator
        # with a frame executing cannot be aclose()d directly.
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        # Now the generator can shut down cleanly without leaking the
        # sleep task or the DB connection.
        await stream.aclose()

    async def test_404_on_unknown_run(self, store) -> None:
        stream = store.stream_events("missing", poll_interval_s=0.01)
        with pytest.raises(RunNotFoundError):
            await stream.__anext__()

    async def test_resumes_from_cursor(self, store, internal_store) -> None:
        await internal_store.create_run("r1", "Agent")
        for i in range(3):
            await internal_store.save_run_event("r1", event_type="tick", sequence_index=i)

        stream = store.stream_events("r1", after_sequence_index=1, poll_interval_s=0.01)
        first = await stream.__anext__()
        await stream.aclose()
        assert first.sequence_index == 2


# ------------------------------------------------------------------
# get_llm_calls
# ------------------------------------------------------------------


class TestGetLLMCalls:
    async def test_returns_public_dataclass(self, store, internal_store) -> None:
        await internal_store.create_run("r1", "Agent")
        await internal_store.save_llm_interaction(
            "r1",
            iteration_index=0,
            usage=UsageStats(
                input_tokens=100,
                output_tokens=20,
                total_tokens=120,
                cache_read_input_tokens=50,
                cache_creation_input_tokens=10,
            ),
            model="claude-sonnet",
            provider="anthropic",
            duration_ms=250,
        )

        calls = await store.get_llm_calls("r1")
        assert len(calls) == 1
        call = calls[0]
        assert isinstance(call, LLMCall)
        assert call.iteration == 0
        assert call.input_tokens == 100
        assert call.output_tokens == 20
        assert call.cache_read_input_tokens == 50
        assert call.cache_creation_input_tokens == 10
        assert call.model == "claude-sonnet"

    async def test_filter_by_iteration(self, store, internal_store) -> None:
        await internal_store.create_run("r1", "Agent")
        for i in range(3):
            await internal_store.save_llm_interaction(
                "r1",
                iteration_index=i,
                usage=UsageStats(input_tokens=1, output_tokens=1, total_tokens=2),
            )

        only = await store.get_llm_calls("r1", iteration=1)
        assert len(only) == 1
        assert only[0].iteration == 1


# ------------------------------------------------------------------
# get_tool_invocations
# ------------------------------------------------------------------


class TestGetToolInvocations:
    async def test_returns_public_dataclass(self, store, internal_store) -> None:
        await internal_store.create_run("r1", "Agent")
        await internal_store.save_tool_call(
            "r1",
            tool_call_id="tc_1",
            provider_tool_call_id="call_abc",
            tool_name="lookup_price",
            target="server",
            params={"ticker": "AAPL"},
            result_payload='"AAPL: $227.50"',
            success=True,
            duration_ms=42,
            iteration_index=0,
        )

        tools = await store.get_tool_invocations("r1")
        assert len(tools) == 1
        tool = tools[0]
        assert isinstance(tool, ToolInvocation)
        assert tool.tool_name == "lookup_price"
        assert tool.success is True
        assert tool.duration_ms == 42
        assert tool.params == {"ticker": "AAPL"}
        assert tool.iteration == 0

    async def test_filter_by_iteration(self, store, internal_store) -> None:
        await internal_store.create_run("r1", "Agent")
        for i in range(2):
            await internal_store.save_tool_call(
                "r1",
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

        only = await store.get_tool_invocations("r1", iteration=1)
        assert len(only) == 1
        assert only[0].iteration == 1


# ------------------------------------------------------------------
# get_pauses — derived from run_events, NEVER from pause_data
# ------------------------------------------------------------------


class TestGetPauses:
    async def test_pairs_paused_and_resumed(self, store, internal_store) -> None:
        await internal_store.create_run("r1", "Agent")
        await internal_store.save_run_event(
            "r1",
            event_type="run.paused",
            sequence_index=0,
            data={
                "status": "waiting_client_tool",
                "pending_tool_calls": [{"id": "tc_1", "name": "read_file", "target": "client"}],
            },
        )
        await internal_store.save_run_event(
            "r1",
            event_type="run.resumed",
            sequence_index=1,
            data={
                "resumed_from": "waiting_client_tool",
                "submitted_results": [{"call_id": "tc_1", "name": "read_file", "success": True}],
            },
        )

        pauses = await store.get_pauses("r1")
        assert len(pauses) == 1
        pause = pauses[0]
        assert isinstance(pause, PausePair)
        assert pause.reason == "waiting_client_tool"
        assert pause.pause_sequence_index == 0
        assert pause.resume_sequence_index == 1
        assert pause.resume_at is not None
        assert pause.pending_tool_calls == [{"id": "tc_1", "name": "read_file", "target": "client"}]

    async def test_unpaired_pause_is_active(self, store, internal_store) -> None:
        """A run.paused event with no following run.resumed is still pending."""
        await internal_store.create_run("r1", "Agent")
        await internal_store.save_run_event(
            "r1",
            event_type="run.paused",
            sequence_index=0,
            data={
                "status": "waiting_approval",
                "pending_tool_calls": [],
            },
        )

        pauses = await store.get_pauses("r1")
        assert len(pauses) == 1
        assert pauses[0].resume_sequence_index is None
        assert pauses[0].resume_at is None
        assert pauses[0].reason == "waiting_approval"

    async def test_does_not_read_pause_data(self, store, internal_store) -> None:
        """get_pauses must derive from run_events; AgentRun.pause_data is
        execution state (unredacted, cleared on finalize) and must not be
        exposed to public readers."""
        await internal_store.create_run("r1", "Agent")
        # Populate pause_data with a marker that should NEVER appear anywhere.
        await internal_store.pause_run(
            "r1",
            status="waiting_client_tool",
            pause_data={
                "secret_marker": "UNREDACTED_EXECUTION_STATE",
                "history": [{"role": "user", "content": "top secret"}],
            },
        )
        # Emit only a run.paused event with observable data.
        await internal_store.save_run_event(
            "r1",
            event_type="run.paused",
            sequence_index=0,
            data={"status": "waiting_client_tool", "pending_tool_calls": []},
        )

        pauses = await store.get_pauses("r1")
        serialized = repr(pauses)
        assert "UNREDACTED_EXECUTION_STATE" not in serialized
        assert "top secret" not in serialized

    async def test_multiple_pauses_in_order(self, store, internal_store) -> None:
        await internal_store.create_run("r1", "Agent")
        # First pause/resume cycle.
        await internal_store.save_run_event(
            "r1",
            event_type="run.paused",
            sequence_index=0,
            data={"status": "waiting_client_tool", "pending_tool_calls": []},
        )
        await internal_store.save_run_event(
            "r1",
            event_type="run.resumed",
            sequence_index=1,
            data={"resumed_from": "waiting_client_tool"},
        )
        # Second pause, still pending.
        await internal_store.save_run_event(
            "r1",
            event_type="run.paused",
            sequence_index=2,
            data={"status": "waiting_approval", "pending_tool_calls": []},
        )

        pauses = await store.get_pauses("r1")
        assert len(pauses) == 2
        assert pauses[0].resume_sequence_index == 1
        assert pauses[1].resume_sequence_index is None
