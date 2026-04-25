"""Integration tests for RunStore — the public read facade across the backend matrix.

Exercises the public contract. RunStore sits over SQLAlchemyStateStore;
these tests verify that the public dataclasses, filters, cursors, and
stream semantics are correct and that ``get_pauses`` derives from
``run_events`` only (never ``pause_data``).

The ``engine`` fixture lives in ``tests/integration/conftest.py`` and
parametrizes across SQLite and Postgres. The local ``store`` fixture
overrides the conftest one because RunStore is the surface-under-test
here, not SQLAlchemyStateStore.
"""

from __future__ import annotations

import asyncio

import pytest

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
    TraceEntry,
)
from dendrux.types import UsageStats

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


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

    async def test_wait_for_timeout_cancels_cleanly(self, store, internal_store) -> None:
        """Cooperative cancellation via ``asyncio.wait_for`` should time out
        and let the generator close without leaking sessions."""
        await internal_store.create_run("r1", "Agent")

        stream = store.stream_events("r1", poll_interval_s=0.5)
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(stream.__anext__(), timeout=0.05)
        await stream.aclose()


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

    async def test_sql_pagination(self, store, internal_store) -> None:
        """limit/offset push to SQL — ordered by iteration_index."""
        await internal_store.create_run("r1", "Agent")
        for i in range(5):
            await internal_store.save_llm_interaction(
                "r1",
                iteration_index=i,
                usage=UsageStats(input_tokens=1, output_tokens=1, total_tokens=2),
            )

        page = await store.get_llm_calls("r1", limit=2, offset=2)
        assert [c.iteration for c in page] == [2, 3]


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

    async def test_sql_pagination(self, store, internal_store) -> None:
        """limit/offset push to SQL — ordered by created_at + id."""
        await internal_store.create_run("r1", "Agent")
        for i in range(4):
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

        page = await store.get_tool_invocations("r1", limit=2, offset=1)
        assert len(page) == 2


# ------------------------------------------------------------------
# get_traces
# ------------------------------------------------------------------


class TestGetTraces:
    async def test_returns_public_dataclass_ordered(self, store, internal_store) -> None:
        await internal_store.create_run("r1", "Agent")
        await internal_store.save_trace("r1", "user", "hello", order_index=0)
        await internal_store.save_trace("r1", "assistant", "hi", order_index=1)
        await internal_store.save_trace(
            "r1", "tool", '{"result": 42}', order_index=2, meta={"call_id": "c1"}
        )

        traces = await store.get_traces("r1")
        assert len(traces) == 3
        assert all(isinstance(t, TraceEntry) for t in traces)
        assert [t.role for t in traces] == ["user", "assistant", "tool"]
        assert [t.order_index for t in traces] == [0, 1, 2]
        assert traces[2].meta == {"call_id": "c1"}

    async def test_unknown_run_returns_empty(self, store) -> None:
        assert await store.get_traces("missing") == []

    async def test_sql_pagination(self, store, internal_store) -> None:
        """limit/offset push to SQL — ordered by order_index."""
        await internal_store.create_run("r1", "Agent")
        for i in range(5):
            await internal_store.save_trace(
                "r1", "user" if i % 2 == 0 else "assistant", f"msg {i}", order_index=i
            )

        page = await store.get_traces("r1", limit=2, offset=1)
        assert [t.order_index for t in page] == [1, 2]


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

    async def test_paused_with_no_status_data(self, store, internal_store) -> None:
        """A paused event without ``status`` in ``data`` still produces a pair."""
        await internal_store.create_run("r1", "Agent")
        await internal_store.save_run_event(
            "r1", event_type="run.paused", sequence_index=0, data={}
        )

        pauses = await store.get_pauses("r1")
        assert len(pauses) == 1
        assert pauses[0].reason == ""
        assert pauses[0].pending_tool_calls == []

    async def test_skips_unrelated_events(self, store, internal_store) -> None:
        """get_pauses filters by event_type at the SQL layer; non-pause/resume
        events do not appear in the pair derivation."""
        await internal_store.create_run("r1", "Agent")
        await internal_store.save_run_event("r1", event_type="run.started", sequence_index=0)
        await internal_store.save_run_event(
            "r1",
            event_type="run.paused",
            sequence_index=1,
            data={"status": "waiting_approval"},
        )
        await internal_store.save_run_event("r1", event_type="llm.completed", sequence_index=2)
        await internal_store.save_run_event(
            "r1", event_type="run.resumed", sequence_index=3, data={}
        )

        pauses = await store.get_pauses("r1")
        assert len(pauses) == 1
        assert pauses[0].pause_sequence_index == 1
        assert pauses[0].resume_sequence_index == 3


# ------------------------------------------------------------------
# RunStore lifecycle — close() and async context manager
# ------------------------------------------------------------------


class TestRunStoreLifecycle:
    async def test_from_database_url_supports_aclose(self) -> None:
        """RunStore created via ``from_database_url`` owns its engine and
        must be closed by the caller."""
        s = RunStore.from_database_url("sqlite+aiosqlite:///:memory:")
        await s.close()

    async def test_async_context_manager_disposes_owned_engine(self) -> None:
        """`async with RunStore.from_database_url(...)` disposes on exit
        and clears the owned-engine reference."""
        async with RunStore.from_database_url("sqlite+aiosqlite:///:memory:") as s:
            assert isinstance(s, RunStore)
            assert s._owned_engine is not None
        # After exit, ownership cleared.
        assert s._owned_engine is None


class TestGetPIIMapping:
    """RunStore.get_pii_mapping exposes the audit key without forcing
    callers to reach into private internals like _resolve_state_store."""

    async def test_returns_mapping_for_run_with_pii(self, store, internal_store) -> None:
        await internal_store.create_run("r1", "Agent")
        await internal_store.pause_run(
            "r1",
            status="waiting_client_tool",
            pause_data={},
            pii_mapping={
                "<<EMAIL_ADDRESS_1>>": "alice@example.com",
                "<<LOCATION_1>>": "San Francisco",
            },
        )

        mapping = await store.get_pii_mapping("r1")
        assert mapping == {
            "<<EMAIL_ADDRESS_1>>": "alice@example.com",
            "<<LOCATION_1>>": "San Francisco",
        }

    async def test_empty_mapping_when_no_pii(self, store, internal_store) -> None:
        await internal_store.create_run("r1", "Agent")
        mapping = await store.get_pii_mapping("r1")
        assert mapping == {}

    async def test_unknown_run_returns_empty(self, store) -> None:
        mapping = await store.get_pii_mapping("does-not-exist")
        assert mapping == {}


class TestOwnedEngineLifecycle:
    async def test_close_is_idempotent(self) -> None:
        """Calling close() twice on an owned-engine store is a no-op
        and must not raise."""
        s = RunStore.from_database_url("sqlite+aiosqlite:///:memory:")
        await s.close()
        await s.close()

    async def test_query_after_close_raises(self) -> None:
        """Operations on a disposed engine surface an error rather than
        silently returning stale or empty data."""
        from sqlalchemy.exc import SQLAlchemyError

        s = RunStore.from_database_url("sqlite+aiosqlite:///:memory:")
        await s.close()
        with pytest.raises(SQLAlchemyError):
            await s.list_runs()

    async def test_from_engine_does_not_dispose_caller_owned(self, engine) -> None:
        """RunStore created via ``from_engine`` must not dispose the engine
        — the caller still owns it."""
        s = RunStore.from_engine(engine)
        await s.close()

        # Engine must still be usable after store close.
        from dendrux.runtime.state import SQLAlchemyStateStore

        check = SQLAlchemyStateStore(engine)
        await check.create_run("after_close", "Agent")
        assert (await check.get_run("after_close")) is not None
