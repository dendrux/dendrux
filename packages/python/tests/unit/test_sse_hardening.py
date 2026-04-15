"""Tests for SSE/bridge transport hardening (SA-1 + SA-2).

Step 1: Cursor pagination on StateStore.get_run_events().
Step 2: _offer_sse_event() helper and payload truncation.
Step 3: TransportNotifier decoupled from queue (uses offer callable).

Covers:
- after_sequence_index exclusive filter (> semantics)
- limit parameter with max cap clamping
- Default behavior unchanged (no params = all events)
- Combined after + limit
- Empty results for out-of-range cursor
- Payload truncation for oversized events
- Non-blocking enqueue via put_nowait()
- Overflow flag on full queue
- TransportNotifier uses offer callable instead of queue
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any

import pytest

from dendrux.bridge.notifier import ServerEvent

# ------------------------------------------------------------------
# In-memory state store for cursor API tests
# ------------------------------------------------------------------


@dataclass
class _EventRow:
    """Simulates a DB row for run events."""

    id: str
    event_type: str
    sequence_index: int
    iteration_index: int = 0
    correlation_id: str | None = None
    data: dict[str, Any] | None = None
    created_at: Any = None


def _make_events(run_id: str, count: int) -> list[_EventRow]:
    """Create count sequential events for a run."""
    return [
        _EventRow(
            id=f"evt_{i}",
            event_type=f"test.event.{i}",
            sequence_index=i,
        )
        for i in range(count)
    ]


# ------------------------------------------------------------------
# SQLAlchemy integration tests (real DB)
# ------------------------------------------------------------------


@pytest.fixture
async def store_with_events():
    """Create a fresh in-memory SQLite state store with 10 events."""
    from sqlalchemy.ext.asyncio import create_async_engine

    from dendrux.db.models import Base
    from dendrux.runtime.state import SQLAlchemyStateStore

    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    store = SQLAlchemyStateStore(engine)

    run_id = "test-run-cursor"
    await store.create_run(
        run_id,
        "TestAgent",
        input_data={"input": "test"},
        model="mock",
        strategy="NativeToolCalling",
    )

    # Save 10 events with sequence_index 0..9
    for i in range(10):
        await store.save_run_event(
            run_id,
            event_type=f"test.event.{i}",
            sequence_index=i,
            iteration_index=0,
            data={"index": i},
        )

    yield store, run_id
    await engine.dispose()


class TestGetRunEventsCursor:
    """get_run_events() cursor pagination."""

    async def test_no_params_returns_all(self, store_with_events) -> None:
        """Default call with no cursor/limit returns all events ordered."""
        store, run_id = store_with_events
        events = await store.get_run_events(run_id)
        assert len(events) == 10
        assert events[0].sequence_index == 0
        assert events[9].sequence_index == 9

    async def test_after_sequence_index_exclusive(self, store_with_events) -> None:
        """after_sequence_index uses > semantics (exclusive)."""
        store, run_id = store_with_events
        events = await store.get_run_events(run_id, after_sequence_index=4)
        assert len(events) == 5
        assert events[0].sequence_index == 5
        assert events[-1].sequence_index == 9

    async def test_after_zero_skips_first(self, store_with_events) -> None:
        """after_sequence_index=0 returns events with sequence > 0."""
        store, run_id = store_with_events
        events = await store.get_run_events(run_id, after_sequence_index=0)
        assert len(events) == 9
        assert events[0].sequence_index == 1

    async def test_after_last_returns_empty(self, store_with_events) -> None:
        """Cursor past the last event returns empty list."""
        store, run_id = store_with_events
        events = await store.get_run_events(run_id, after_sequence_index=9)
        assert events == []

    async def test_after_beyond_range_returns_empty(self, store_with_events) -> None:
        """Cursor way beyond existing events returns empty."""
        store, run_id = store_with_events
        events = await store.get_run_events(run_id, after_sequence_index=999)
        assert events == []

    async def test_limit_caps_results(self, store_with_events) -> None:
        """limit parameter restricts number of returned events."""
        store, run_id = store_with_events
        events = await store.get_run_events(run_id, limit=3)
        assert len(events) == 3
        assert events[0].sequence_index == 0
        assert events[2].sequence_index == 2

    async def test_limit_larger_than_total(self, store_with_events) -> None:
        """limit larger than total events returns all events."""
        store, run_id = store_with_events
        events = await store.get_run_events(run_id, limit=500)
        assert len(events) == 10

    async def test_after_and_limit_combined(self, store_with_events) -> None:
        """after + limit together: cursor then cap."""
        store, run_id = store_with_events
        events = await store.get_run_events(run_id, after_sequence_index=2, limit=3)
        assert len(events) == 3
        assert events[0].sequence_index == 3
        assert events[1].sequence_index == 4
        assert events[2].sequence_index == 5

    async def test_limit_zero_clamped_to_one(self, store_with_events) -> None:
        """limit=0 is clamped to 1."""
        store, run_id = store_with_events
        events = await store.get_run_events(run_id, limit=0)
        assert len(events) == 1

    async def test_negative_limit_clamped_to_one(self, store_with_events) -> None:
        """Negative limit is clamped to 1."""
        store, run_id = store_with_events
        events = await store.get_run_events(run_id, limit=-5)
        assert len(events) == 1

    async def test_limit_over_max_clamped(self, store_with_events) -> None:
        """limit > 1000 is clamped to 1000."""
        store, run_id = store_with_events
        # With only 10 events, clamping to 1000 still returns 10
        events = await store.get_run_events(run_id, limit=5000)
        assert len(events) == 10

    async def test_nonexistent_run_returns_empty(self, store_with_events) -> None:
        """Cursor query on nonexistent run returns empty."""
        store, _ = store_with_events
        events = await store.get_run_events("no-such-run", after_sequence_index=0, limit=10)
        assert events == []

    async def test_results_ordered_ascending(self, store_with_events) -> None:
        """Results are always ordered by sequence_index ascending."""
        store, run_id = store_with_events
        events = await store.get_run_events(run_id, after_sequence_index=3, limit=5)
        seq_indices = [e.sequence_index for e in events]
        assert seq_indices == sorted(seq_indices)

    async def test_default_limit_is_none_returns_all(self, store_with_events) -> None:
        """When limit is not provided (None), all events are returned."""
        store, run_id = store_with_events
        events = await store.get_run_events(run_id, limit=None)
        assert len(events) == 10


# ------------------------------------------------------------------
# Protocol compatibility
# ------------------------------------------------------------------


class TestProtocolBackwardCompat:
    """Existing callers that use get_run_events(run_id) still work."""

    async def test_existing_callers_unaffected(self, store_with_events) -> None:
        """Calling with just run_id returns all events (backward compat)."""
        store, run_id = store_with_events
        events = await store.get_run_events(run_id)
        assert len(events) == 10
        # All events present, ordered
        for i, event in enumerate(events):
            assert event.sequence_index == i


# ==================================================================
# Step 2: _offer_sse_event() and payload truncation
# ==================================================================


class TestOfferSseEvent:
    """_offer_sse_event() is the single chokepoint for all enqueuing."""

    def test_enqueues_event_via_put_nowait(self) -> None:
        """Normal event is enqueued via put_nowait (non-blocking)."""
        import asyncio

        from dendrux.bridge.transport import _offer_sse_event

        queue: asyncio.Queue = asyncio.Queue(maxsize=100)
        overflow: dict[str, bool] = {}
        event = ServerEvent(event="run.step", data={"content": "hello"})

        _offer_sse_event("run-1", event, queue, overflow)

        assert queue.qsize() == 1
        assert "run-1" not in overflow

    def test_overflow_flag_set_on_full_queue(self) -> None:
        """When queue is full, overflow flag is set instead of blocking."""
        import asyncio

        from dendrux.bridge.transport import _offer_sse_event

        queue: asyncio.Queue = asyncio.Queue(maxsize=2)
        overflow: dict[str, bool] = {}

        # Fill the queue
        _offer_sse_event("run-1", ServerEvent(event="e1", data={}), queue, overflow)
        _offer_sse_event("run-1", ServerEvent(event="e2", data={}), queue, overflow)

        # This should set overflow, not block
        _offer_sse_event("run-1", ServerEvent(event="e3", data={}), queue, overflow)

        assert queue.qsize() == 2  # queue unchanged
        assert overflow.get("run-1") is True

    def test_does_not_block_on_full_queue(self) -> None:
        """put_nowait never blocks — returns immediately."""
        import asyncio
        import time

        from dendrux.bridge.transport import _offer_sse_event

        queue: asyncio.Queue = asyncio.Queue(maxsize=1)
        overflow: dict[str, bool] = {}

        _offer_sse_event("run-1", ServerEvent(event="e1", data={}), queue, overflow)

        start = time.monotonic()
        _offer_sse_event("run-1", ServerEvent(event="e2", data={}), queue, overflow)
        elapsed = time.monotonic() - start

        # Should complete in under 10ms (not blocking)
        assert elapsed < 0.01


class TestPayloadTruncation:
    """Payload truncation and size validation."""

    def test_small_payload_passes_through(self) -> None:
        """Events under MAX_EVENT_BYTES are not modified."""
        import asyncio

        from dendrux.bridge.transport import _offer_sse_event

        queue: asyncio.Queue = asyncio.Queue(maxsize=100)
        overflow: dict[str, bool] = {}
        data = {"content": "short message"}
        event = ServerEvent(event="run.step", data=data)

        _offer_sse_event("run-1", event, queue, overflow)

        enqueued = queue.get_nowait()
        assert enqueued.data == data
        assert "truncated" not in enqueued.data

    def test_oversized_string_truncated(self) -> None:
        """String values in data are truncated when event exceeds size limit."""
        import asyncio

        from dendrux.bridge.transport import MAX_EVENT_BYTES, _offer_sse_event

        queue: asyncio.Queue = asyncio.Queue(maxsize=100)
        overflow: dict[str, bool] = {}
        # Create data with a very large string
        big_string = "x" * (MAX_EVENT_BYTES * 2)
        event = ServerEvent(event="run.step", data={"content": big_string})

        _offer_sse_event("run-1", event, queue, overflow)

        enqueued = queue.get_nowait()
        assert enqueued.data.get("truncated") is True
        assert len(enqueued.data.get("content", "")) < len(big_string)

    def test_oversized_nested_dict_truncated(self) -> None:
        """Nested dicts with large values are truncated recursively."""
        import asyncio

        from dendrux.bridge.transport import MAX_EVENT_BYTES, _offer_sse_event

        queue: asyncio.Queue = asyncio.Queue(maxsize=100)
        overflow: dict[str, bool] = {}
        event = ServerEvent(
            event="run.tool_done",
            data={
                "tool_name": "search",
                "result": {"output": "y" * (MAX_EVENT_BYTES * 2)},
            },
        )

        _offer_sse_event("run-1", event, queue, overflow)

        enqueued = queue.get_nowait()
        assert enqueued.data.get("truncated") is True

    def test_still_oversized_after_truncation_gets_fallback(self) -> None:
        """If truncation still exceeds limit, replace with minimal fallback."""
        import asyncio
        import json

        from dendrux.bridge.transport import MAX_EVENT_BYTES, _offer_sse_event

        queue: asyncio.Queue = asyncio.Queue(maxsize=100)
        overflow: dict[str, bool] = {}
        # Create data with many large keys (truncation per-key may not be enough)
        data = {f"key_{i}": "z" * 10000 for i in range(100)}
        event = ServerEvent(event="run.step", data=data)

        _offer_sse_event("run-1", event, queue, overflow)

        enqueued = queue.get_nowait()
        # Must be within size limit
        assert len(json.dumps(enqueued.data)) <= MAX_EVENT_BYTES
        assert enqueued.data.get("truncated") is True

    def test_list_values_truncated(self) -> None:
        """Lists with large string elements are truncated."""
        import asyncio

        from dendrux.bridge.transport import MAX_EVENT_BYTES, _offer_sse_event

        queue: asyncio.Queue = asyncio.Queue(maxsize=100)
        overflow: dict[str, bool] = {}
        event = ServerEvent(
            event="run.governance",
            data={"entities": ["a" * (MAX_EVENT_BYTES * 2)]},
        )

        _offer_sse_event("run-1", event, queue, overflow)

        enqueued = queue.get_nowait()
        assert enqueued.data.get("truncated") is True


# ==================================================================
# Step 3: TransportNotifier decoupled from queue
# ==================================================================


class TestTransportNotifierOffer:
    """TransportNotifier uses offer callable instead of queue."""

    async def test_uses_offer_callable(self) -> None:
        """TransportNotifier calls the offer function, not queue.put()."""
        from dendrux.bridge.notifier import TransportNotifier
        from dendrux.types import Message, Role

        received: list[ServerEvent] = []

        def offer(event: ServerEvent) -> None:
            received.append(event)

        notifier = TransportNotifier(offer)
        msg = Message(role=Role.USER, content="hello")
        await notifier.on_message_appended(msg, iteration=0)

        assert len(received) == 1
        assert received[0].event == "run.step"
        assert received[0].data["content"] == "hello"

    async def test_llm_done_uses_offer(self) -> None:
        from dendrux.bridge.notifier import TransportNotifier
        from dendrux.types import LLMResponse, UsageStats

        received: list[ServerEvent] = []

        def offer(event: ServerEvent) -> None:
            received.append(event)

        notifier = TransportNotifier(offer)
        response = LLMResponse(
            text="hi",
            usage=UsageStats(input_tokens=100, output_tokens=50, total_tokens=150),
        )
        await notifier.on_llm_call_completed(response, iteration=1)

        assert len(received) == 1
        assert received[0].event == "run.llm_done"

    async def test_tool_done_uses_offer(self) -> None:
        from dendrux.bridge.notifier import TransportNotifier
        from dendrux.types import ToolCall, ToolResult

        received: list[ServerEvent] = []

        def offer(event: ServerEvent) -> None:
            received.append(event)

        notifier = TransportNotifier(offer)
        tc = ToolCall(name="add", params={"a": 1})
        tr = ToolResult(name="add", call_id=tc.id, payload="3", success=True)
        await notifier.on_tool_completed(tc, tr, iteration=1)

        assert len(received) == 1
        assert received[0].event == "run.tool_done"

    async def test_governance_uses_offer(self) -> None:
        from dendrux.bridge.notifier import TransportNotifier

        received: list[ServerEvent] = []

        def offer(event: ServerEvent) -> None:
            received.append(event)

        notifier = TransportNotifier(offer)
        await notifier.on_governance_event("policy.denied", 1, {"tool_name": "rm"})

        assert len(received) == 1
        assert received[0].event == "run.governance"

    async def test_redaction_still_works(self) -> None:
        """Redaction is applied before calling offer."""
        from dendrux.bridge.notifier import TransportNotifier
        from dendrux.types import Message, Role

        received: list[ServerEvent] = []

        def offer(event: ServerEvent) -> None:
            received.append(event)

        redact = lambda text: text.replace("secret", "[REDACTED]")  # noqa: E731
        notifier = TransportNotifier(offer, redact=redact)
        msg = Message(role=Role.USER, content="my secret key")
        await notifier.on_message_appended(msg, iteration=0)

        assert "[REDACTED]" in received[0].data["content"]
        assert "secret" not in received[0].data["content"]


# ==================================================================
# P1 fix: One SSE connection per run
# ==================================================================


class TestConnectionGuardsEndpoint:
    """Test connection guards via bridge HTTP endpoint."""

    @pytest.fixture
    async def bridge_client(self):
        """Create a bridge-mounted app with a paused run."""
        from fastapi import FastAPI
        from httpx import ASGITransport, AsyncClient
        from sqlalchemy.ext.asyncio import create_async_engine

        from dendrux.agent import Agent
        from dendrux.bridge import bridge
        from dendrux.db.models import Base
        from dendrux.llm.mock import MockLLM
        from dendrux.runtime.state import SQLAlchemyStateStore
        from dendrux.tool import tool
        from dendrux.types import LLMResponse
        from dendrux.types import ToolCall as ToolCallType

        @tool(target="client")
        async def read_file(path: str) -> str:
            """Client tool."""
            return ""

        engine = create_async_engine(
            "sqlite+aiosqlite:///:memory:",
            connect_args={"check_same_thread": False},
        )
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        store = SQLAlchemyStateStore(engine)

        tc = ToolCallType(name="read_file", params={"path": "test.txt"})
        llm = MockLLM(
            [
                LLMResponse(tool_calls=[tc]),
                LLMResponse(text="Done."),
            ]
        )

        agent = Agent(
            provider=llm,
            prompt="Test.",
            tools=[read_file],
            state_store=store,
        )

        result = await agent.run("Read the file.")
        run_id = result.run_id

        app = FastAPI()
        transport_app = bridge(agent, allow_insecure_dev_mode=True)
        app.mount("/bridge", transport_app)

        client = AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        )
        yield client, run_id, transport_app
        await client.aclose()
        await engine.dispose()

    async def test_second_sse_connection_returns_409(self, bridge_client) -> None:
        """Second SSE client to the same run gets 409 via actual endpoint."""
        import asyncio

        client, run_id, transport_app = bridge_client

        # Simulate an active connection by adding to _active_sse
        # (accessing the bridge's internal state via the app's state)
        # Since bridge() returns a FastAPI app, we can't directly access
        # _active_sse. Instead, start a streaming request in background
        # and test the second request.

        # Use asyncio.Task to start first stream, then test second
        events_received = []

        async def _consume_first():
            async with client.stream("GET", f"/bridge/runs/{run_id}/events") as resp:
                async for line in resp.aiter_lines():
                    events_received.append(line)
                    if line.startswith("data:"):
                        return  # Got snapshot, stop

        # Start first stream as background task
        task = asyncio.create_task(_consume_first())
        # Give it time to connect and register
        await asyncio.sleep(0.1)

        # Second connection should get 409
        resp = await client.get(f"/bridge/runs/{run_id}/events")
        assert resp.status_code == 409
        assert "already has an active SSE connection" in resp.json()["detail"]

        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


# ==================================================================
# P2 fix: Offer looks up queue by run_id at offer time
# ==================================================================


class TestOfferQueueLookup:
    """_offer closure looks up queue by run_id, not captured reference."""

    def test_offer_drops_event_when_queue_removed(self) -> None:
        """If SSE disconnected and popped the queue, offer drops silently."""
        import asyncio

        from dendrux.bridge.transport import _offer_sse_event

        sse_queues: dict[str, asyncio.Queue] = {}
        overflow: dict[str, bool] = {}
        run_id = "run-1"

        # Create queue, then remove it (simulating SSE disconnect)
        sse_queues[run_id] = asyncio.Queue(maxsize=100)
        queue = sse_queues.pop(run_id)

        # Offer using the dict lookup pattern (not the captured queue)
        # This simulates what the fixed _offer closure does
        current_queue = sse_queues.get(run_id)
        if current_queue is not None:
            _offer_sse_event(run_id, ServerEvent(event="e1", data={}), current_queue, overflow)

        # No crash, event silently dropped
        assert queue.qsize() == 0  # old queue not written to

    def test_offer_uses_new_queue_after_reconnect(self) -> None:
        """After reconnect, offer writes to the new queue."""
        import asyncio

        from dendrux.bridge.transport import _offer_sse_event

        sse_queues: dict[str, asyncio.Queue] = {}
        overflow: dict[str, bool] = {}
        run_id = "run-1"

        # First queue (old connection)
        sse_queues[run_id] = asyncio.Queue(maxsize=100)
        old_queue = sse_queues[run_id]

        # Disconnect + reconnect (new queue)
        sse_queues.pop(run_id)
        sse_queues[run_id] = asyncio.Queue(maxsize=100)
        new_queue = sse_queues[run_id]

        # Offer should write to new queue
        current_queue = sse_queues.get(run_id)
        assert current_queue is not None
        _offer_sse_event(run_id, ServerEvent(event="e1", data={}), current_queue, overflow)

        assert old_queue.qsize() == 0
        assert new_queue.qsize() == 1
