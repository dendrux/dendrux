"""Tests for bridge infrastructure — observer and task manager."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from dendrux.bridge.observer import CompositeObserver, ServerEvent, TransportObserver
from dendrux.bridge.tasks import RunTaskManager
from dendrux.types import Message, Role, ToolCall, ToolResult


class TestTransportObserver:
    async def test_queues_events(self) -> None:
        queue: asyncio.Queue[ServerEvent] = asyncio.Queue()
        obs = TransportObserver(queue)

        msg = Message(role=Role.USER, content="hello")
        await obs.on_message_appended(msg, iteration=0)

        event = await asyncio.wait_for(queue.get(), timeout=1.0)
        assert event.event == "run.step"
        assert event.data["role"] == "user"
        assert event.data["content"] == "hello"

    async def test_redaction_applied_to_content(self) -> None:
        """H-006: TransportObserver redacts message content when redact is set."""
        queue: asyncio.Queue[ServerEvent] = asyncio.Queue()
        redact = lambda text: text.replace("secret-key-123", "[REDACTED]")  # noqa: E731
        obs = TransportObserver(queue, redact=redact)

        msg = Message(role=Role.ASSISTANT, content="The API key is secret-key-123")
        await obs.on_message_appended(msg, iteration=0)

        event = await asyncio.wait_for(queue.get(), timeout=1.0)
        assert "secret-key-123" not in event.data["content"]
        assert "[REDACTED]" in event.data["content"]

    async def test_no_redaction_when_none(self) -> None:
        """Without redact, content passes through unchanged."""
        queue: asyncio.Queue[ServerEvent] = asyncio.Queue()
        obs = TransportObserver(queue)

        msg = Message(role=Role.ASSISTANT, content="The API key is secret-key-123")
        await obs.on_message_appended(msg, iteration=0)

        event = await asyncio.wait_for(queue.get(), timeout=1.0)
        assert "secret-key-123" in event.data["content"]

    async def test_tool_completion_event(self) -> None:
        queue: asyncio.Queue[ServerEvent] = asyncio.Queue()
        obs = TransportObserver(queue)

        tc = ToolCall(name="add", params={"a": 1})
        tr = ToolResult(name="add", call_id=tc.id, payload="3", success=True)
        await obs.on_tool_completed(tc, tr, iteration=1)

        event = await asyncio.wait_for(queue.get(), timeout=1.0)
        assert event.event == "run.tool_done"
        assert event.data["tool_name"] == "add"
        assert event.data["success"] is True


class TestCompositeObserver:
    async def test_fans_out_to_all(self) -> None:
        q1: asyncio.Queue[ServerEvent] = asyncio.Queue()
        q2: asyncio.Queue[ServerEvent] = asyncio.Queue()
        obs = CompositeObserver([TransportObserver(q1), TransportObserver(q2)])

        msg = Message(role=Role.USER, content="hi")
        await obs.on_message_appended(msg, iteration=0)

        e1 = await asyncio.wait_for(q1.get(), timeout=1.0)
        e2 = await asyncio.wait_for(q2.get(), timeout=1.0)
        assert e1.event == "run.step"
        assert e2.event == "run.step"

    async def test_one_failure_doesnt_block_others(self) -> None:
        """If one observer fails, the others still fire."""

        class FailingObserver:
            async def on_message_appended(self, message: Any, iteration: int) -> None:
                raise RuntimeError("boom")

            async def on_llm_call_completed(
                self, response: Any, iteration: int, **kwargs: Any
            ) -> None:
                pass

            async def on_tool_completed(self, tc: Any, tr: Any, iteration: int) -> None:
                pass

        queue: asyncio.Queue[ServerEvent] = asyncio.Queue()
        obs = CompositeObserver([FailingObserver(), TransportObserver(queue)])  # type: ignore[list-item]

        msg = Message(role=Role.USER, content="hi")
        await obs.on_message_appended(msg, iteration=0)

        event = await asyncio.wait_for(queue.get(), timeout=1.0)
        assert event.event == "run.step"


class TestTaskManager:
    async def test_spawn_and_complete(self) -> None:
        mgr = RunTaskManager()

        async def _work() -> str:
            return "done"

        task = mgr.spawn("r1", _work())
        result = await task
        assert result == "done"
        assert not mgr.is_running("r1")

    async def test_terminal_event_not_auto_buffered(self) -> None:
        """_run_wrapper does NOT buffer terminal events — CAS winner does."""
        mgr = RunTaskManager()

        async def _work() -> str:
            return "done"

        task = mgr.spawn("r1", _work())
        await task
        await asyncio.sleep(0.05)

        event = mgr.get_terminal_event("r1")
        assert event is None

    async def test_buffer_terminal_event(self) -> None:
        """CAS winner buffers terminal event via buffer_terminal_event."""
        mgr = RunTaskManager()
        mgr.buffer_terminal_event("r1", {"event": "run.completed", "data": {"run_id": "r1"}})
        event = mgr.get_terminal_event("r1")
        assert event is not None
        assert event["event"] == "run.completed"

    async def test_terminal_event_ttl_expiry(self) -> None:
        """Terminal events expire after TTL."""
        mgr = RunTaskManager(terminal_ttl_seconds=0.1)
        mgr.buffer_terminal_event("r1", {"event": "run.completed", "data": {}})
        assert mgr.get_terminal_event("r1") is not None

        await asyncio.sleep(0.15)
        assert mgr.get_terminal_event("r1") is None

    async def test_terminal_event_max_size_eviction(self) -> None:
        """Oldest terminal events are evicted when buffer is full."""
        mgr = RunTaskManager(max_terminal_events=3)
        mgr.buffer_terminal_event("r1", {"event": "run.completed", "data": {}})
        mgr.buffer_terminal_event("r2", {"event": "run.completed", "data": {}})
        mgr.buffer_terminal_event("r3", {"event": "run.completed", "data": {}})

        mgr.buffer_terminal_event("r4", {"event": "run.completed", "data": {}})
        assert mgr.get_terminal_event("r1") is None
        assert mgr.get_terminal_event("r4") is not None
        assert mgr.terminal_event_count == 3

    async def test_exception_observed(self) -> None:
        mgr = RunTaskManager()

        async def _fail() -> None:
            raise RuntimeError("oops")

        task = mgr.spawn("r1", _fail())
        with pytest.raises(RuntimeError, match="oops"):
            await task

        assert not mgr.is_running("r1")

    async def test_cancel(self) -> None:
        mgr = RunTaskManager()

        async def _slow() -> None:
            await asyncio.sleep(100)

        mgr.spawn("r1", _slow())
        await asyncio.sleep(0.05)
        assert mgr.is_running("r1")

        cancelled = mgr.cancel("r1")
        assert cancelled is True
        await asyncio.sleep(0.1)
        assert not mgr.is_running("r1")

    async def test_cleanup_on_complete(self) -> None:
        mgr = RunTaskManager()

        async def _work() -> str:
            return "ok"

        task = mgr.spawn("r1", _work())
        await task
        await asyncio.sleep(0.05)

        assert len(mgr) == 0
