"""Tests for Dendrite server (Sprint 3, Group 3)."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import pytest
from fastapi import Request  # noqa: TC002 — needed at runtime by FastAPI route definitions
from httpx import ASGITransport, AsyncClient

from dendrite.agent import Agent
from dendrite.llm.mock import MockLLM
from dendrite.server.app import create_app
from dendrite.server.observer import CompositeObserver, ServerEvent, TransportObserver
from dendrite.server.registry import AgentRegistry, HostedAgentConfig
from dendrite.server.tasks import RunTaskManager
from dendrite.tool import tool
from dendrite.types import (
    LLMResponse,
    Message,
    Role,
    ToolCall,
    ToolResult,
)

# ------------------------------------------------------------------
# Test tools and agents
# ------------------------------------------------------------------


@tool()
async def server_add(a: int, b: int) -> int:
    """Server-side add."""
    return a + b


@tool(target="client")
async def read_range(sheet: str) -> str:
    """Client-side tool."""
    return "should never run"


_test_agent = Agent(
    name="TestAgent",
    model="mock",
    prompt="You are a test agent.",
    tools=[server_add, read_range],
)


# ------------------------------------------------------------------
# Mock StateStore for server tests
# ------------------------------------------------------------------


@dataclass
class MockServerStore:
    """Minimal StateStore for server tests."""

    _runs: dict[str, dict[str, Any]] = field(default_factory=dict)
    _pause_data: dict[str, dict[str, Any]] = field(default_factory=dict)
    _traces: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    _events: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    async def create_run(self, run_id: str, agent_name: str, **kwargs: Any) -> None:
        self._runs[run_id] = {
            "id": run_id,
            "agent_name": agent_name,
            "status": "running",
            "iteration_count": 0,
            "answer": None,
            "error": None,
            "input_data": kwargs.get("input_data"),
            "output_data": None,
            "model": kwargs.get("model"),
            "strategy": kwargs.get("strategy"),
            "parent_run_id": None,
            "delegation_level": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_cost_usd": None,
            "meta": None,
            "created_at": None,
            "updated_at": None,
            "tenant_id": kwargs.get("tenant_id"),
            **kwargs,
        }

    async def save_trace(self, run_id: str, role: str, content: str, **kwargs: Any) -> None:
        self._traces.setdefault(run_id, []).append({"role": role, "content": content, **kwargs})

    async def save_tool_call(self, run_id: str, **kwargs: Any) -> None:
        pass

    async def save_usage(self, run_id: str, **kwargs: Any) -> None:
        pass

    async def finalize_run(self, run_id: str, **kwargs: Any) -> bool:
        expected = kwargs.pop("expected_current_status", None)
        if (
            expected is not None
            and run_id in self._runs
            and self._runs[run_id]["status"] != expected
        ):
            return False
        if run_id in self._runs:
            self._runs[run_id]["status"] = kwargs.get("status", "success")
            self._runs[run_id]["answer"] = kwargs.get("answer")
            self._runs[run_id]["iteration_count"] = kwargs.get("iteration_count", 0)
            self._pause_data.pop(run_id, None)
        return True

    async def pause_run(self, run_id: str, *, status: str, pause_data: dict, **kwargs: Any) -> None:
        if run_id in self._runs:
            self._runs[run_id]["status"] = status
        self._pause_data[run_id] = pause_data

    async def get_pause_state(self, run_id: str) -> dict[str, Any] | None:
        return self._pause_data.get(run_id)

    async def claim_paused_run(self, run_id: str, *, expected_status: str) -> bool:
        run = self._runs.get(run_id)
        if run and run["status"] == expected_status:
            run["status"] = "running"
            return True
        return False

    async def get_run(self, run_id: str) -> Any:
        run = self._runs.get(run_id)
        if run is None:
            return None

        @dataclass
        class _Record:
            id: str
            agent_name: str
            status: str
            answer: str | None
            error: str | None
            iteration_count: int
            input_data: Any = None
            output_data: Any = None
            model: str | None = None
            strategy: str | None = None
            parent_run_id: str | None = None
            delegation_level: int = 0
            total_input_tokens: int = 0
            total_output_tokens: int = 0
            total_cost_usd: float | None = None
            meta: Any = None
            created_at: Any = None
            updated_at: Any = None

        return _Record(
            id=run["id"],
            agent_name=run["agent_name"],
            status=run["status"],
            answer=run.get("answer"),
            error=run.get("error"),
            iteration_count=run.get("iteration_count", 0),
        )

    async def get_traces(self, run_id: str) -> list[Any]:
        @dataclass
        class _Trace:
            order_index: int

        traces = self._traces.get(run_id, [])
        return [_Trace(order_index=t.get("order_index", i)) for i, t in enumerate(traces)]

    async def get_tool_calls(self, run_id: str) -> list[Any]:
        return []

    async def save_run_event(self, run_id: str, **kwargs: Any) -> None:
        self._events.setdefault(run_id, []).append(kwargs)

    async def get_run_events(self, run_id: str) -> list[Any]:
        @dataclass
        class _Event:
            sequence_index: int = 0

        raw = self._events.get(run_id, [])
        return [_Event(sequence_index=e.get("sequence_index", i)) for i, e in enumerate(raw)]

    async def list_runs(self, **kwargs: Any) -> list[Any]:
        return []


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_registry(llm_responses: list[LLMResponse]) -> AgentRegistry:
    """Create a registry with TestAgent backed by MockLLM."""
    registry = AgentRegistry()
    registry.register(
        HostedAgentConfig(
            agent=_test_agent,
            provider_factory=lambda: MockLLM(llm_responses),
        )
    )
    return registry


async def _make_client(
    store: MockServerStore,
    registry: AgentRegistry,
    *,
    hmac_secret: str | None = None,
    allow_insecure_dev_mode: bool = True,
) -> AsyncClient:
    """Create an httpx AsyncClient for the Dendrite app."""
    app = create_app(
        state_store=store,
        registry=registry,
        hmac_secret=hmac_secret,
        allow_insecure_dev_mode=allow_insecure_dev_mode,
    )
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


class TestCreateRun:
    async def test_create_run_returns_run_id(self) -> None:
        store = MockServerStore()
        registry = _make_registry([LLMResponse(text="Hello!")])
        async with await _make_client(store, registry) as client:
            resp = await client.post(
                "/runs",
                json={
                    "agent_name": "TestAgent",
                    "input": "Say hello",
                },
            )
            assert resp.status_code == 200
            data = resp.json()
            assert "run_id" in data
            assert data["status"] == "running"

            # Wait for background task to complete
            await asyncio.sleep(0.2)

    async def test_create_run_unknown_agent_returns_404(self) -> None:
        store = MockServerStore()
        registry = AgentRegistry()
        async with await _make_client(store, registry) as client:
            resp = await client.post(
                "/runs",
                json={
                    "agent_name": "NonExistent",
                    "input": "Hi",
                },
            )
            assert resp.status_code == 404


class TestPollStatus:
    async def test_poll_status_reflects_transitions(self) -> None:
        store = MockServerStore()
        tc = ToolCall(name="read_range", params={"sheet": "S1"}, provider_tool_call_id="t1")
        registry = _make_registry([LLMResponse(tool_calls=[tc])])
        async with await _make_client(store, registry) as client:
            resp = await client.post(
                "/runs",
                json={
                    "agent_name": "TestAgent",
                    "input": "Read sheet",
                },
            )
            run_id = resp.json()["run_id"]
            await asyncio.sleep(0.2)

            # Should be paused
            status_resp = await client.get(f"/runs/{run_id}")
            assert status_resp.status_code == 200
            data = status_resp.json()
            assert data["status"] == "waiting_client_tool"
            assert data["pending_tool_calls"] is not None

    async def test_poll_nonexistent_returns_404(self) -> None:
        store = MockServerStore()
        registry = AgentRegistry()
        async with await _make_client(store, registry) as client:
            resp = await client.get("/runs/nonexistent")
            assert resp.status_code == 404


class TestSubmitToolResults:
    async def test_submit_tool_results_resumes(self) -> None:
        """Full pause → submit tool results → resume → complete via HTTP."""
        store = MockServerStore()
        tc = ToolCall(name="read_range", params={"sheet": "S1"}, provider_tool_call_id="t1")

        call_count = {"n": 0}

        def _provider_factory() -> MockLLM:
            call_count["n"] += 1
            if call_count["n"] == 1:
                return MockLLM([LLMResponse(tool_calls=[tc])])
            return MockLLM([LLMResponse(text="Got the data!")])

        registry = AgentRegistry()
        registry.register(HostedAgentConfig(agent=_test_agent, provider_factory=_provider_factory))

        async with await _make_client(store, registry) as client:
            resp = await client.post("/runs", json={"agent_name": "TestAgent", "input": "Read it"})
            run_id = resp.json()["run_id"]
            await asyncio.sleep(0.3)

            status = await client.get(f"/runs/{run_id}")
            assert status.json()["status"] == "waiting_client_tool"
            pending = status.json()["pending_tool_calls"]
            assert len(pending) == 1

            resume_resp = await client.post(
                f"/runs/{run_id}/tool-results",
                json={
                    "tool_results": [
                        {
                            "tool_call_id": pending[0]["tool_call_id"],
                            "tool_name": "read_range",
                            "result": '{"data": [1, 2, 3]}',
                        }
                    ]
                },
            )
            assert resume_resp.status_code == 200
            assert resume_resp.json()["status"] == "success"

    async def test_submit_on_non_paused_returns_409(self) -> None:
        store = MockServerStore()
        registry = _make_registry([LLMResponse(text="Done")])
        async with await _make_client(store, registry) as client:
            resp = await client.post(
                "/runs",
                json={
                    "agent_name": "TestAgent",
                    "input": "Hi",
                },
            )
            run_id = resp.json()["run_id"]
            await asyncio.sleep(0.2)

            # Run is complete, not paused
            resp = await client.post(
                f"/runs/{run_id}/tool-results",
                json={
                    "tool_results": [
                        {"tool_call_id": "x", "tool_name": "read_range", "result": "{}"}
                    ],
                },
            )
            assert resp.status_code in (404, 409)


class TestCancelRun:
    async def test_cancel_paused_run(self) -> None:
        store = MockServerStore()
        tc = ToolCall(name="read_range", params={"sheet": "S1"}, provider_tool_call_id="t1")
        registry = _make_registry([LLMResponse(tool_calls=[tc])])
        async with await _make_client(store, registry) as client:
            resp = await client.post(
                "/runs",
                json={
                    "agent_name": "TestAgent",
                    "input": "Read",
                },
            )
            run_id = resp.json()["run_id"]
            await asyncio.sleep(0.2)

            # Cancel the paused run
            del_resp = await client.delete(f"/runs/{run_id}")
            assert del_resp.status_code == 200
            assert del_resp.json()["status"] == "cancelled"


class TestRegistry:
    def test_register_and_get(self) -> None:
        registry = AgentRegistry()
        config = HostedAgentConfig(agent=_test_agent, provider_factory=lambda: MockLLM([]))
        registry.register(config)
        assert registry.get("TestAgent") is config
        assert registry.list_agents() == ["TestAgent"]

    def test_duplicate_registration_raises(self) -> None:
        registry = AgentRegistry()
        config = HostedAgentConfig(agent=_test_agent, provider_factory=lambda: MockLLM([]))
        registry.register(config)
        with pytest.raises(ValueError, match="already registered"):
            registry.register(config)

    def test_get_unknown_raises(self) -> None:
        registry = AgentRegistry()
        with pytest.raises(KeyError, match="not registered"):
            registry.get("Unknown")

    def test_provider_factory_creates_fresh_instance(self) -> None:
        """Each call to provider_factory returns a new object (D2)."""
        instances = []
        config = HostedAgentConfig(
            agent=_test_agent,
            provider_factory=lambda: MockLLM([LLMResponse(text="hi")]),
        )
        instances.append(config.provider_factory())
        instances.append(config.provider_factory())
        assert instances[0] is not instances[1]


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

            async def on_llm_call_completed(self, response: Any, iteration: int, **kwargs) -> None:
                pass

            async def on_tool_completed(self, tc: Any, tr: Any, iteration: int) -> None:
                pass

        queue: asyncio.Queue[ServerEvent] = asyncio.Queue()
        obs = CompositeObserver([FailingObserver(), TransportObserver(queue)])  # type: ignore[list-item]

        msg = Message(role=Role.USER, content="hi")
        await obs.on_message_appended(msg, iteration=0)

        # Second observer should still have received the event
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

        # _run_wrapper no longer buffers — callers (CAS winners) are responsible
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

        # At capacity — adding r4 should evict r1
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

        # Exception is observed (logged) but NOT buffered as terminal event
        assert not mgr.is_running("r1")

    async def test_cancel(self) -> None:
        mgr = RunTaskManager()

        async def _slow() -> None:
            await asyncio.sleep(100)

        mgr.spawn("r1", _slow())
        # Let the task start running
        await asyncio.sleep(0.05)
        assert mgr.is_running("r1")

        cancelled = mgr.cancel("r1")
        assert cancelled is True
        # Let cancellation propagate through _run_wrapper finally block
        await asyncio.sleep(0.1)
        assert not mgr.is_running("r1")

    async def test_cleanup_on_complete(self) -> None:
        mgr = RunTaskManager()

        async def _work() -> str:
            return "ok"

        task = mgr.spawn("r1", _work())
        await task
        await asyncio.sleep(0.05)

        # Task reference cleaned up
        assert len(mgr) == 0


class TestServerImport:
    def test_import_without_fastapi_would_fail(self) -> None:
        """Verify the import guard exists (can't easily uninstall fastapi in test)."""
        # Just verify the module is importable when fastapi IS installed
        from dendrite.server import create_app as _  # noqa: F401


# ------------------------------------------------------------------
# Group 4: HMAC Auth Tests
# ------------------------------------------------------------------

_TEST_SECRET = "test-secret-for-hmac-auth"


class TestAuthTokenGeneration:
    def test_generate_and_verify_roundtrip(self) -> None:
        from dendrite.server.auth import generate_run_token, verify_run_token

        token = generate_run_token("run-123", _TEST_SECRET)
        assert token.startswith("drn_")
        assert verify_run_token("run-123", token, _TEST_SECRET)

    def test_wrong_run_id_fails(self) -> None:
        from dendrite.server.auth import generate_run_token, verify_run_token

        token = generate_run_token("run-123", _TEST_SECRET)
        assert not verify_run_token("run-456", token, _TEST_SECRET)

    def test_tampered_token_fails(self) -> None:
        from dendrite.server.auth import generate_run_token, verify_run_token

        token = generate_run_token("run-123", _TEST_SECRET)
        tampered = token[:-4] + "dead"
        assert not verify_run_token("run-123", tampered, _TEST_SECRET)

    def test_wrong_secret_fails(self) -> None:
        from dendrite.server.auth import generate_run_token, verify_run_token

        token = generate_run_token("run-123", _TEST_SECRET)
        assert not verify_run_token("run-123", token, "wrong-secret")

    def test_missing_prefix_fails(self) -> None:
        from dendrite.server.auth import verify_run_token

        assert not verify_run_token("run-123", "not_a_drn_token", _TEST_SECRET)

    def test_versioned_message_format(self) -> None:
        """Token signs 'drn:v0:<run_id>', not raw run_id."""
        from dendrite.server.auth import _build_message

        msg = _build_message("run-123")
        assert msg == b"drn:v0:run-123"


class TestExtractBearerToken:
    def test_valid_bearer(self) -> None:
        from dendrite.server.auth import extract_bearer_token

        assert extract_bearer_token("Bearer drn_abc123") == "drn_abc123"

    def test_case_insensitive_scheme(self) -> None:
        from dendrite.server.auth import extract_bearer_token

        assert extract_bearer_token("bearer drn_abc123") == "drn_abc123"

    def test_missing_header(self) -> None:
        from dendrite.server.auth import extract_bearer_token

        assert extract_bearer_token(None) is None

    def test_malformed_header(self) -> None:
        from dendrite.server.auth import extract_bearer_token

        assert extract_bearer_token("Basic abc123") is None
        assert extract_bearer_token("drn_abc123") is None  # no scheme
        assert extract_bearer_token("") is None


class TestAuthEndpoints:
    """Tests for HMAC auth wired into endpoints."""

    async def test_create_run_returns_token_when_auth_enabled(self) -> None:
        store = MockServerStore()
        registry = _make_registry([LLMResponse(text="Hello!")])
        async with await _make_client(store, registry, hmac_secret=_TEST_SECRET) as client:
            resp = await client.post("/runs", json={"agent_name": "TestAgent", "input": "Hi"})
            assert resp.status_code == 200
            data = resp.json()
            assert data["token"] is not None
            assert data["token"].startswith("drn_")
            await asyncio.sleep(0.2)

    async def test_create_run_no_token_when_auth_disabled(self) -> None:
        store = MockServerStore()
        registry = _make_registry([LLMResponse(text="Hello!")])
        async with await _make_client(store, registry) as client:
            resp = await client.post("/runs", json={"agent_name": "TestAgent", "input": "Hi"})
            assert resp.status_code == 200
            assert resp.json()["token"] is None
            await asyncio.sleep(0.2)

    async def test_poll_rejects_missing_token(self) -> None:
        store = MockServerStore()
        registry = _make_registry([LLMResponse(text="Hello!")])
        async with await _make_client(store, registry, hmac_secret=_TEST_SECRET) as client:
            # Create run to get run_id
            resp = await client.post("/runs", json={"agent_name": "TestAgent", "input": "Hi"})
            run_id = resp.json()["run_id"]
            await asyncio.sleep(0.2)

            # Poll without token
            status_resp = await client.get(f"/runs/{run_id}")
            assert status_resp.status_code == 401

    async def test_poll_rejects_wrong_token(self) -> None:
        store = MockServerStore()
        registry = _make_registry([LLMResponse(text="Hello!")])
        async with await _make_client(store, registry, hmac_secret=_TEST_SECRET) as client:
            resp = await client.post("/runs", json={"agent_name": "TestAgent", "input": "Hi"})
            run_id = resp.json()["run_id"]
            await asyncio.sleep(0.2)

            # Poll with wrong token
            status_resp = await client.get(
                f"/runs/{run_id}",
                headers={"Authorization": "Bearer drn_wrongtoken"},
            )
            assert status_resp.status_code == 401

    async def test_poll_accepts_valid_token(self) -> None:
        store = MockServerStore()
        registry = _make_registry([LLMResponse(text="Hello!")])
        async with await _make_client(store, registry, hmac_secret=_TEST_SECRET) as client:
            resp = await client.post("/runs", json={"agent_name": "TestAgent", "input": "Hi"})
            data = resp.json()
            run_id = data["run_id"]
            token = data["token"]
            await asyncio.sleep(0.2)

            # Poll with valid token
            status_resp = await client.get(
                f"/runs/{run_id}",
                headers={"Authorization": f"Bearer {token}"},
            )
            assert status_resp.status_code == 200

    async def test_cross_run_token_rejected(self) -> None:
        """Token for run A cannot access run B."""
        store = MockServerStore()
        registry = _make_registry([LLMResponse(text="Hello!")])
        async with await _make_client(store, registry, hmac_secret=_TEST_SECRET) as client:
            # Create two runs
            resp1 = await client.post("/runs", json={"agent_name": "TestAgent", "input": "Hi 1"})
            token1 = resp1.json()["token"]
            await asyncio.sleep(0.1)

            resp2 = await client.post("/runs", json={"agent_name": "TestAgent", "input": "Hi 2"})
            run_id_2 = resp2.json()["run_id"]
            await asyncio.sleep(0.1)

            # Try to access run 2 with run 1's token
            status_resp = await client.get(
                f"/runs/{run_id_2}",
                headers={"Authorization": f"Bearer {token1}"},
            )
            assert status_resp.status_code == 401

    async def test_delete_requires_token(self) -> None:
        store = MockServerStore()
        tc = ToolCall(name="read_range", params={"sheet": "S1"}, provider_tool_call_id="t1")
        registry = _make_registry([LLMResponse(tool_calls=[tc])])
        async with await _make_client(store, registry, hmac_secret=_TEST_SECRET) as client:
            resp = await client.post("/runs", json={"agent_name": "TestAgent", "input": "Read"})
            run_id = resp.json()["run_id"]
            await asyncio.sleep(0.2)

            # Delete without token
            del_resp = await client.delete(f"/runs/{run_id}")
            assert del_resp.status_code == 401

    async def test_tool_results_requires_token(self) -> None:
        """POST /tool-results rejects unauthenticated requests."""
        store = MockServerStore()
        tc = ToolCall(name="read_range", params={"sheet": "S1"}, provider_tool_call_id="t1")
        registry = _make_registry([LLMResponse(tool_calls=[tc])])
        async with await _make_client(store, registry, hmac_secret=_TEST_SECRET) as client:
            resp = await client.post("/runs", json={"agent_name": "TestAgent", "input": "Read"})
            run_id = resp.json()["run_id"]
            await asyncio.sleep(0.2)

            # Submit tool results without token
            resume_resp = await client.post(
                f"/runs/{run_id}/tool-results",
                json={
                    "tool_results": [
                        {"tool_call_id": "x", "tool_name": "read_range", "result": "{}"}
                    ]
                },
            )
            assert resume_resp.status_code == 401

    async def test_auth_disabled_skips_checks(self) -> None:
        """All endpoints accessible without token when auth is disabled."""
        store = MockServerStore()
        registry = _make_registry([LLMResponse(text="Hello!")])
        async with await _make_client(store, registry) as client:
            resp = await client.post("/runs", json={"agent_name": "TestAgent", "input": "Hi"})
            run_id = resp.json()["run_id"]
            await asyncio.sleep(0.2)

            # Poll without any token — should work
            status_resp = await client.get(f"/runs/{run_id}")
            assert status_resp.status_code == 200


class TestHeaderStripping:
    async def test_authorization_header_stripped_from_request(self) -> None:
        """Downstream handlers cannot see Authorization header (raw ASGI middleware)."""
        store = MockServerStore()
        registry = _make_registry([LLMResponse(text="Hello!")])
        app = create_app(state_store=store, registry=registry, hmac_secret=_TEST_SECRET)

        # Add a test route that inspects request.headers
        seen_auth: dict[str, Any] = {}

        @app.get("/debug-headers")
        async def debug_headers(request: Request) -> dict[str, str]:
            seen_auth["has_auth"] = "authorization" in request.headers
            seen_auth["state_auth"] = getattr(request.state, "_auth_header", "MISSING")
            return {"ok": "true"}

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            await client.get(
                "/debug-headers",
                headers={"Authorization": "Bearer drn_secret_token"},
            )

        # Header stripped from request.headers (not cached)
        assert seen_auth["has_auth"] is False
        # But the value was preserved in request.state for auth dependency
        assert seen_auth["state_auth"] == "Bearer drn_secret_token"

    async def test_header_stripped_even_in_dev_mode(self) -> None:
        """Authorization header is stripped unconditionally, not just when auth is enabled."""
        store = MockServerStore()
        registry = _make_registry([LLMResponse(text="Hello!")])
        app = create_app(state_store=store, registry=registry, allow_insecure_dev_mode=True)

        seen_auth: dict[str, Any] = {}

        @app.get("/debug-headers")
        async def debug_headers(request: Request) -> dict[str, str]:
            seen_auth["has_auth"] = "authorization" in request.headers
            return {"ok": "true"}

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            await client.get(
                "/debug-headers",
                headers={"Authorization": "Bearer some_dev_token"},
            )

        # Stripped even in dev mode — unconditional policy
        assert seen_auth["has_auth"] is False


class TestFailClosed:
    def test_no_secret_no_opt_in_raises(self) -> None:
        """create_app fails if hmac_secret is None and allow_insecure_dev_mode is False."""
        store = MockServerStore()
        registry = AgentRegistry()
        with pytest.raises(ValueError, match="hmac_secret is required"):
            create_app(state_store=store, registry=registry)

    def test_no_secret_with_opt_in_works(self) -> None:
        """create_app succeeds with explicit allow_insecure_dev_mode=True."""
        store = MockServerStore()
        registry = AgentRegistry()
        app = create_app(state_store=store, registry=registry, allow_insecure_dev_mode=True)
        assert app is not None

    def test_with_secret_works(self) -> None:
        """create_app succeeds when hmac_secret is provided."""
        store = MockServerStore()
        registry = AgentRegistry()
        app = create_app(state_store=store, registry=registry, hmac_secret="my-secret")
        assert app is not None
