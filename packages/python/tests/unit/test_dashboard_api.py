"""Tests for the dashboard read-only API."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from httpx import ASGITransport, AsyncClient

from dendrite.dashboard.api import create_dashboard_api

# ------------------------------------------------------------------
# Mock store for API tests
# ------------------------------------------------------------------

_T0 = datetime(2026, 3, 15, 10, 0, 0)


@dataclass
class _Event:
    id: str = ""
    event_type: str = ""
    sequence_index: int = 0
    iteration_index: int = 0
    correlation_id: str | None = None
    data: dict[str, Any] | None = None
    created_at: datetime | None = None


@dataclass
class _Run:
    id: str = ""
    agent_name: str = ""
    status: str = "success"
    input_data: dict[str, Any] | None = None
    output_data: dict[str, Any] | None = None
    answer: str | None = None
    error: str | None = None
    iteration_count: int = 0
    model: str | None = None
    strategy: str | None = None
    parent_run_id: str | None = None
    delegation_level: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float | None = None
    meta: dict[str, Any] | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass
class _Trace:
    id: str = ""
    role: str = ""
    content: str = ""
    order_index: int = 0
    meta: dict[str, Any] | None = None
    created_at: datetime | None = None


@dataclass
class _ToolCall:
    id: str = ""
    tool_call_id: str = ""
    provider_tool_call_id: str | None = None
    tool_name: str = ""
    target: str = "server"
    params: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    success: bool = True
    duration_ms: int | None = None
    iteration_index: int | None = None
    error_message: str | None = None
    created_at: datetime | None = None


@dataclass
class _LLMInteraction:
    id: str = ""
    iteration_index: int = 0
    model: str | None = None
    provider: str | None = None
    semantic_request: dict[str, Any] | None = None
    semantic_response: dict[str, Any] | None = None
    provider_request: dict[str, Any] | None = None
    provider_response: dict[str, Any] | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None
    duration_ms: int | None = None
    created_at: datetime | None = None


@dataclass
class DashboardMockStore:
    """Mock store for dashboard API tests."""

    _runs: list[_Run] = field(default_factory=list)
    _events: dict[str, list[_Event]] = field(default_factory=dict)
    _traces: dict[str, list[_Trace]] = field(default_factory=dict)
    _tool_calls: dict[str, list[_ToolCall]] = field(default_factory=dict)
    _llm_interactions: dict[str, list[_LLMInteraction]] = field(default_factory=dict)

    async def get_run(self, run_id: str) -> _Run | None:
        for r in self._runs:
            if r.id == run_id:
                return r
        return None

    async def get_run_events(self, run_id: str) -> list[_Event]:
        return self._events.get(run_id, [])

    async def get_traces(self, run_id: str) -> list[_Trace]:
        return self._traces.get(run_id, [])

    async def get_tool_calls(self, run_id: str) -> list[_ToolCall]:
        return self._tool_calls.get(run_id, [])

    async def get_llm_interactions(self, run_id: str) -> list[_LLMInteraction]:
        return self._llm_interactions.get(run_id, [])

    async def list_runs(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        status: str | None = None,
        tenant_id: str | None = None,
    ) -> list[_Run]:
        runs = list(self._runs)
        if status:
            runs = [r for r in runs if r.status == status]
        return runs[:limit]


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


class TestHealthEndpoint:
    async def test_health(self) -> None:
        store = DashboardMockStore()
        app = create_dashboard_api(store)  # type: ignore[arg-type]
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/health")
            assert resp.status_code == 200
            assert resp.json() == {"status": "ok"}


class TestListRuns:
    async def test_list_runs_empty(self) -> None:
        store = DashboardMockStore()
        app = create_dashboard_api(store)  # type: ignore[arg-type]
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/runs")
            assert resp.status_code == 200
            assert resp.json()["runs"] == []

    async def test_list_runs_with_data(self) -> None:
        store = DashboardMockStore(
            _runs=[
                _Run(
                    id="r1",
                    agent_name="TestAgent",
                    status="success",
                    total_input_tokens=500,
                    total_output_tokens=200,
                    model="claude-sonnet",
                    created_at=_T0,
                ),
            ],
        )
        app = create_dashboard_api(store)  # type: ignore[arg-type]
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/runs")
            data = resp.json()
            assert len(data["runs"]) == 1
            assert data["runs"][0]["run_id"] == "r1"
            assert data["runs"][0]["agent_name"] == "TestAgent"
            assert data["runs"][0]["pause_count"] == 0

    async def test_list_runs_with_pause_count(self) -> None:
        store = DashboardMockStore(
            _runs=[_Run(id="r1", agent_name="A", status="success")],
            _events={
                "r1": [
                    _Event(event_type="run.started"),
                    _Event(event_type="run.paused"),
                    _Event(event_type="run.resumed"),
                    _Event(event_type="run.paused"),
                    _Event(event_type="run.resumed"),
                    _Event(event_type="run.completed"),
                ]
            },
        )
        app = create_dashboard_api(store)  # type: ignore[arg-type]
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/runs")
            assert resp.json()["runs"][0]["pause_count"] == 2

    async def test_filter_by_status(self) -> None:
        store = DashboardMockStore(
            _runs=[
                _Run(id="r1", agent_name="A", status="success"),
                _Run(id="r2", agent_name="A", status="error"),
            ],
        )
        app = create_dashboard_api(store)  # type: ignore[arg-type]
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/runs?status=error")
            assert len(resp.json()["runs"]) == 1
            assert resp.json()["runs"][0]["run_id"] == "r2"

    async def test_filter_by_agent_with_correct_total(self) -> None:
        """Agent filter returns correct total, not page size."""
        store = DashboardMockStore(
            _runs=[
                _Run(id="r1", agent_name="AgentA", status="success"),
                _Run(id="r2", agent_name="AgentB", status="success"),
                _Run(id="r3", agent_name="AgentA", status="success"),
                _Run(id="r4", agent_name="AgentB", status="error"),
                _Run(id="r5", agent_name="AgentA", status="error"),
            ],
        )
        app = create_dashboard_api(store)  # type: ignore[arg-type]
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/runs?agent=AgentA")
            data = resp.json()
            assert data["total"] == 3
            assert len(data["runs"]) == 3
            assert all(r["agent_name"] == "AgentA" for r in data["runs"])


class TestRunDetail:
    async def test_run_detail_returns_timeline(self) -> None:
        store = DashboardMockStore(
            _runs=[
                _Run(
                    id="r1",
                    agent_name="TestAgent",
                    status="success",
                    input_data={"input": "hello"},
                    answer="world",
                ),
            ],
            _events={
                "r1": [
                    _Event(
                        id="e0",
                        event_type="run.started",
                        sequence_index=0,
                        data={"agent_name": "TestAgent"},
                    ),
                    _Event(
                        id="e1",
                        event_type="run.completed",
                        sequence_index=1,
                        data={"status": "success"},
                    ),
                ]
            },
        )
        app = create_dashboard_api(store)  # type: ignore[arg-type]
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/runs/r1")
            assert resp.status_code == 200
            data = resp.json()
            assert data["summary"]["run_id"] == "r1"
            assert data["summary"]["answer"] == "world"
            assert len(data["nodes"]) == 2
            assert data["nodes"][0]["type"] == "run_started"
            assert data["nodes"][1]["type"] == "finish"

    async def test_run_detail_404(self) -> None:
        store = DashboardMockStore()
        app = create_dashboard_api(store)  # type: ignore[arg-type]
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/runs/nonexistent")
            assert resp.status_code == 404


class TestRunEvents:
    async def test_raw_events(self) -> None:
        store = DashboardMockStore(
            _runs=[_Run(id="r1", agent_name="A")],
            _events={
                "r1": [
                    _Event(
                        id="e0",
                        event_type="run.started",
                        sequence_index=0,
                        data={"agent_name": "A"},
                        created_at=_T0,
                    ),
                ]
            },
        )
        app = create_dashboard_api(store)  # type: ignore[arg-type]
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/runs/r1/events")
            assert resp.status_code == 200
            assert len(resp.json()["events"]) == 1
            assert resp.json()["events"][0]["event_type"] == "run.started"


class TestRunTraces:
    async def test_traces(self) -> None:
        store = DashboardMockStore(
            _runs=[_Run(id="r1", agent_name="A")],
            _traces={
                "r1": [
                    _Trace(id="t0", role="user", content="hello", order_index=0),
                    _Trace(id="t1", role="assistant", content="world", order_index=1),
                ]
            },
        )
        app = create_dashboard_api(store)  # type: ignore[arg-type]
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/runs/r1/traces")
            assert resp.status_code == 200
            assert len(resp.json()["traces"]) == 2

    async def test_traces_404(self) -> None:
        store = DashboardMockStore()
        app = create_dashboard_api(store)  # type: ignore[arg-type]
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/runs/nonexistent/traces")
            assert resp.status_code == 404


class TestRunToolCalls:
    async def test_tool_calls(self) -> None:
        store = DashboardMockStore(
            _runs=[_Run(id="r1", agent_name="A")],
            _tool_calls={
                "r1": [
                    _ToolCall(
                        id="tc0",
                        tool_call_id="tc_123",
                        tool_name="lookup",
                        target="server",
                        params={"ticker": "AAPL"},
                        result={"price": 227.50},
                        success=True,
                        duration_ms=23,
                    ),
                ]
            },
        )
        app = create_dashboard_api(store)  # type: ignore[arg-type]
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/runs/r1/tool-calls")
            assert resp.status_code == 200
            tc = resp.json()["tool_calls"][0]
            assert tc["tool_name"] == "lookup"
            assert tc["params"] == {"ticker": "AAPL"}
            assert tc["duration_ms"] == 23


# ------------------------------------------------------------------
# LLM Calls endpoint (Sprint 3.5)
# ------------------------------------------------------------------


class TestLLMCallsEndpoint:
    async def test_llm_calls_returns_interactions(self) -> None:
        store = DashboardMockStore(
            _runs=[_Run(id="r1", agent_name="Agent")],
            _llm_interactions={
                "r1": [
                    _LLMInteraction(
                        id="li_1",
                        iteration_index=1,
                        model="claude-sonnet-4-6",
                        provider="Anthropic",
                        input_tokens=100,
                        output_tokens=50,
                        cost_usd=0.005,
                        semantic_request={"messages": [{"role": "user", "content": "hi"}]},
                        semantic_response={"text": "hello"},
                        provider_request={"model": "claude-sonnet-4-6"},
                        provider_response={"id": "msg_123"},
                    )
                ]
            },
        )
        app = create_dashboard_api(store)  # type: ignore[arg-type]
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/runs/r1/llm-calls")
            assert resp.status_code == 200
            calls = resp.json()["llm_calls"]
            assert len(calls) == 1
            assert calls[0]["model"] == "claude-sonnet-4-6"
            assert calls[0]["input_tokens"] == 100
            assert calls[0]["semantic_request"]["messages"][0]["content"] == "hi"
            assert calls[0]["provider_response"]["id"] == "msg_123"

    async def test_llm_calls_404_for_missing_run(self) -> None:
        store = DashboardMockStore()
        app = create_dashboard_api(store)  # type: ignore[arg-type]
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/runs/nonexistent/llm-calls")
            assert resp.status_code == 404

    async def test_llm_calls_empty(self) -> None:
        store = DashboardMockStore(_runs=[_Run(id="r1", agent_name="Agent")])
        app = create_dashboard_api(store)  # type: ignore[arg-type]
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/runs/r1/llm-calls")
            assert resp.status_code == 200
            assert resp.json()["llm_calls"] == []
