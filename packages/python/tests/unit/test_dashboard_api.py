"""Tests for the dashboard read-only API."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from httpx import ASGITransport, AsyncClient

from dendrux.dashboard.api import create_dashboard_api
from dendrux.runtime.state import (
    DelegationInfo,
    ParentRef,
    RunBrief,
    SubtreeSummary,
)

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

    async def get_delegation_info(self, run_id: str) -> DelegationInfo | None:
        """In-memory delegation info — mirrors SQLAlchemy logic."""
        run = await self.get_run(run_id)
        if run is None:
            return None

        runs_by_id = {r.id: r for r in self._runs}

        # Parent
        parent: ParentRef | None = None
        if run.parent_run_id:
            p = runs_by_id.get(run.parent_run_id)
            if p is not None:
                parent = ParentRef(
                    run_id=p.id,
                    resolved=True,
                    agent_name=p.agent_name,
                    status=p.status,
                    delegation_level=p.delegation_level,
                )
            else:
                parent = ParentRef(run_id=run.parent_run_id, resolved=False)

        # Ancestry (walk up)
        ancestry: list[RunBrief] = []
        ancestry_complete = True
        cur = run.parent_run_id
        visited: set[str] = {run_id}
        while cur and cur not in visited:
            visited.add(cur)
            a = runs_by_id.get(cur)
            if a is None:
                ancestry_complete = False
                break
            ancestry.append(
                RunBrief(
                    run_id=a.id,
                    agent_name=a.agent_name,
                    status=a.status,
                    delegation_level=a.delegation_level,
                )
            )
            cur = a.parent_run_id
        if cur and cur in visited:
            ancestry_complete = False
        ancestry.reverse()

        # Direct children
        children = [
            RunBrief(
                run_id=c.id,
                agent_name=c.agent_name,
                status=c.status,
                delegation_level=c.delegation_level,
            )
            for c in self._runs
            if c.parent_run_id == run_id
        ]

        # Subtree (BFS downward, with cycle guard)
        queue = [run_id]
        subtree_runs: list[_Run] = []
        max_depth = 0
        depth_map = {run_id: 0}
        seen: set[str] = {run_id}
        while queue:
            nid = queue.pop(0)
            r = runs_by_id.get(nid)
            if r is not None:
                subtree_runs.append(r)
                for c in self._runs:
                    if c.parent_run_id == nid and c.id not in seen:
                        seen.add(c.id)
                        d = depth_map[nid] + 1
                        depth_map[c.id] = d
                        if d > max_depth:
                            max_depth = d
                        queue.append(c.id)

        status_counts: dict[str, int] = {}
        total_in = 0
        total_out = 0
        total_cost = 0.0
        has_cost = False
        unknown_cost_count = 0
        for sr in subtree_runs:
            total_in += sr.total_input_tokens
            total_out += sr.total_output_tokens
            if sr.total_cost_usd is not None:
                total_cost += sr.total_cost_usd
                has_cost = True
            else:
                unknown_cost_count += 1
            status_counts[sr.status] = status_counts.get(sr.status, 0) + 1

        return DelegationInfo(
            parent=parent,
            children=children,
            ancestry=ancestry,
            subtree_summary=SubtreeSummary(
                direct_child_count=len(children),
                descendant_count=len(subtree_runs) - 1,
                max_depth=max_depth,
                subtree_input_tokens=total_in,
                subtree_output_tokens=total_out,
                subtree_cost_usd=total_cost if has_cost else None,
                unknown_cost_count=unknown_cost_count,
                status_counts=status_counts,
            ),
            ancestry_complete=ancestry_complete,
        )


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

    async def test_list_runs_includes_delegation_fields(self) -> None:
        """API returns parent_run_id and delegation_level for each run."""
        store = DashboardMockStore(
            _runs=[
                _Run(id="parent", agent_name="Orchestrator", status="success"),
                _Run(
                    id="child",
                    agent_name="Worker",
                    status="success",
                    parent_run_id="parent",
                    delegation_level=1,
                ),
            ],
        )
        app = create_dashboard_api(store)  # type: ignore[arg-type]
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/runs")
            runs = resp.json()["runs"]

            parent = next(r for r in runs if r["run_id"] == "parent")
            assert parent["parent_run_id"] is None
            assert parent["delegation_level"] == 0

            child = next(r for r in runs if r["run_id"] == "child")
            assert child["parent_run_id"] == "parent"
            assert child["delegation_level"] == 1

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


# ------------------------------------------------------------------
# Delegation block on run detail (Post-Sprint 4)
# ------------------------------------------------------------------


class TestDelegationBlock:
    """Tests for the delegation block on GET /api/runs/{run_id}."""

    async def test_root_run_has_delegation_block(self) -> None:
        """A root run with no children still gets a delegation block."""
        store = DashboardMockStore(
            _runs=[
                _Run(
                    id="r1",
                    agent_name="Root",
                    status="success",
                    total_input_tokens=100,
                    total_output_tokens=50,
                    total_cost_usd=0.01,
                )
            ],
            _events={
                "r1": [
                    _Event(
                        id="e0",
                        event_type="run.started",
                        sequence_index=0,
                        data={"agent_name": "Root"},
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
            d = resp.json()["delegation"]
            assert d is not None
            assert d["parent"] is None
            assert d["children"] == []
            assert d["ancestry"] == []
            assert d["ancestry_complete"] is True
            # Subtree = self only
            ss = d["subtree_summary"]
            assert ss["direct_child_count"] == 0
            assert ss["descendant_count"] == 0
            assert ss["max_depth"] == 0
            assert ss["subtree_input_tokens"] == 100
            assert ss["subtree_output_tokens"] == 50
            assert ss["subtree_cost_usd"] == 0.01
            assert ss["status_counts"] == {"success": 1}

    async def test_parent_child_delegation(self) -> None:
        """Child run shows resolved parent; parent shows child in children list."""
        store = DashboardMockStore(
            _runs=[
                _Run(
                    id="root",
                    agent_name="Orchestrator",
                    status="success",
                    total_input_tokens=200,
                    total_output_tokens=100,
                    total_cost_usd=0.02,
                ),
                _Run(
                    id="child1",
                    agent_name="Worker",
                    status="success",
                    parent_run_id="root",
                    delegation_level=1,
                    total_input_tokens=300,
                    total_output_tokens=150,
                    total_cost_usd=0.03,
                ),
            ],
            _events={
                "root": [
                    _Event(
                        id="e0",
                        event_type="run.started",
                        sequence_index=0,
                        data={"agent_name": "Orchestrator"},
                    ),
                    _Event(
                        id="e1",
                        event_type="run.completed",
                        sequence_index=1,
                        data={"status": "success"},
                    ),
                ],
                "child1": [
                    _Event(
                        id="e2",
                        event_type="run.started",
                        sequence_index=0,
                        data={"agent_name": "Worker"},
                    ),
                    _Event(
                        id="e3",
                        event_type="run.completed",
                        sequence_index=1,
                        data={"status": "success"},
                    ),
                ],
            },
        )
        app = create_dashboard_api(store)  # type: ignore[arg-type]
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # Check parent (root) detail
            resp = await client.get("/api/runs/root")
            d = resp.json()["delegation"]
            assert d["parent"] is None
            assert len(d["children"]) == 1
            assert d["children"][0]["run_id"] == "child1"
            assert d["children"][0]["agent_name"] == "Worker"
            ss = d["subtree_summary"]
            assert ss["direct_child_count"] == 1
            assert ss["descendant_count"] == 1
            assert ss["max_depth"] == 1
            assert ss["subtree_input_tokens"] == 500  # 200 + 300
            assert ss["subtree_output_tokens"] == 250  # 100 + 150
            assert ss["subtree_cost_usd"] == 0.05  # 0.02 + 0.03

            # Check child detail
            resp = await client.get("/api/runs/child1")
            d = resp.json()["delegation"]
            assert d["parent"]["run_id"] == "root"
            assert d["parent"]["resolved"] is True
            assert d["parent"]["agent_name"] == "Orchestrator"
            assert d["ancestry"] == [
                {
                    "run_id": "root",
                    "agent_name": "Orchestrator",
                    "status": "success",
                    "delegation_level": 0,
                },
            ]
            assert d["ancestry_complete"] is True
            assert d["children"] == []

    async def test_broken_parent_chain(self) -> None:
        """Parent run_id exists but parent row is missing."""
        store = DashboardMockStore(
            _runs=[
                _Run(
                    id="orphan",
                    agent_name="Worker",
                    status="success",
                    parent_run_id="missing_parent",
                    delegation_level=1,
                    total_input_tokens=100,
                    total_output_tokens=50,
                ),
            ],
            _events={
                "orphan": [
                    _Event(
                        id="e0",
                        event_type="run.started",
                        sequence_index=0,
                        data={"agent_name": "Worker"},
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
            resp = await client.get("/api/runs/orphan")
            d = resp.json()["delegation"]
            # Parent is unresolved
            assert d["parent"]["run_id"] == "missing_parent"
            assert d["parent"]["resolved"] is False
            assert d["parent"]["agent_name"] is None
            # Ancestry is incomplete
            assert d["ancestry"] == []
            assert d["ancestry_complete"] is False

    async def test_deep_tree_subtree_summary(self) -> None:
        """Three-level tree: root → child → grandchild."""
        store = DashboardMockStore(
            _runs=[
                _Run(
                    id="root",
                    agent_name="Orch",
                    status="success",
                    total_input_tokens=100,
                    total_output_tokens=50,
                    total_cost_usd=0.01,
                ),
                _Run(
                    id="mid",
                    agent_name="Research",
                    status="success",
                    parent_run_id="root",
                    delegation_level=1,
                    total_input_tokens=200,
                    total_output_tokens=100,
                    total_cost_usd=0.02,
                ),
                _Run(
                    id="leaf",
                    agent_name="Fact",
                    status="error",
                    parent_run_id="mid",
                    delegation_level=2,
                    total_input_tokens=50,
                    total_output_tokens=25,
                    total_cost_usd=0.005,
                ),
            ],
            _events={
                "root": [
                    _Event(
                        id="e0",
                        event_type="run.started",
                        sequence_index=0,
                        data={"agent_name": "Orch"},
                    ),
                    _Event(
                        id="e1",
                        event_type="run.completed",
                        sequence_index=1,
                        data={"status": "success"},
                    ),
                ],
                "mid": [
                    _Event(
                        id="e2",
                        event_type="run.started",
                        sequence_index=0,
                        data={"agent_name": "Research"},
                    ),
                    _Event(
                        id="e3",
                        event_type="run.completed",
                        sequence_index=1,
                        data={"status": "success"},
                    ),
                ],
                "leaf": [
                    _Event(
                        id="e4",
                        event_type="run.started",
                        sequence_index=0,
                        data={"agent_name": "Fact"},
                    ),
                    _Event(
                        id="e5",
                        event_type="run.completed",
                        sequence_index=1,
                        data={"status": "error"},
                    ),
                ],
            },
        )
        app = create_dashboard_api(store)  # type: ignore[arg-type]
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # Root sees full tree
            resp = await client.get("/api/runs/root")
            ss = resp.json()["delegation"]["subtree_summary"]
            assert ss["descendant_count"] == 2
            assert ss["max_depth"] == 2
            assert ss["subtree_input_tokens"] == 350  # 100+200+50
            assert ss["status_counts"] == {"success": 2, "error": 1}

            # Middle node sees itself + leaf
            resp = await client.get("/api/runs/mid")
            d = resp.json()["delegation"]
            assert d["parent"]["run_id"] == "root"
            assert d["parent"]["resolved"] is True
            assert len(d["ancestry"]) == 1
            assert d["ancestry"][0]["run_id"] == "root"
            ss = d["subtree_summary"]
            assert ss["descendant_count"] == 1
            assert ss["max_depth"] == 1
            assert ss["direct_child_count"] == 1

            # Leaf has ancestry [root, mid]
            resp = await client.get("/api/runs/leaf")
            d = resp.json()["delegation"]
            assert len(d["ancestry"]) == 2
            assert d["ancestry"][0]["run_id"] == "root"
            assert d["ancestry"][1]["run_id"] == "mid"
            assert d["subtree_summary"]["descendant_count"] == 0

    async def test_multiple_children_status_counts(self) -> None:
        """Status counts include self + all descendants."""
        store = DashboardMockStore(
            _runs=[
                _Run(
                    id="root",
                    agent_name="Orch",
                    status="success",
                    total_input_tokens=100,
                    total_output_tokens=50,
                ),
                _Run(
                    id="c1",
                    agent_name="W1",
                    status="success",
                    parent_run_id="root",
                    delegation_level=1,
                    total_input_tokens=50,
                    total_output_tokens=25,
                ),
                _Run(
                    id="c2",
                    agent_name="W2",
                    status="error",
                    parent_run_id="root",
                    delegation_level=1,
                    total_input_tokens=60,
                    total_output_tokens=30,
                ),
                _Run(
                    id="c3",
                    agent_name="W3",
                    status="success",
                    parent_run_id="root",
                    delegation_level=1,
                    total_input_tokens=70,
                    total_output_tokens=35,
                ),
            ],
            _events={
                "root": [
                    _Event(
                        id="e0",
                        event_type="run.started",
                        sequence_index=0,
                        data={"agent_name": "Orch"},
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
            resp = await client.get("/api/runs/root")
            ss = resp.json()["delegation"]["subtree_summary"]
            assert ss["direct_child_count"] == 3
            assert ss["descendant_count"] == 3
            assert ss["status_counts"] == {"success": 3, "error": 1}
            assert ss["subtree_input_tokens"] == 280  # 100+50+60+70

    async def test_cyclic_parent_chain_marks_incomplete(self) -> None:
        """A→B→A cycle in parent_run_id sets ancestry_complete=False."""
        store = DashboardMockStore(
            _runs=[
                _Run(
                    id="a", agent_name="A", status="success", parent_run_id="b", delegation_level=1
                ),
                _Run(
                    id="b", agent_name="B", status="success", parent_run_id="a", delegation_level=1
                ),
            ],
            _events={
                "a": [
                    _Event(
                        id="e0",
                        event_type="run.started",
                        sequence_index=0,
                        data={"agent_name": "A"},
                    ),
                    _Event(
                        id="e1",
                        event_type="run.completed",
                        sequence_index=1,
                        data={"status": "success"},
                    ),
                ],
            },
        )
        app = create_dashboard_api(store)  # type: ignore[arg-type]
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/runs/a")
            d = resp.json()["delegation"]
            # Cycle detected — ancestry is incomplete
            assert d["ancestry_complete"] is False
            # B was walked, then its parent (a) is in visited → stop
            assert len(d["ancestry"]) == 1
            assert d["ancestry"][0]["run_id"] == "b"

    async def test_mixed_null_cost_tracks_unknown_count(self) -> None:
        """Subtree with some known and some unknown costs."""
        store = DashboardMockStore(
            _runs=[
                _Run(
                    id="root",
                    agent_name="Orch",
                    status="success",
                    total_input_tokens=100,
                    total_output_tokens=50,
                    total_cost_usd=0.01,
                ),
                _Run(
                    id="c1",
                    agent_name="W1",
                    status="success",
                    parent_run_id="root",
                    delegation_level=1,
                    total_input_tokens=200,
                    total_output_tokens=100,
                    total_cost_usd=None,
                ),  # Unknown cost
                _Run(
                    id="c2",
                    agent_name="W2",
                    status="success",
                    parent_run_id="root",
                    delegation_level=1,
                    total_input_tokens=300,
                    total_output_tokens=150,
                    total_cost_usd=0.03,
                ),
            ],
            _events={
                "root": [
                    _Event(
                        id="e0",
                        event_type="run.started",
                        sequence_index=0,
                        data={"agent_name": "Orch"},
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
            resp = await client.get("/api/runs/root")
            ss = resp.json()["delegation"]["subtree_summary"]
            # Cost only includes root (0.01) + c2 (0.03) = 0.04
            assert ss["subtree_cost_usd"] == 0.04
            # c1 has unknown cost
            assert ss["unknown_cost_count"] == 1
            # Tokens still sum everything
            assert ss["subtree_input_tokens"] == 600  # 100+200+300

    async def test_all_unknown_cost(self) -> None:
        """When no runs have cost, subtree_cost_usd is None."""
        store = DashboardMockStore(
            _runs=[
                _Run(
                    id="root",
                    agent_name="Orch",
                    status="success",
                    total_input_tokens=100,
                    total_output_tokens=50,
                    total_cost_usd=None,
                ),
            ],
            _events={
                "root": [
                    _Event(
                        id="e0",
                        event_type="run.started",
                        sequence_index=0,
                        data={"agent_name": "Orch"},
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
            resp = await client.get("/api/runs/root")
            ss = resp.json()["delegation"]["subtree_summary"]
            assert ss["subtree_cost_usd"] is None
            assert ss["unknown_cost_count"] == 1


# ------------------------------------------------------------------
# Auth token tests
# ------------------------------------------------------------------


class TestDashboardAuth:
    """Tests for the --auth-token middleware."""

    async def test_no_token_configured_allows_all(self) -> None:
        """Without auth_token, all API requests succeed."""
        store = DashboardMockStore()
        app = create_dashboard_api(store, auth_token=None)  # type: ignore[arg-type]
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/runs")
            assert resp.status_code == 200

    async def test_missing_token_returns_401(self) -> None:
        """Request without auth header returns 401 when token is set."""
        store = DashboardMockStore()
        app = create_dashboard_api(store, auth_token="secret123")  # type: ignore[arg-type]
        transport = ASGITransport(app=app, raise_app_exceptions=False)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/runs")
            assert resp.status_code == 401
            assert "auth token" in resp.json()["detail"].lower()

    async def test_wrong_token_returns_401(self) -> None:
        """Request with wrong token returns 401."""
        store = DashboardMockStore()
        app = create_dashboard_api(store, auth_token="secret123")  # type: ignore[arg-type]
        transport = ASGITransport(app=app, raise_app_exceptions=False)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/api/runs",
                headers={"Authorization": "Bearer wrong"},
            )
            assert resp.status_code == 401

    async def test_correct_token_returns_200(self) -> None:
        """Request with correct token succeeds."""
        store = DashboardMockStore()
        app = create_dashboard_api(store, auth_token="secret123")  # type: ignore[arg-type]
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/api/runs",
                headers={"Authorization": "Bearer secret123"},
            )
            assert resp.status_code == 200

    async def test_static_files_bypass_auth(self) -> None:
        """Static file requests don't require auth."""
        store = DashboardMockStore()
        app = create_dashboard_api(store, auth_token="secret123")  # type: ignore[arg-type]
        transport = ASGITransport(app=app, raise_app_exceptions=False)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/")
            assert resp.status_code == 200
