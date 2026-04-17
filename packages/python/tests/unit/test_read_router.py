"""Tests for ``make_read_router`` — the public mountable read router.

Exercises the HTTP contract over a real RunStore + in-memory SQLite,
with a small auth dependency injected by the test.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI, HTTPException
from httpx import ASGITransport, AsyncClient
from sqlalchemy import event
from sqlalchemy.ext.asyncio import create_async_engine

from dendrux.db.models import Base
from dendrux.http import make_read_router
from dendrux.runtime.state import SQLAlchemyStateStore
from dendrux.store import RunStore
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


def _build_app(store: RunStore, authorize=None) -> FastAPI:
    """Wrap make_read_router in a minimal FastAPI app for the test client."""
    if authorize is None:

        async def _allow() -> None:
            return None

        authorize = _allow

    app = FastAPI()
    app.include_router(make_read_router(store=store, authorize=authorize))
    return app


async def _client(app: FastAPI) -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


# ------------------------------------------------------------------
# /health
# ------------------------------------------------------------------


class TestHealth:
    async def test_health_ok(self, store) -> None:
        app = _build_app(store)
        async with await _client(app) as http:
            resp = await http.get("/health")
            assert resp.status_code == 200
            assert resp.json() == {"status": "ok"}

    async def test_health_bypasses_auth(self, store) -> None:
        """Health probes (k8s liveness/readiness) must reach /health even
        when auth would otherwise deny."""

        async def deny() -> None:
            raise HTTPException(status_code=401, detail="nope")

        app = _build_app(store, authorize=deny)
        async with await _client(app) as http:
            resp = await http.get("/health")
            assert resp.status_code == 200


# ------------------------------------------------------------------
# Auth dependency
# ------------------------------------------------------------------


class TestAuthDependency:
    async def test_authorize_deny_returns_401(self, store) -> None:
        async def deny() -> None:
            raise HTTPException(status_code=401, detail="nope")

        app = _build_app(store, authorize=deny)
        async with await _client(app) as http:
            resp = await http.get("/runs")
            assert resp.status_code == 401
            assert resp.json()["detail"] == "nope"

    async def test_authorize_runs_on_every_authenticated_route(self, store, internal_store) -> None:
        """The dep must gate every data endpoint. /health is intentionally
        unauth'd and must not invoke the dependency."""
        await internal_store.create_run("r1", "Agent")

        call_count = 0

        async def counter() -> None:
            nonlocal call_count
            call_count += 1

        app = _build_app(store, authorize=counter)
        async with await _client(app) as http:
            await http.get("/health")  # must NOT run the dep
            await http.get("/runs")
            await http.get("/runs/r1")
            await http.get("/runs/r1/events")
            await http.get("/runs/r1/llm-calls")
            await http.get("/runs/r1/tool-calls")
            await http.get("/runs/r1/traces")
            await http.get("/runs/r1/pauses")

        assert call_count == 7


# ------------------------------------------------------------------
# GET /runs — list with filters and envelope
# ------------------------------------------------------------------


class TestListRunsEndpoint:
    async def test_returns_envelope(self, store, internal_store) -> None:
        await internal_store.create_run("r1", "Alpha")
        await internal_store.create_run("r2", "Beta")

        app = _build_app(store)
        async with await _client(app) as http:
            resp = await http.get("/runs")
            assert resp.status_code == 200
            body = resp.json()
            assert set(body.keys()) == {"items", "total", "limit", "offset"}
            assert body["total"] == 2
            assert len(body["items"]) == 2
            # Items serialize the public RunSummary shape.
            item = body["items"][0]
            assert "run_id" in item
            assert "agent_name" in item
            assert "status" in item

    async def test_filters_agent_name(self, store, internal_store) -> None:
        await internal_store.create_run("r1", "Alpha")
        await internal_store.create_run("r2", "Beta")
        await internal_store.create_run("r3", "Alpha")

        app = _build_app(store)
        async with await _client(app) as http:
            resp = await http.get("/runs?agent_name=Alpha")
            body = resp.json()
            assert body["total"] == 2
            assert {r["run_id"] for r in body["items"]} == {"r1", "r3"}

    async def test_pagination_reflects_cursor(self, store, internal_store) -> None:
        for i in range(5):
            await internal_store.create_run(f"r_{i}", "Agent")

        app = _build_app(store)
        async with await _client(app) as http:
            resp = await http.get("/runs?limit=2&offset=0")
            body = resp.json()
            assert body["total"] == 5
            assert len(body["items"]) == 2
            assert body["limit"] == 2
            assert body["offset"] == 0

    async def test_status_list_via_repeated_query_param(self, store, internal_store) -> None:
        """`?status=success&status=error` → list of statuses."""
        await internal_store.create_run("r1", "Agent")
        await internal_store.create_run("r2", "Agent")
        await internal_store.create_run("r3", "Agent")
        await internal_store.finalize_run("r1", status="success")
        await internal_store.finalize_run("r2", status="error")

        app = _build_app(store)
        async with await _client(app) as http:
            resp = await http.get("/runs?status=success&status=error")
            body = resp.json()
            assert {r["run_id"] for r in body["items"]} == {"r1", "r2"}


# ------------------------------------------------------------------
# GET /runs/{run_id}
# ------------------------------------------------------------------


class TestGetRunEndpoint:
    async def test_returns_detail(self, store, internal_store) -> None:
        await internal_store.create_run("r1", "Analyst", model="claude-sonnet")
        await internal_store.finalize_run("r1", status="success", answer="done")

        app = _build_app(store)
        async with await _client(app) as http:
            resp = await http.get("/runs/r1")
            assert resp.status_code == 200
            body = resp.json()
            assert body["run_id"] == "r1"
            assert body["status"] == "success"
            assert body["answer"] == "done"
            assert body["model"] == "claude-sonnet"

    async def test_unknown_returns_404(self, store) -> None:
        app = _build_app(store)
        async with await _client(app) as http:
            resp = await http.get("/runs/missing")
            assert resp.status_code == 404


# ------------------------------------------------------------------
# GET /runs/{run_id}/events — cursor envelope
# ------------------------------------------------------------------


class TestGetEventsEndpoint:
    async def test_envelope_with_cursor(self, store, internal_store) -> None:
        await internal_store.create_run("r1", "Agent")
        for i in range(3):
            await internal_store.save_run_event("r1", event_type="tick", sequence_index=i)

        app = _build_app(store)
        async with await _client(app) as http:
            resp = await http.get("/runs/r1/events")
            body = resp.json()
            assert set(body.keys()) == {"items", "next_cursor"}
            assert [e["sequence_index"] for e in body["items"]] == [0, 1, 2]
            assert body["next_cursor"] == 2

    async def test_cursor_param(self, store, internal_store) -> None:
        await internal_store.create_run("r1", "Agent")
        for i in range(5):
            await internal_store.save_run_event("r1", event_type="tick", sequence_index=i)

        app = _build_app(store)
        async with await _client(app) as http:
            resp = await http.get("/runs/r1/events?after=2")
            body = resp.json()
            assert [e["sequence_index"] for e in body["items"]] == [3, 4]
            assert body["next_cursor"] == 4

    async def test_empty_cursor_is_none(self, store, internal_store) -> None:
        """No events and no cursor => next_cursor is null."""
        await internal_store.create_run("r1", "Agent")

        app = _build_app(store)
        async with await _client(app) as http:
            resp = await http.get("/runs/r1/events")
            body = resp.json()
            assert body["items"] == []
            assert body["next_cursor"] is None

    async def test_empty_cursor_echoes_after(self, store, internal_store) -> None:
        """Empty batch with a cursor echoes the cursor back — a poller
        must be able to resume without losing its place."""
        await internal_store.create_run("r1", "Agent")
        await internal_store.save_run_event("r1", event_type="tick", sequence_index=0)

        app = _build_app(store)
        async with await _client(app) as http:
            resp = await http.get("/runs/r1/events?after=0")
            body = resp.json()
            assert body["items"] == []
            assert body["next_cursor"] == 0

    async def test_unknown_run_returns_404(self, store) -> None:
        app = _build_app(store)
        async with await _client(app) as http:
            resp = await http.get("/runs/missing/events")
            assert resp.status_code == 404

    async def test_limit_above_max_returns_422(self, store, internal_store) -> None:
        """Query validation rejects limits beyond the cap."""
        await internal_store.create_run("r1", "Agent")

        app = _build_app(store)
        async with await _client(app) as http:
            resp = await http.get("/runs/r1/events?limit=5000")
            assert resp.status_code == 422

    async def test_datetime_serialized_as_iso_string(self, store, internal_store) -> None:
        """Datetime fields round-trip through JSON as ISO strings."""
        await internal_store.create_run("r1", "Agent")
        await internal_store.save_run_event("r1", event_type="run.started", sequence_index=0)

        app = _build_app(store)
        async with await _client(app) as http:
            resp = await http.get("/runs/r1/events")
            body = resp.json()
            assert body["items"]
            ts = body["items"][0]["created_at"]
            assert isinstance(ts, str)
            # ISO 8601 — must parse back.
            from datetime import datetime as _dt

            _dt.fromisoformat(ts.replace("Z", "+00:00"))


# ------------------------------------------------------------------
# GET /runs/{run_id}/llm-calls
# ------------------------------------------------------------------


class TestLLMCallsEndpoint:
    async def test_returns_items_envelope(self, store, internal_store) -> None:
        await internal_store.create_run("r1", "Agent")
        await internal_store.save_llm_interaction(
            "r1",
            iteration_index=0,
            usage=UsageStats(
                input_tokens=10,
                output_tokens=5,
                total_tokens=15,
                cache_read_input_tokens=2,
            ),
            model="claude-sonnet",
            provider="anthropic",
            duration_ms=100,
        )

        app = _build_app(store)
        async with await _client(app) as http:
            resp = await http.get("/runs/r1/llm-calls")
            body = resp.json()
            assert set(body.keys()) == {"items", "limit", "offset"}
            assert len(body["items"]) == 1
            call = body["items"][0]
            assert call["iteration"] == 0
            assert call["input_tokens"] == 10
            assert call["cache_read_input_tokens"] == 2

    async def test_iteration_filter(self, store, internal_store) -> None:
        await internal_store.create_run("r1", "Agent")
        for i in range(3):
            await internal_store.save_llm_interaction(
                "r1",
                iteration_index=i,
                usage=UsageStats(input_tokens=1, output_tokens=1, total_tokens=2),
            )

        app = _build_app(store)
        async with await _client(app) as http:
            resp = await http.get("/runs/r1/llm-calls?iteration=1")
            body = resp.json()
            assert len(body["items"]) == 1
            assert body["items"][0]["iteration"] == 1

    async def test_unknown_run_404(self, store) -> None:
        app = _build_app(store)
        async with await _client(app) as http:
            resp = await http.get("/runs/missing/llm-calls")
            assert resp.status_code == 404


# ------------------------------------------------------------------
# GET /runs/{run_id}/tool-calls
# ------------------------------------------------------------------


class TestToolCallsEndpoint:
    async def test_returns_items_envelope(self, store, internal_store) -> None:
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

        app = _build_app(store)
        async with await _client(app) as http:
            resp = await http.get("/runs/r1/tool-calls")
            body = resp.json()
            assert set(body.keys()) == {"items", "limit", "offset"}
            assert len(body["items"]) == 1
            tc = body["items"][0]
            assert tc["tool_name"] == "lookup_price"
            assert tc["params"] == {"ticker": "AAPL"}
            assert tc["success"] is True

    async def test_iteration_filter(self, store, internal_store) -> None:
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

        app = _build_app(store)
        async with await _client(app) as http:
            resp = await http.get("/runs/r1/tool-calls?iteration=1")
            body = resp.json()
            assert len(body["items"]) == 1
            assert body["items"][0]["iteration"] == 1

    async def test_unknown_run_404(self, store) -> None:
        app = _build_app(store)
        async with await _client(app) as http:
            resp = await http.get("/runs/missing/tool-calls")
            assert resp.status_code == 404


# ------------------------------------------------------------------
# GET /runs/{run_id}/traces
# ------------------------------------------------------------------


class TestTracesEndpoint:
    async def test_returns_items_envelope(self, store, internal_store) -> None:
        await internal_store.create_run("r1", "Agent")
        await internal_store.save_trace("r1", "user", "hello", order_index=0)
        await internal_store.save_trace("r1", "assistant", "hi back", order_index=1)

        app = _build_app(store)
        async with await _client(app) as http:
            resp = await http.get("/runs/r1/traces")
            body = resp.json()
            assert set(body.keys()) == {"items", "limit", "offset"}
            assert len(body["items"]) == 2
            assert [t["role"] for t in body["items"]] == ["user", "assistant"]
            assert [t["order_index"] for t in body["items"]] == [0, 1]
            assert body["items"][0]["content"] == "hello"

    async def test_unknown_run_404(self, store) -> None:
        app = _build_app(store)
        async with await _client(app) as http:
            resp = await http.get("/runs/missing/traces")
            assert resp.status_code == 404

    async def test_envelope_echoes_limit_and_offset(self, store, internal_store) -> None:
        """Drill-down pagination echoes limit/offset so clients can page
        without tracking their own state."""
        await internal_store.create_run("r1", "Agent")
        for i in range(5):
            await internal_store.save_trace("r1", "user", f"msg {i}", order_index=i)

        app = _build_app(store)
        async with await _client(app) as http:
            resp = await http.get("/runs/r1/traces?limit=2&offset=1")
            body = resp.json()
            assert body["limit"] == 2
            assert body["offset"] == 1
            assert [t["order_index"] for t in body["items"]] == [1, 2]


# ------------------------------------------------------------------
# GET /runs/{run_id}/pauses
# ------------------------------------------------------------------


class TestPausesEndpoint:
    async def test_pairs_derived_from_events(self, store, internal_store) -> None:
        await internal_store.create_run("r1", "Agent")
        await internal_store.save_run_event(
            "r1",
            event_type="run.paused",
            sequence_index=0,
            data={
                "status": "waiting_client_tool",
                "pending_tool_calls": [{"id": "tc_1", "name": "read_file"}],
            },
        )
        await internal_store.save_run_event(
            "r1",
            event_type="run.resumed",
            sequence_index=1,
            data={"resumed_from": "waiting_client_tool"},
        )

        app = _build_app(store)
        async with await _client(app) as http:
            resp = await http.get("/runs/r1/pauses")
            body = resp.json()
            assert len(body["items"]) == 1
            pause = body["items"][0]
            assert pause["reason"] == "waiting_client_tool"
            assert pause["pause_sequence_index"] == 0
            assert pause["resume_sequence_index"] == 1

    async def test_empty(self, store, internal_store) -> None:
        await internal_store.create_run("r1", "Agent")

        app = _build_app(store)
        async with await _client(app) as http:
            resp = await http.get("/runs/r1/pauses")
            body = resp.json()
            assert body == {"items": []}

    async def test_unknown_run_404(self, store) -> None:
        app = _build_app(store)
        async with await _client(app) as http:
            resp = await http.get("/runs/missing/pauses")
            assert resp.status_code == 404
