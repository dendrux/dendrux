"""Tests for ``make_read_router`` — the public mountable read router.

Exercises the HTTP contract over a real RunStore + in-memory SQLite,
with a small auth dependency injected by the test.
"""

from __future__ import annotations

import asyncio
import contextlib

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


# ------------------------------------------------------------------
# GET /runs/{run_id}/events/stream — DB-backed SSE
# ------------------------------------------------------------------


def _parse_sse_frames(raw: str) -> list[dict[str, str]]:
    """Parse a block of raw SSE text into a list of frame dicts.

    Each frame ends with a blank line. Comment lines (``: foo``) produce
    a frame with ``{"comment": "foo"}``. Data lines are concatenated.
    """
    import json as _json

    frames: list[dict[str, str]] = []
    current: dict[str, str] = {}
    data_lines: list[str] = []
    for line in raw.splitlines():
        if line == "":
            if data_lines:
                current["data"] = "\n".join(data_lines)
                with contextlib.suppress(_json.JSONDecodeError, KeyError):
                    current["json"] = _json.loads(current["data"])
            if current:
                frames.append(current)
            current = {}
            data_lines = []
            continue
        if line.startswith(":"):
            frames.append({"comment": line[1:].lstrip()})
            continue
        field, _, value = line.partition(":")
        value = value.lstrip()
        if field == "data":
            data_lines.append(value)
        else:
            current[field] = value
    return frames


class TestSseGenerator:
    """Unit tests for the SSE generator function — decoupled from HTTP
    transport so timing is deterministic."""

    async def test_yields_existing_events(self, store, internal_store, monkeypatch) -> None:
        import dendrux.http.read_router as _rr

        monkeypatch.setattr(_rr, "_SSE_POLL_INTERVAL_S", 0.01)

        await internal_store.create_run("r1", "Agent")
        await internal_store.save_run_event("r1", event_type="run.started", sequence_index=0)
        await internal_store.save_run_event("r1", event_type="llm.completed", sequence_index=1)

        gen = _rr._sse_event_generator(store, "r1", cursor=None)
        frame1 = await gen.__anext__()
        frame2 = await gen.__anext__()
        await gen.aclose()

        assert "id: 0" in frame1
        assert "event: message" in frame1
        assert '"sequence_index": 0' in frame1
        assert '"event_type": "run.started"' in frame1

        assert "id: 1" in frame2
        assert '"sequence_index": 1' in frame2

    async def test_cursor_skips_events(self, store, internal_store, monkeypatch) -> None:
        import dendrux.http.read_router as _rr

        monkeypatch.setattr(_rr, "_SSE_POLL_INTERVAL_S", 0.01)

        await internal_store.create_run("r1", "Agent")
        for i in range(3):
            await internal_store.save_run_event("r1", event_type="tick", sequence_index=i)

        gen = _rr._sse_event_generator(store, "r1", cursor=1)
        frame = await gen.__anext__()
        await gen.aclose()

        assert '"sequence_index": 2' in frame
        assert "id: 2" in frame

    async def test_live_events_delivered(self, store, internal_store, monkeypatch) -> None:
        import dendrux.http.read_router as _rr

        monkeypatch.setattr(_rr, "_SSE_POLL_INTERVAL_S", 0.01)

        await internal_store.create_run("r1", "Agent")

        gen = _rr._sse_event_generator(store, "r1", cursor=None)

        async def _write_later() -> None:
            await asyncio.sleep(0.05)
            await internal_store.save_run_event("r1", event_type="run.completed", sequence_index=0)

        write_task = asyncio.create_task(_write_later())
        frame = await asyncio.wait_for(gen.__anext__(), timeout=2.0)
        await write_task
        await gen.aclose()

        assert '"event_type": "run.completed"' in frame

    async def test_heartbeat_when_idle(self, store, internal_store, monkeypatch) -> None:
        import dendrux.http.read_router as _rr

        monkeypatch.setattr(_rr, "_SSE_HEARTBEAT_INTERVAL_S", 0.01)
        monkeypatch.setattr(_rr, "_SSE_POLL_INTERVAL_S", 0.005)

        await internal_store.create_run("r1", "Agent")

        gen = _rr._sse_event_generator(store, "r1", cursor=None)
        frame = await asyncio.wait_for(gen.__anext__(), timeout=2.0)
        await gen.aclose()

        assert frame.startswith(": keepalive")

    async def test_cancellation_cleans_up(self, store, internal_store) -> None:
        """Cancelling the generator while it's waiting on a sleep must
        close without leaving tasks or sessions behind."""
        from dendrux.http.read_router import _sse_event_generator

        await internal_store.create_run("r1", "Agent")

        gen = _sse_event_generator(store, "r1", cursor=None)
        task = asyncio.create_task(gen.__anext__())
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await gen.aclose()


class TestSseHttpEndpoint:
    """Smoke tests over HTTP — verify the endpoint is wired. Deep
    behavior is covered by ``TestSseGenerator`` above, which exercises
    the generator directly without HTTP streaming quirks."""

    async def test_404_before_stream_opens(self, store) -> None:
        """Unknown run_id returns 404 without starting the stream."""
        app = _build_app(store)
        async with await _client(app) as http:
            resp = await http.get("/runs/missing/events/stream")
            assert resp.status_code == 404

    async def test_auth_dep_applies(self, store, internal_store) -> None:
        """SSE endpoint is gated by the same auth dep as other routes."""

        async def deny() -> None:
            raise HTTPException(status_code=401, detail="nope")

        await internal_store.create_run("r1", "Agent")

        app = _build_app(store, authorize=deny)
        async with await _client(app) as http:
            resp = await http.get("/runs/r1/events/stream")
            assert resp.status_code == 401


class TestResolveSseCursor:
    """Direct unit tests for cursor resolution — avoids HTTP streaming
    gymnastics in the test client."""

    def test_header_int_wins(self) -> None:
        from dendrux.http.read_router import _resolve_sse_cursor

        assert _resolve_sse_cursor("42", after=100) == 42

    def test_no_header_uses_after(self) -> None:
        from dendrux.http.read_router import _resolve_sse_cursor

        assert _resolve_sse_cursor(None, after=7) == 7

    def test_empty_header_uses_after(self) -> None:
        from dendrux.http.read_router import _resolve_sse_cursor

        assert _resolve_sse_cursor("", after=7) == 7

    def test_malformed_header_falls_back_to_after(self) -> None:
        from dendrux.http.read_router import _resolve_sse_cursor

        assert _resolve_sse_cursor("not-a-number", after=5) == 5
        assert _resolve_sse_cursor("not-a-number", after=None) is None

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
