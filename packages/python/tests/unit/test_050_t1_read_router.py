"""0.5.0 T1: read router — tenant resolver (B1, D-002), payload exposure (B2,
D-006), bounded reads (D-058, D-121) and SSE close/caps/shared polling (D-057).
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from fastapi import FastAPI, Request
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import create_async_engine

import dendrux.http.read_router as rr
from dendrux.db.models import Base
from dendrux.http import make_read_router
from dendrux.runtime.state import UNSCOPED, SQLAlchemyStateStore
from dendrux.store import RunStore
from dendrux.types import UsageStats


@pytest.fixture
async def engine(tmp_path: Any) -> Any:
    eng = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'router.db'}")
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield eng
    await eng.dispose()


@pytest.fixture
def state(engine: Any) -> SQLAlchemyStateStore:
    return SQLAlchemyStateStore(engine)


@pytest.fixture
def store(engine: Any) -> RunStore:
    return RunStore.from_engine(engine)


@pytest.fixture(autouse=True)
def fast_sse(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rr, "_SSE_POLL_INTERVAL_S", 0.01)
    monkeypatch.setattr(rr, "_SSE_POLL_MAX_S", 0.02)
    monkeypatch.setattr(rr, "_SSE_STATUS_EVERY", 1)


async def _allow() -> None:
    return None


def _tenant_from_header(request: Request) -> Any:
    value = request.headers.get("x-tenant")
    if value == "*":
        return UNSCOPED
    if value == "-":
        return None
    return value


def _app(store: RunStore, resolver: Any = _tenant_from_header, **kw: Any) -> FastAPI:
    app = FastAPI()
    app.include_router(
        make_read_router(store=store, authorize=_allow, tenant_resolver=resolver, **kw)
    )
    return app


def _client(app: FastAPI) -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://t")


async def _seed(state: Any) -> None:
    for run_id, tenant in (("ra", "A"), ("rb", "B"), ("rn", None)):
        await state.create_run(run_id, "agent", tenant_id=tenant)
        await state.save_trace(run_id, "user", f"{run_id} private chat", order_index=0)
        await state.save_run_event(
            run_id, event_type="run.started", data={"agent_name": "a", "system_prompt": "SECRET"}
        )
        await state.save_tool_call(
            run_id,
            tool_call_id=f"tc-{run_id}",
            provider_tool_call_id=None,
            tool_name="lookup",
            target="server",
            params={"card": "4111"},
            result_payload='"balance 9"',
            success=True,
            duration_ms=1,
            iteration_index=0,
        )
        await state.save_llm_interaction(
            run_id,
            iteration_index=0,
            usage=UsageStats(input_tokens=1, output_tokens=1),
            provider_request={"messages": ["raw"]},
            provider_response={"raw": True},
        )
        await state.save_run_event(
            run_id,
            event_type="run.paused",
            data={
                "status": "waiting_client_tool",
                "pending_tool_calls": [{"id": "c1", "name": "lookup", "params": {"card": "4111"}}],
            },
        )


_RUN_ROUTES = (
    "/runs/{id}",
    "/runs/{id}/events",
    "/runs/{id}/llm-calls",
    "/runs/{id}/tool-calls",
    "/runs/{id}/traces",
    "/runs/{id}/pauses",
    "/runs/{id}/events/stream",
)


class TestTenantResolver:
    async def test_list_shows_only_the_callers_tenant(self, state: Any, store: RunStore) -> None:
        await _seed(state)
        async with _client(_app(store)) as http:
            body = (await http.get("/runs", headers={"x-tenant": "A"})).json()
            assert [i["run_id"] for i in body["items"]] == ["ra"]
            assert body["total"] == 1
            assert body["items"][0]["tenant_id"] == "A"
            untenanted = (await http.get("/runs", headers={"x-tenant": "-"})).json()
            assert [i["run_id"] for i in untenanted["items"]] == ["rn"]

    async def test_query_tenant_cannot_widen_a_scoped_caller(
        self, state: Any, store: RunStore
    ) -> None:
        await _seed(state)
        async with _client(_app(store)) as http:
            body = (await http.get("/runs?tenant_id=B", headers={"x-tenant": "A"})).json()
            assert body == {"items": [], "total": 0, "limit": 50, "offset": 0}
            same = (await http.get("/runs?tenant_id=A", headers={"x-tenant": "A"})).json()
            assert [i["run_id"] for i in same["items"]] == ["ra"]

    async def test_unscoped_caller_sees_all_and_may_narrow(
        self, state: Any, store: RunStore
    ) -> None:
        await _seed(state)
        async with _client(_app(store)) as http:
            everything = (await http.get("/runs", headers={"x-tenant": "*"})).json()
            assert everything["total"] == 3
            narrowed = (await http.get("/runs?tenant_id=B", headers={"x-tenant": "*"})).json()
            assert [i["run_id"] for i in narrowed["items"]] == ["rb"]

    @pytest.mark.parametrize("route", _RUN_ROUTES)
    async def test_foreign_and_missing_runs_get_the_same_404(
        self, state: Any, store: RunStore, route: str
    ) -> None:
        await _seed(state)
        for run_id in ("ra", "rb", "rn"):  # terminal, so the stream route returns
            await state.finalize_run(run_id, status="success")
        async with _client(_app(store)) as http:
            foreign = await http.get(route.format(id="rb"), headers={"x-tenant": "A"})
            missing = await http.get(route.format(id="rb"), headers={"x-tenant": "Z"})
            assert foreign.status_code == missing.status_code == 404
            assert foreign.json() == missing.json() == {"detail": "Run 'rb' not found."}
            own = await http.get(route.format(id="ra"), headers={"x-tenant": "A"})
            assert own.status_code in (200, 204)

    @pytest.mark.parametrize("bad", [42, "", ["A"], object()])
    async def test_invalid_resolver_value_fails_closed(self, store: RunStore, bad: Any) -> None:
        async def resolver() -> Any:
            return bad

        async with _client(_app(store, resolver)) as http:
            for path in ("/runs", "/runs/ra", "/runs/ra/events/stream"):
                resp = await http.get(path)
                assert resp.status_code == 500
                assert resp.json() == {"detail": "Tenant scope could not be resolved."}

    def test_generator_resolver_rejected(self, store: RunStore) -> None:
        async def agen() -> Any:
            yield "A"

        def gen() -> Any:
            yield "A"

        for resolver in (agen, gen):
            with pytest.raises(TypeError, match="yield"):
                make_read_router(store=store, authorize=_allow, tenant_resolver=resolver)

    async def test_sync_resolver_with_request(self, state: Any, store: RunStore) -> None:
        await _seed(state)

        def resolver(request: Request) -> str:
            return request.query_params.get("who", "A")

        async with _client(_app(store, resolver)) as http:
            assert (await http.get("/runs/ra")).status_code == 200
            assert (await http.get("/runs/ra?who=B")).status_code == 404


class TestPayloads:
    async def test_hidden_by_default(self, state: Any, store: RunStore) -> None:
        await _seed(state)
        async with _client(_app(store, UNSCOPED)) as http:
            llm = (await http.get("/runs/ra/llm-calls")).json()["items"][0]
            assert llm["provider_request"] is None and llm["provider_response"] is None
            tc = (await http.get("/runs/ra/tool-calls")).json()["items"][0]
            assert tc["params"] is None and tc["result"] is None
            assert tc["tool_name"] == "lookup"
            trace = (await http.get("/runs/ra/traces")).json()["items"][0]
            assert trace["content"] is None and trace["role"] == "user"
            events = (await http.get("/runs/ra/events")).json()["items"]
            started = next(e for e in events if e["event_type"] == "run.started")
            assert "system_prompt" not in started["data"]
            assert started["data"]["agent_name"] == "a"
            paused = next(e for e in events if e["event_type"] == "run.paused")
            assert paused["data"]["pending_tool_calls"] == [{"id": "c1", "name": "lookup"}]
            pauses = (await http.get("/runs/ra/pauses")).json()["items"]
            assert pauses[0]["pending_tool_calls"] == [{"id": "c1", "name": "lookup"}]
            raw = json.dumps([llm, tc, trace, events, pauses])
            assert "4111" not in raw and "SECRET" not in raw and "private chat" not in raw

    async def test_included_when_enabled(self, state: Any, store: RunStore) -> None:
        await _seed(state)
        async with _client(_app(store, UNSCOPED, include_payloads=True)) as http:
            assert (await http.get("/runs/ra/llm-calls")).json()["items"][0][
                "provider_request"
            ] == {"messages": ["raw"]}
            assert (await http.get("/runs/ra/tool-calls")).json()["items"][0]["params"] == {
                "card": "4111"
            }
            assert (await http.get("/runs/ra/traces")).json()["items"][0]["content"] == (
                "ra private chat"
            )
            events = (await http.get("/runs/ra/events")).json()["items"]
            assert events[0]["data"]["system_prompt"] == "SECRET"

    async def test_sse_frames_are_redacted_by_default(self, state: Any, store: RunStore) -> None:
        await _seed(state)
        await state.finalize_run("ra", status="success")
        async with _client(_app(store, UNSCOPED)) as http:
            body = (await http.get("/runs/ra/events/stream")).text
        assert "SECRET" not in body and "4111" not in body
        assert '"event_type": "run.started"' in body


class TestBounds:
    async def test_naive_datetime_is_422(self, store: RunStore) -> None:
        async with _client(_app(store, UNSCOPED)) as http:
            resp = await http.get("/runs?started_after=2026-01-01T00:00:00")
            assert resp.status_code == 422
            ok = await http.get("/runs?started_after=2026-01-01T00:00:00%2B05:00")
            assert ok.status_code == 200

    async def test_offsets_are_capped(self, store: RunStore) -> None:
        async with _client(_app(store, UNSCOPED)) as http:
            assert (await http.get("/runs?offset=10001")).status_code == 422
            assert (await http.get("/runs?offset=10000")).status_code == 200

    async def test_llm_calls_default_50_max_200(self, state: Any, store: RunStore) -> None:
        await state.create_run("r1", "a")
        async with _client(_app(store, UNSCOPED)) as http:
            assert (await http.get("/runs/r1/llm-calls")).json()["limit"] == 50
            assert (await http.get("/runs/r1/llm-calls?limit=200")).status_code == 200
            assert (await http.get("/runs/r1/llm-calls?limit=201")).status_code == 422

    async def test_empty_tenant_query_is_422(self, store: RunStore) -> None:
        async with _client(_app(store, UNSCOPED)) as http:
            assert (await http.get("/runs?tenant_id=")).status_code == 422

    def test_sse_timestamp_is_rfc3339_utc(self) -> None:
        import datetime as dt

        from dendrux.store import StoredEvent

        ev = StoredEvent(
            sequence_index=0,
            iteration_index=0,
            event_type="x",
            correlation_id=None,
            data=None,
            created_at=dt.datetime(2026, 1, 2, 3, 4, 5, tzinfo=dt.timezone(dt.timedelta(hours=2))),
        )
        frame = rr._format_sse_frame(ev)
        payload = json.loads(frame.split("data: ", 1)[1])
        assert payload["timestamp"] == "2026-01-02T01:04:05Z"


def _frames(raw: str) -> list[tuple[str, Any]]:
    out: list[tuple[str, Any]] = []
    for block in raw.split("\n\n"):
        if not block.strip() or block.startswith(":"):
            continue
        fields = dict(line.split(": ", 1) for line in block.splitlines() if ": " in line)
        out.append((fields.get("event", "message"), json.loads(fields["data"])))
    return out


class TestSseClose:
    async def test_terminal_run_closes_after_draining(self, state: Any, store: RunStore) -> None:
        await state.create_run("r1", "a")
        for i in range(3):
            await state.save_run_event("r1", event_type=f"e{i}")
        await state.finalize_with_event(
            "r1", status="success", allowed_current_statuses=["running"], event_type="run.completed"
        )
        async with _client(_app(store, UNSCOPED)) as http:
            resp = await http.get("/runs/r1/events/stream")
        frames = _frames(resp.text)
        assert [f[1]["sequence_index"] for f in frames[:-1]] == [0, 1, 2, 3]
        assert frames[-1] == ("end", {"reason": "terminal", "cursor": 3})

    async def test_reconnect_to_a_drained_terminal_run_is_204(
        self, state: Any, store: RunStore
    ) -> None:
        await state.create_run("r1", "a")
        await state.save_run_event("r1", event_type="run.completed")
        await state.finalize_run("r1", status="success")
        async with _client(_app(store, UNSCOPED)) as http:
            resp = await http.get("/runs/r1/events/stream", headers={"last-event-id": "0"})
            assert resp.status_code == 204
            partial = await http.get("/runs/r1/events/stream?after=-1")
            assert partial.status_code == 200
            assert _frames(partial.text)[-1][0] == "end"

    async def test_live_run_ends_once_it_finishes(self, state: Any, store: RunStore) -> None:
        await state.create_run("r1", "a")
        gen = rr._sse_event_generator(store, "r1", None)

        async def finish() -> None:
            await asyncio.sleep(0.05)
            await state.save_run_event("r1", event_type="tick")
            await state.finalize_with_event(
                "r1",
                status="error",
                allowed_current_statuses=["running"],
                event_type="run.error",
            )

        writer = asyncio.create_task(finish())
        frames = [f async for f in gen if not f.startswith(":")]
        await writer
        assert [_frames(f)[0][1].get("sequence_index") for f in frames[:-1]] == [0, 1]
        assert _frames(frames[-1])[0] == ("end", {"reason": "terminal", "cursor": 1})

    async def test_status_flip_without_event_still_ends(self, state: Any, store: RunStore) -> None:
        await state.create_run("r1", "a")
        gen = rr._sse_event_generator(store, "r1", None)

        async def sweep_like() -> None:
            await asyncio.sleep(0.05)
            await state.finalize_run("r1", status="error")

        writer = asyncio.create_task(sweep_like())
        frames = [f async for f in gen if not f.startswith(":")]
        await writer
        assert _frames(frames[-1])[0][1]["reason"] == "terminal"

    async def test_follow_keeps_streaming_after_terminal(self, state: Any, store: RunStore) -> None:
        await state.create_run("r1", "a")
        await state.save_run_event("r1", event_type="run.completed")
        await state.finalize_run("r1", status="success")
        gen = rr._sse_event_generator(store, "r1", None, follow=True)
        first = await asyncio.wait_for(gen.__anext__(), 5)
        assert '"sequence_index": 0' in first
        await state.save_run_event("r1", event_type="late")
        nxt = await asyncio.wait_for(gen.__anext__(), 5)
        assert '"event_type": "late"' in nxt
        await gen.aclose()


class TestSseConnectionCap:
    async def test_full_router_returns_503_and_releases_slots(
        self, state: Any, store: RunStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        await state.create_run("r1", "a")
        await state.save_run_event("r1", event_type="run.completed")
        await state.finalize_run("r1", status="success")
        feeds: list[Any] = []
        make = rr._make_feed

        def capture(*args: Any, **kwargs: Any) -> Any:
            feeds.append(make(*args, **kwargs))
            return feeds[-1]

        monkeypatch.setattr(rr, "_make_feed", capture)
        app = _app(store, UNSCOPED, sse_max_connections=1)
        async with _client(app) as http:
            assert (await http.get("/runs/r1/events/stream?after=-1")).status_code == 200
            (feed,) = feeds
            assert feed.max_followers == 1 and feed.followers == 0  # released after the stream
            feed._acquire()  # someone holds the only slot
            resp = await http.get("/runs/r1/events/stream?after=-1")
            assert resp.status_code == 503
            assert resp.headers["retry-after"] == "5"
            feed._release()
            assert (await http.get("/runs/r1/events/stream?after=-1")).status_code == 200
        assert feed.followers == 0


@pytest.mark.parametrize("disconnect", [True, False])
async def test_sse_disconnect_and_stalled_send_release_slot(
    state: Any,
    store: RunStore,
    disconnect: bool,
) -> None:
    from dendrux.store import EventFeed

    await state.create_run("disconnect", "a")
    await state.save_run_event("disconnect", event_type="tick")
    feed = EventFeed(store, poll_interval_s=0.01, poll_max_s=0.02)
    stream = feed.follow("disconnect", tenant_id=None, max_duration_s=0.03)
    await stream.start()
    sent = asyncio.Event()

    async def receive() -> Any:
        await sent.wait()
        if disconnect:
            return {"type": "http.disconnect"}
        await asyncio.Event().wait()

    async def send(message: Any) -> None:
        if message["type"] == "http.response.body":
            sent.set()
            await asyncio.Event().wait()

    response = rr._SseResponse(
        rr._sse_frames(stream), close=stream.aclose, deadline=stream._deadline
    )
    try:
        async with asyncio.timeout(2):
            if disconnect:
                await response({"type": "http", "asgi": {"spec_version": "2.0"}}, receive, send)
            else:
                with pytest.raises((TimeoutError, ExceptionGroup)):
                    await response({"type": "http", "asgi": {"spec_version": "2.0"}}, receive, send)
        assert feed.followers == 0
    finally:
        await feed.aclose()
    assert feed.watched_runs == 0
