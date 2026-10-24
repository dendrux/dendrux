"""Dashboard hardening: Host check, CORS, bearer token, bounded pages."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from dendrux.dashboard import api as dashboard_api
from dendrux.dashboard import normalizer
from dendrux.dashboard.api import create_dashboard_api
from dendrux.dashboard.security import (
    TOKEN_ENV,
    is_loopback_host,
    normalize_origins,
    resolve_token,
)
from tests.unit.test_dashboard_api import (
    DashboardMockStore,
    _Event,
    _LLMInteraction,
    _Run,
    _ToolCall,
    _Trace,
)

TOKEN = "s3cret-token-value"


def _client(app: Any, base_url: str = "http://localhost:8001") -> AsyncClient:
    return AsyncClient(
        transport=ASGITransport(app=app, raise_app_exceptions=False), base_url=base_url
    )


def _store() -> DashboardMockStore:
    return DashboardMockStore(_runs=[_Run(id="R1", agent_name="a", status="success")])


class TestHelpers:
    @pytest.mark.parametrize(
        ("host", "expected"),
        [
            ("127.0.0.1", True),
            ("127.1.2.3", True),
            ("::1", True),
            ("[::1]", True),
            ("localhost", True),
            ("app.localhost", True),
            ("0.0.0.0", False),
            ("::", False),
            ("", False),
            ("192.168.1.5", False),
            ("dash.example.com", False),
            ("localhost.evil.example", False),
        ],
    )
    def test_is_loopback_host(self, host: str, expected: bool) -> None:
        assert is_loopback_host(host) is expected

    def test_resolve_token_prefers_explicit_then_env(self) -> None:
        assert resolve_token("a", {TOKEN_ENV: "b"}) == "a"
        assert resolve_token(None, {TOKEN_ENV: "b"}) == "b"
        assert resolve_token(None, {}) is None

    def test_empty_values_mean_not_configured(self) -> None:
        assert resolve_token("", {TOKEN_ENV: ""}) is None
        assert resolve_token("", {TOKEN_ENV: "b"}) == "b"

    def test_normalize_origins_accepts_plain_origins(self) -> None:
        assert normalize_origins(["http://localhost:5173/", "https://a.example"]) == [
            "http://localhost:5173",
            "https://a.example",
        ]

    @pytest.mark.parametrize(
        "bad", ["*", "https://*.example.com", "localhost:5173", "https://a.example/path", ""]
    )
    def test_normalize_origins_rejects_wildcards_and_paths(self, bad: str) -> None:
        with pytest.raises(ValueError, match="allowed origins"):
            normalize_origins([bad])


class TestHostCheck:
    @pytest.mark.parametrize(
        "base_url",
        [
            "http://localhost:8001",
            "http://127.0.0.1:8001",
            "http://[::1]:8001",
            "http://192.168.1.20:8001",
        ],
    )
    async def test_loopback_and_ip_literal_hosts_pass(self, base_url: str) -> None:
        app = create_dashboard_api(_store())  # type: ignore[arg-type]
        async with _client(app, base_url) as c:
            assert (await c.get("/api/health")).status_code == 200

    async def test_foreign_hostname_is_rejected(self) -> None:
        """A rebound DNS name (evil.example -> 127.0.0.1) must not reach the data."""
        app = create_dashboard_api(_store())  # type: ignore[arg-type]
        async with _client(app, "http://evil.example:8001") as c:
            resp = await c.get("/api/runs")
            assert resp.status_code == 403
            assert "R1" not in resp.text

    async def test_host_check_covers_static_and_api(self) -> None:
        app = create_dashboard_api(_store())  # type: ignore[arg-type]
        async with _client(app, "http://evil.example") as c:
            assert (await c.get("/")).status_code == 403
            assert (await c.get("/api/health")).status_code == 403

    async def test_allowed_hosts_extends_the_list(self) -> None:
        app = create_dashboard_api(_store(), allowed_hosts=["dash.example.com"])  # type: ignore[arg-type]
        async with _client(app, "http://dash.example.com") as c:
            assert (await c.get("/api/health")).status_code == 200
        async with _client(app, "http://other.example.com") as c:
            assert (await c.get("/api/health")).status_code == 403

    async def test_star_switches_the_check_off(self) -> None:
        app = create_dashboard_api(_store(), allowed_hosts=["*"])  # type: ignore[arg-type]
        async with _client(app, "http://anything.example") as c:
            assert (await c.get("/api/health")).status_code == 200

    async def test_bind_host_name_is_accepted(self) -> None:
        app = create_dashboard_api(
            _store(),  # type: ignore[arg-type]
            auth_token=TOKEN,
            bind_host="dash.internal",
        )
        async with _client(app, "http://dash.internal:8001") as c:
            resp = await c.get("/api/health", headers={"Authorization": f"Bearer {TOKEN}"})
            assert resp.status_code == 200

    async def test_missing_host_header_is_rejected(self) -> None:
        app = create_dashboard_api(_store())  # type: ignore[arg-type]
        sent: list[dict[str, Any]] = []

        async def receive() -> dict[str, Any]:
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message: dict[str, Any]) -> None:
            sent.append(message)

        scope = {
            "type": "http",
            "method": "GET",
            "path": "/api/health",
            "raw_path": b"/api/health",
            "root_path": "",
            "query_string": b"",
            "headers": [],
            "scheme": "http",
            "http_version": "1.1",
        }
        await app(scope, receive, send)
        assert sent[0]["status"] == 403


class TestCors:
    async def test_no_cors_headers_by_default(self) -> None:
        app = create_dashboard_api(_store())  # type: ignore[arg-type]
        async with _client(app) as c:
            resp = await c.get("/api/runs", headers={"Origin": "https://evil.example"})
            assert "access-control-allow-origin" not in resp.headers

    async def test_preflight_not_answered_by_default(self) -> None:
        app = create_dashboard_api(_store())  # type: ignore[arg-type]
        async with _client(app) as c:
            resp = await c.options(
                "/api/runs",
                headers={
                    "Origin": "https://evil.example",
                    "Access-Control-Request-Method": "GET",
                },
            )
            assert "access-control-allow-origin" not in resp.headers

    async def test_listed_origin_allowed_and_others_not(self) -> None:
        app = create_dashboard_api(
            _store(),  # type: ignore[arg-type]
            allow_origins=["http://localhost:5173"],
        )
        async with _client(app) as c:
            ok = await c.get("/api/runs", headers={"Origin": "http://localhost:5173"})
            assert ok.headers["access-control-allow-origin"] == "http://localhost:5173"
            bad = await c.get("/api/runs", headers={"Origin": "https://evil.example"})
            assert "access-control-allow-origin" not in bad.headers

    async def test_wildcard_origin_is_refused(self) -> None:
        with pytest.raises(ValueError, match="allowed origins"):
            create_dashboard_api(_store(), allow_origins=["*"])  # type: ignore[arg-type]

    async def test_unauthorized_response_still_carries_cors_headers(self) -> None:
        """The browser has to read the 401 to prompt for the token."""
        app = create_dashboard_api(
            _store(),  # type: ignore[arg-type]
            auth_token=TOKEN,
            allow_origins=["http://localhost:5173"],
        )
        async with _client(app) as c:
            resp = await c.get("/api/runs", headers={"Origin": "http://localhost:5173"})
            assert resp.status_code == 401
            assert resp.headers["access-control-allow-origin"] == "http://localhost:5173"


class TestToken:
    async def test_non_ascii_authorization_header_is_401_not_500(self) -> None:
        app = create_dashboard_api(_store(), auth_token=TOKEN)  # type: ignore[arg-type]
        async with _client(app) as c:
            resp = await c.get("/api/runs", headers={"Authorization": b"Bearer s3cr\xe9t"})
            assert resp.status_code == 401

    async def test_utf8_header_bytes_are_401(self) -> None:
        app = create_dashboard_api(_store(), auth_token=TOKEN)  # type: ignore[arg-type]
        async with _client(app) as c:
            resp = await c.get("/api/runs", headers={"Authorization": "Bearer ünïcode".encode()})
            assert resp.status_code == 401

    async def test_correct_and_incorrect_tokens(self) -> None:
        app = create_dashboard_api(_store(), auth_token=TOKEN)  # type: ignore[arg-type]
        async with _client(app) as c:
            good = await c.get("/api/runs", headers={"Authorization": f"Bearer {TOKEN}"})
            assert good.status_code == 200
            for header in (f"Bearer {TOKEN}x", f"bearer {TOKEN}", TOKEN, "Bearer ", ""):
                resp = await c.get("/api/runs", headers={"Authorization": header})
                assert resp.status_code == 401, header

    async def test_401_names_no_secret_and_asks_for_bearer(self) -> None:
        app = create_dashboard_api(_store(), auth_token=TOKEN)  # type: ignore[arg-type]
        async with _client(app) as c:
            resp = await c.get("/api/runs")
            assert resp.headers["www-authenticate"] == "Bearer"
            assert TOKEN not in resp.text

    async def test_comparison_is_constant_time(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import hmac

        calls: list[tuple[bytes, bytes]] = []
        real = hmac.compare_digest

        def spy(a: Any, b: Any) -> bool:
            calls.append((a, b))
            return real(a, b)

        monkeypatch.setattr("dendrux.dashboard.security.hmac.compare_digest", spy)
        app = create_dashboard_api(_store(), auth_token=TOKEN)  # type: ignore[arg-type]
        async with _client(app) as c:
            await c.get("/api/runs", headers={"Authorization": "Bearer nope"})
        assert calls and all(isinstance(a, bytes) and isinstance(b, bytes) for a, b in calls)

    async def test_every_api_path_needs_the_token(self) -> None:
        app = create_dashboard_api(_store(), auth_token=TOKEN)  # type: ignore[arg-type]
        async with _client(app) as c:
            for path in (
                "/api/health",
                "/api/runs",
                "/api/runs/R1",
                "/api/runs/R1/events",
                "/api/runs/R1/traces",
                "/api/runs/R1/tool-calls",
                "/api/runs/R1/llm-calls",
                "/api/nope",
            ):
                assert (await c.get(path)).status_code == 401, path

    async def test_token_still_enforced_when_mounted_under_a_prefix(self) -> None:
        """Path matching uses the route path, so a mount cannot bypass the check."""
        dash = create_dashboard_api(_store(), auth_token=TOKEN)  # type: ignore[arg-type]
        parent = FastAPI()
        parent.mount("/dash", dash)
        async with _client(parent) as c:
            assert (await c.get("/dash/api/runs")).status_code == 401
            ok = await c.get("/dash/api/runs", headers={"Authorization": f"Bearer {TOKEN}"})
            assert ok.status_code == 200

    async def test_interactive_docs_and_schema_are_not_served(self) -> None:
        app = create_dashboard_api(_store())  # type: ignore[arg-type]
        async with _client(app) as c:
            for path in ("/docs", "/redoc", "/openapi.json"):
                assert (await c.get(path)).status_code == 404, path

    def test_non_loopback_bind_without_token_is_refused(self) -> None:
        for host in ("0.0.0.0", "192.168.1.5", "dash.example.com", "::"):
            with pytest.raises(ValueError, match="token"):
                create_dashboard_api(_store(), bind_host=host)  # type: ignore[arg-type]
            with pytest.raises(ValueError, match="token"):
                create_dashboard_api(_store(), auth_token="", bind_host=host)  # type: ignore[arg-type]

    def test_non_loopback_bind_with_token_is_allowed(self) -> None:
        create_dashboard_api(_store(), auth_token=TOKEN, bind_host="0.0.0.0")  # type: ignore[arg-type]

    async def test_rejections_are_logged_without_the_token(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        app = create_dashboard_api(_store(), auth_token=TOKEN)  # type: ignore[arg-type]
        with caplog.at_level("WARNING", logger="dendrux.dashboard"):
            async with _client(app) as c:
                await c.get("/api/runs", headers={"Authorization": "Bearer guess"})
            async with _client(app, "http://evil.example") as c:
                await c.get("/api/runs")
        text = " ".join(r.getMessage() for r in caplog.records)
        assert "dashboard.auth_rejected" in text
        assert "dashboard.host_rejected" in text
        assert "guess" not in text


class TestTenantScope:
    async def test_no_tenant_param_leaves_the_argument_out(self) -> None:
        store = _store()
        seen: list[dict[str, Any]] = []
        original = store.list_runs

        async def spy(**kwargs: Any) -> Any:
            seen.append(kwargs)
            return await original(**kwargs)

        store.list_runs = spy  # type: ignore[method-assign]
        app = create_dashboard_api(store)  # type: ignore[arg-type]
        async with _client(app) as c:
            assert (await c.get("/api/runs")).status_code == 200
            assert (await c.get("/api/runs?tenant=acme")).status_code == 200
        direct = [k for k in seen if k["limit"] <= 200]  # count_runs' own scan uses 10_000
        assert "tenant_id" not in direct[0]
        assert direct[1]["tenant_id"] == "acme"


class TestBoundedPages:
    @staticmethod
    def _big_store(n: int) -> DashboardMockStore:
        store = DashboardMockStore(_runs=[_Run(id="R1", agent_name="a")])
        store._events["R1"] = [
            _Event(id=f"e{i}", event_type="x", sequence_index=i, iteration_index=0)
            for i in range(n)
        ]
        store._traces["R1"] = [
            _Trace(id=f"t{i}", role="user", content="c", order_index=i) for i in range(n)
        ]
        store._tool_calls["R1"] = [
            _ToolCall(id=f"c{i}", tool_call_id=f"tc{i}", tool_name="t") for i in range(n)
        ]
        store._llm_interactions["R1"] = [
            _LLMInteraction(id=f"l{i}", iteration_index=i) for i in range(n)
        ]
        return store

    async def test_events_page_and_cursor(self) -> None:
        app = create_dashboard_api(self._big_store(25))  # type: ignore[arg-type]
        async with _client(app) as c:
            first = (await c.get("/api/runs/R1/events?limit=10")).json()
            assert [e["sequence_index"] for e in first["events"]] == list(range(10))
            assert first["has_more"] is True
            assert first["next_after"] == 9
            nxt = (await c.get(f"/api/runs/R1/events?limit=10&after={first['next_after']}")).json()
            assert nxt["events"][0]["sequence_index"] == 10
            last = (await c.get("/api/runs/R1/events?limit=10&after=19")).json()
            assert len(last["events"]) == 5
            assert last["has_more"] is False
            assert last["next_after"] is None

    async def test_events_default_and_max_limit(self) -> None:
        n = dashboard_api.DRILLDOWN_MAX_LIMIT + 50
        app = create_dashboard_api(self._big_store(n))  # type: ignore[arg-type]
        async with _client(app) as c:
            default = (await c.get("/api/runs/R1/events")).json()
            assert len(default["events"]) == dashboard_api.DRILLDOWN_DEFAULT_LIMIT
            huge = (await c.get("/api/runs/R1/events?limit=1000000")).json()
            assert len(huge["events"]) == dashboard_api.DRILLDOWN_MAX_LIMIT
            assert huge["has_more"] is True
            floor = (await c.get("/api/runs/R1/events?limit=-4")).json()
            assert len(floor["events"]) == 1

    async def test_events_unknown_run_is_404(self) -> None:
        app = create_dashboard_api(self._big_store(1))  # type: ignore[arg-type]
        async with _client(app) as c:
            assert (await c.get("/api/runs/nope/events")).status_code == 404
            assert (await c.get("/api/runs/nope/events?after=3")).status_code == 404

    @pytest.mark.parametrize(
        ("path", "key"),
        [
            ("traces", "traces"),
            ("tool-calls", "tool_calls"),
            ("llm-calls", "llm_calls"),
        ],
    )
    async def test_offset_pages(self, path: str, key: str) -> None:
        app = create_dashboard_api(self._big_store(12))  # type: ignore[arg-type]
        async with _client(app) as c:
            page = (await c.get(f"/api/runs/R1/{path}?limit=5&offset=5")).json()
            assert len(page[key]) == 5
            assert page["has_more"] is True
            assert page["next_offset"] == 10
            tail = (await c.get(f"/api/runs/R1/{path}?limit=5&offset=10")).json()
            assert len(tail[key]) == 2
            assert tail["has_more"] is False
            assert tail["next_offset"] is None

    async def test_llm_calls_never_exceed_their_max(self) -> None:
        n = dashboard_api.LLM_CALLS_MAX_LIMIT + 20
        app = create_dashboard_api(self._big_store(n))  # type: ignore[arg-type]
        async with _client(app) as c:
            resp = (await c.get("/api/runs/R1/llm-calls?limit=999999")).json()
            assert len(resp["llm_calls"]) == dashboard_api.LLM_CALLS_MAX_LIMIT
            assert resp["has_more"] is True

    async def test_runs_list_clamps_limit_and_offset(self) -> None:
        store = DashboardMockStore(_runs=[_Run(id=f"R{i}", agent_name="a") for i in range(5)])
        app = create_dashboard_api(store)  # type: ignore[arg-type]
        async with _client(app) as c:
            body = (await c.get("/api/runs?limit=999999&offset=99999999")).json()
            assert body["runs"] == []
            body = (await c.get("/api/runs?limit=0&offset=-5")).json()
            assert len(body["runs"]) == 1

    async def test_timeline_flags_truncation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(normalizer, "MAX_TIMELINE_ROWS", 7)
        app = create_dashboard_api(self._big_store(20))  # type: ignore[arg-type]
        async with _client(app) as c:
            body = (await c.get("/api/runs/R1")).json()
        assert body["truncated"] is True

    async def test_timeline_at_the_cap_is_not_truncated(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(normalizer, "MAX_TIMELINE_ROWS", 7)
        app = create_dashboard_api(self._big_store(7))  # type: ignore[arg-type]
        async with _client(app) as c:
            body = (await c.get("/api/runs/R1")).json()
        assert body["truncated"] is False

    async def test_timeline_reads_more_than_one_store_page(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(normalizer, "_STORE_PAGE", 4)
        monkeypatch.setattr(normalizer, "MAX_TIMELINE_ROWS", 50)
        store = self._big_store(10)
        app = create_dashboard_api(store)  # type: ignore[arg-type]
        async with _client(app) as c:
            body = (await c.get("/api/runs/R1")).json()
        assert body["truncated"] is False
        assert len(body["messages_by_iteration"]["0"]) == 10
