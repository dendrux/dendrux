"""Dashboard hardening: Host check, CORS, bearer token, bounded pages."""

from __future__ import annotations

from pathlib import Path
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

    async def test_runs_list_clamps_limit_and_rejects_bad_offsets(self) -> None:
        store = DashboardMockStore(_runs=[_Run(id=f"R{i}", agent_name="a") for i in range(5)])
        app = create_dashboard_api(store)  # type: ignore[arg-type]
        async with _client(app) as c:
            assert len((await c.get("/api/runs?limit=999999")).json()["runs"]) == 5
            assert len((await c.get("/api/runs?limit=0")).json()["runs"]) == 1
            assert (await c.get("/api/runs?offset=-5")).status_code == 422
            over = dashboard_api.RUNS_MAX_OFFSET + 1
            assert (await c.get(f"/api/runs?offset={over}")).status_code == 422

    @pytest.mark.parametrize("path", ["traces", "tool-calls", "llm-calls"])
    async def test_offset_beyond_the_cap_is_an_error_not_a_repeat(self, path: str) -> None:
        app = create_dashboard_api(self._big_store(3))  # type: ignore[arg-type]
        async with _client(app) as c:
            over = dashboard_api.RUNS_MAX_OFFSET + 1
            assert (await c.get(f"/api/runs/R1/{path}?offset={over}")).status_code == 422
            assert (await c.get(f"/api/runs/R1/{path}?offset=-1")).status_code == 422

    async def test_paging_stops_explicitly_at_the_offset_cap(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(dashboard_api, "RUNS_MAX_OFFSET", 30)
        app = create_dashboard_api(self._big_store(100))  # type: ignore[arg-type]
        async with _client(app) as c:
            seen: list[str] = []
            offset: int | None = 0
            pages = 0
            while offset is not None:
                body = (await c.get(f"/api/runs/R1/traces?limit=10&offset={offset}")).json()
                seen += [t["id"] for t in body["traces"]]
                offset = body["next_offset"]
                pages += 1
                assert pages < 10, "pagination did not terminate"
            assert body["has_more"] is True
            assert body["capped"] is True
            assert body["next_offset"] is None
            assert len(seen) == len(set(seen))  # no page repeated

    async def test_end_of_data_is_not_capped(self) -> None:
        app = create_dashboard_api(self._big_store(5))  # type: ignore[arg-type]
        async with _client(app) as c:
            body = (await c.get("/api/runs/R1/traces?limit=10")).json()
        assert body["has_more"] is False
        assert body["capped"] is False

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


class TestRootPathBoundary:
    """The auth decision must agree with Starlette's routing and fail closed."""

    @staticmethod
    async def _status(path: str, root_path: str, *, token: str | None = TOKEN) -> int:
        from dendrux.dashboard.security import DashboardGuard

        reached: list[bool] = []

        async def inner(scope: Any, receive: Any, send: Any) -> None:
            reached.append(True)
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b""})

        sent: list[dict[str, Any]] = []

        async def send(message: dict[str, Any]) -> None:
            sent.append(message)

        async def receive() -> dict[str, Any]:
            return {"type": "http.request", "body": b"", "more_body": False}

        guard = DashboardGuard(inner, allowed_hosts=["*"], token=token)
        scope = {
            "type": "http",
            "method": "GET",
            "path": path,
            "root_path": root_path,
            "headers": [(b"host", b"localhost")],
        }
        await guard(scope, receive, send)
        return int(sent[0]["status"])

    @pytest.mark.parametrize(
        ("path", "root_path"),
        [
            ("/api/runs", ""),
            ("/api/runs", "/"),  # not a boundary strip: Starlette routes /api/runs
            ("/api/runs", "/a"),
            ("/api/runs", "/ap"),
            ("/dash/api/runs", "/dash"),
            ("/dash/api/runs", "/dash/"),  # trailing slash on the root: still an API path
            ("/api/runs", "/dash"),  # unprefixed with a root set: never "already stripped"
            ("/api/runs", "/dashboard"),
            ("/dash/api", "/dash"),
            ("/api", ""),
            ("/api/dashboard/api/runs", "/api/dashboard"),  # mount prefix is not the API
            ("/api/dashboard/api/health", "/api/dashboard"),
        ],
    )
    async def test_api_paths_need_the_token(self, path: str, root_path: str) -> None:
        assert await self._status(path, root_path) == 401, (path, root_path)

    @pytest.mark.parametrize(
        ("path", "root_path"),
        [
            ("/", ""),
            ("/assets/app.js", "/dash"),
            ("/dash/", "/dash"),
            ("/dash", "/dash"),
            ("/api/dashboard/", "/api/dashboard"),  # HTML under an `api` mount prefix
            ("/api/dashboard", "/api/dashboard"),
            ("/api/dashboard/assets/index.js", "/api/dashboard"),
            ("/api/dashboard/index.html", "/api/dashboard"),
        ],
    )
    async def test_static_paths_stay_open(self, path: str, root_path: str) -> None:
        assert await self._status(path, root_path) == 200, (path, root_path)

    @pytest.mark.parametrize(
        ("path", "root_path"),
        [
            ("/api/runs", "/api"),  # the router sees /runs
            ("/dashboard/api/runs", "/dash"),  # no boundary: the router sees it unchanged
        ],
    )
    async def test_paths_the_router_cannot_serve_as_api_are_left_to_it(
        self, path: str, root_path: str
    ) -> None:
        """The guard agrees with Starlette; the router then serves no data there."""
        assert await self._status(path, root_path) == 200
        dash = create_dashboard_api(_store(), auth_token=TOKEN)  # type: ignore[arg-type]
        transport = ASGITransport(app=dash, root_path=root_path, raise_app_exceptions=False)
        async with AsyncClient(transport=transport, base_url="http://localhost") as c:
            resp = await c.get(path)
        assert resp.status_code in {401, 404}
        assert "R1" not in resp.text

    async def test_valid_token_passes_for_every_shape(self) -> None:
        from dendrux.dashboard.security import DashboardGuard

        results = []
        for path, root in (("/api/runs", ""), ("/dash/api/runs", "/dash"), ("/api/runs", "/a")):
            sent: list[dict[str, Any]] = []

            async def inner(scope: Any, receive: Any, send: Any) -> None:
                await send({"type": "http.response.start", "status": 200, "headers": []})

            async def send(message: dict[str, Any], sent: list[dict[str, Any]] = sent) -> None:
                sent.append(message)

            async def receive() -> dict[str, Any]:
                return {"type": "http.request", "body": b""}

            guard = DashboardGuard(inner, allowed_hosts=["*"], token=TOKEN)
            await guard(
                {
                    "type": "http",
                    "method": "GET",
                    "path": path,
                    "root_path": root,
                    "headers": [(b"host", b"x"), (b"authorization", f"Bearer {TOKEN}".encode())],
                },
                receive,
                send,
            )
            results.append(sent[0]["status"])
        assert results == [200, 200, 200]

    async def test_dashboard_mounted_under_an_api_prefix_loads_and_stays_protected(self) -> None:
        dash = create_dashboard_api(_store(), auth_token=TOKEN)  # type: ignore[arg-type]
        parent = FastAPI()
        parent.mount("/api/dashboard", dash)
        asset = next((Path(dashboard_api.__file__).parent / "static" / "assets").glob("index-*.js"))
        auth = {"Authorization": f"Bearer {TOKEN}"}
        async with _client(parent) as c:
            html = await c.get("/api/dashboard/")
            assert html.status_code == 200
            assert "<html" in html.text.lower()
            js = await c.get(f"/api/dashboard/assets/{asset.name}")
            assert js.status_code == 200
            assert len(js.content) > 1000
            # The API under the same prefix is still locked.
            for path in ("/api/dashboard/api/runs", "/api/dashboard/api/health"):
                assert (await c.get(path)).status_code == 401, path
                assert (
                    await c.get(path, headers={"Authorization": "Bearer nope"})
                ).status_code == 401
                assert (await c.get(path, headers=auth)).status_code == 200, path
            assert (await c.get("/api/dashboard/api/runs/R1")).status_code == 401

    async def test_the_api_routes_enforce_the_token_without_the_middleware(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The router dependency is authoritative: a bypassed guard still cannot serve data."""
        from dendrux.dashboard.security import DashboardGuard

        async def passthrough(self: Any, scope: Any, receive: Any, send: Any) -> None:
            await self._app(scope, receive, send)

        monkeypatch.setattr(DashboardGuard, "__call__", passthrough)
        dash = create_dashboard_api(_store(), auth_token=TOKEN)  # type: ignore[arg-type]
        async with _client(dash) as c:
            for path in ("/api/runs", "/api/runs/R1", "/api/health", "/api/runs/R1/events"):
                assert (await c.get(path)).status_code == 401, path
                bad = await c.get(path, headers={"Authorization": b"Bearer s3cr\xe9t"})
                assert bad.status_code == 401, path
                ok = await c.get(path, headers={"Authorization": f"Bearer {TOKEN}"})
                assert ok.status_code == 200, path

    def test_route_path_matches_starlette(self) -> None:
        from starlette._utils import get_route_path

        from dendrux.dashboard.security import DashboardGuard

        cases = [
            ("/api/runs", ""),
            ("/api/runs", "/"),
            ("/api/runs", "/a"),
            ("/dash/api/runs", "/dash"),
            ("/dash/api/runs", "/dash/"),
            ("/dash", "/dash"),
            ("/dashboard/x", "/dash"),
            ("/x", "/dash"),
        ]
        for path, root in cases:
            scope = {"type": "http", "path": path, "root_path": root}
            assert DashboardGuard._route_path(scope) == get_route_path(scope), (path, root)

    async def test_partial_prefix_root_path_does_not_bypass_a_real_app(self) -> None:
        """Embedded under FastAPI(root_path="/a"), the dashboard still demands the token."""
        dash = create_dashboard_api(_store(), auth_token=TOKEN)  # type: ignore[arg-type]
        for root in ("/a", "/ap", "/"):
            parent = FastAPI(root_path=root)
            parent.mount("/", dash)
            async with _client(parent) as c:
                assert (await c.get("/api/runs")).status_code == 401, root
                ok = await c.get("/api/runs", headers={"Authorization": f"Bearer {TOKEN}"})
                assert ok.status_code == 200, root


class TestDelegationBound:
    """Delegation is built from bounded reads; no test lowers a threshold."""

    @staticmethod
    def _store(runs: list[_Run]) -> DashboardMockStore:
        store = DashboardMockStore(_runs=runs)

        async def forbidden(run_id: str) -> Any:
            raise AssertionError("the unbounded subtree walk must never run")

        store.get_delegation_info = forbidden  # type: ignore[method-assign]
        return store

    async def _block(self, store: DashboardMockStore, run_id: str) -> dict[str, Any]:
        app = create_dashboard_api(store)  # type: ignore[arg-type]
        async with _client(app) as c:
            resp = await c.get(f"/api/runs/{run_id}")
        assert resp.status_code == 200
        return resp.json()["delegation"]

    async def test_small_tree_is_complete_and_not_truncated(self) -> None:
        runs = [_Run(id="P", agent_name="p", total_input_tokens=10, total_output_tokens=1)]
        runs += [
            _Run(
                id=f"C{i}",
                agent_name="c",
                parent_run_id="P",
                total_input_tokens=5,
                total_output_tokens=2,
                total_cost_usd=0.5,
            )
            for i in range(2)
        ]
        block = await self._block(self._store(runs), "P")
        assert block["truncated"] is False
        assert len(block["children"]) == 2
        ss = block["subtree_summary"]
        assert ss["descendant_count"] == 2
        assert ss["subtree_input_tokens"] == 20
        assert ss["subtree_cost_usd"] == 1.0
        assert ss["unknown_cost_count"] == 1  # the parent has no cost

    async def test_wide_second_level_is_flagged_without_walking_it(self) -> None:
        """One child that itself has 2,000 children: the 10,001-row shape in miniature."""
        runs = [_Run(id="P", agent_name="p"), _Run(id="C", agent_name="c", parent_run_id="P")]
        runs += [_Run(id=f"G{i}", agent_name="g", parent_run_id="C") for i in range(2000)]
        store = self._store(runs)
        block = await self._block(store, "P")
        assert block["truncated"] is True
        assert [c["run_id"] for c in block["children"]] == ["C"]
        # The grandchildren are never loaded, so the summary does not claim them.
        assert block["subtree_summary"]["descendant_count"] == 1

    async def test_more_children_than_the_cap_are_cut_and_flagged(self) -> None:
        n = dashboard_api.MAX_DELEGATION_CHILDREN + 100
        runs = [_Run(id="P", agent_name="p")]
        runs += [_Run(id=f"C{i}", agent_name="c", parent_run_id="P") for i in range(n)]
        block = await self._block(self._store(runs), "P")
        assert block["truncated"] is True
        assert len(block["children"]) == dashboard_api.MAX_DELEGATION_CHILDREN
        assert block["subtree_summary"]["direct_child_count"] == n

    async def test_children_beyond_the_probe_budget_are_flagged(self) -> None:
        n = dashboard_api.MAX_DELEGATION_PROBES + 1
        runs = [_Run(id="P", agent_name="p")]
        runs += [_Run(id=f"C{i}", agent_name="c", parent_run_id="P") for i in range(n)]
        block = await self._block(self._store(runs), "P")
        assert len(block["children"]) == n
        assert block["truncated"] is True  # not every child could be checked for descendants

    async def test_deep_ancestor_chain_is_capped_and_flagged(self) -> None:
        depth = dashboard_api.MAX_DELEGATION_ANCESTRY + 15
        runs = [
            _Run(id=f"N{i}", agent_name="n", parent_run_id=f"N{i + 1}", delegation_level=depth - i)
            for i in range(depth)
        ]
        runs.append(_Run(id=f"N{depth}", agent_name="root"))
        block = await self._block(self._store(runs), "N0")
        assert len(block["ancestry"]) == dashboard_api.MAX_DELEGATION_ANCESTRY
        assert block["ancestry_complete"] is False
        assert block["truncated"] is True
        assert block["parent"]["run_id"] == "N1"
        assert block["parent"]["resolved"] is True

    async def test_short_chain_is_complete(self) -> None:
        runs = [
            _Run(id="A", agent_name="a"),
            _Run(id="B", agent_name="b", parent_run_id="A"),
            _Run(id="C", agent_name="c", parent_run_id="B"),
        ]
        block = await self._block(self._store(runs), "C")
        assert [a["run_id"] for a in block["ancestry"]] == ["A", "B"]
        assert block["ancestry_complete"] is True
        assert block["truncated"] is False

    async def test_cycle_and_missing_parent_are_reported_incomplete(self) -> None:
        cyc = [
            _Run(id="A", agent_name="a", parent_run_id="B"),
            _Run(id="B", agent_name="b", parent_run_id="A"),
        ]
        block = await self._block(self._store(cyc), "A")
        assert block["ancestry_complete"] is False
        orphan = [_Run(id="X", agent_name="x", parent_run_id="GONE")]
        block = await self._block(self._store(orphan), "X")
        assert block["ancestry_complete"] is False
        assert block["parent"]["resolved"] is False


class TestShippedFrontendShowsTruncation:
    """The committed build must carry the notices for every bounded read."""

    def test_built_bundle_contains_the_notices(self) -> None:
        static = Path(dashboard_api.__file__).parent / "static" / "assets"
        bundle = "".join(p.read_text(errors="ignore") for p in static.glob("index-*.js"))
        for text in (
            "more events than the dashboard reads",
            "delegation tree is very large",
            "more LLM calls than the dashboard loads",
            "most recent of",
        ):
            assert text in bundle, text
