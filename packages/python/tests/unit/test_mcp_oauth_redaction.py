"""Value-free errors, events, representations, logs, and stored runs."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import httpx2
import pytest

from dendrux.mcp import (
    MCPOAuth,
    MCPOAuthClient,
    MCPOAuthMemoryStore,
    MCPRuntime,
    MCPRuntimeEvent,
    MCPSource,
)
from dendrux.mcp._credentials import (
    MCPAuthKey,
    MCPCredentialProvider,
    MCPRefreshingCredentialProvider,
)
from dendrux.mcp._errors import (
    MCPAuthorizationRequired,
    MCPConnectionError,
    MCPCredentialError,
    MCPError,
    MCPOAuthError,
    MCPOAuthFlowError,
    MCPOAuthRefreshError,
)
from dendrux.mcp._observability import (
    MCPConnectionEvent,
    MCPOAuthRefreshed,
    MCPOAuthRefreshFailed,
    MCPOAuthRequired,
    MCPOAuthRevoked,
)
from tests.unit.mcp_oauth_harness import (
    RS_SENTINEL,
    ASConfig,
    OAuthHarness,
    RawHTTPServer,
    RawRequest,
    raw_response,
    scan_chain,
)

if TYPE_CHECKING:
    import asyncio

SENTINEL = "SENTINEL-redaction-4be1"


def test_runtime_module_keeps_reexporting_the_credential_protocols() -> None:
    from dendrux.mcp import _runtime

    assert _runtime.MCPCredentialProvider is MCPCredentialProvider
    assert _runtime.MCPRefreshingCredentialProvider is MCPRefreshingCredentialProvider


@pytest.mark.parametrize(
    ("connection_key", "tenant_key"),
    [("", None), ("  ", None), ("a\nb", None), ("ok", ""), ("ok", "t\x00"), (3, None)],
)
def test_auth_key_validation(connection_key: object, tenant_key: object) -> None:
    with pytest.raises(ValueError):
        MCPAuthKey(connection_key, tenant_key=tenant_key)  # type: ignore[arg-type]


def test_auth_key_equality_is_by_value() -> None:
    assert MCPAuthKey("c", "t") == MCPAuthKey("c", tenant_key="t")
    assert MCPAuthKey("c") != MCPAuthKey("c", "t")


def test_authorization_required_is_value_free() -> None:
    key = MCPAuthKey(SENTINEL, tenant_key=SENTINEL)
    error = MCPAuthorizationRequired(
        key,
        "insufficient_scope",
        scopes=(SENTINEL,),
        resource_metadata_url=f"https://rs.example/{SENTINEL}",
        status_code=403,
    )
    assert isinstance(error, MCPConnectionError)
    assert not isinstance(error, MCPCredentialError)
    assert SENTINEL not in str(error)
    assert SENTINEL not in repr(error)
    assert error.reason == "insufficient_scope"
    assert error.scopes == (SENTINEL,)
    assert error.key is key
    assert error.status_code == 403


def test_flow_error_carries_reason_and_status_only() -> None:
    error = MCPOAuthFlowError("token_exchange_failed", status_code=502)
    assert isinstance(error, MCPOAuthError) and isinstance(error, MCPError)
    assert error.reason == "token_exchange_failed"
    assert error.status_code == 502
    assert "token_exchange_failed" in str(error)


def test_refresh_error_is_a_transient_credential_error() -> None:
    error = MCPOAuthRefreshError("MCP OAuth refresh is held by another worker.")
    assert isinstance(error, MCPOAuthError)
    assert isinstance(error, MCPCredentialError)
    assert isinstance(error, MCPConnectionError)


def _identity() -> dict[str, str]:
    return {"tenant_key": SENTINEL, "connection_key": SENTINEL, "source_name": "demo"}


@pytest.mark.parametrize(
    ("event", "expected"),
    [
        (
            MCPOAuthRequired(**_identity(), reason="expired", scope_count=2),
            "MCPOAuthRequired(reason='expired', scope_count=2)",
        ),
        (MCPOAuthRefreshed(**_identity(), expires_in=60.0), "MCPOAuthRefreshed(expires_in=60.0)"),
        (
            MCPOAuthRefreshFailed(**_identity(), outcome="uncertain", cause=SENTINEL),
            "MCPOAuthRefreshFailed(outcome='uncertain')",
        ),
        (MCPOAuthRevoked(**_identity(), remote="confirmed"), "MCPOAuthRevoked(remote='confirmed')"),
    ],
)
def test_oauth_event_reprs_show_outcomes_and_counts_only(
    event: MCPConnectionEvent, expected: str
) -> None:
    assert isinstance(event, MCPConnectionEvent)
    assert repr(event) == expected
    assert event.identity == (SENTINEL, SENTINEL)


# --------------------------------------------------------------------------
# Sentinel scans over real flows (acceptance 6)
# --------------------------------------------------------------------------

CALLBACK = "http://127.0.0.1:9/callback"
KEY = MCPAuthKey("conn", tenant_key="user")
HOSTILE = "SENTINEL-hostile-0b7d"


class Collected:
    """Everything an application or operator could observe."""

    def __init__(self) -> None:
        self.events: list[MCPRuntimeEvent] = []
        self.errors: list[BaseException] = []
        self.texts: list[str] = []

    def on_event(self, event: MCPRuntimeEvent) -> None:
        self.events.append(event)

    async def capture(self, awaitable: Any) -> Any:
        try:
            return await awaitable
        except Exception as exc:
            self.errors.append(exc)
            return None

    def rendered(self, caplog: pytest.LogCaptureFixture) -> str:
        parts = [repr(event) for event in self.events]
        parts.extend(scan_chain(error) for error in self.errors)
        parts.extend(str(getattr(error, "transport_detail", "") or "") for error in self.errors)
        parts.extend(
            record.getMessage() + " " + str(record.exc_text or "")
            for record in caplog.records
            if _scanned(record)
        )
        parts.extend(self.texts)
        return "\n".join(parts)


# Third-party loggers whose DEBUG output is documented as out of scope.
_THIRD_PARTY_DEBUG = ("httpcore2", "httpx2", "mcp")


def _scanned(record: logging.LogRecord) -> bool:
    """Every Dendrux record at any level; everything else at INFO and above,
    except the documented third-party DEBUG loggers."""
    if record.name == "dendrux" or record.name.startswith("dendrux."):
        return True
    if record.levelno >= logging.INFO:
        return True
    return not any(
        record.name == name or record.name.startswith(f"{name}.") for name in _THIRD_PARTY_DEBUG
    )


def _assert_detached(errors: list[BaseException]) -> None:
    """Every Dendrux-translated error carries no implicit or explicit chain."""
    for error in errors:
        if isinstance(error, (MCPAuthorizationRequired, MCPOAuthError, MCPCredentialError)):
            assert error.__cause__ is None, type(error).__name__
            assert error.__context__ is None, type(error).__name__


def _secrets_of(store: MCPOAuthMemoryStore) -> list[str]:
    needles: list[str] = []
    for record in store._backend.rows.values():
        credential = record.credential
        if credential is None:
            continue
        needles.append(credential.access_token)
        if credential.refresh_token:
            needles.append(credential.refresh_token)
        if credential.client.client_secret:
            needles.append(credential.client.client_secret)
    for flow in store._backend.flows.values():
        needles.extend([flow.state, flow.code_verifier])
    return needles


async def test_sentinel_scan_across_sign_in_refresh_runtime_and_failures(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    caplog.set_level(logging.DEBUG)
    seen_secrets: list[str] = []
    collected = Collected()
    config = ASConfig(echo_sentinel=HOSTILE)
    async with OAuthHarness(config=config) as harness:
        harness.prm_override = {
            "resource_name": HOSTILE,
            "resource_documentation": f"https://d.example/{HOSTILE}",
        }
        harness.challenge_override = (
            f'Bearer error="invalid_token", error_description="{HOSTILE}", '
            f'resource_metadata="http://127.0.0.1:{harness.rs_port}'
            f'/.well-known/oauth-protected-resource/mcp?x={HOSTILE}"'
        )
        store = MCPOAuthMemoryStore()
        oauth = MCPOAuth(
            store=store,
            client=MCPOAuthClient.dynamic(client_name="t", redirect_uris=[CALLBACK]),
        )
        src = MCPSource.http("demo", harness.rs_url)

        sends: list[str] = []
        original_send = httpx2.AsyncClient.send

        async def tracking_send(
            self: httpx2.AsyncClient, request: httpx2.Request, **kw: Any
        ) -> Any:
            sends.append(request.url.path)
            return await original_send(self, request, **kw)

        monkeypatch.setattr(httpx2.AsyncClient, "send", tracking_send)

        url = await oauth.begin_authorization(KEY, src, redirect_uri=CALLBACK)
        seen_secrets.extend(_secrets_of(store))
        callback = await harness.browser(url)
        seen_secrets.append(callback["code"] or "")
        await oauth.complete_authorization(
            key=KEY, state=callback["state"], code=callback["code"], iss=callback["iss"]
        )
        seen_secrets.extend(_secrets_of(store))
        collected.texts.append(repr(await oauth.status(KEY)))
        collected.texts.append(repr(await store.load(KEY)))

        # Failed refresh echoing the refresh token in error_description.
        harness.config.token_status_override = 500
        harness.config.token_body_override = json.dumps(
            {"error": "server_error", "error_description": seen_secrets[-1]}
        ).encode()
        credentials = oauth.credentials(KEY, src)
        oauth._clock = lambda: time.time() + 3700
        await collected.capture(credentials.get_auth())
        harness.config.token_status_override = None
        assert sends == [], "OAuth requests must bypass httpx2.AsyncClient"

        # Fresh sign-in, then the managed runtime with its observer.
        oauth._clock = time.time
        url = await oauth.begin_authorization(KEY, src, redirect_uri=CALLBACK)
        seen_secrets.extend(_secrets_of(store))
        callback = await harness.browser(url)
        seen_secrets.append(callback["code"] or "")
        await oauth.complete_authorization(
            key=KEY, state=callback["state"], code=callback["code"], iss=callback["iss"]
        )
        seen_secrets.extend(_secrets_of(store))
        async with MCPRuntime(observer=collected) as runtime:
            connection = runtime.bind(
                connection_key="conn", tenant_key="user", source=src, oauth=oauth
            )
            await connection.discover()
            harness.reject_all_tokens = True
            await collected.capture(
                runtime.refresh_authorization(connection_key="conn", tenant_key="user")
            )
            await collected.capture(connection.discover())
            harness.reject_all_tokens = False
            seen_secrets.extend(_secrets_of(store))
            await collected.capture(
                runtime.revoke_authorization(connection_key="conn", tenant_key="user")
            )
        await collected.capture(oauth.status(MCPAuthKey("other")))
        await collected.capture(
            oauth.complete_authorization(key=KEY, state=seen_secrets[0], code=HOSTILE)
        )
        await oauth.aclose()

    rendered = collected.rendered(caplog)
    _assert_detached(collected.errors)
    assert collected.errors, "the scan must cover failures"
    assert collected.events, "the scan must cover events"
    for needle in {s for s in seen_secrets if s}:
        assert needle not in rendered
    assert HOSTILE not in "\n".join(scan_chain(e) for e in collected.errors)
    assert HOSTILE not in "\n".join(repr(e) for e in collected.events)


async def test_hostile_reason_phrase_on_the_token_endpoint_is_never_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    echoed: list[str] = []

    async def token(request: RawRequest, writer: asyncio.StreamWriter) -> None:
        form = request.form
        secret = form.get("refresh_token") or form.get("code") or ""
        echoed.append(secret)
        if form.get("grant_type") == "refresh_token":
            writer.write(raw_response(503, b"{}", reason=f"Unavailable {secret} {HOSTILE}"))
            return
        body = json.dumps(
            {"access_token": "at-raw-" + secret[:8], "refresh_token": "rt-raw-1", "expires_in": 60}
        ).encode()
        writer.write(raw_response(200, body, reason=f"OK {secret} {HOSTILE}"))

    async with RawHTTPServer(token) as raw, OAuthHarness() as harness:
        harness.as_metadata_override = {"token_endpoint": raw.url + "/token"}
        store = MCPOAuthMemoryStore()
        oauth = MCPOAuth(
            store=store, client=MCPOAuthClient.dynamic(client_name="t", redirect_uris=[CALLBACK])
        )
        src = MCPSource.http("demo", harness.rs_url)
        url = await oauth.begin_authorization(KEY, src, redirect_uri=CALLBACK)
        callback = await harness.browser(url)
        await oauth.complete_authorization(
            key=KEY, state=callback["state"], code=callback["code"], iss=callback["iss"]
        )
        oauth._clock = lambda: time.time() + 3600
        error: BaseException | None = None
        try:
            await oauth.credentials(KEY, src).get_auth()
        except Exception as exc:
            error = exc
        await oauth.aclose()
    assert error is not None and echoed and all(echoed)
    logged = "\n".join(record.getMessage() for record in caplog.records if _scanned(record))
    _assert_detached([error])
    for needle in [*echoed, HOSTILE]:
        assert needle not in logged
        assert needle not in scan_chain(error)


# --------------------------------------------------------------------------
# Application surface
# --------------------------------------------------------------------------


async def _signed_in(harness: OAuthHarness) -> tuple[MCPOAuth, MCPOAuthMemoryStore]:
    store = MCPOAuthMemoryStore()
    oauth = MCPOAuth(
        store=store, client=MCPOAuthClient.dynamic(client_name="t", redirect_uris=[CALLBACK])
    )
    url = await oauth.begin_authorization(
        KEY, MCPSource.http("demo", harness.rs_url), redirect_uri=CALLBACK
    )
    callback = await harness.browser(url)
    await oauth.complete_authorization(
        key=KEY, state=callback["state"], code=callback["code"], iss=callback["iss"]
    )
    return oauth, store


@pytest.mark.parametrize("failure_mode", ["strict", "best_effort"])
async def test_agent_failure_modes_when_not_signed_in(failure_mode: str) -> None:
    from dendrux import Agent
    from dendrux.llm.mock import MockLLM
    from dendrux.types import LLMResponse

    async with OAuthHarness() as harness:
        oauth = MCPOAuth(
            store=MCPOAuthMemoryStore(),
            client=MCPOAuthClient.dynamic(client_name="t", redirect_uris=[CALLBACK]),
        )
        async with MCPRuntime() as runtime:
            connection = runtime.bind(
                connection_key="conn",
                tenant_key="user",
                source=MCPSource.http("demo", harness.rs_url, failure_mode=failure_mode),  # type: ignore[arg-type]
                oauth=oauth,
            )
            agent = Agent(
                provider=MockLLM([LLMResponse(text="done")]),
                prompt="p",
                tool_sources=[connection.tools()],
            )
            try:
                if failure_mode == "strict":
                    with pytest.raises(MCPAuthorizationRequired) as caught:
                        await agent.get_tool_lookups()
                    assert caught.value.reason == "not_signed_in"
                else:
                    result = await agent.run("hi")
                    skipped = result.meta["mcp_skipped_sources"]
                    assert skipped[0]["error_type"] == "MCPAuthorizationRequired"
                    assert skipped[0]["error"] == "MCP authorization required (not_signed_in)."
            finally:
                await agent.close()
        await oauth.aclose()


async def test_tool_error_and_run_store_stay_value_free(tmp_path: Any) -> None:
    from sqlalchemy import MetaData, select
    from sqlalchemy.ext.asyncio import create_async_engine

    from dendrux import Agent
    from dendrux.db.models import Base
    from dendrux.llm.mock import MockLLM
    from dendrux.runtime.state import SQLAlchemyStateStore
    from dendrux.types import LLMResponse, ToolCall

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'oauth.db'}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    async with OAuthHarness() as harness:
        oauth, store = await _signed_in(harness)
        secrets_before = _secrets_of(store)
        async with MCPRuntime() as runtime:
            connection = runtime.bind(
                connection_key="conn",
                tenant_key="user",
                source=MCPSource.http("demo", harness.rs_url),
                oauth=oauth,
            )
            agent = Agent(
                provider=MockLLM(
                    [
                        LLMResponse(
                            tool_calls=[ToolCall(name="demo__admin_echo", params={"text": "x"})]
                        ),
                        LLMResponse(
                            tool_calls=[ToolCall(name="demo__echo", params={"text": "ok"})]
                        ),
                        LLMResponse(text="done"),
                    ]
                ),
                prompt="p",
                tool_sources=[connection.tools()],
                state_store=SQLAlchemyStateStore(engine),
            )
            try:
                result = await agent.run("hi")
            finally:
                await agent.close()
        secrets = {s for s in [*secrets_before, *_secrets_of(store)] if s}
        await oauth.aclose()

    assert result.answer == "done"
    metadata = MetaData()
    dump: list[str] = []
    async with engine.connect() as conn:
        await conn.run_sync(metadata.reflect)
        for table in metadata.sorted_tables:
            rows = (await conn.execute(select(table))).fetchall()
            dump.extend(repr(tuple(row)) for row in rows)
    await engine.dispose()
    persisted = "\n".join(dump)
    assert "MCP authorization required (insufficient_scope)." in persisted
    for needle in secrets:
        assert needle not in persisted


async def test_tool_call_authorization_error_has_no_cause_chain() -> None:
    from dendrux.mcp._runtime import _ViewToolSource

    async with OAuthHarness() as harness:
        oauth, store = await _signed_in(harness)
        async with MCPRuntime() as runtime:
            connection = runtime.bind(
                connection_key="conn",
                tenant_key="user",
                source=MCPSource.http("demo", harness.rs_url),
                oauth=oauth,
            )
            view = _ViewToolSource(connection.tools())
            await view._discover()
            executor = view._create_executor("admin_echo")
            with pytest.raises(MCPAuthorizationRequired) as caught:
                await executor(text="x")
            await view.close()
        error = caught.value
        assert error.__cause__ is None
        assert error.__context__ is None
        # The resource server's 403 carried RS_SENTINEL in its challenge and
        # body; an implicit context would have dragged that text along.
        assert RS_SENTINEL not in scan_chain(error)
        for needle in _secrets_of(store):
            assert needle not in scan_chain(error)
        await oauth.aclose()


def test_status_and_record_reprs_hide_identifiers_from_servers() -> None:
    from dendrux.mcp import MCPOAuthStatus

    status = MCPOAuthStatus(
        state="retired",
        scope=HOSTILE,
        expires_at=1.0,
        issuer=f"https://{HOSTILE}.example",
        resource=f"https://{HOSTILE}.example/mcp",
    )
    assert HOSTILE not in repr(status)
    assert replace(status, state="active").state == "active"


def test_internal_carriers_have_redacting_reprs() -> None:
    from dendrux.mcp._oauth import _Token
    from dendrux.mcp._oauth_http import OAuthResponse
    from dendrux.mcp._runtime import _Rejection

    token = _Token(SENTINEL, "Bearer", 60, SENTINEL, SENTINEL)
    response = OAuthResponse(200, httpx2.Headers({"x-echo": SENTINEL}), SENTINEL.encode())
    rejection = _Rejection(401, None, f"Bearer {SENTINEL}")
    for value in (token, response, rejection):
        assert SENTINEL not in repr(value)
        assert SENTINEL not in str(value)
    assert repr(response) == "OAuthResponse(status=200, body=<9 bytes>)".replace(
        "9", str(len(SENTINEL))
    )
    assert repr(rejection) == "_Rejection(status=401)"
    assert repr(token) == "_Token(expires_in=60, refresh_token=<set>)"
