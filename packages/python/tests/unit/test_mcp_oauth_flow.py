"""Sign-in, discovery, registration, callbacks, ordering, status, and sign-out."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import replace
from typing import Any
from urllib.parse import parse_qs, urlsplit

import pytest

pytest.importorskip("uvicorn")

from dendrux.mcp import MCPSource
from dendrux.mcp._credentials import MCPAuthKey
from dendrux.mcp._errors import MCPOAuthFlowError
from dendrux.mcp._oauth import (
    MCPOAuth,
    MCPOAuthClaim,
    MCPOAuthClient,
    MCPOAuthMemoryStore,
    MCPOAuthStore,
)
from tests.unit.mcp_oauth_harness import (
    ASConfig,
    OAuthHarness,
    RawHTTPServer,
    RawRequest,
    basic_header,
    raw_response,
)

CALLBACK = "http://127.0.0.1:9/callback"
KEY = MCPAuthKey("github-1", tenant_key="user-a")
OTHER_KEY = MCPAuthKey("github-1", tenant_key="user-b")


def dynamic(**kwargs: Any) -> MCPOAuthClient:
    return MCPOAuthClient.dynamic(client_name="dendrux-test", redirect_uris=[CALLBACK], **kwargs)


def make_oauth(store: MCPOAuthStore | None = None, **kwargs: Any) -> MCPOAuth:
    kwargs.setdefault("client", dynamic())
    return MCPOAuth(store=store or MCPOAuthMemoryStore(), **kwargs)


def source(harness: OAuthHarness, url: str | None = None) -> MCPSource:
    return MCPSource.http("demo", url or harness.rs_url)


async def sign_in(
    oauth: MCPOAuth,
    harness: OAuthHarness,
    key: MCPAuthKey = KEY,
    *,
    src: MCPSource | None = None,
    **begin: Any,
) -> dict[str, Any]:
    url = await oauth.begin_authorization(
        key, src or source(harness), redirect_uri=CALLBACK, **begin
    )
    callback = await harness.browser(url)
    return {"url": url, **callback}


async def complete(oauth: MCPOAuth, callback: dict[str, Any], key: MCPAuthKey = KEY) -> MCPAuthKey:
    return await oauth.complete_authorization(
        key=key, state=callback["state"], code=callback["code"], iss=callback["iss"]
    )


async def reason_of(awaitable: Any) -> str:
    with pytest.raises(MCPOAuthFlowError) as caught:
        await awaitable
    return caught.value.reason


# -- acceptance 2 and the basic round trip -----------------------------------


async def test_begin_on_one_instance_completes_on_another() -> None:
    async with OAuthHarness() as harness:
        backend = MCPOAuthMemoryStore()
        first = make_oauth(backend.client())
        second = make_oauth(backend.client())
        callback = await sign_in(first, harness)
        assert await complete(second, callback) == KEY
        status = await first.status(KEY)
        assert status.state == "active"
        assert status.resource == harness.rs_url
        assert status.issuer == harness.as_url
        assert status.scope == "mcp"
        query = parse_qs(urlsplit(callback["url"]).query)
        assert query["resource"] == [harness.rs_url]
        assert query["code_challenge_method"] == ["S256"]
        assert len(query["state"][0]) >= 43
        token_form = harness.provider.token_request_headers  # type: ignore[union-attr]
        assert token_form
        await first.aclose()
        await second.aclose()


async def test_authorization_url_keeps_an_existing_endpoint_query() -> None:
    async with OAuthHarness() as harness:
        harness.as_metadata_override = {"authorization_endpoint": f"{harness.as_url}/authorize?x=1"}
        oauth = make_oauth()
        url = await oauth.begin_authorization(KEY, source(harness), redirect_uri=CALLBACK)
        assert url.startswith(f"{harness.as_url}/authorize?x=1&response_type=code")
        await oauth.aclose()


# -- principal binding and state -------------------------------------------


async def test_flow_completed_with_another_key_is_consumed_without_a_token_request() -> None:
    async with OAuthHarness() as harness:
        oauth = make_oauth()
        callback = await sign_in(oauth, harness)
        assert await reason_of(complete(oauth, callback, OTHER_KEY)) == "key_mismatch"
        assert await reason_of(complete(oauth, callback)) == "unknown_state"
        assert harness.counters["token"] == 0
        await oauth.aclose()


async def test_unknown_and_expired_state() -> None:
    async with OAuthHarness() as harness:
        oauth = make_oauth(flow_ttl=5.0)
        assert (
            await reason_of(oauth.complete_authorization(key=KEY, state="nope", code="c"))
            == "unknown_state"
        )
        callback = await sign_in(oauth, harness)
        real_now = oauth._clock()
        oauth._clock = lambda: real_now + 10
        assert await reason_of(complete(oauth, callback)) == "expired_state"
        assert harness.counters["token"] == 0
        await oauth.aclose()


@pytest.mark.parametrize(
    ("error", "code", "expected"),
    [
        ("access_denied", None, "access_denied"),
        ("server_error", None, "invalid_callback"),
        ("access_denied", "code", "access_denied"),
        (None, None, "invalid_callback"),
        (None, "bad\ncode", "invalid_callback"),
    ],
)
async def test_error_callbacks(error: str | None, code: str | None, expected: str) -> None:
    async with OAuthHarness() as harness:
        oauth = make_oauth()
        callback = await sign_in(oauth, harness)
        reason = await reason_of(
            oauth.complete_authorization(
                key=KEY, state=callback["state"], code=code, iss=callback["iss"], error=error
            )
        )
        assert reason == expected
        assert harness.counters["token"] == 0
        await oauth.aclose()


# -- iss (RFC 9207) ------------------------------------------------------------


async def test_iss_mismatch_rejected_even_without_advertisement() -> None:
    async with OAuthHarness() as harness:
        oauth = make_oauth()
        callback = await sign_in(oauth, harness)
        callback["iss"] = "http://127.0.0.1:1"
        assert await reason_of(complete(oauth, callback)) == "issuer_mismatch"
        await oauth.aclose()


async def test_iss_trailing_slash_is_a_mismatch() -> None:
    async with OAuthHarness() as harness:
        oauth = make_oauth()
        callback = await sign_in(oauth, harness)
        callback["iss"] = harness.as_url + "/"
        assert await reason_of(complete(oauth, callback)) == "issuer_mismatch"
        await oauth.aclose()


async def test_iss_required_when_advertised() -> None:
    async with OAuthHarness(config=ASConfig(advertise_iss=True, include_iss=False)) as harness:
        oauth = make_oauth()
        callback = await sign_in(oauth, harness)
        assert callback["iss"] is None
        assert await reason_of(complete(oauth, callback)) == "issuer_mismatch"
        await oauth.aclose()


async def test_iss_without_advertisement_may_be_absent() -> None:
    async with OAuthHarness(config=ASConfig(include_iss=False)) as harness:
        oauth = make_oauth()
        callback = await sign_in(oauth, harness)
        await complete(oauth, callback)
        assert (await oauth.status(KEY)).state == "active"
        await oauth.aclose()


async def test_iss_is_checked_on_error_callbacks_too() -> None:
    async with OAuthHarness() as harness:
        oauth = make_oauth()
        callback = await sign_in(oauth, harness)
        reason = await reason_of(
            oauth.complete_authorization(
                key=KEY, state=callback["state"], code=None, iss="http://evil.invalid", error="x"
            )
        )
        assert reason == "issuer_mismatch"
        await oauth.aclose()


# -- ordering against revocation ----------------------------------------------


async def test_begin_while_absent_then_revoke_supersedes_the_flow() -> None:
    async with OAuthHarness() as harness:
        oauth = make_oauth()
        callback = await sign_in(oauth, harness)
        revocation = await oauth.revoke(KEY)
        assert (revocation.local, revocation.remote) == ("already_signed_out", "skipped")
        assert await reason_of(complete(oauth, callback)) == "superseded"
        assert harness.counters["token"] == 0
        assert (await oauth.status(KEY)).state == "signed_out"
        await oauth.aclose()


async def test_second_flow_superseded_after_first_completes_and_revoke() -> None:
    async with OAuthHarness() as harness:
        oauth = make_oauth()
        first = await sign_in(oauth, harness)
        second = await sign_in(oauth, harness)
        await complete(oauth, first)
        assert (await oauth.revoke(KEY)).local == "signed_out"
        assert await reason_of(complete(oauth, second)) == "superseded"
        assert harness.counters["code_exchange"] == 1
        await oauth.aclose()


async def test_flow_begun_against_a_tombstone_is_superseded_by_another_revoke() -> None:
    async with OAuthHarness() as harness:
        oauth = make_oauth()
        await oauth.revoke(KEY)
        before = await oauth.store.load(KEY)
        callback = await sign_in(oauth, harness)
        revocation = await oauth.revoke(KEY)
        assert revocation.local == "already_signed_out"
        after = await oauth.store.load(KEY)
        assert before is not None and after is not None
        assert after.version.generation != before.version.generation
        assert await reason_of(complete(oauth, callback)) == "superseded"
        assert harness.counters["token"] == 0
        await oauth.aclose()


async def test_reverse_completion_supersedes_the_older_flow() -> None:
    async with OAuthHarness() as harness:
        oauth = make_oauth()
        first = await sign_in(oauth, harness)
        second = await sign_in(oauth, harness)
        await complete(oauth, second)
        assert await reason_of(complete(oauth, first)) == "superseded"
        await oauth.aclose()


class _BumpOnFirstPublish(MCPOAuthMemoryStore):
    """Simulates a refresh landing between the completion's load and save."""

    bumped = False

    async def save(self, key, record, *, expected):  # type: ignore[no-untyped-def]
        current = self._backend.rows.get(key)
        if not self.bumped and current is not None and record.version.revision == 0:
            self.bumped = True
            self._backend.rows[key] = replace(
                current,
                version=replace(current.version, revision=current.version.revision + 1),
            )
        return await super().save(key, record, expected=expected)


async def test_refresh_contention_during_completion_still_publishes() -> None:
    async with OAuthHarness() as harness:
        store = _BumpOnFirstPublish()
        oauth = make_oauth(store)
        await complete(oauth, await sign_in(oauth, harness))
        store.bumped = False
        before = await store.load(KEY)
        callback = await sign_in(oauth, harness)
        await complete(oauth, callback)
        after = await store.load(KEY)
        assert store.bumped
        assert before is not None and after is not None
        assert after.version.generation != before.version.generation
        assert after.status == "active"
        await oauth.aclose()


class _SlowStore(MCPOAuthMemoryStore):
    delay = 0.0

    async def load(self, key):  # type: ignore[no-untyped-def]
        await asyncio.sleep(self.delay)
        return await super().load(key)

    async def save(self, key, record, *, expected):  # type: ignore[no-untyped-def]
        await asyncio.sleep(self.delay)
        return await super().save(key, record, expected=expected)


@pytest.mark.parametrize("delay", [0.0, 0.01, 0.03])
async def test_completion_racing_revoke_has_a_single_winner(delay: float) -> None:
    async with OAuthHarness() as harness:
        store = _SlowStore()
        oauth = make_oauth(store)
        callback = await sign_in(oauth, harness)
        store.delay = delay
        outcomes = await asyncio.gather(
            complete(oauth, callback), oauth.revoke(KEY), return_exceptions=True
        )
        row = await store.load(KEY)
        assert row is not None, "absence is never fabricated"
        assert row.status == "signed_out"
        completed, revocation = outcomes
        if isinstance(completed, MCPOAuthFlowError):
            assert completed.reason == "superseded"
            assert revocation.local == "already_signed_out"  # type: ignore[union-attr]
        else:
            assert completed == KEY
            assert revocation.local == "signed_out"  # type: ignore[union-attr]
        await oauth.aclose()


# -- discovery -------------------------------------------------------------------


@pytest.mark.parametrize(
    "resource_suffix",
    ["/mcp2", "/mcp/", "/mcp?tenant=A", "/other"],
)
async def test_prm_identity_must_match_exactly(resource_suffix: str) -> None:
    async with OAuthHarness() as harness:
        harness.prm_override = {"resource": f"http://127.0.0.1:{harness.rs_port}{resource_suffix}"}
        oauth = make_oauth()
        assert await reason_of(sign_in(oauth, harness)) == "discovery_failed"
        await oauth.aclose()


async def test_prm_normalized_looking_resource_is_a_raw_mismatch() -> None:
    async with OAuthHarness() as harness:
        harness.prm_override = {"resource": f"HTTP://127.0.0.1:{harness.rs_port}/mcp"}
        oauth = make_oauth()
        assert await reason_of(sign_in(oauth, harness)) == "discovery_failed"
        harness.prm_override = {"resource": f"http://127.0.0.1:{harness.rs_port}/x/../mcp"}
        assert await reason_of(sign_in(oauth, harness)) == "discovery_failed"
        await oauth.aclose()


async def test_source_query_must_match_the_prm_resource() -> None:
    async with OAuthHarness() as harness:
        oauth = make_oauth()
        src = source(harness, harness.rs_url + "?tenant=B")
        assert await reason_of(sign_in(oauth, harness, src=src)) == "discovery_failed"
        harness.prm_override = {"resource": harness.rs_url + "?tenant=B"}
        await complete(oauth, await sign_in(oauth, harness, src=src))
        assert (await oauth.status(KEY)).resource == harness.rs_url + "?tenant=B"
        await oauth.aclose()


@pytest.mark.parametrize("url", ["https://rs.example/a/../b", "https://rs.example/a/%2e%2e/b"])
async def test_dot_segment_sources_are_rejected_before_io(url: str) -> None:
    oauth = make_oauth()
    reason = await reason_of(
        oauth.begin_authorization(KEY, MCPSource.http("d", url), redirect_uri=CALLBACK)
    )
    assert reason == "invalid_argument"
    await oauth.aclose()


async def test_header_only_metadata_from_the_probe_and_its_challenged_scopes() -> None:
    async with OAuthHarness() as harness:
        harness.prm_path = "/meta/prm"
        harness.challenge_override = (
            f'Bearer error="invalid_token", resource_metadata="http://127.0.0.1:{harness.rs_port}'
            '/meta/prm", scope="mcp admin"'
        )
        oauth = make_oauth()
        callback = await sign_in(oauth, harness)
        query = parse_qs(urlsplit(callback["url"]).query)
        assert query["scope"] == ["mcp admin"]
        assert harness.provider.registration_requests[-1]["scope"] == "mcp admin"  # type: ignore[union-attr]
        await complete(oauth, callback)
        await oauth.aclose()


async def test_explicit_scopes_win_over_challenge_and_prm() -> None:
    async with OAuthHarness() as harness:
        oauth = make_oauth()
        callback = await sign_in(oauth, harness, scopes=["mcp", "admin"])
        assert parse_qs(urlsplit(callback["url"]).query)["scope"] == ["mcp admin"]
        assert (
            await reason_of(
                oauth.begin_authorization(KEY, source(harness), redirect_uri=CALLBACK, scopes="mcp")  # type: ignore[arg-type]
            )
            == "invalid_argument"
        )
        await oauth.aclose()


@pytest.mark.parametrize("status", [302, 429, 503])
async def test_transient_prm_failures_fail_discovery(status: int) -> None:
    async with OAuthHarness() as harness:
        harness.prm_status = status
        oauth = make_oauth(allow_legacy_discovery=True)
        with pytest.raises(MCPOAuthFlowError) as caught:
            await sign_in(oauth, harness)
        assert caught.value.reason == "discovery_failed"
        assert caught.value.status_code == status
        await oauth.aclose()


async def test_prm_404_without_legacy_flag_fails() -> None:
    async with OAuthHarness() as harness:
        harness.prm_status = 404
        oauth = make_oauth()
        assert await reason_of(sign_in(oauth, harness)) == "discovery_failed"
        await oauth.aclose()


def _legacy_server(as_status: int | None, metadata: bool) -> RawHTTPServer:
    server = RawHTTPServer()

    async def handle(request: RawRequest, writer: asyncio.StreamWriter) -> None:
        if request.path.startswith("/.well-known/oauth-protected-resource"):
            writer.write(raw_response(404, b"{}"))
        elif request.path == "/.well-known/oauth-authorization-server":
            if as_status is not None:
                writer.write(raw_response(as_status, b"{}"))
            elif metadata:
                body = {
                    "issuer": server.url,
                    "authorization_endpoint": f"{server.url}/oauth/authorize",
                    "token_endpoint": f"{server.url}/oauth/token",
                    "code_challenge_methods_supported": ["S256"],
                    "response_types_supported": ["code"],
                }
                writer.write(raw_response(200, json.dumps(body).encode()))
            else:
                writer.write(raw_response(404, b"{}"))
        else:
            writer.write(raw_response(404, b"{}"))

    server.handler = handle
    return server


async def test_legacy_discovery_with_flag() -> None:
    async with _legacy_server(None, metadata=True) as server:
        client = MCPOAuthClient.preregistered("legacy", issuer=server.url)
        oauth = make_oauth(client=client, allow_legacy_discovery=True)
        url = await oauth.begin_authorization(
            KEY, MCPSource.http("d", server.url + "/mcp"), redirect_uri=CALLBACK
        )
        assert url.startswith(server.url + "/oauth/authorize?")
        await oauth.aclose()


async def test_legacy_discovery_synthesizes_endpoints_when_metadata_is_absent() -> None:
    async with _legacy_server(None, metadata=False) as server:
        client = MCPOAuthClient.preregistered("legacy", issuer=server.url)
        oauth = make_oauth(client=client, allow_legacy_discovery=True)
        url = await oauth.begin_authorization(
            KEY, MCPSource.http("d", server.url + "/mcp"), redirect_uri=CALLBACK
        )
        assert url.startswith(server.url + "/authorize?")
        await oauth.aclose()


@pytest.mark.parametrize("status", [503, 302, 429])
async def test_legacy_as_metadata_transient_fails(status: int) -> None:
    async with _legacy_server(status, metadata=False) as server:
        client = MCPOAuthClient.preregistered("legacy", issuer=server.url)
        oauth = make_oauth(client=client, allow_legacy_discovery=True)
        with pytest.raises(MCPOAuthFlowError) as caught:
            await oauth.begin_authorization(
                KEY, MCPSource.http("d", server.url + "/mcp"), redirect_uri=CALLBACK
            )
        assert caught.value.reason == "discovery_failed"
        await oauth.aclose()


async def test_oidc_metadata_path() -> None:
    async with OAuthHarness() as harness:
        harness.as_metadata_path = "/.well-known/openid-configuration"
        oauth = make_oauth()
        await complete(oauth, await sign_in(oauth, harness))
        assert harness.counters["as_metadata"] == 2
        await oauth.aclose()


async def test_missing_s256_is_pkce_unsupported() -> None:
    async with OAuthHarness() as harness:
        harness.as_metadata_override = {"code_challenge_methods_supported": ["plain"]}
        oauth = make_oauth()
        assert await reason_of(sign_in(oauth, harness)) == "pkce_unsupported"
        harness.as_metadata_override = {"code_challenge_methods_supported": None}
        assert await reason_of(sign_in(oauth, harness)) == "pkce_unsupported"
        await oauth.aclose()


async def test_issuer_with_query_is_rejected() -> None:
    async with OAuthHarness() as harness:
        harness.prm_override = {"authorization_servers": [harness.as_url + "?tenant=x"]}
        oauth = make_oauth()
        assert await reason_of(sign_in(oauth, harness)) == "discovery_failed"
        await oauth.aclose()


@pytest.mark.parametrize("issuer_suffix", ["/", ":80"])
async def test_raw_issuer_must_equal_the_prm_entry(issuer_suffix: str) -> None:
    async with OAuthHarness() as harness:
        harness.as_metadata_override = {"issuer": harness.as_url + issuer_suffix}
        oauth = make_oauth()
        assert await reason_of(sign_in(oauth, harness)) == "discovery_failed"
        await oauth.aclose()


async def test_uppercase_issuer_is_a_raw_mismatch() -> None:
    async with OAuthHarness() as harness:
        harness.as_metadata_override = {"issuer": harness.as_url.replace("http", "HTTP")}
        oauth = make_oauth()
        assert await reason_of(sign_in(oauth, harness)) == "discovery_failed"
        await oauth.aclose()


async def test_authorization_server_argument_must_be_a_raw_entry() -> None:
    async with OAuthHarness() as harness:
        oauth = make_oauth()
        assert (
            await reason_of(sign_in(oauth, harness, authorization_server=harness.as_url + "/"))
            == "discovery_failed"
        )
        await complete(oauth, await sign_in(oauth, harness, authorization_server=harness.as_url))
        await oauth.aclose()


async def test_insecure_urls_are_rejected_before_io() -> None:
    oauth = make_oauth()
    assert (
        await reason_of(
            oauth.begin_authorization(
                KEY,
                MCPSource.http("d", "https://rs.example/mcp"),
                redirect_uri="http://app.example/cb",
            )
        )
        == "insecure_url"
    )
    assert (
        await reason_of(
            oauth.begin_authorization(
                KEY, MCPSource.http("d", "http://rs.example/mcp"), redirect_uri=CALLBACK
            )
        )
        == "insecure_url"
    )
    await oauth.aclose()


async def test_insecure_endpoint_in_metadata_is_rejected() -> None:
    async with OAuthHarness() as harness:
        harness.as_metadata_override = {"token_endpoint": "http://as.example/token"}
        oauth = make_oauth()
        assert await reason_of(sign_in(oauth, harness)) == "insecure_url"
        await oauth.aclose()


async def test_destination_policy_denies_the_authorization_server() -> None:
    async with OAuthHarness() as harness:
        as_port = harness.as_port

        async def policy(destination) -> bool:
            return destination.port != as_port

        oauth = make_oauth(destination_policy=policy)
        assert await reason_of(sign_in(oauth, harness)) == "destination_denied"
        assert harness.counters["as_metadata"] == 0
        await oauth.aclose()


# -- registration ----------------------------------------------------------------


async def test_dynamic_registration_request_shape() -> None:
    async with OAuthHarness() as harness:
        oauth = make_oauth(client=dynamic(client_uri="https://app.example", contacts=["a@b.c"]))
        await sign_in(oauth, harness)
        request = harness.provider.registration_requests[-1]  # type: ignore[union-attr]
        assert request["redirect_uris"][0] == CALLBACK
        assert request["application_type"] == "web"
        assert request["scope"] == "mcp"
        assert request["response_types"] == ["code"]
        assert request["grant_types"] == ["authorization_code", "refresh_token"]
        assert request["token_endpoint_auth_method"] == "none"
        assert request["contacts"] == ["a@b.c"]
        await oauth.aclose()


@pytest.mark.parametrize(
    ("override", "drop"),
    [
        ({"token_endpoint_auth_method": "private_key_jwt"}, ()),
        ({"token_endpoint_auth_method": "tls_client_auth"}, ()),
        ({"token_endpoint_auth_method": "client_secret_post", "client_secret": None}, ()),
        ({"redirect_uris": ["http://127.0.0.1:9/other"]}, ()),
        ({}, ("redirect_uris",)),
        ({"client_secret": "s", "client_secret_expires_at": 1}, ("token_endpoint_auth_method",)),
    ],
)
async def test_unusable_registrations_fail(override: dict[str, Any], drop: tuple[str, ...]) -> None:
    async with OAuthHarness() as harness:
        harness.registration_override = override
        harness.registration_drop = drop
        oauth = make_oauth()
        assert await reason_of(sign_in(oauth, harness)) == "registration_failed"
        await oauth.aclose()


async def test_omitted_method_means_basic_and_zero_expiry_never_expires() -> None:
    async with OAuthHarness() as harness:
        harness.registration_override = {
            "client_secret": "dcr-secret",
            "client_secret_expires_at": 0,
        }
        harness.registration_drop = ("token_endpoint_auth_method",)
        oauth = make_oauth()
        callback = await sign_in(oauth, harness)
        await complete(oauth, callback)
        row = await oauth.store.load(KEY)
        assert row is not None and row.credential is not None
        client = row.credential.client
        assert client.token_endpoint_auth_method == "client_secret_basic"
        assert client.client_secret_expires_at is None
        assert harness.provider.token_request_headers[-1]["authorization"].startswith("Basic ")  # type: ignore[union-attr]
        await oauth.aclose()


async def test_registration_rejected_by_server() -> None:
    async with OAuthHarness() as harness:
        oauth = make_oauth(client=dynamic(token_endpoint_auth_method="client_secret_basic"))
        harness.registration_override = {"client_secret": None}
        assert await reason_of(sign_in(oauth, harness)) == "registration_failed"
        await oauth.aclose()


async def test_no_registration_endpoint_and_no_cimd_fails() -> None:
    async with OAuthHarness() as harness:
        harness.as_metadata_override = {"registration_endpoint": None}
        oauth = make_oauth()
        assert await reason_of(sign_in(oauth, harness)) == "registration_failed"
        await oauth.aclose()


async def test_cimd_client_id_when_supported() -> None:
    async with OAuthHarness() as harness:
        harness.as_metadata_override = {"client_id_metadata_document_supported": True}
        cimd = "https://app.example/oauth/client.json"
        oauth = make_oauth(client=dynamic(cimd_url=cimd))
        url = await oauth.begin_authorization(KEY, source(harness), redirect_uri=CALLBACK)
        assert parse_qs(urlsplit(url).query)["client_id"] == [cimd]
        assert harness.counters["register"] == 0
        await oauth.aclose()


async def test_cimd_falls_back_to_registration_when_unsupported() -> None:
    async with OAuthHarness() as harness:
        oauth = make_oauth(client=dynamic(cimd_url="https://app.example/oauth/client.json"))
        await sign_in(oauth, harness)
        assert harness.counters["register"] == 1
        await oauth.aclose()


@pytest.mark.parametrize(
    "cimd",
    [
        "http://app.example/client.json",
        "https://app.example/",
        "https://app.example",
        "https://app.example/c.json?x=1",
        "https://app.example/c.json#f",
        "https://u@app.example/c.json",
    ],
)
def test_cimd_url_shape_rules(cimd: str) -> None:
    with pytest.raises(ValueError):
        dynamic(cimd_url=cimd)


def test_client_constructor_validation() -> None:
    with pytest.raises(ValueError):
        MCPOAuthClient.preregistered(
            "id", "secret", issuer="https://as.example", token_endpoint_auth_method="none"
        )
    with pytest.raises(ValueError):
        MCPOAuthClient.preregistered(
            "id", issuer="https://as.example", token_endpoint_auth_method="client_secret_basic"
        )
    with pytest.raises(ValueError):
        MCPOAuthClient.preregistered("id", issuer="https://as.example/?q=1")
    with pytest.raises(ValueError):
        MCPOAuthClient.preregistered("", issuer="https://as.example")
    assert (
        MCPOAuthClient.preregistered(
            "id", "s", issuer="https://as.example"
        ).token_endpoint_auth_method
        == "client_secret_post"
    )
    assert (
        MCPOAuthClient.preregistered("id", issuer="https://as.example").token_endpoint_auth_method
        == "none"
    )
    with pytest.raises(ValueError):
        MCPOAuthClient.dynamic(client_name="x", redirect_uris=[])
    with pytest.raises(ValueError):
        MCPOAuthClient.dynamic(client_name="x", redirect_uris="https://a.example/cb")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        MCPOAuthClient.dynamic(client_name="x", redirect_uris=["https://a.example/cb#f"])
    with pytest.raises(ValueError):
        MCPOAuthClient.dynamic(client_name="", redirect_uris=["https://a.example/cb"])
    assert "secret" not in repr(
        MCPOAuthClient.preregistered("id", "secret", issuer="https://as.example")
    )


async def test_preregistered_secret_only_for_its_exact_issuer() -> None:
    async with OAuthHarness() as harness:
        client = MCPOAuthClient.preregistered(
            "pre",
            "secret",
            issuer=harness.as_url + "/",
            token_endpoint_auth_method="client_secret_basic",
        )
        oauth = make_oauth(client=client)
        assert await reason_of(sign_in(oauth, harness)) == "registration_failed"
        await oauth.aclose()


async def test_preregistered_basic_secret_reaches_the_cross_origin_token_endpoint() -> None:
    async with OAuthHarness() as harness:
        harness.preregister("pre", "pre-secret", "client_secret_basic", redirect_uris=[CALLBACK])
        client = MCPOAuthClient.preregistered(
            "pre",
            "pre-secret",
            issuer=harness.as_url,
            token_endpoint_auth_method="client_secret_basic",
        )
        oauth = make_oauth(client=client)
        await complete(oauth, await sign_in(oauth, harness))
        assert harness.provider.token_request_headers[-1]["authorization"] == basic_header(
            "pre", "pre-secret"
        )  # type: ignore[union-attr]
        assert harness.counters["register"] == 0
        await oauth.aclose()


async def test_preregistered_post_secret() -> None:
    async with OAuthHarness() as harness:
        harness.preregister("pre", "pre-secret", "client_secret_post", redirect_uris=[CALLBACK])
        client = MCPOAuthClient.preregistered("pre", "pre-secret", issuer=harness.as_url)
        oauth = make_oauth(client=client)
        await complete(oauth, await sign_in(oauth, harness))
        assert "authorization" not in harness.provider.token_request_headers[-1]  # type: ignore[union-attr]
        await oauth.aclose()


async def test_expired_client_secret_at_completion() -> None:
    async with OAuthHarness() as harness:
        harness.registration_override = {
            "client_secret": "s",
            "client_secret_expires_at": int(time.time()) + 100,
            "token_endpoint_auth_method": "client_secret_post",
        }
        oauth = make_oauth()
        callback = await sign_in(oauth, harness)
        oauth._clock = lambda: time.time() + 200
        assert await reason_of(complete(oauth, callback)) == "registration_failed"
        assert harness.counters["token"] == 0
        await oauth.aclose()


# -- token exchange --------------------------------------------------------------


@pytest.mark.parametrize("status", [400, 500])
async def test_token_exchange_failure_carries_status(status: int) -> None:
    async with OAuthHarness() as harness:
        oauth = make_oauth()
        callback = await sign_in(oauth, harness)
        harness.config.token_status_override = status
        with pytest.raises(MCPOAuthFlowError) as caught:
            await complete(oauth, callback)
        assert (caught.value.reason, caught.value.status_code) == ("token_exchange_failed", status)
        await oauth.aclose()


@pytest.mark.parametrize(
    "body",
    [
        b"not json",
        b'{"token_type": "Bearer"}',
        b'{"access_token": "a b", "token_type": "Bearer"}',
        b'{"access_token": "ok", "token_type": "DPoP"}',
        b'{"access_token": "ok", "expires_in": -5}',
        b'{"access_token": "ok", "refresh_token": "r\\nX"}',
    ],
)
async def test_unusable_token_responses(body: bytes) -> None:
    async with OAuthHarness() as harness:
        oauth = make_oauth()
        callback = await sign_in(oauth, harness)
        harness.config.token_status_override = 200
        harness.config.token_body_override = body
        assert await reason_of(complete(oauth, callback)) == "token_exchange_failed"
        await oauth.aclose()


async def test_sign_in_without_refresh_token_publishes_retired() -> None:
    async with OAuthHarness(config=ASConfig(issue_refresh_tokens=False)) as harness:
        oauth = make_oauth()
        await complete(oauth, await sign_in(oauth, harness))
        row = await oauth.store.load(KEY)
        assert row is not None and row.status == "retired" and row.retired_claim_id is None
        assert (await oauth.status(KEY)).state == "retired"
        await oauth.aclose()


async def test_exchange_expiry_counts_from_before_the_request() -> None:
    async with OAuthHarness(config=ASConfig(access_token_lifetime=100)) as harness:
        oauth = make_oauth()
        callback = await sign_in(oauth, harness)
        times = iter([1000.0, 1000.0, 1000.0, 1050.0, 1050.0, 1050.0, 1050.0])
        oauth._clock = lambda: next(times, 1050.0)
        await complete(oauth, callback)
        row = await oauth.store.load(KEY)
        assert row is not None and row.credential is not None
        assert row.credential.expires_at is not None and row.credential.expires_at <= 1100.0
        await oauth.aclose()


# -- status ------------------------------------------------------------------------


async def test_status_states() -> None:
    async with OAuthHarness() as harness:
        oauth = make_oauth()
        assert (await oauth.status(KEY)).state == "signed_out"
        await complete(oauth, await sign_in(oauth, harness))
        assert (await oauth.status(KEY)).state == "active"
        row = await oauth.store.load(KEY)
        assert row is not None and row.credential is not None
        expires_at = row.credential.expires_at
        assert expires_at is not None
        real = oauth._clock
        oauth._clock = lambda: expires_at + 1
        assert (await oauth.status(KEY)).state == "expired"
        oauth._clock = real
        claimed = replace(
            row,
            version=replace(row.version, revision=1),
            refresh_claim=MCPOAuthClaim(id="x", expires_at=real() + 100),
        )
        assert await oauth.store.save(KEY, claimed, expected=row.version)
        assert (await oauth.status(KEY)).state == "refreshing"
        retired = replace(
            claimed,
            version=replace(row.version, revision=2),
            status="retired",
            refresh_claim=None,
            credential=replace(row.credential, refresh_token=None),
        )
        assert await oauth.store.save(KEY, retired, expected=claimed.version)
        assert (await oauth.status(KEY)).state == "retired"
        oauth._clock = lambda: expires_at + 1
        assert (await oauth.status(KEY)).state == "needs_sign_in"
        await oauth.aclose()


# -- revocation --------------------------------------------------------------------


async def test_revocation_confirmed_with_refresh_token_hint() -> None:
    async with OAuthHarness() as harness:
        oauth = make_oauth()
        await complete(oauth, await sign_in(oauth, harness))
        row = await oauth.store.load(KEY)
        assert row is not None and row.credential is not None
        revocation = await oauth.revoke(KEY)
        assert (revocation.local, revocation.remote) == ("signed_out", "confirmed")
        request = harness.provider.revocation_requests[-1]  # type: ignore[union-attr]
        assert request["token"] == row.credential.refresh_token
        assert request["token_type_hint"] == "refresh_token"
        assert request["client_id"] == row.credential.client.client_id
        assert harness.provider.revoked_families  # type: ignore[union-attr]
        await oauth.aclose()


async def test_revocation_with_access_token_when_no_refresh_token() -> None:
    async with OAuthHarness(config=ASConfig(issue_refresh_tokens=False)) as harness:
        oauth = make_oauth()
        await complete(oauth, await sign_in(oauth, harness))
        await oauth.revoke(KEY)
        request = harness.provider.revocation_requests[-1]  # type: ignore[union-attr]
        assert request["token_type_hint"] == "access_token"
        await oauth.aclose()


async def test_revocation_failure_still_signs_out_locally() -> None:
    async with OAuthHarness(config=ASConfig(revocation_status=503)) as harness:
        oauth = make_oauth()
        await complete(oauth, await sign_in(oauth, harness))
        revocation = await oauth.revoke(KEY)
        assert (revocation.local, revocation.remote) == ("signed_out", "failed")
        assert (await oauth.status(KEY)).state == "signed_out"
        await oauth.aclose()


async def test_revocation_unsupported_without_endpoint() -> None:
    async with OAuthHarness() as harness:
        harness.as_metadata_override = {"revocation_endpoint": None}
        oauth = make_oauth()
        await complete(oauth, await sign_in(oauth, harness))
        assert (await oauth.revoke(KEY)).remote == "unsupported"
        await oauth.aclose()


async def test_already_signed_out_revoke_mints_a_new_generation() -> None:
    oauth = make_oauth()
    await oauth.revoke(KEY)
    first = await oauth.store.load(KEY)
    revocation = await oauth.revoke(KEY)
    second = await oauth.store.load(KEY)
    assert (revocation.local, revocation.remote) == ("already_signed_out", "skipped")
    assert first is not None and second is not None
    assert first.version.generation != second.version.generation
    await oauth.aclose()


class _AlwaysLoses(MCPOAuthMemoryStore):
    async def save(self, key, record, *, expected):  # type: ignore[no-untyped-def]
        return False


async def test_revoke_contention_is_reported() -> None:
    oauth = make_oauth(_AlwaysLoses())
    assert await reason_of(oauth.revoke(KEY)) == "contention"
    await oauth.aclose()


async def test_caller_cancelled_during_remote_revoke_keeps_the_tombstone() -> None:
    async with OAuthHarness() as harness:
        oauth = make_oauth()
        await complete(oauth, await sign_in(oauth, harness))
        harness.revocation_latency = 0.3
        task = asyncio.create_task(oauth.revoke(KEY))
        await asyncio.sleep(0.15)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert (await oauth.status(KEY)).state == "signed_out"
        await oauth.aclose()
        assert harness.counters["revoke"] == 1


async def test_aclose_cancelling_a_remote_revoke_keeps_the_tombstone() -> None:
    async with OAuthHarness() as harness:
        oauth = make_oauth(oauth_timeout=0.3, claim_ttl=100)
        await complete(oauth, await sign_in(oauth, harness))
        harness.revocation_latency = 5.0
        task = asyncio.create_task(oauth.revoke(KEY))
        await asyncio.sleep(0.1)
        await oauth.aclose()
        assert (await oauth.store.load(KEY)).status == "signed_out"  # type: ignore[union-attr]
        # The per-request deadline or aclose's own deadline ends the POST;
        # either way the remote outcome is never reported as confirmed.
        try:
            revocation = await task
        except asyncio.CancelledError:
            pass
        else:
            assert revocation.remote == "failed"


# -- lifecycle -----------------------------------------------------------------------


async def test_aclose_rejects_new_operations() -> None:
    oauth = make_oauth()
    await oauth.aclose()
    await oauth.aclose()
    assert await reason_of(oauth.status(KEY)) == "closed"
    assert await reason_of(oauth.revoke(KEY)) == "closed"


async def test_aclose_waits_for_an_in_flight_exchange_and_its_save() -> None:
    async with OAuthHarness() as harness:
        oauth = make_oauth()
        callback = await sign_in(oauth, harness)
        harness.token_latency = 0.3
        harness.token_entered = asyncio.Event()
        task = asyncio.create_task(complete(oauth, callback))
        await harness.token_entered.wait()
        await oauth.aclose()
        assert task.done() and task.result() == KEY
        assert (await oauth.store.load(KEY)).status == "active"  # type: ignore[union-attr]


async def test_caller_cancelled_during_exchange_still_publishes() -> None:
    async with OAuthHarness() as harness:
        oauth = make_oauth()
        callback = await sign_in(oauth, harness)
        harness.token_latency = 0.3
        harness.token_entered = asyncio.Event()
        task = asyncio.create_task(complete(oauth, callback))
        await harness.token_entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await oauth.aclose()
        assert (await oauth.store.load(KEY)).status == "active"  # type: ignore[union-attr]


def test_constructor_validation() -> None:
    store = MCPOAuthMemoryStore()
    with pytest.raises(ValueError):
        MCPOAuth(store=store, client=dynamic(), oauth_timeout=10, claim_ttl=70)
    with pytest.raises(ValueError):
        MCPOAuth(store=store, client=dynamic(), oauth_timeout=10, claim_ttl=60)
    MCPOAuth(store=store, client=dynamic(), oauth_timeout=10, claim_ttl=70.5)
    with pytest.raises(ValueError):
        MCPOAuth(store=object(), client=dynamic())  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        MCPOAuth(store=store, client=object())  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        MCPOAuth(store=store, client=dynamic(), oauth_timeout=0)
    with pytest.raises(ValueError):
        MCPOAuth(store=store, client=dynamic(), refresh_skew=float("nan"))
    with pytest.raises(ValueError):
        MCPOAuth(store=store, client=dynamic(), destination_policy=3)  # type: ignore[arg-type]
    assert repr(MCPOAuth(store=store, client=dynamic())).startswith("MCPOAuth(")


class _BrokenStore(MCPOAuthMemoryStore):
    async def load(self, key):  # type: ignore[no-untyped-def]
        raise RuntimeError("SENTINEL-store-failure https://db.internal/?password=hunter2")


async def test_store_failures_are_translated_and_detached() -> None:
    from dendrux.mcp._errors import MCPOAuthError

    oauth = make_oauth(_BrokenStore())
    with pytest.raises(MCPOAuthError) as caught:
        await oauth.status(KEY)
    assert "hunter2" not in str(caught.value)
    assert caught.value.__cause__ is None and caught.value.__context__ is None
    await oauth.aclose()


@pytest.mark.parametrize(
    "expires_in", ['"\u00b2"', '"\u0661\u0662"', '"12345678901234"', '"1e3"', "1.5", "true"]
)
async def test_malformed_expires_in_is_an_exchange_failure(expires_in: str) -> None:
    async with OAuthHarness() as harness:
        oauth = make_oauth()
        callback = await sign_in(oauth, harness)
        harness.config.token_status_override = 200
        harness.config.token_body_override = (
            '{"access_token": "x", "expires_in": ' + expires_in + "}"
        ).encode()
        with pytest.raises(MCPOAuthFlowError) as caught:
            await complete(oauth, callback)
        assert caught.value.reason == "token_exchange_failed"
        assert caught.value.__cause__ is None and caught.value.__context__ is None
        await oauth.aclose()


def test_token_parser_never_raises() -> None:
    from dendrux.mcp._oauth import _parse_token

    for body in (
        b'{"access_token": "x", "expires_in": "\\u00b2"}',
        b'{"access_token": "x", "expires_in": 1e400}',
        b'{"access_token": "x", "scope": 7}',
        b'{"access_token": "x", "token_type": null}',
        b"[1, 2]",
        b"\xff\xfe",
        b"",
    ):
        assert _parse_token(body) is None
    assert _parse_token(b'{"access_token": "x", "expires_in": "60"}').expires_in == 60  # type: ignore[union-attr]
