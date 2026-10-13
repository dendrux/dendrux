"""Security boundaries at the HTTP request and TCP dial layers."""

from __future__ import annotations

import asyncio
from ipaddress import ip_address
from unittest.mock import AsyncMock

import httpcore2
import httpx2
import pytest

from dendrux.mcp import MCPDestination, MCPDestinationDeniedError, MCPSource
from dendrux.mcp._http import _PolicyBackend, create_http_client


@pytest.mark.parametrize(
    "target",
    [
        "https://other.example/mcp",
        "http://source.example/mcp",
        "https://source.example:444/mcp",
    ],
)
async def test_cross_origin_headers_are_removed_on_every_hop(target):
    requests = []

    def handle(request):
        requests.append(request)
        if len(requests) == 1:
            return httpx2.Response(307, headers={"location": target})
        if len(requests) == 2:
            return httpx2.Response(307, headers={"location": target + "/again"})
        return httpx2.Response(200)

    source = MCPSource.http(
        "test",
        "https://source.example/mcp",
        headers={"X-Custom-Key": "secret", "Cookie": "session=secret"},
        auth=("user", "password"),
    )
    async with create_http_client(source, transport=httpx2.MockTransport(handle)) as client:
        response = await client.post(source.url, json={"hello": "world"})
    assert response.status_code == 200
    assert requests[0].headers["x-custom-key"] == "secret"
    assert "authorization" in requests[0].headers
    for request in requests[1:]:
        assert "x-custom-key" not in request.headers
        assert "cookie" not in request.headers
        assert "authorization" not in request.headers


async def test_same_origin_and_default_port_preserve_credentials():
    requests = []

    def handle(request):
        requests.append(request)
        if len(requests) == 1:
            return httpx2.Response(307, headers={"location": "https://SOURCE.example:443/next"})
        return httpx2.Response(200)

    source = MCPSource.http("test", "https://source.example/mcp", headers={"X-Key": "secret"})
    async with create_http_client(source, transport=httpx2.MockTransport(handle)) as client:
        await client.get(source.url)
    assert len(requests) == 2
    assert all(request.headers["x-key"] == "secret" for request in requests)


async def test_redirects_can_be_disabled():
    handler = AsyncMock(return_value=httpx2.Response(307, headers={"location": "http://127.0.0.1"}))
    source = MCPSource.http("test", "https://source.example", follow_redirects=False)
    async with create_http_client(source, transport=httpx2.MockTransport(handler)) as client:
        response = await client.get(source.url)
    assert response.status_code == 307
    assert handler.await_count == 1


async def test_redirect_limit():
    handler = AsyncMock(return_value=httpx2.Response(307, headers={"location": "/again"}))
    source = MCPSource.http("test", "https://source.example", max_redirects=1)
    async with create_http_client(source, transport=httpx2.MockTransport(handler)) as client:
        with pytest.raises(httpx2.TooManyRedirects):
            await client.get(source.url)
    assert handler.await_count == 2


async def test_auth_generated_custom_header_is_scoped():
    class CustomAuth(httpx2.Auth):
        def auth_flow(self, request):
            request.headers["X-Generated-Key"] = "secret"
            response = yield request
            if response.status_code == 401:
                request = httpx2.Request(
                    "GET", "https://other.example/retry", headers={"X-Generated-Key": "refreshed"}
                )
                yield request

    requests = []

    def handle(request):
        requests.append(request)
        if len(requests) == 1:
            return httpx2.Response(307, headers={"location": "https://other.example/mcp"})
        return httpx2.Response(401 if len(requests) == 2 else 200)

    source = MCPSource.http("test", "https://source.example", auth=CustomAuth())
    async with create_http_client(source, transport=httpx2.MockTransport(handle)) as client:
        await client.get(source.url)
    assert requests[0].headers["x-generated-key"] == "secret"
    assert len(requests) == 3
    # The redirect hop is built by the HTTP client, not the auth flow.
    assert "x-generated-key" not in requests[1].headers
    # A new request the auth flow itself sends is auth-owned and keeps its headers.
    assert requests[2].headers["x-generated-key"] == "refreshed"


class TokenRequestAuth(httpx2.Auth):
    """Auth whose flow makes its own off-origin token request, like an OAuth client."""

    def __init__(self, token_url="https://auth.example/token"):
        self.token_url = token_url

    def auth_flow(self, request):
        response = yield request
        if response.status_code != 401:
            return
        token_request = httpx2.Request(
            "POST",
            self.token_url,
            headers={"Authorization": "Basic Y2xpZW50OnNlY3JldA=="},
            data={"grant_type": "client_credentials"},
        )
        token_response = yield token_request
        if token_response.status_code == 200:
            request.headers["Authorization"] = "Bearer fresh"
            yield request


async def test_auth_owned_off_origin_request_keeps_its_headers():
    requests = []

    def handle(request):
        requests.append(request)
        if request.url.host == "auth.example":
            return httpx2.Response(200, json={"access_token": "fresh"})
        if request.headers.get("authorization") == "Bearer fresh":
            return httpx2.Response(200)
        return httpx2.Response(401)

    source = MCPSource.http(
        "test",
        "https://source.example/mcp",
        headers={"X-Custom-Key": "static"},
        auth=TokenRequestAuth(),
    )
    async with create_http_client(source, transport=httpx2.MockTransport(handle)) as client:
        response = await client.post(source.url, json={"hello": "world"})
    assert response.status_code == 200
    assert [r.url.host for r in requests] == ["source.example", "auth.example", "source.example"]
    assert requests[1].headers["authorization"] == "Basic Y2xpZW50OnNlY3JldA=="
    # Static source headers are never added to a request the auth flow built.
    assert "x-custom-key" not in requests[1].headers
    assert requests[2].headers["authorization"] == "Bearer fresh"


class CopyingAuth(httpx2.Auth):
    """Auth flow that re-targets a copy of the original request to another origin."""

    def auth_flow(self, request):
        response = yield request
        if response.status_code != 401:
            return
        copied = httpx2.Request(
            "POST", "https://auth.example/token", headers=request.headers, content=b"x"
        )
        copied.headers["Authorization"] = "Basic Y2xpZW50OnNlY3JldA=="
        yield copied


async def test_auth_owned_request_keeps_only_headers_its_flow_introduced():
    requests = []

    def handle(request):
        requests.append(request)
        return httpx2.Response(200 if request.url.host == "auth.example" else 401)

    source = MCPSource.http(
        "test",
        "https://source.example/mcp",
        headers={"X-Api-Key": "static-key", "Cookie": "session=static"},
        auth=CopyingAuth(),
    )
    async with create_http_client(source, transport=httpx2.MockTransport(handle)) as client:
        await client.post(source.url, json={})
    assert [r.url.host for r in requests] == ["source.example", "auth.example"]
    assert requests[0].headers["x-api-key"] == "static-key"
    token_request = requests[1]
    assert token_request.headers["authorization"] == "Basic Y2xpZW50OnNlY3JldA=="
    assert "x-api-key" not in token_request.headers
    assert "cookie" not in token_request.headers


async def test_auth_owned_copy_of_an_unchanged_credential_is_stripped():
    """A credential present before the flow ran is not the flow's to forward."""
    requests = []

    class ForwardingAuth(httpx2.Auth):
        def auth_flow(self, request):
            response = yield request
            if response.status_code == 401:
                yield httpx2.Request("GET", "https://auth.example/probe", headers=request.headers)

    def handle(request):
        requests.append(request)
        return httpx2.Response(200 if request.url.host == "auth.example" else 401)

    source = MCPSource.http(
        "test",
        "https://source.example/mcp",
        headers={"Authorization": "Bearer static-token"},
        auth=ForwardingAuth(),
    )
    async with create_http_client(source, transport=httpx2.MockTransport(handle)) as client:
        await client.post(source.url, json={})
    assert requests[0].headers["authorization"] == "Bearer static-token"
    assert "authorization" not in requests[1].headers


async def test_sdk_oauth_client_provider_token_request_keeps_basic_auth_end_to_end():
    """The unmanaged ``MCPSource(auth=OAuthClientProvider)`` path with a cross-origin AS."""
    import base64
    import json
    from urllib.parse import parse_qs, urlsplit

    from mcp.client.auth import OAuthClientProvider
    from mcp.server.mcpserver import MCPServer as SDKServer
    from mcp.shared.auth import (
        AuthorizationCodeResult,
        OAuthClientInformationFull,
        OAuthClientMetadata,
    )

    from dendrux.mcp._client import MCPClientAdapter
    from tests.unit.test_mcp_streamable_http_auth import _serve

    bases = {}
    token_requests = []
    expected_basic = "Basic " + base64.b64encode(b"client:s3cret").decode()

    async def respond(send, status, payload=None, headers=()):
        body = b"" if payload is None else json.dumps(payload).encode()
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [(b"content-type", b"application/json"), *headers],
            }
        )
        await send({"type": "http.response.body", "body": body})

    async def auth_server(scope, receive, send):
        if scope["type"] != "http":
            return
        base = bases["as"]
        if scope["path"] == "/.well-known/oauth-authorization-server":
            await respond(
                send,
                200,
                {
                    "issuer": base,
                    "authorization_endpoint": f"{base}/authorize",
                    "token_endpoint": f"{base}/token",
                    "response_types_supported": ["code"],
                    "code_challenge_methods_supported": ["S256"],
                    "token_endpoint_auth_methods_supported": ["client_secret_basic"],
                },
            )
        elif scope["path"] == "/token":
            authorization = dict(scope["headers"]).get(b"authorization", b"").decode()
            token_requests.append(authorization)
            if authorization != expected_basic:
                await respond(send, 401, {"error": "invalid_client"})
            else:
                await respond(
                    send,
                    200,
                    {"access_token": "at-1", "token_type": "Bearer", "expires_in": 3600},
                )
        else:
            await respond(send, 404)

    mcp_server = SDKServer(name="oauth-rs")

    @mcp_server.tool()
    def echo(text: str) -> str:
        return text

    inner = mcp_server.streamable_http_app()

    async def resource_server(scope, receive, send):
        if scope["type"] != "http":
            await inner(scope, receive, send)
            return
        base = bases["rs"]
        if scope["path"].startswith("/.well-known/oauth-protected-resource"):
            await respond(
                send, 200, {"resource": f"{base}/mcp", "authorization_servers": [bases["as"]]}
            )
            return
        if dict(scope["headers"]).get(b"authorization") != b"Bearer at-1":
            challenge = (
                f'Bearer resource_metadata="{base}/.well-known/oauth-protected-resource/mcp"'
            )
            await respond(send, 401, headers=[(b"www-authenticate", challenge.encode())])
            return
        await inner(scope, receive, send)

    class Storage:
        def __init__(self):
            self.tokens = None
            self.client = OAuthClientInformationFull(
                client_id="client",
                client_secret="s3cret",
                token_endpoint_auth_method="client_secret_basic",
                redirect_uris=["http://localhost/callback"],
            )

        async def get_tokens(self):
            return self.tokens

        async def set_tokens(self, tokens):
            self.tokens = tokens

        async def get_client_info(self):
            return self.client

        async def set_client_info(self, client):
            self.client = client

    authorization_urls = []

    async def redirect_handler(url):
        authorization_urls.append(url)

    async def callback_handler():
        state = parse_qs(urlsplit(authorization_urls[-1]).query)["state"][0]
        return AuthorizationCodeResult(code="code-1", state=state)

    async with _serve(auth_server) as as_url, _serve(resource_server) as rs_url:
        bases["as"] = as_url.removesuffix("/mcp")
        bases["rs"] = rs_url.removesuffix("/mcp")
        provider = OAuthClientProvider(
            server_url=rs_url,
            client_metadata=OAuthClientMetadata(
                redirect_uris=["http://localhost/callback"],
                token_endpoint_auth_method="client_secret_basic",
            ),
            storage=Storage(),
            redirect_handler=redirect_handler,
            callback_handler=callback_handler,
        )
        adapter = MCPClientAdapter(MCPSource.http("demo", rs_url, auth=provider))
        await adapter.connect()
        try:
            tools = await adapter.list_tools()
        finally:
            await adapter.close()
    assert [tool.name for tool in tools] == ["echo"]
    assert token_requests == [expected_basic]


async def test_redirect_of_an_auth_owned_request_still_strips_headers():
    requests = []

    def handle(request):
        requests.append(request)
        if request.url.host == "auth.example":
            return httpx2.Response(307, headers={"location": "https://third.example/token"})
        if request.url.host == "third.example":
            return httpx2.Response(400)
        return httpx2.Response(401)

    source = MCPSource.http("test", "https://source.example/mcp", auth=TokenRequestAuth())
    async with create_http_client(source, transport=httpx2.MockTransport(handle)) as client:
        await client.post(source.url, json={})
    assert [r.url.host for r in requests] == ["source.example", "auth.example", "third.example"]
    assert "authorization" in requests[1].headers
    assert "authorization" not in requests[2].headers


async def test_redirect_of_the_mcp_request_to_another_origin_still_strips_auth():
    requests = []

    def handle(request):
        requests.append(request)
        if request.url.host == "source.example":
            return httpx2.Response(307, headers={"location": "https://other.example/mcp"})
        return httpx2.Response(200)

    class BearerAuth(httpx2.Auth):
        def auth_flow(self, request):
            request.headers["Authorization"] = "Bearer secret"
            yield request

    source = MCPSource.http("test", "https://source.example/mcp", auth=BearerAuth())
    async with create_http_client(source, transport=httpx2.MockTransport(handle)) as client:
        await client.post(source.url, json={})
    assert [r.url.host for r in requests] == ["source.example", "other.example"]
    assert requests[0].headers["authorization"] == "Bearer secret"
    assert "authorization" not in requests[1].headers


async def test_original_request_reyielded_by_auth_is_never_auth_owned():
    """Re-sending the original request after a redirect keeps the redirect's scoping."""
    requests = []

    class RetryAuth(httpx2.Auth):
        def auth_flow(self, request):
            request.headers["Authorization"] = "Bearer one"
            response = yield request
            if response.status_code == 401:
                request.headers["Authorization"] = "Bearer two"
                yield request

    def handle(request):
        requests.append(request)
        if request.url.host == "source.example":
            return httpx2.Response(307, headers={"location": "https://other.example/mcp"})
        return httpx2.Response(401)

    source = MCPSource.http("test", "https://source.example/mcp", auth=RetryAuth())
    async with create_http_client(source, transport=httpx2.MockTransport(handle)) as client:
        await client.post(source.url, json={})
    assert [r.url.host for r in requests] == ["source.example", "other.example"] * 2
    assert all("authorization" not in r.headers for r in requests[1::2])
    assert requests[2].headers["authorization"] == "Bearer two"


async def allow_public(destination):
    return destination.ip.is_global


def backend(monkeypatch, addresses, policy=allow_public):
    result = _PolicyBackend(policy)
    resolver = AsyncMock(side_effect=addresses)
    dial = AsyncMock(return_value=object())
    monkeypatch.setattr(result, "_resolve", resolver)
    monkeypatch.setattr(result._backend, "connect_tcp", dial)
    return result, resolver, dial


async def test_dns_change_between_connections_is_rechecked(monkeypatch):
    net, resolver, dial = backend(monkeypatch, [["8.8.8.8"], ["127.0.0.1"]])
    await net.connect_tcp("source.example", 443)
    with pytest.raises(MCPDestinationDeniedError):
        await net.connect_tcp("source.example", 443)
    assert resolver.await_count == 2
    assert dial.await_count == 1
    assert dial.call_args.kwargs["host"] == "8.8.8.8"


@pytest.mark.parametrize(
    "address", ["127.0.0.1", "10.0.0.1", "169.254.169.254", "::1", "fc00::1", "::ffff:127.0.0.1"]
)
async def test_private_addresses_are_never_dialed(monkeypatch, address):
    net, _, dial = backend(monkeypatch, [[address]])
    with pytest.raises(MCPDestinationDeniedError):
        await net.connect_tcp("source.example", 443)
    dial.assert_not_awaited()


@pytest.mark.parametrize("answer", [False, None, 1, "yes"])
async def test_policy_fails_closed(monkeypatch, answer):
    net, _, dial = backend(monkeypatch, [["8.8.8.8"]], AsyncMock(return_value=answer))
    with pytest.raises(MCPDestinationDeniedError):
        await net.connect_tcp("source.example", 443)
    dial.assert_not_awaited()


async def test_policy_exception_is_sanitized(monkeypatch):
    net, _, dial = backend(
        monkeypatch, [["8.8.8.8"]], AsyncMock(side_effect=RuntimeError("secret"))
    )
    with pytest.raises(MCPDestinationDeniedError) as caught:
        await net.connect_tcp("source.example", 443)
    assert "secret" not in str(caught.value)
    assert caught.value.__context__ is None
    dial.assert_not_awaited()


async def test_mixed_dns_answers_only_dial_approved_ips(monkeypatch):
    net, _, dial = backend(monkeypatch, [["127.0.0.1", "8.8.8.8"]])
    await net.connect_tcp("source.example", 443)
    assert dial.await_count == 1
    assert dial.call_args.kwargs["host"] == "8.8.8.8"


async def test_dns_and_policy_share_connect_deadline(monkeypatch):
    async def slow_policy(destination):
        await asyncio.sleep(1)
        return True

    net, _, dial = backend(monkeypatch, [["8.8.8.8"]], slow_policy)
    with pytest.raises(httpcore2.ConnectTimeout):
        await net.connect_tcp("source.example", 443, timeout=0.01)
    dial.assert_not_awaited()


async def test_redirect_to_private_ip_never_opens_socket(monkeypatch):
    # Real HTTP parsing and redirect handling; only the network boundary is faked.
    calls = []

    async def connect(self, host, port, **kwargs):
        calls.append(host)
        return httpcore2.AsyncMockStream(
            [
                b"HTTP/1.1 307 Temporary Redirect\r\nLocation: http://127.0.0.1/mcp\r\n"
                b"Content-Length: 0\r\n\r\n"
            ]
        )

    monkeypatch.setattr(httpcore2.AnyIOBackend, "connect_tcp", connect)
    monkeypatch.setattr(
        _PolicyBackend, "_resolve", AsyncMock(side_effect=[["8.8.8.8"], ["127.0.0.1"]])
    )
    source = MCPSource.http("test", "http://source.example/mcp", destination_policy=allow_public)
    async with create_http_client(source) as client:
        with pytest.raises(MCPDestinationDeniedError):
            await client.get(source.url)
    assert calls == ["8.8.8.8"]


def test_public_destination_contract():
    destination = MCPDestination("source.example", 443, ip_address("8.8.8.8"))
    assert destination.ip.is_global


@pytest.mark.parametrize(
    "kwargs",
    [
        {"follow_redirects": "yes"},
        {"max_redirects": -1},
        {"max_redirects": True},
        {"destination_policy": "bad"},
    ],
)
def test_source_rejects_invalid_transport_settings(kwargs):
    with pytest.raises(ValueError):
        MCPSource.http("test", "https://source.example", **kwargs)


async def test_real_dns_resolution_and_numeric_literals(monkeypatch):
    import socket

    net = _PolicyBackend(allow_public)
    resolver = AsyncMock(
        return_value=[
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", 443)),
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", 443)),
            (socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("2606:4700:4700::1111", 443, 0, 0)),
        ]
    )
    monkeypatch.setattr(asyncio.get_running_loop(), "getaddrinfo", resolver)
    assert await net._resolve("example.com", 443) == ["8.8.8.8", "2606:4700:4700::1111"]
    assert await net._resolve("8.8.8.8", 443) == ["8.8.8.8"]
    assert await net._resolve("::1", 443) == ["::1"]
    assert resolver.await_count == 1


async def test_failed_candidate_falls_back_under_same_deadline(monkeypatch):
    net, _, dial = backend(monkeypatch, [["8.8.8.8", "1.1.1.1"]])
    dial.side_effect = [httpcore2.ConnectError("unreachable"), object()]
    await net.connect_tcp("source.example", 443, timeout=1)
    assert [call.kwargs["host"] for call in dial.call_args_list] == ["8.8.8.8", "1.1.1.1"]
    assert (
        0
        < dial.call_args_list[1].kwargs["timeout"]
        <= dial.call_args_list[0].kwargs["timeout"]
        <= 1
    )


async def test_all_allowed_candidates_fail(monkeypatch):
    net, _, dial = backend(monkeypatch, [["8.8.8.8"]])
    dial.side_effect = httpcore2.ConnectError("unreachable")
    with pytest.raises(httpcore2.ConnectError):
        await net.connect_tcp("source.example", 443)


async def test_dns_error_and_dns_timeout(monkeypatch):
    net, resolver, dial = backend(monkeypatch, [OSError("DNS failure")])
    with pytest.raises(httpcore2.ConnectError):
        await net.connect_tcp("source.example", 443)

    async def slow(*args):
        await asyncio.sleep(1)
        return ["8.8.8.8"]

    resolver.side_effect = slow
    with pytest.raises(httpcore2.ConnectTimeout):
        await net.connect_tcp("source.example", 443, timeout=0.01)
    dial.assert_not_awaited()


async def test_policy_cancellation_is_not_swallowed(monkeypatch):
    net, _, dial = backend(
        monkeypatch, [["8.8.8.8"]], AsyncMock(side_effect=asyncio.CancelledError)
    )
    with pytest.raises(asyncio.CancelledError):
        await net.connect_tcp("source.example", 443)
    dial.assert_not_awaited()


async def test_empty_dns_response_fails_closed(monkeypatch):
    net, _, dial = backend(monkeypatch, [[]])
    with pytest.raises(MCPDestinationDeniedError):
        await net.connect_tcp("source.example", 443)
    dial.assert_not_awaited()


async def test_environment_proxy_cannot_bypass_policy(monkeypatch):
    seen = []

    async def policy(destination):
        seen.append(destination.host)
        return False

    monkeypatch.setenv("HTTP_PROXY", "http://proxy.example:9999")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example:9999")
    monkeypatch.setenv("ALL_PROXY", "http://proxy.example:9999")
    monkeypatch.setenv("NO_PROXY", "")
    monkeypatch.setattr(_PolicyBackend, "_resolve", AsyncMock(return_value=["8.8.8.8"]))
    dial = AsyncMock()
    monkeypatch.setattr(httpcore2.AnyIOBackend, "connect_tcp", dial)
    source = MCPSource.http("test", "http://source.example", destination_policy=policy)
    async with create_http_client(source) as client:
        with pytest.raises(MCPDestinationDeniedError):
            await client.get(source.url)
    assert seen == ["source.example"]
    dial.assert_not_awaited()


async def test_redirect_userinfo_is_rejected_before_send():
    handler = AsyncMock(
        return_value=httpx2.Response(
            307, headers={"location": "https://user:secret@other.example/mcp"}
        )
    )
    source = MCPSource.http("test", "https://source.example")
    async with create_http_client(source, transport=httpx2.MockTransport(handler)) as client:
        with pytest.raises(MCPDestinationDeniedError):
            await client.get(source.url)
    assert handler.await_count == 1


def test_policy_identity_is_part_of_connection_config():
    from dendrux.mcp._runtime import _same_connection_config

    class Policy:
        def __eq__(self, other):
            return True

        async def __call__(self, destination):
            return True

    first = MCPSource.http("test", "https://source.example", destination_policy=Policy())
    second = MCPSource.http("test", "https://source.example", destination_policy=Policy())
    assert not _same_connection_config(first, second)
    assert _same_connection_config(first, first)


@pytest.mark.parametrize(
    "kwargs",
    [{"follow_redirects": False}, {"max_redirects": 0}, {"destination_policy": allow_public}],
)
def test_http_security_settings_rejected_for_stdio(kwargs):
    with pytest.raises(ValueError, match="HTTP security"):
        MCPSource("test", command=("echo",), **kwargs)


@pytest.mark.parametrize("hostname,success", [("mcp.test", True), ("wrong.test", False)])
async def test_real_tls_preserves_hostname_verification(monkeypatch, hostname, success):
    import ssl
    from contextlib import suppress
    from pathlib import Path

    fixtures = Path(__file__).parents[1] / "fixtures"
    server_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    server_context.load_cert_chain(fixtures / "mcp_test_cert.pem", fixtures / "mcp_test_key.pem")
    client_context = ssl.create_default_context(cafile=str(fixtures / "mcp_test_cert.pem"))
    monkeypatch.setattr(httpx2, "create_ssl_context", lambda **kwargs: client_context)
    monkeypatch.setattr(_PolicyBackend, "_resolve", AsyncMock(return_value=["127.0.0.1"]))
    requests = []
    tasks = set()

    async def handle(reader, writer):
        try:
            requests.append(await reader.readuntil(b"\r\n\r\n"))
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok")
            await writer.drain()
        finally:
            writer.close()
            with suppress(ConnectionResetError):
                await writer.wait_closed()

    def connected(reader, writer):
        task = asyncio.create_task(handle(reader, writer))
        tasks.add(task)
        task.add_done_callback(tasks.discard)

    server = await asyncio.start_server(connected, "127.0.0.1", 0, ssl=server_context)
    port = server.sockets[0].getsockname()[1]
    policy = AsyncMock(return_value=True)
    try:
        source = MCPSource.http("test", f"https://{hostname}:{port}/mcp", destination_policy=policy)
        async with create_http_client(source) as client:
            if success:
                response = await client.get(source.url)
                assert response.text == "ok"
                assert f"Host: {hostname}:{port}".encode() in requests[0]
            else:
                with pytest.raises(httpx2.ConnectError, match="CERTIFICATE_VERIFY_FAILED"):
                    await client.get(source.url)
                assert not requests
        assert policy.call_args.args[0].ip == ip_address("127.0.0.1")
        assert policy.call_args.args[0].host == hostname
    finally:
        server.close()
        await server.wait_closed()
        if tasks:
            await asyncio.gather(*tasks)


async def test_connection_pool_rechecks_dns_after_server_closes_socket(monkeypatch):
    resolver = AsyncMock(side_effect=[["8.8.8.8"], ["127.0.0.1"]])
    monkeypatch.setattr(_PolicyBackend, "_resolve", resolver)
    dial = AsyncMock(
        return_value=httpcore2.AsyncMockStream(
            [b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok"]
        )
    )
    monkeypatch.setattr(httpcore2.AnyIOBackend, "connect_tcp", dial)
    source = MCPSource.http("test", "http://source.example", destination_policy=allow_public)
    async with create_http_client(source) as client:
        assert (await client.get(source.url)).text == "ok"
        with pytest.raises(MCPDestinationDeniedError):
            await client.get(source.url)
    assert resolver.await_count == 2
    assert dial.await_count == 1


async def test_real_cross_origin_redirect_never_receives_custom_credentials():
    from tests.unit.test_mcp_streamable_http_auth import _serve

    received = []

    async def target(scope, receive, send):
        if scope["type"] != "http":
            return
        received.append(dict(scope["headers"]))
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    async with _serve(target) as target_url:

        async def redirect(scope, receive, send):
            if scope["type"] != "http":
                return
            await send(
                {
                    "type": "http.response.start",
                    "status": 307,
                    "headers": [(b"location", target_url.encode())],
                }
            )
            await send({"type": "http.response.body", "body": b""})

        async with _serve(redirect) as source_url:
            source = MCPSource.http(
                "test",
                source_url,
                headers={"X-Custom-Key": "secret"},
                destination_policy=AsyncMock(return_value=True),
            )
            async with create_http_client(source) as client:
                assert (await client.post(source.url, json={})).status_code == 200
    assert len(received) == 1
    assert b"x-custom-key" not in received[0]


async def test_same_then_cross_origin_redirect_after_auth_owned_request_strips_auth():
    """An earlier auth-owned request must not let a later redirect keep credentials."""
    requests = []

    def handle(request):
        requests.append(request)
        if request.url.host == "auth.example":
            return httpx2.Response(200, json={"access_token": "fresh"})
        if request.url.host == "other.example":
            return httpx2.Response(200)
        if request.headers.get("authorization") != "Bearer fresh":
            return httpx2.Response(401)
        if request.url.path == "/mcp":
            return httpx2.Response(307, headers={"location": "/next"})
        return httpx2.Response(307, headers={"location": "https://other.example/mcp"})

    source = MCPSource.http("test", "https://source.example/mcp", auth=TokenRequestAuth())
    async with create_http_client(source, transport=httpx2.MockTransport(handle)) as client:
        await client.post(source.url, json={})
    assert [(r.url.host, r.url.path) for r in requests] == [
        ("source.example", "/mcp"),
        ("auth.example", "/token"),
        ("source.example", "/mcp"),
        ("source.example", "/next"),
        ("other.example", "/mcp"),
    ]
    assert requests[1].headers["authorization"].startswith("Basic ")
    assert requests[3].headers["authorization"] == "Bearer fresh"
    assert "authorization" not in requests[4].headers


async def test_real_sdk_same_then_cross_origin_redirect_never_carries_auth():
    """The SDK follows the same-origin hop itself; the cross-origin hop is never credentialed."""
    from dendrux.mcp import MCPConnectionError
    from dendrux.mcp._client import MCPClientAdapter
    from tests.unit.test_mcp_streamable_http_auth import _serve

    seen = {"auth": [], "source": [], "target": []}

    def recorder(name, status, headers=()):
        async def app(scope, receive, send):
            if scope["type"] != "http":
                return
            seen[name].append((scope["path"], dict(scope["headers"]).get(b"authorization")))
            await send({"type": "http.response.start", "status": status, "headers": list(headers)})
            await send({"type": "http.response.body", "body": b'{"access_token": "fresh"}'})

        return app

    async with (
        _serve(recorder("auth", 200)) as auth_url,
        _serve(recorder("target", 200)) as target_url,
    ):

        async def source_app(scope, receive, send):
            if scope["type"] != "http":
                return
            authorization = dict(scope["headers"]).get(b"authorization")
            seen["source"].append((scope["path"], authorization))
            if authorization != b"Bearer fresh":
                status, headers = 401, []
            elif scope["path"] == "/mcp":
                status, headers = 307, [(b"location", b"/next")]
            else:
                status, headers = 307, [(b"location", target_url.encode())]
            await send({"type": "http.response.start", "status": status, "headers": headers})
            await send({"type": "http.response.body", "body": b""})

        async with _serve(source_app) as source_url:
            adapter = MCPClientAdapter(
                MCPSource.http("demo", source_url, auth=TokenRequestAuth(auth_url))
            )
            with pytest.raises(MCPConnectionError):
                await adapter.connect()
            await adapter.close()

    assert seen["auth"] and all(value.startswith(b"Basic ") for _, value in seen["auth"])
    assert ("/next", b"Bearer fresh") in seen["source"]
    # The SDK does not follow the cross-origin hop at all.
    assert seen["target"] == []


async def test_destination_policy_still_applies_to_auth_owned_requests():
    from tests.unit.test_mcp_streamable_http_auth import _serve

    reached = []

    async def app(scope, receive, send):
        if scope["type"] != "http":
            return
        reached.append(scope["server"][1])
        await send({"type": "http.response.start", "status": 401, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    async with _serve(app) as source_url, _serve(app) as auth_url:
        source_port = httpx2.URL(source_url).port
        auth_port = httpx2.URL(auth_url).port
        dialed = []

        async def policy(destination):
            dialed.append(destination.port)
            return destination.port == source_port

        source = MCPSource.http(
            "test",
            source_url,
            auth=TokenRequestAuth(auth_url),
            destination_policy=policy,
        )
        async with create_http_client(source) as client:
            with pytest.raises(MCPDestinationDeniedError):
                await client.post(source.url, json={})
    assert dialed == [source_port, auth_port]
    assert reached == [source_port]


async def test_auth_owned_marker_does_not_outlive_its_flow():
    """A token request reused later as an original request keeps nothing off-origin."""
    from dendrux.mcp._http import _AUTH_OWNED_EXTENSION

    built = []

    class TokenOnce(httpx2.Auth):
        def auth_flow(self, request):
            response = yield request
            if request.url.host == "source.example" and response.status_code == 401:
                token = httpx2.Request(
                    "POST", "https://auth.example/token", headers={"Authorization": "Basic X"}
                )
                built.append(token)
                yield token

    requests = []

    def handle(request):
        requests.append(request)
        return httpx2.Response(200 if request.url.host != "source.example" else 401)

    source = MCPSource.http("test", "https://source.example/mcp", auth=TokenOnce())
    async with create_http_client(source, transport=httpx2.MockTransport(handle)) as client:
        await client.post(source.url, json={})
        assert requests[1].headers["authorization"] == "Basic X"
        stale = built[0]
        stale.url = httpx2.URL("https://third.example/steal")
        await client.send(stale)
    assert requests[-1].url.host == "third.example"
    assert "authorization" not in requests[-1].headers
    assert _AUTH_OWNED_EXTENSION not in stale.extensions


async def test_planted_auth_owned_marker_is_cleared_on_flow_entry():
    from dendrux.mcp._http import _AUTH_OWNED_EXTENSION, _AuthOwned

    class PassThrough(httpx2.Auth):
        def auth_flow(self, request):
            yield request

    requests = []

    def handle(request):
        requests.append(request)
        return httpx2.Response(200)

    source = MCPSource.http("test", "https://source.example/mcp", auth=PassThrough())
    async with create_http_client(source, transport=httpx2.MockTransport(handle)) as client:
        request = client.build_request(
            "GET", "https://other.example/x", headers={"Authorization": "Bearer S"}
        )
        request.extensions[_AUTH_OWNED_EXTENSION] = _AuthOwned(
            request, frozenset({"authorization"})
        )
        await client.send(request)
    assert "authorization" not in requests[0].headers
    assert _AUTH_OWNED_EXTENSION not in request.extensions


async def test_original_request_retargeted_by_its_flow_keeps_nothing():
    class Retarget(httpx2.Auth):
        def auth_flow(self, request):
            request.headers["Authorization"] = "Basic X"
            request.url = httpx2.URL("https://other.example/mcp")
            yield request

    requests = []

    def handle(request):
        requests.append(request)
        return httpx2.Response(200)

    source = MCPSource.http("test", "https://source.example/mcp", auth=Retarget())
    async with create_http_client(source, transport=httpx2.MockTransport(handle)) as client:
        await client.post(source.url, json={})
    assert requests[0].url.host == "other.example"
    assert "authorization" not in requests[0].headers
