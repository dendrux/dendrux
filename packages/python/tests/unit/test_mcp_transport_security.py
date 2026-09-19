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
    assert all("x-generated-key" not in r.headers for r in requests[1:])


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
