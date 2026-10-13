"""The direct OAuth transport: URL identity, deadlines, caps, classification."""

from __future__ import annotations

import asyncio
import logging

import httpx2
import pytest

from dendrux.mcp._oauth_http import (
    MAX_BODY_BYTES,
    OAuthHTTP,
    OAuthTransportError,
    SendAttempt,
    URLRejectedError,
    canonical_url,
    origin_of,
    validate_url,
)
from tests.unit.mcp_oauth_harness import RawHTTPServer, RawRequest, free_port, raw_response

SENTINEL = "SENTINEL-oauth-http-7f3a"


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://RS.Example:443/mcp", "https://rs.example/mcp"),
        ("http://127.0.0.1:80/", "http://127.0.0.1"),
        ("https://rs.example/", "https://rs.example"),
        ("https://rs.example", "https://rs.example"),
        ("https://rs.example:8443/A/b?x=Y&z", "https://rs.example:8443/A/b?x=Y&z"),
        ("https://rs.example/mcp/", "https://rs.example/mcp/"),
        ("http://[::1]:8080/mcp", "http://[::1]:8080/mcp"),
        ("https://rs.example/a%2Fb", "https://rs.example/a%2Fb"),
    ],
)
def test_canonical_form(url: str, expected: str) -> None:
    assert canonical_url(url) == expected


@pytest.mark.parametrize(
    "url",
    [
        "ftp://rs.example/mcp",
        "https:///mcp",
        "https://user:pw@rs.example/mcp",
        "https://user@rs.example/mcp",
        "https://rs.example/mcp#frag",
        "https://rs.example/mcp#",
        "https://rs.example/a/../b",
        "https://rs.example/a/./b",
        "https://rs.example/a/%2e%2e/b",
        "https://rs.example/a/%2E%2e/b",
        "https://rs.example/a/.%2E/b",
        "https://rs.example/a/%2e/b",
        "https://rs.example/..",
        "https://rs.example/mcp\x00",
        "https://rs.example/m cp",
        "https://rs.example/mé",
        "https://rs.example/" + "a" * 2050,
        "https://rs.example:99999/mcp",
        "https://rs.exa%41mple/mcp",
        "https://rs.example\\@evil/",
        "",
        "mcp",
    ],
)
def test_canonical_rejections(url: str) -> None:
    with pytest.raises(URLRejectedError) as caught:
        canonical_url(url)
    assert url not in str(caught.value) or not url


def test_canonical_rejects_non_strings() -> None:
    with pytest.raises(URLRejectedError):
        canonical_url(None)


def test_validate_url_requires_https_off_loopback() -> None:
    assert validate_url("http://localhost:9/cb", allow_insecure=False) == "http://localhost:9/cb"
    assert validate_url("http://127.0.0.5/cb", allow_insecure=False) == "http://127.0.0.5/cb"
    assert validate_url("http://[::1]/cb", allow_insecure=False) == "http://[::1]/cb"
    with pytest.raises(URLRejectedError) as caught:
        validate_url("http://app.example/cb", allow_insecure=False)
    assert caught.value.insecure
    assert validate_url("http://app.example/cb", allow_insecure=True) == "http://app.example/cb"


def test_issuers_may_not_carry_a_query() -> None:
    with pytest.raises(URLRejectedError):
        validate_url("https://as.example/?tenant=a", allow_insecure=False, issuer=True)
    assert validate_url("https://as.example/t", allow_insecure=False, issuer=True)


def test_origin_of() -> None:
    assert origin_of("https://AS.example:443/a/b?q") == "https://as.example"
    assert origin_of("http://127.0.0.1:8000/x") == "http://127.0.0.1:8000"


def _handler(response: bytes):
    async def handle(request: RawRequest, writer: asyncio.StreamWriter) -> None:
        writer.write(response)

    return handle


async def test_send_reads_body_and_never_follows_redirects() -> None:
    async with RawHTTPServer(
        _handler(raw_response(302, b"", headers={"Location": "http://127.0.0.1:9/x"}))
    ) as server:
        client = OAuthHTTP(policy=None, timeout=2.0)
        response = await client.send("GET", server.url + "/meta")
        await client.aclose()
    assert response.status == 302
    assert len(server.requests) == 1


async def test_form_post_and_hostile_reason_phrase_is_never_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)

    async def echo(request: RawRequest, writer: asyncio.StreamWriter) -> None:
        token = request.form["refresh_token"]
        writer.write(raw_response(200, b'{"ok": true}', reason=f"OK {token}"))

    async with RawHTTPServer(echo) as server:
        client = OAuthHTTP(policy=None, timeout=2.0)
        response = await client.send(
            "POST", server.url + "/token", form={"refresh_token": SENTINEL}
        )
        await client.aclose()
    assert response.status == 200
    assert response.body == b'{"ok": true}'
    info_and_above = [r for r in caplog.records if r.levelno >= logging.INFO]
    assert all(SENTINEL not in r.getMessage() for r in info_and_above)
    assert server.requests[0].form["refresh_token"] == SENTINEL


async def test_connection_refused_is_unsent() -> None:
    client = OAuthHTTP(policy=None, timeout=2.0)
    with pytest.raises(OAuthTransportError) as caught:
        await client.send("POST", f"http://127.0.0.1:{free_port()}/token", form={"a": "b"})
    await client.aclose()
    assert caught.value.outcome == "unsent"
    assert caught.value.cause == "ConnectError"
    assert caught.value.__cause__ is None and caught.value.__context__ is None


async def test_destination_policy_denies_before_any_connection() -> None:
    calls = []

    async def deny(destination) -> bool:
        calls.append(destination)
        return False

    async with RawHTTPServer(_handler(raw_response(200, b"{}"))) as server:
        client = OAuthHTTP(policy=deny, timeout=2.0)
        with pytest.raises(OAuthTransportError) as caught:
            await client.send("GET", server.url + "/m")
        await client.aclose()
    assert caught.value.outcome == "unsent" and caught.value.denied
    assert server.connections == 0
    assert calls and calls[0].ip.is_loopback


async def test_policy_exception_fails_closed() -> None:
    async def broken(destination) -> bool:
        raise RuntimeError(SENTINEL)

    async with RawHTTPServer(_handler(raw_response(200, b"{}"))) as server:
        client = OAuthHTTP(policy=broken, timeout=2.0)
        with pytest.raises(OAuthTransportError) as caught:
            await client.send("GET", server.url + "/m")
        await client.aclose()
    assert caught.value.denied
    assert SENTINEL not in repr(caught.value)


async def test_policy_precheck_runs_once_per_host_and_socket_check_stays() -> None:
    calls = []

    async def allow(destination) -> bool:
        calls.append(destination)
        return True

    async with RawHTTPServer(_handler(raw_response(200, b"{}"))) as server:
        client = OAuthHTTP(policy=allow, timeout=2.0)
        await client.send("GET", server.url + "/a")
        await client.send("GET", server.url + "/b")
        await client.aclose()
    # one precheck, plus the socket-level check for each new connection
    assert len(calls) == 3


async def test_body_over_cap_is_uncertain() -> None:
    big = b"x" * (MAX_BODY_BYTES + 10)
    async with RawHTTPServer(_handler(raw_response(200, big))) as server:
        client = OAuthHTTP(policy=None, timeout=5.0)
        with pytest.raises(OAuthTransportError) as caught:
            await client.send("POST", server.url + "/token", form={})
        await client.aclose()
    assert caught.value.outcome == "uncertain"
    assert caught.value.cause == "BodyTooLarge"


async def test_trickling_body_hits_the_overall_deadline() -> None:
    async def trickle(request: RawRequest, writer: asyncio.StreamWriter) -> None:
        writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 1000\r\n\r\n")
        for _ in range(100):
            writer.write(b"x")
            await writer.drain()
            await asyncio.sleep(0.05)

    async with RawHTTPServer(trickle) as server:
        client = OAuthHTTP(policy=None, timeout=0.4)
        loop = asyncio.get_running_loop()
        started = loop.time()
        with pytest.raises(OAuthTransportError) as caught:
            await client.send("POST", server.url + "/token", form={})
        await client.aclose()
    assert loop.time() - started < 2.0
    assert caught.value.outcome == "uncertain"


async def test_dropped_connection_after_request_is_uncertain() -> None:
    async def drop(request: RawRequest, writer: asyncio.StreamWriter) -> None:
        writer.transport.abort()

    async with RawHTTPServer(drop) as server:
        client = OAuthHTTP(policy=None, timeout=2.0)
        with pytest.raises(OAuthTransportError) as caught:
            await client.send("POST", server.url + "/token", form={"x": "y"})
        await client.aclose()
    assert caught.value.outcome == "uncertain"


async def test_invalid_url_is_unsent_without_io() -> None:
    client = OAuthHTTP(policy=None, timeout=1.0)
    with pytest.raises(OAuthTransportError) as caught:
        await client.send("GET", "https://user@as.example/x")
    assert caught.value.outcome == "unsent"
    await client.aclose()
    with pytest.raises(OAuthTransportError):
        await client.send("GET", "https://as.example/x")


async def test_cancellation_reports_whether_the_send_began() -> None:
    entered = asyncio.Event()

    async def hang(request: RawRequest, writer: asyncio.StreamWriter) -> None:
        entered.set()
        await asyncio.sleep(10)

    async with RawHTTPServer(hang) as server:
        client = OAuthHTTP(policy=None, timeout=10.0)
        attempt = SendAttempt()
        task = asyncio.create_task(
            client.send("POST", server.url + "/token", form={}, attempt=attempt)
        )
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert attempt.started

        gate = asyncio.Event()

        async def slow_policy(destination) -> bool:
            await gate.wait()
            return True

        slow = OAuthHTTP(policy=slow_policy, timeout=10.0)
        attempt = SendAttempt()
        task = asyncio.create_task(slow.send("GET", server.url + "/x", attempt=attempt))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not attempt.started
        await slow.aclose()
        await client.aclose()


async def test_failed_responses_release_their_pool_slots() -> None:
    async with RawHTTPServer(
        _handler(
            raw_response(200, b"x" * (MAX_BODY_BYTES + 1), headers={"Connection": "keep-alive"})
        )
    ) as server:
        client = OAuthHTTP(policy=None, timeout=2.0)
        for _ in range(3):
            with pytest.raises(OAuthTransportError):
                await client.send("GET", server.url + "/x")
        pool = client._transport._pool
        # An unclosed response would keep its request registered in the pool.
        assert not pool._requests
        await client.aclose()


async def test_oauth_requests_never_use_async_client(monkeypatch: pytest.MonkeyPatch) -> None:
    async def forbidden(*args: object, **kwargs: object) -> httpx2.Response:
        raise AssertionError("OAuth request went through httpx2.AsyncClient")

    monkeypatch.setattr(httpx2.AsyncClient, "send", forbidden)
    async with RawHTTPServer(_handler(raw_response(200, b"{}"))) as server:
        client = OAuthHTTP(policy=None, timeout=2.0)
        response = await client.send("GET", server.url + "/x")
        await client.aclose()
    assert response.status == 200
