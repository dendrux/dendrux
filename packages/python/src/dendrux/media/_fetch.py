"""SSRF-hardened, fetch-once URL reader for ``Image.from_url``.

The built-in fetcher never consults the environment (no proxies, ``.netrc`` or
CA overrides), resolves every host itself, rejects the whole answer set when
any address is non-public, and connects the socket to the validated address.
TLS then runs on that socket against the original host name (SNI, certificate
and ``Host`` header). Redirects are followed manually and every hop is
re-checked. Errors carry the host only, never the URL path, query or
credentials, and are raised with no exception chain.
"""

from __future__ import annotations

import asyncio
import socket
import ssl
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable
from ipaddress import IPv4Address, IPv4Network, IPv6Address, IPv6Network, ip_address
from typing import TYPE_CHECKING, Any

import httpcore
import httpx

from dendrux.errors import ImageFetchError

if TYPE_CHECKING:
    from dendrux.media._policy import UrlPolicy

Resolver = Callable[[str, int], Awaitable[list[str]]]

BLOCKED_HOSTNAMES = frozenset({"metadata.google.internal", "metadata", "instance-data"})
_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
_DEFAULT_PORTS = {"http": 80, "https": 443}

_V6_BLOCKED_NETS = (
    IPv6Network("::/96"),  # IPv4-compatible (and ::, ::1)
    IPv6Network("64:ff9b::/96"),  # NAT64
    IPv6Network("64:ff9b:1::/48"),  # local-use NAT64
    IPv6Network("fc00::/7"),
    IPv6Network("fe80::/10"),
    IPv6Network("fec0::/10"),
)
_V4_EXTRA_BLOCKED = (IPv4Network("100.64.0.0/10"), IPv4Network("255.255.255.255/32"))


class _FetchDeniedError(Exception):
    """Internal refusal; converted to ImageFetchError outside every handler."""

    def __init__(self, reason: str, *, status_code: int | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.status_code = status_code


def is_blocked_address(ip: IPv4Address | IPv6Address) -> bool:
    """True for every address that is not plainly public unicast."""
    if isinstance(ip, IPv6Address):
        if ip.ipv4_mapped is not None:
            return True
        if any(ip in net for net in _V6_BLOCKED_NETS):
            return True
        if ip.sixtofour is not None:
            return is_blocked_address(ip.sixtofour)
        if ip.teredo is not None:
            return True
    elif any(ip in net for net in _V4_EXTRA_BLOCKED):
        return True
    return bool(
        not ip.is_global
        or ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def _literal_ip(host: str) -> IPv4Address | IPv6Address | None:
    try:
        return ip_address(host)
    except ValueError:
        pass
    # Legacy IPv4 forms libc would accept ("2130706433", "0x7f.1", "0177.0.0.1").
    if host and all(c in "0123456789abcdefxABCDEFX." for c in host) and not host.endswith("."):
        try:
            return IPv4Address(socket.inet_aton(host))
        except OSError:
            return None
    return None


async def _system_resolve(host: str, port: int) -> list[str]:
    infos = await asyncio.get_running_loop().getaddrinfo(
        host, port, family=socket.AF_UNSPEC, type=socket.SOCK_STREAM
    )
    return list(dict.fromkeys(str(info[4][0]) for info in infos))


def host_allowed(host: str, allowed: Iterable[str]) -> bool:
    """Exact host match, or ``"*.suffix"`` matching any subdomain of ``suffix``."""
    for pattern in allowed:
        if pattern.startswith("*."):
            if host.endswith(pattern[1:]) and len(host) > len(pattern) - 1:
                return True
        elif host == pattern:
            return True
    return False


class _PinnedBackend(httpcore.AsyncNetworkBackend):
    """Resolves and validates every answer, then connects to a validated address."""

    def __init__(
        self,
        *,
        allow_private: bool,
        resolver: Resolver,
        inner: httpcore.AsyncNetworkBackend,
    ) -> None:
        self._allow_private = allow_private
        self._resolver = resolver
        self._inner = inner

    async def _addresses(self, host: str, port: int) -> list[IPv4Address | IPv6Address]:
        literal = _literal_ip(host)
        if literal is not None:
            return [literal]
        try:
            answers = await self._resolver(host, port)
        except (OSError, UnicodeError):
            answers = []
        ips: list[IPv4Address | IPv6Address] = []
        for answer in answers:
            try:
                ips.append(ip_address(answer))
            except ValueError:
                raise _FetchDeniedError("dns") from None
        if not ips:
            raise _FetchDeniedError("dns")
        return ips

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Iterable[Any] | None = None,
    ) -> httpcore.AsyncNetworkStream:
        ips = await self._addresses(host, port)
        if not self._allow_private and any(is_blocked_address(ip) for ip in ips):
            raise _FetchDeniedError("blocked_address")
        last: Exception | None = None
        for ip in ips:
            try:
                return await self._inner.connect_tcp(
                    str(ip),
                    port,
                    timeout=timeout,
                    local_address=local_address,
                    socket_options=socket_options,
                )
            except (httpcore.ConnectError, httpcore.ConnectTimeout) as exc:
                last = exc
        assert last is not None
        raise last

    async def connect_unix_socket(
        self,
        path: str,
        timeout: float | None = None,
        socket_options: Iterable[Any] | None = None,
    ) -> httpcore.AsyncNetworkStream:
        raise _FetchDeniedError("scheme")

    async def sleep(self, seconds: float) -> None:
        await self._inner.sleep(seconds)


class _ResponseStream(httpx.AsyncByteStream):
    def __init__(self, stream: Any) -> None:
        self._stream = stream

    async def __aiter__(self) -> AsyncIterator[bytes]:
        async for chunk in self._stream:
            yield chunk

    async def aclose(self) -> None:
        await self._stream.aclose()


class _PinnedTransport(httpx.AsyncBaseTransport):
    def __init__(self, backend: httpcore.AsyncNetworkBackend, ssl_context: ssl.SSLContext) -> None:
        self._pool = httpcore.AsyncConnectionPool(
            ssl_context=ssl_context,
            network_backend=backend,
            http1=True,
            http2=False,
            retries=0,
            max_connections=4,
        )

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        assert isinstance(request.stream, httpx.AsyncByteStream)
        response = await self._pool.handle_async_request(
            httpcore.Request(
                method=request.method,
                url=httpcore.URL(
                    scheme=request.url.raw_scheme,
                    host=request.url.raw_host,
                    port=request.url.port,
                    target=request.url.raw_path,
                ),
                headers=request.headers.raw,
                content=request.stream,
                extensions=request.extensions,
            )
        )
        return httpx.Response(
            response.status,
            headers=response.headers,
            stream=_ResponseStream(response.stream),
            extensions=response.extensions,
        )

    async def aclose(self) -> None:
        await self._pool.aclose()


def _check_hop(raw: str, policy: UrlPolicy) -> tuple[httpx.URL, str]:
    """Validate scheme, credentials, port and host of one hop; return (url, host)."""
    try:
        url = httpx.URL(raw)
        host = url.raw_host.decode("ascii").lower().rstrip(".")
    except (httpx.InvalidURL, UnicodeError, ValueError, TypeError):
        raise _FetchDeniedError("scheme") from None
    if url.scheme.lower() not in policy.allowed_schemes:
        raise _FetchDeniedError("scheme")
    if not host:
        raise _FetchDeniedError("host_not_allowed")
    if url.userinfo:
        raise _FetchDeniedError("credentials")
    port = url.port if url.port is not None else _DEFAULT_PORTS[url.scheme.lower()]
    if port not in policy.allowed_ports:
        raise _FetchDeniedError("port")
    if policy.allowed_hosts is not None and not host_allowed(host, policy.allowed_hosts):
        raise _FetchDeniedError("host_not_allowed")
    if not policy.allow_private_networks:
        if host in BLOCKED_HOSTNAMES:
            raise _FetchDeniedError("blocked_address")
        literal = _literal_ip(host)
        if literal is not None and is_blocked_address(literal):
            raise _FetchDeniedError("blocked_address")
    return url, host


async def _fetch_hops(
    client: httpx.AsyncClient, url: str, policy: UrlPolicy, max_bytes: int, hop_host: list[str]
) -> bytes:
    current = url
    for _ in range(policy.max_redirects + 1):
        target, host = _check_hop(current, policy)
        hop_host[0] = host
        async with client.stream(
            "GET",
            target,
            headers={"Accept-Encoding": "identity", "Accept": "image/*"},
        ) as response:
            status = response.status_code
            if status in _REDIRECT_STATUSES:
                location = response.headers.get("location")
                if not location:
                    raise _FetchDeniedError("http_status", status_code=status)
                try:
                    current = str(response.url.join(location))
                except (httpx.InvalidURL, ValueError, TypeError):
                    raise _FetchDeniedError("redirect") from None
                continue
            if not 200 <= status < 300:
                raise _FetchDeniedError("http_status", status_code=status)
            encoding = response.headers.get("content-encoding")
            if encoding is not None and encoding.strip().lower() not in ("", "identity"):
                raise _FetchDeniedError("content_encoding")
            declared = response.headers.get("content-length")
            if declared is not None and declared.strip().isdigit() and int(declared) > max_bytes:
                raise _FetchDeniedError("too_large")
            body = bytearray()
            async for chunk in response.aiter_raw():
                body += chunk
                if len(body) > max_bytes:
                    raise _FetchDeniedError("too_large")
            return bytes(body)
    raise _FetchDeniedError("redirect")


async def fetch_url(
    url: str,
    *,
    policy: UrlPolicy,
    max_bytes: int,
    resolver: Resolver | None = None,
    network_backend: httpcore.AsyncNetworkBackend | None = None,
    ssl_context: ssl.SSLContext | None = None,
) -> bytes:
    """Fetch ``url`` under ``policy`` with a hard ``max_bytes`` ceiling.

    ``resolver``, ``network_backend`` and ``ssl_context`` exist for tests; the
    defaults are the system resolver, a plain TCP backend and a verifying TLS
    context that ignores environment overrides.

    Raises:
        ImageFetchError: on any refusal or failure; only the host is reported.
    """
    if not policy.fetch:
        raise ImageFetchError("disabled")
    if policy.fetcher is not None:
        data = await policy.fetcher(url, max_bytes)
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError("UrlPolicy.fetcher must return bytes")
        if len(data) > max_bytes:
            raise ImageFetchError("too_large", host=_host_label(url))
        return bytes(data)

    backend = _PinnedBackend(
        allow_private=policy.allow_private_networks,
        resolver=resolver or _system_resolve,
        inner=network_backend or httpcore.AnyIOBackend(),
    )
    context = ssl_context or httpx.create_ssl_context(trust_env=False)
    transport = _PinnedTransport(backend, context)
    hop_host = [_host_label(url) or ""]
    reason: str | None = None
    status_code: int | None = None
    data_out: bytes | None = None
    try:
        async with (
            asyncio.timeout(policy.total_timeout),
            httpx.AsyncClient(
                transport=transport,
                trust_env=False,
                follow_redirects=False,
                timeout=httpx.Timeout(
                    connect=policy.connect_timeout,
                    read=policy.read_timeout,
                    write=policy.read_timeout,
                    pool=policy.connect_timeout,
                ),
            ) as client,
        ):
            data_out = await _fetch_hops(client, url, policy, max_bytes, hop_host)
    except _FetchDeniedError as denied:
        reason, status_code = denied.reason, denied.status_code
    except (TimeoutError, httpx.TimeoutException, httpcore.TimeoutException):
        reason = "timeout"
    except (
        httpx.HTTPError,
        httpx.InvalidURL,
        httpcore.NetworkError,
        httpcore.ProtocolError,
        httpcore.UnsupportedProtocol,
        ssl.SSLError,
        OSError,
    ):
        reason = "connect"
    if reason is not None:
        raise ImageFetchError(reason, host=hop_host[0] or None, status_code=status_code)
    assert data_out is not None
    return data_out


def _host_label(url: str) -> str | None:
    try:
        return httpx.URL(url).raw_host.decode("ascii").lower().rstrip(".") or None
    except (httpx.InvalidURL, UnicodeError, ValueError, TypeError):
        return None
