"""HTTP credential boundaries and pre-connect destination enforcement."""

from __future__ import annotations

import asyncio
import socket
from contextlib import asynccontextmanager
from ipaddress import IPv6Address, ip_address
from typing import TYPE_CHECKING, Any

import httpcore2
import httpx2

from dendrux.mcp._destination import MCPDestination, MCPDestinationPolicy
from dendrux.mcp._errors import MCPDestinationDeniedError

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Iterable

    from dendrux.mcp._source import MCPSource


class _PolicyBackend(httpcore2.AsyncNetworkBackend):
    def __init__(
        self,
        policy: MCPDestinationPolicy,
        on_denied: Callable[[], None] | None = None,
    ) -> None:
        self._on_denied = on_denied
        self._policy = policy
        self._backend = httpcore2.AnyIOBackend()

    async def _resolve(self, host: str, port: int) -> list[str]:
        try:
            return [str(ip_address(host))]
        except ValueError:
            pass
        answers = await asyncio.get_running_loop().getaddrinfo(
            host,
            port,
            family=socket.AF_UNSPEC,
            type=socket.SOCK_STREAM,
        )
        return list(dict.fromkeys(str(answer[4][0]) for answer in answers))

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Iterable[tuple[Any, ...]] | None = None,
    ) -> httpcore2.AsyncNetworkStream:
        try:
            return await self._connect_tcp(host, port, timeout, local_address, socket_options)
        except MCPDestinationDeniedError:
            if self._on_denied is not None:
                self._on_denied()
            raise

    async def _connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None,
        local_address: str | None,
        socket_options: Iterable[tuple[Any, ...]] | None,
    ) -> httpcore2.AsyncNetworkStream:
        try:
            async with asyncio.timeout(timeout) as deadline:
                addresses = await self._resolve(host, port)
                last_error: Exception | None = None
                for address in addresses:
                    ip = ip_address(address)
                    if isinstance(ip, IPv6Address) and ip.ipv4_mapped is not None:
                        ip = ip.ipv4_mapped
                    policy_failed = False
                    try:
                        allowed = await self._policy(MCPDestination(host, port, ip))
                    except Exception:
                        policy_failed = True
                        allowed = False
                    if policy_failed:
                        # Raise outside the handler so callback secrets cannot enter the chain.
                        raise MCPDestinationDeniedError("MCP destination policy failed.")
                    if allowed is not True:
                        continue
                    expires = deadline.when()
                    remaining = (
                        None
                        if expires is None
                        else max(0.0, expires - asyncio.get_running_loop().time())
                    )
                    try:
                        return await self._backend.connect_tcp(
                            host=str(ip),
                            port=port,
                            timeout=remaining,
                            local_address=local_address,
                            socket_options=socket_options,
                        )
                    except (httpcore2.ConnectError, httpcore2.ConnectTimeout) as exc:
                        last_error = exc
                if last_error is not None:
                    raise last_error
                raise MCPDestinationDeniedError("MCP destination policy denied the connection.")
        except TimeoutError:
            raise httpcore2.ConnectTimeout("MCP destination connection timed out.") from None
        except OSError:
            raise httpcore2.ConnectError("MCP destination resolution failed.") from None


_CORE_ERRORS = {
    httpcore2.ConnectTimeout: httpx2.ConnectTimeout,
    httpcore2.ReadTimeout: httpx2.ReadTimeout,
    httpcore2.WriteTimeout: httpx2.WriteTimeout,
    httpcore2.PoolTimeout: httpx2.PoolTimeout,
    httpcore2.ConnectError: httpx2.ConnectError,
    httpcore2.ReadError: httpx2.ReadError,
    httpcore2.WriteError: httpx2.WriteError,
    httpcore2.RemoteProtocolError: httpx2.RemoteProtocolError,
    httpcore2.LocalProtocolError: httpx2.LocalProtocolError,
    httpcore2.UnsupportedProtocol: httpx2.UnsupportedProtocol,
    httpcore2.ProxyError: httpx2.ProxyError,
}


@asynccontextmanager
async def _map_errors() -> AsyncIterator[None]:
    try:
        yield
    except tuple(_CORE_ERRORS) as exc:
        for core, public in _CORE_ERRORS.items():
            if isinstance(exc, core):
                raise public(str(exc)) from exc
        raise


class _ResponseStream(httpx2.AsyncByteStream):
    def __init__(self, stream: Any) -> None:
        self._stream = stream

    async def __aiter__(self) -> AsyncIterator[bytes]:
        async with _map_errors():
            async for chunk in self._stream:
                yield chunk

    async def aclose(self) -> None:
        async with _map_errors():
            await self._stream.aclose()


class _PolicyTransport(httpx2.AsyncBaseTransport):
    def __init__(
        self,
        policy: MCPDestinationPolicy,
        on_denied: Callable[[], None] | None = None,
    ) -> None:
        self._pool = httpcore2.AsyncConnectionPool(
            ssl_context=httpx2.create_ssl_context(trust_env=False),
            network_backend=_PolicyBackend(policy, on_denied),
            max_connections=100,
            max_keepalive_connections=20,
            keepalive_expiry=5.0,
        )

    async def handle_async_request(self, request: httpx2.Request) -> httpx2.Response:
        async with _map_errors():
            response = await self._pool.handle_async_request(
                httpcore2.Request(
                    method=request.method,
                    url=httpcore2.URL(
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
        return httpx2.Response(
            response.status,
            headers=response.headers,
            stream=_ResponseStream(response.stream),
            extensions=response.extensions,
        )

    async def aclose(self) -> None:
        async with _map_errors():
            await self._pool.aclose()


_Origin = tuple[str, str, int | None]


def _origin(url: httpx2.URL) -> _Origin:
    port = url.port if url.port is not None else {"http": 80, "https": 443}.get(url.scheme)
    return url.scheme, url.host, port


_AUTH_OWNED_EXTENSION = "dendrux_auth_owned"


class _AuthOwned:
    """Provenance of one request an auth flow built for another origin."""

    __slots__ = ("headers", "request")

    def __init__(self, request: httpx2.Request, headers: frozenset[str]) -> None:
        self.request = request
        self.headers = headers


def _auth_owned_headers(request: httpx2.Request) -> frozenset[str]:
    """Header names an auth flow introduced on exactly this request.

    Redirect requests copy their parent's extensions, so the marker names
    the exact request object it was set on; a copy never matches and keeps
    nothing.
    """
    marker = request.extensions.get(_AUTH_OWNED_EXTENSION)
    if isinstance(marker, _AuthOwned) and marker.request is request:
        return marker.headers
    return frozenset()


class _ScopedAuth(httpx2.Auth):
    """Track auth-generated headers and tag the auth flow's own requests.

    A request the wrapped flow builds itself for another origin (for
    example an OAuth token request to an authorization server) is tagged
    auth-owned together with the header names the flow introduced on it:
    absent from, or different to, the original request before the flow ran,
    and not a value the flow set on the original request in place. Only
    those headers survive the cross-origin scoping; anything copied from the
    original request is still stripped. The original request, and anything
    a redirect builds, is never tagged.

    The marker lives only while its request is being sent by this flow: any
    marker already on the original request is removed on entry, every yield
    sets or removes it, and it is removed again once the request returns.
    """

    def __init__(self, auth: httpx2.Auth, sensitive: set[str], origin: _Origin) -> None:
        self._auth = auth
        self._sensitive = sensitive
        self._origin = origin

    async def async_auth_flow(
        self,
        request: httpx2.Request,
    ) -> AsyncGenerator[httpx2.Request, httpx2.Response]:
        request.extensions.pop(_AUTH_OWNED_EXTENSION, None)
        initial = dict(request.headers)
        before = initial
        # (name, value) pairs the flow set on the original request in place.
        # Such a value is the source's credential, never the flow's own for
        # another origin, even when copied onto a new request.
        set_on_original: set[tuple[str, str]] = set()
        flow = self._auth.async_auth_flow(request)
        try:
            outgoing = await anext(flow)
            while True:
                set_on_original.update(
                    (name, value)
                    for name, value in request.headers.items()
                    if initial.get(name) != value
                )
                if outgoing is not request and _origin(outgoing.url) != self._origin:
                    introduced = frozenset(
                        name
                        for name, value in outgoing.headers.items()
                        if initial.get(name) != value and (name, value) not in set_on_original
                    )
                    outgoing.extensions[_AUTH_OWNED_EXTENSION] = _AuthOwned(outgoing, introduced)
                else:
                    outgoing.extensions.pop(_AUTH_OWNED_EXTENSION, None)
                self._sensitive.update(
                    name for name, value in outgoing.headers.items() if before.get(name) != value
                )
                try:
                    response = yield outgoing
                finally:
                    outgoing.extensions.pop(_AUTH_OWNED_EXTENSION, None)
                before = dict(outgoing.headers)
                try:
                    outgoing = await flow.asend(response)
                except StopAsyncIteration:
                    break
        finally:
            await flow.aclose()


def create_http_client(
    source: MCPSource,
    *,
    request_hook: Callable[[httpx2.Request], Awaitable[None]] | None = None,
    response_hook: Callable[[httpx2.Response], Awaitable[None]] | None = None,
    transport: httpx2.AsyncBaseTransport | None = None,
    on_denied: Callable[[], None] | None = None,
) -> httpx2.AsyncClient:
    """Build the source's SDK HTTP client with transport security boundaries."""
    assert source.url is not None
    origin = _origin(httpx2.URL(source.url))
    sensitive = {name.lower() for name in source.headers}
    sensitive.update({"authorization", "proxy-authorization", "cookie"})

    async def scope_credentials(request: httpx2.Request) -> None:
        if request_hook is not None:
            await request_hook(request)
        if request.url.scheme not in ("http", "https") or request.url.userinfo:
            if on_denied is not None:
                on_denied()
            raise MCPDestinationDeniedError("MCP request has an unsupported destination URL.")
        if _origin(request.url) != origin:
            keep = _auth_owned_headers(request)
            for name in sensitive - keep:
                request.headers.pop(name, None)

    if transport is None and source.destination_policy is not None:
        transport = _PolicyTransport(source.destination_policy, on_denied)
    client = httpx2.AsyncClient(
        headers=dict(source.headers),
        auth=source.auth,
        timeout=httpx2.Timeout(source.connect_timeout, read=source.call_timeout),
        follow_redirects=source.follow_redirects,
        # MCP 2.2 follows same-origin redirects itself; its loop uses this
        # budget even when the HTTP client has follow_redirects disabled.
        max_redirects=source.max_redirects if source.follow_redirects else 0,
        trust_env=source.destination_policy is None,
        transport=transport,
        event_hooks={
            "request": [scope_credentials],
            "response": [response_hook] if response_hook is not None else [],
        },
    )
    if client.auth is not None:
        client.auth = _ScopedAuth(client.auth, sensitive, origin)
    return client
