"""Direct-transport HTTP client for OAuth requests, and URL identity rules.

OAuth requests never go through ``httpx2.AsyncClient``: its ``send`` logs the
URL and the server's reason phrase at INFO, and a hostile authorization
server can echo a submitted refresh token into that phrase. Requests are
built as ``httpx2.Request`` objects and sent straight through a pool-backed
transport with no redirects, auth objects, event hooks, or cookies. Every
request runs under one overall deadline covering send, headers, and the full
body read; bodies are capped and every response is closed.
"""

from __future__ import annotations

import asyncio
import logging
import socket
from dataclasses import dataclass
from ipaddress import IPv6Address, ip_address
from typing import TYPE_CHECKING, Literal
from urllib.parse import urlsplit

import httpx2

from dendrux.mcp._destination import MCPDestination, MCPDestinationPolicy
from dendrux.mcp._errors import MCPDestinationDeniedError
from dendrux.mcp._http import _PolicyTransport

if TYPE_CHECKING:
    from collections.abc import Mapping

logger = logging.getLogger(__name__)

MAX_BODY_BYTES = 1 << 20
_MAX_URL_LENGTH = 2048
_LOOPBACK_NAMES = frozenset({"localhost"})

SendOutcome = Literal["unsent", "uncertain"]

# Raised while waiting for (or opening) a connection, before any request
# handler on the far side could have run.
_UNSENT_ERRORS: tuple[type[BaseException], ...] = (
    httpx2.ConnectError,
    httpx2.ConnectTimeout,
    httpx2.PoolTimeout,
    MCPDestinationDeniedError,
)


class URLRejectedError(ValueError):
    """A URL failed identity validation. The message never carries the URL."""

    def __init__(self, *, insecure: bool = False) -> None:
        self.insecure = insecure
        super().__init__(
            "URL must use https (or a loopback host)."
            if insecure
            else "URL is not a valid absolute http(s) identity."
        )


def _valid_host(host: str, bracketed: bool) -> bool:
    if bracketed:
        try:
            return isinstance(ip_address(host), IPv6Address)
        except ValueError:
            return False
    return bool(host) and all(char.isalnum() or char in ".-_" for char in host)


def canonical_url(url: object) -> str:
    """Return the canonical identity form of an absolute http(s) URL.

    Rejects non-http(s), missing host, userinfo, fragments, over-long URLs,
    dot segments (including any percent-encoded form), and control or
    non-ASCII characters. Scheme and host are lower-cased, the default port is
    removed, a path of exactly ``/`` becomes empty, and other paths and the
    query are kept verbatim.
    """
    if not isinstance(url, str) or not url or len(url) > _MAX_URL_LENGTH:
        raise URLRejectedError
    if any(ord(char) <= 0x20 or ord(char) >= 0x7F for char in url):
        raise URLRejectedError
    if "#" in url or "\\" in url:
        raise URLRejectedError
    try:
        parts = urlsplit(url)
        port = parts.port
        host = parts.hostname
    except ValueError:
        raise URLRejectedError from None
    scheme = parts.scheme.lower()
    if scheme not in ("http", "https") or host is None or "@" in parts.netloc:
        raise URLRejectedError
    bracketed = parts.netloc.startswith("[")
    if not _valid_host(host, bracketed):
        raise URLRejectedError
    for segment in parts.path.split("/"):
        if segment.lower().replace("%2e", ".") in (".", ".."):
            raise URLRejectedError
    default_port = 443 if scheme == "https" else 80
    authority = f"[{host}]" if bracketed else host
    if port is not None and port != default_port:
        authority = f"{authority}:{port}"
    path = "" if parts.path == "/" else parts.path
    query = f"?{parts.query}" if parts.query else ""
    return f"{scheme}://{authority}{path}{query}"


def is_loopback_host(host: str | None) -> bool:
    """Whether a URL host is a loopback name or address."""
    if host is None:
        return False
    if host.lower() in _LOOPBACK_NAMES:
        return True
    try:
        return ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False


def validate_url(url: object, *, allow_insecure: bool, issuer: bool = False) -> str:
    """Validate a URL before any I/O; return its canonical form.

    https is required unless the host is loopback or ``allow_insecure``.
    Issuers additionally may not carry a query.
    """
    canonical = canonical_url(url)
    parts = urlsplit(canonical)
    if issuer and parts.query:
        raise URLRejectedError
    if parts.scheme != "https" and not allow_insecure and not is_loopback_host(parts.hostname):
        raise URLRejectedError(insecure=True)
    return canonical


def origin_of(url: str) -> str:
    """``scheme://host[:port]`` of a validated URL, in canonical form."""
    parts = urlsplit(canonical_url(url))
    return f"{parts.scheme}://{parts.netloc}"


@dataclass(frozen=True, slots=True)
class OAuthResponse:
    """A fully read, already closed OAuth HTTP response."""

    status: int
    headers: httpx2.Headers
    body: bytes


class OAuthTransportError(Exception):
    """A value-free OAuth transport failure, classified at the boundary.

    ``outcome`` is ``unsent`` when the request provably never reached a
    request handler, else ``uncertain``. ``cause`` is a class name.
    """

    def __init__(self, outcome: SendOutcome, cause: str, *, denied: bool = False) -> None:
        self.outcome: SendOutcome = outcome
        self.cause = cause
        self.denied = denied
        super().__init__(f"OAuth request failed ({outcome}).")


class SendAttempt:
    """Whether a send call began, for classifying a cancellation."""

    __slots__ = ("started",)

    def __init__(self) -> None:
        self.started = False


class _BodyTooLargeError(Exception):
    pass


class OAuthHTTP:
    """Owner of one pool-backed transport for OAuth requests."""

    def __init__(self, *, policy: MCPDestinationPolicy | None, timeout: float) -> None:
        self._policy = policy
        self._timeout = timeout
        self._transport = _PolicyTransport(policy)
        self._checked_hosts: set[tuple[str, int]] = set()
        self._closing: set[asyncio.Task[None]] = set()
        self._closed = False

    async def _precheck(self, url: httpx2.URL) -> None:
        """Consult the destination policy once per host before its first request.

        The socket-level check in the transport remains authoritative; this
        fails fast before any byte is sent when no address is permitted.
        """
        policy = self._policy
        if policy is None:
            return
        port = url.port or (443 if url.scheme == "https" else 80)
        host = url.host
        if (host, port) in self._checked_hosts:
            return
        try:
            addresses = [ip_address(host)]
        except ValueError:
            try:
                answers = await asyncio.get_running_loop().getaddrinfo(
                    host, port, family=socket.AF_UNSPEC, type=socket.SOCK_STREAM
                )
            except OSError:
                raise httpx2.ConnectError("resolution failed") from None
            addresses = [ip_address(str(answer[4][0])) for answer in answers]
        allowed = False
        for address in dict.fromkeys(addresses):
            if isinstance(address, IPv6Address) and address.ipv4_mapped is not None:
                address = address.ipv4_mapped
            failed = False
            try:
                verdict = await policy(MCPDestination(host, port, address))
            except Exception:
                failed = True
                verdict = False
            if failed:
                break
            if verdict is True:
                allowed = True
                break
        if not allowed:
            raise MCPDestinationDeniedError("OAuth destination denied.")
        self._checked_hosts.add((host, port))

    async def send(
        self,
        method: str,
        url: str,
        *,
        form: Mapping[str, str] | None = None,
        json_body: object = None,
        headers: Mapping[str, str] | None = None,
        attempt: SendAttempt | None = None,
    ) -> OAuthResponse:
        """Send one request and return the closed, fully read response.

        Raises :class:`OAuthTransportError` (detached, value-free) on any
        failure. A ``CancelledError`` propagates; ``attempt.started`` then
        says whether the send call had begun.
        """
        attempt = attempt if attempt is not None else SendAttempt()
        request: httpx2.Request | None = None
        try:
            canonical_url(url)
            request = httpx2.Request(
                method,
                url,
                headers={"Accept": "application/json", **(headers or {})},
                data=dict(form) if form is not None else None,
                json=json_body,
                extensions={
                    "timeout": {
                        "connect": self._timeout,
                        "read": self._timeout,
                        "write": self._timeout,
                        "pool": self._timeout,
                    }
                },
            )
        except Exception:
            request = None
        if request is None or self._closed:
            raise OAuthTransportError("unsent", "InvalidRequest")
        response: httpx2.Response | None = None
        failure: BaseException | None = None
        try:
            async with asyncio.timeout(self._timeout):
                await self._precheck(request.url)
                attempt.started = True
                response = await self._transport.handle_async_request(request)
                body = bytearray()
                async for chunk in response.aiter_bytes():
                    body.extend(chunk)
                    if len(body) > MAX_BODY_BYTES:
                        raise _BodyTooLargeError
                status = response.status_code
                response_headers = response.headers
        except Exception as exc:
            failure = exc
        finally:
            if response is not None:
                await self._close_response(response)
        if failure is None:
            return OAuthResponse(status, response_headers, bytes(body))
        # Classified on the exception the transport actually raised, then
        # raised outside the handler so third-party text never chains.
        denied = isinstance(failure, MCPDestinationDeniedError)
        outcome: SendOutcome = (
            "unsent" if not attempt.started or isinstance(failure, _UNSENT_ERRORS) else "uncertain"
        )
        cause = (
            "BodyTooLarge" if isinstance(failure, _BodyTooLargeError) else type(failure).__name__
        )
        failure = None
        raise OAuthTransportError(outcome, cause, denied=denied)

    async def _close_response(self, response: httpx2.Response) -> None:
        """Close a response within ``timeout``; a slow close finishes in the background.

        The close runs as its own task, so neither a stalled server nor the
        caller's cancellation can abandon it or hold ``send()`` past its bound.
        """

        async def close() -> None:
            try:
                await response.aclose()
            except Exception:
                logger.debug("OAuth response close failed")

        task = asyncio.get_running_loop().create_task(close())
        self._closing.add(task)
        task.add_done_callback(self._closing.discard)
        await asyncio.wait({task}, timeout=self._timeout)

    async def aclose(self) -> None:
        """Close the transport; later sends fail as unsent.

        Response closes still running get one more ``timeout`` to finish.
        """
        self._closed = True
        if self._closing:
            await asyncio.wait(set(self._closing), timeout=self._timeout)
        try:
            await self._transport.aclose()
        except Exception:
            logger.debug("OAuth transport close failed")
