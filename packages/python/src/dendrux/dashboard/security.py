"""Request guards for the dashboard: Host check, CORS origins, bearer token.

The dashboard serves run payloads (prompts, tool results, raw provider
traffic), so it is closed by default:

- a ``Host`` check rejects DNS-rebinding requests (a hostile page that
  re-points its own domain at ``127.0.0.1``);
- cross-origin access is off unless origins are listed explicitly;
- a bearer token is mandatory when the server is bound beyond loopback.

Everything here is pure ASGI (no ``BaseHTTPMiddleware``) and works on the raw
header bytes, so a malformed header can never raise.
"""

from __future__ import annotations

import hmac
import ipaddress
import json
import logging
import os
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterable, Mapping, MutableMapping

    Scope = MutableMapping[str, Any]
    Receive = Callable[[], Awaitable[MutableMapping[str, Any]]]
    Send = Callable[[MutableMapping[str, Any]], Awaitable[None]]

__all__ = [
    "TOKEN_ENV",
    "DashboardGuard",
    "bearer_matches",
    "is_loopback_host",
    "normalize_origins",
    "resolve_token",
]

logger = logging.getLogger("dendrux.dashboard")

TOKEN_ENV = "DENDRUX_DASHBOARD_TOKEN"
ANY_HOST = "*"
_WILDCARD_BINDS = frozenset({"", "0.0.0.0", "::", "[::]"})


def is_loopback_host(host: str) -> bool:
    """True when binding to ``host`` exposes the server to this machine only."""
    name = host.strip().strip("[]").lower()
    if not name or name in _WILDCARD_BINDS:
        return False
    if name == "localhost" or name.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(name).is_loopback
    except ValueError:
        return False


def resolve_token(cli_token: str | None, environ: Mapping[str, str] | None = None) -> str | None:
    """The configured token: the explicit value if given, else ``DENDRUX_DASHBOARD_TOKEN``.

    An empty string counts as "not configured" so an empty variable can never
    switch the check into "any empty Authorization header matches".
    """
    env = os.environ if environ is None else environ
    for candidate in (cli_token, env.get(TOKEN_ENV)):
        if candidate:
            return candidate
    return None


def normalize_origins(origins: Iterable[str]) -> list[str]:
    """Validate ``scheme://host[:port]`` origins; reject wildcards and paths."""
    cleaned: list[str] = []
    for raw in origins:
        origin = raw.strip().rstrip("/")
        parts = urlsplit(origin)
        if (
            "*" in origin
            or parts.scheme not in {"http", "https"}
            or not parts.netloc
            or parts.path
            or parts.query
            or parts.fragment
        ):
            raise ValueError(
                "allowed origins must look like 'https://host[:port]'; wildcards and paths "
                f"are not allowed (got {raw!r})."
            )
        cleaned.append(origin)
    return cleaned


def _host_only(header: str) -> str:
    """Hostname from a ``Host`` header value: no port, no IPv6 brackets, lower case."""
    value = header.strip().lower()
    if value.startswith("["):
        end = value.find("]")
        return value[1:end] if end != -1 else value
    return value.rsplit(":", 1)[0] if value.count(":") == 1 else value


def _is_ip_literal(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
    except ValueError:
        return False
    return True


def bearer_matches(headers: Iterable[tuple[bytes, bytes]], expected: bytes | None) -> bool:
    """True when ``expected`` (``b"Bearer <token>"``) matches the first Authorization header.

    Works on raw header bytes with a constant-time compare, so a non-ASCII
    value is simply a mismatch. ``expected=None`` means no token is configured.
    """
    if expected is None:
        return True
    for key, value in headers:
        if key == b"authorization":
            return hmac.compare_digest(value, expected)
    return False


def expected_bearer(token: str | None) -> bytes | None:
    """The exact ``Authorization`` header bytes a client must send, or None."""
    return None if not token else b"Bearer " + token.encode("utf-8")


class DashboardGuard:
    """ASGI middleware: Host check on every request, bearer token on the API.

    The token decision is made on the **app-relative** path, computed the way
    Starlette's router computes it, so a parent application's mount prefix
    (for example ``/api/dashboard``) never counts as part of the API. The API
    routes also enforce the token themselves (a router dependency in
    ``create_dashboard_api``); this middleware is the first line, the
    dependency the authoritative one.

    An IP-literal ``Host`` is always accepted (DNS rebinding needs a DNS name);
    ``localhost`` names are accepted; any other name must be in
    ``allowed_hosts``. ``allowed_hosts=["*"]`` switches the Host check off, for
    embedders whose own gateway already validates it.
    """

    def __init__(
        self,
        app: Callable[[Scope, Receive, Send], Awaitable[None]],
        *,
        allowed_hosts: Iterable[str] = (),
        token: str | None = None,
    ) -> None:
        self._app = app
        hosts = {h.strip().lower().strip("[]") for h in allowed_hosts if h.strip()}
        self._any_host = ANY_HOST in hosts
        self._hosts = frozenset(hosts)
        self._expected = expected_bearer(token)

    def _host_ok(self, headers: list[tuple[bytes, bytes]]) -> bool:
        if self._any_host:
            return True
        for key, value in headers:
            if key == b"host":
                host = _host_only(value.decode("latin-1"))
                break
        else:
            return False
        return is_loopback_host(host) or _is_ip_literal(host) or host in self._hosts

    @staticmethod
    def _route_path(scope: Scope) -> str:
        """The path Starlette's router will match, computed the way Starlette does.

        ``root_path`` is stripped only when the path starts with it **on a
        ``/`` boundary**; otherwise the path is used unchanged (never assumed
        to be already stripped).
        """
        path: str = scope.get("path", "")
        root: str = scope.get("root_path", "")
        if not root or not path.startswith(root):
            return path
        if path == root:
            return ""
        return path[len(root) :] if path[len(root)] == "/" else path

    @classmethod
    def _is_api(cls, scope: Scope) -> bool:
        """True when the app-relative path is under ``/api``.

        Checks Starlette's own route path and, to fail closed on a trailing-slash
        ``root_path``, the path stripped with the slash removed from the root.
        """
        candidates = {cls._route_path(scope)}
        root: str = scope.get("root_path", "")
        if root.endswith("/") and root != "/":
            candidates.add(cls._route_path({**scope, "root_path": root.rstrip("/")}))
        return any(p == "/api" or p.startswith("/api/") for p in candidates)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in {"http", "websocket"}:
            await self._app(scope, receive, send)
            return

        headers: list[tuple[bytes, bytes]] = list(scope.get("headers") or [])
        if not self._host_ok(headers):
            logger.warning("dashboard.host_rejected")
            await self._reject(scope, send, 403, "Host not allowed.")
            return

        needs_token = self._is_api(scope)
        if scope["type"] == "websocket":
            needs_token = True
        if (
            needs_token
            and scope.get("method") != "OPTIONS"
            and self._expected is not None
            and not bearer_matches(headers, self._expected)
        ):
            logger.warning("dashboard.auth_rejected")
            await self._reject(
                scope, send, 401, "Invalid or missing auth token.", authenticate=True
            )
            return

        await self._app(scope, receive, send)

    @staticmethod
    async def _reject(
        scope: Scope, send: Send, status: int, detail: str, *, authenticate: bool = False
    ) -> None:
        if scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": 1008})
            return
        body = json.dumps({"detail": detail}).encode()
        headers = [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode()),
        ]
        if authenticate:
            headers.append((b"www-authenticate", b"Bearer"))
        await send({"type": "http.response.start", "status": status, "headers": headers})
        await send({"type": "http.response.body", "body": body})
