"""Real loopback servers for MCP OAuth tests.

``RawHTTPServer`` is a raw-socket HTTP/1.1 server whose responses are written
byte for byte, so tests can control reason phrases, trickle bodies, and
dropped connections that uvicorn would normalize. ``OAuthHarness`` runs an
authorization server (the SDK's own auth routes over an in-memory provider)
and an MCP resource server (the SDK's ``MCPServer`` with bearer auth) on two
uvicorn instances with different ports.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import secrets
import socket
import time
import warnings
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import parse_qs, urlencode, urlsplit

import httpx2

Handler = Callable[["RawRequest", asyncio.StreamWriter], Awaitable[None]]


@dataclass
class RawRequest:
    method: str
    path: str
    headers: dict[str, str]
    body: bytes

    @property
    def form(self) -> dict[str, str]:
        return {key: values[0] for key, values in parse_qs(self.body.decode()).items()}


def raw_response(
    status: int,
    body: bytes = b"",
    *,
    reason: str = "OK",
    headers: dict[str, str] | None = None,
) -> bytes:
    lines = [f"HTTP/1.1 {status} {reason}"]
    merged = {"Content-Length": str(len(body)), "Connection": "close", **(headers or {})}
    lines.extend(f"{name}: {value}" for name, value in merged.items())
    return ("\r\n".join(lines) + "\r\n\r\n").encode("latin-1") + body


class RawHTTPServer:
    """Loopback HTTP server whose handler writes raw bytes."""

    def __init__(self, handler: Handler | None = None) -> None:
        self.handler = handler
        self.requests: list[RawRequest] = []
        self.connections = 0
        self._server: asyncio.base_events.Server | None = None
        self.port = 0

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    async def __aenter__(self) -> RawHTTPServer:
        self._server = await asyncio.start_server(self._serve, "127.0.0.1", 0)
        self.port = self._server.sockets[0].getsockname()[1]
        return self

    async def __aexit__(self, *exc: object) -> None:
        assert self._server is not None
        self._server.close()
        with contextlib.suppress(Exception):
            await asyncio.wait_for(self._server.wait_closed(), 1.0)

    async def _serve(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self.connections += 1
        try:
            head = await reader.readuntil(b"\r\n\r\n")
            lines = head.decode("latin-1").split("\r\n")
            method, path, _ = lines[0].split(" ", 2)
            headers: dict[str, str] = {}
            for line in lines[1:]:
                if ":" in line:
                    name, value = line.split(":", 1)
                    headers[name.strip().lower()] = value.strip()
            length = int(headers.get("content-length", "0"))
            body = await reader.readexactly(length) if length else b""
            request = RawRequest(method, path, headers, body)
            self.requests.append(request)
            if self.handler is not None:
                await self.handler(request, writer)
            with contextlib.suppress(Exception):
                await writer.drain()
        except (asyncio.IncompleteReadError, ConnectionError):
            pass
        finally:
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def scan_chain(error: BaseException) -> str:
    """Render an exception, its chains, and groups for sentinel scans."""
    seen: set[int] = set()
    parts: list[str] = []
    stack: list[BaseException] = [error]
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        attributes = vars(node) if hasattr(node, "__dict__") else ""
        parts.append(f"{type(node).__name__}: {node!s} {node!r} {attributes}")
        if isinstance(node, BaseExceptionGroup):
            stack.extend(node.exceptions)
        for linked in (node.__cause__, node.__context__):
            if linked is not None:
                stack.append(linked)
    return "\n".join(parts)


# --------------------------------------------------------------------------
# Authorization server + resource server harness
# --------------------------------------------------------------------------


@dataclass
class ASConfig:
    access_token_lifetime: int | None = 3600
    issue_refresh_tokens: bool = True
    rotate_refresh_tokens: bool = True
    keep_access_token_string: bool = False
    reuse_detection: bool = False
    include_iss: bool = True
    iss_override: str | None = None
    advertise_iss: bool = False
    refresh_latency: float = 0.0
    refresh_barrier: asyncio.Barrier | None = None
    refresh_gate: asyncio.Event | None = None
    refresh_entered: asyncio.Event | None = None
    consume_then_drop: bool = False
    consume_then_status: int | None = None
    token_status_override: int | None = None
    token_body_override: bytes | None = None
    revocation_status: int = 200
    echo_sentinel: str | None = None


class InMemoryAS:
    """In-memory ``OAuthAuthorizationServerProvider`` with test hooks."""

    def __init__(self, config: ASConfig) -> None:
        from mcp.server.auth.provider import (
            AccessToken,
            AuthorizationCode,
            RefreshToken,
        )

        self._AccessToken = AccessToken
        self._AuthorizationCode = AuthorizationCode
        self._RefreshToken = RefreshToken
        self.config = config
        self.issuer = ""
        self.clients: dict[str, Any] = {}
        self.codes: dict[str, Any] = {}
        self.access_tokens: dict[str, Any] = {}
        self.refresh_tokens: dict[str, Any] = {}
        self.consumed_refresh: set[str] = set()
        self.families: dict[str, str] = {}
        self.revoked_families: set[str] = set()
        self.counters: dict[str, int] = {
            "authorize": 0,
            "token": 0,
            "code_exchange": 0,
            "refresh": 0,
            "refresh_redeemed": 0,
            "refresh_replayed": 0,
            "register": 0,
            "revoke": 0,
            "prm": 0,
            "as_metadata": 0,
        }
        self.token_request_headers: list[dict[str, str]] = []
        self.revocation_requests: list[dict[str, str]] = []
        self.registration_requests: list[dict[str, Any]] = []

    async def get_client(self, client_id: str) -> Any:
        return self.clients.get(client_id)

    async def register_client(self, client_info: Any) -> None:
        self.counters["register"] += 1
        self.clients[client_info.client_id] = client_info

    async def authorize(self, client: Any, params: Any) -> str:
        from mcp.server.auth.provider import construct_redirect_uri

        self.counters["authorize"] += 1
        code = secrets.token_urlsafe(24)
        self.codes[code] = self._AuthorizationCode(
            code=code,
            scopes=params.scopes or [],
            expires_at=time.time() + 300,
            client_id=client.client_id,
            code_challenge=params.code_challenge,
            redirect_uri=params.redirect_uri,
            redirect_uri_provided_explicitly=params.redirect_uri_provided_explicitly,
            resource=params.resource,
        )
        iss = self.config.iss_override if self.config.iss_override is not None else self.issuer
        return construct_redirect_uri(
            str(params.redirect_uri),
            code=code,
            state=params.state,
            iss=iss if self.config.include_iss else None,
        )

    async def load_authorization_code(self, client: Any, code: str) -> Any:
        return self.codes.get(code)

    def _mint(self, client_id: str, scopes: list[str], resource: str | None, family: str) -> Any:
        from mcp.shared.auth import OAuthToken

        access = "at-" + secrets.token_urlsafe(16)
        lifetime = self.config.access_token_lifetime
        self.access_tokens[access] = self._AccessToken(
            token=access,
            client_id=client_id,
            scopes=scopes,
            expires_at=int(time.time() + lifetime) if lifetime is not None else None,
            resource=resource,
        )
        self.families[access] = family
        refresh = None
        if self.config.issue_refresh_tokens:
            refresh = "rt-" + secrets.token_urlsafe(16)
            self.refresh_tokens[refresh] = self._RefreshToken(
                token=refresh, client_id=client_id, scopes=scopes, resource=resource
            )
            self.families[refresh] = family
        return OAuthToken(
            access_token=access,
            token_type="Bearer",
            expires_in=lifetime,
            scope=" ".join(scopes) or None,
            refresh_token=refresh,
        )

    async def exchange_authorization_code(self, client: Any, code: Any) -> Any:
        self.counters["code_exchange"] += 1
        self.codes.pop(code.code, None)
        return self._mint(client.client_id, code.scopes, code.resource, secrets.token_hex(8))

    async def load_refresh_token(self, client: Any, refresh_token: str) -> Any:
        if refresh_token in self.consumed_refresh:
            self.counters["refresh_replayed"] += 1
            if self.config.reuse_detection:
                family = self.families.get(refresh_token)
                if family is not None:
                    self.revoked_families.add(family)
            return None
        token = self.refresh_tokens.get(refresh_token)
        if token is None:
            return None
        if self.families.get(refresh_token) in self.revoked_families:
            return None
        return token

    async def exchange_refresh_token(
        self, client: Any, refresh_token: Any, scopes: list[str]
    ) -> Any:
        cfg = self.config
        self.counters["refresh_redeemed"] += 1
        if cfg.refresh_entered is not None:
            cfg.refresh_entered.set()
        if cfg.refresh_barrier is not None:
            await cfg.refresh_barrier.wait()
        if cfg.refresh_gate is not None:
            await cfg.refresh_gate.wait()
        if cfg.refresh_latency:
            await asyncio.sleep(cfg.refresh_latency)
        family = self.families.get(refresh_token.token, secrets.token_hex(8))
        if cfg.rotate_refresh_tokens:
            self.consumed_refresh.add(refresh_token.token)
            self.refresh_tokens.pop(refresh_token.token, None)
        if cfg.consume_then_drop:
            await asyncio.sleep(3600)
        token = self._mint(client.client_id, scopes, refresh_token.resource, family)
        if not cfg.rotate_refresh_tokens:
            self.refresh_tokens.pop(token.refresh_token, None)
            token.refresh_token = None
        if cfg.keep_access_token_string:
            previous = next(
                (t for t, fam in self.families.items() if fam == family and t.startswith("at-")),
                None,
            )
            if previous is not None and previous != token.access_token:
                self.access_tokens[previous] = self.access_tokens.pop(token.access_token)
                self.access_tokens[previous].token = previous
                token.access_token = previous
        return token

    async def load_access_token(self, token: str) -> Any:
        access = self.access_tokens.get(token)
        if access is None:
            return None
        if self.families.get(token) in self.revoked_families:
            return None
        if access.expires_at is not None and access.expires_at < time.time():
            return None
        return access

    async def revoke_token(self, token: Any) -> None:
        family = self.families.get(token.token)
        if family is not None:
            self.revoked_families.add(family)

    def expire_access_tokens(self) -> None:
        for access in self.access_tokens.values():
            access.expires_at = int(time.time()) - 10


class _Middleware:
    def __init__(self, app: Any, hook: Callable[..., Awaitable[bool]]) -> None:
        self.app = app
        self.hook = hook

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        chunks: list[bytes] = []
        more = True
        while more:
            message = await receive()
            chunks.append(message.get("body", b""))
            more = message.get("more_body", False)
        body = b"".join(chunks)
        replayed = False

        async def replay() -> dict[str, Any]:
            nonlocal replayed
            if replayed:
                return await receive()
            replayed = True
            return {"type": "http.request", "body": body, "more_body": False}

        if await self.hook(scope, body, send):
            return
        await self.app(scope, replay, send)


async def _send_json(
    send: Any, status: int, payload: Any, headers: list[tuple[bytes, bytes]] | None = None
) -> None:
    body = json.dumps(payload).encode()
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [(b"content-type", b"application/json"), *(headers or [])],
        }
    )
    await send({"type": "http.response.body", "body": body})


@dataclass
class OAuthHarness:
    """An authorization server and an MCP resource server on loopback."""

    config: ASConfig = field(default_factory=ASConfig)
    as_port: int = 0
    rs_port: int = 0
    provider: InMemoryAS | None = None
    prm_override: dict[str, Any] | None = None
    as_metadata_override: dict[str, Any] | None = None
    as_metadata_status: int | None = None
    prm_status: int | None = None
    challenge_override: str | None = None
    preregistered: dict[str, Any] = field(default_factory=dict)
    rs_tool_calls: int = 0
    rs_auth_headers: list[list[str]] = field(default_factory=list)
    _tasks: list[asyncio.Task[None]] = field(default_factory=list)
    _servers: list[Any] = field(default_factory=list)

    @property
    def as_url(self) -> str:
        return f"http://127.0.0.1:{self.as_port}"

    @property
    def rs_url(self) -> str:
        return f"http://127.0.0.1:{self.rs_port}/mcp"

    @property
    def counters(self) -> dict[str, int]:
        assert self.provider is not None
        return self.provider.counters

    def preregister(
        self,
        client_id: str,
        secret: str | None,
        method: str,
        *,
        scope: str = "mcp admin",
        redirect_uris: list[str] | None = None,
    ) -> None:
        from mcp.shared.auth import OAuthClientInformationFull

        assert self.provider is not None
        self.provider.clients[client_id] = OAuthClientInformationFull(
            client_id=client_id,
            client_secret=secret,
            token_endpoint_auth_method=method,
            redirect_uris=redirect_uris or ["http://127.0.0.1:9/callback"],
            scope=scope,
            grant_types=["authorization_code", "refresh_token"],
        )

    def _as_app(self) -> Any:
        from mcp.server.auth.routes import create_auth_routes
        from mcp.server.auth.settings import ClientRegistrationOptions, RevocationOptions
        from mcp.shared.auth import OAuthMetadata
        from starlette.applications import Starlette

        assert self.provider is not None
        provider = self.provider
        routes = create_auth_routes(
            provider=provider,
            issuer_url=self.as_url,  # type: ignore[arg-type]  # pydantic coerces the str
            client_registration_options=ClientRegistrationOptions(
                enabled=True, valid_scopes=["mcp", "admin"], default_scopes=["mcp"]
            ),
            revocation_options=RevocationOptions(enabled=True),
        )
        inner = Starlette(routes=routes)
        harness = self

        async def hook(scope: dict[str, Any], body: bytes, send: Any) -> bool:
            path = scope["path"]
            headers = {k.decode().lower(): v.decode() for k, v in scope.get("headers", [])}
            cfg = provider.config
            if path.startswith("/.well-known/oauth-authorization-server") or path.startswith(
                "/.well-known/openid-configuration"
            ):
                provider.counters["as_metadata"] += 1
                if harness.as_metadata_status is not None:
                    await _send_json(send, harness.as_metadata_status, {"error": "nope"})
                    return True
                if path != "/.well-known/oauth-authorization-server":
                    await _send_json(send, 404, {"error": "not_found"})
                    return True
                base = f"{harness.as_url}"
                meta: dict[str, Any] = {
                    "issuer": base,
                    "authorization_endpoint": f"{base}/authorize",
                    "token_endpoint": f"{base}/token",
                    "registration_endpoint": f"{base}/register",
                    "revocation_endpoint": f"{base}/revoke",
                    "response_types_supported": ["code"],
                    "grant_types_supported": ["authorization_code", "refresh_token"],
                    "code_challenge_methods_supported": ["S256"],
                    "token_endpoint_auth_methods_supported": [
                        "none",
                        "client_secret_post",
                        "client_secret_basic",
                    ],
                    "scopes_supported": ["mcp", "admin"],
                }
                if cfg.advertise_iss:
                    meta["authorization_response_iss_parameter_supported"] = True
                if cfg.echo_sentinel:
                    meta["service_documentation"] = f"https://docs.example/{cfg.echo_sentinel}"
                    meta["op_policy_uri"] = f"https://x.example/?t={cfg.echo_sentinel}"
                if harness.as_metadata_override is not None:
                    meta.update(harness.as_metadata_override)
                    meta = {k: v for k, v in meta.items() if v is not None}
                OAuthMetadata.model_validate(meta)  # harness sanity
                await _send_json(send, 200, meta)
                return True
            if path == "/token":
                provider.counters["token"] += 1
                provider.token_request_headers.append(headers)
                form = {k: v[0] for k, v in parse_qs(body.decode()).items()}
                if form.get("grant_type") == "refresh_token":
                    provider.counters["refresh"] += 1
                if cfg.token_status_override is not None:
                    payload = (
                        cfg.token_body_override
                        or json.dumps(
                            {"error": "server_error", "error_description": cfg.echo_sentinel or "x"}
                        ).encode()
                    )
                    await send(
                        {
                            "type": "http.response.start",
                            "status": cfg.token_status_override,
                            "headers": [(b"content-type", b"application/json")],
                        }
                    )
                    await send({"type": "http.response.body", "body": payload})
                    return True
                if (
                    cfg.consume_then_status is not None
                    and form.get("grant_type") == "refresh_token"
                ):
                    token = form.get("refresh_token", "")
                    if token in provider.refresh_tokens:
                        provider.counters["refresh_redeemed"] += 1
                        provider.consumed_refresh.add(token)
                        provider.refresh_tokens.pop(token, None)
                    elif token in provider.consumed_refresh:
                        provider.counters["refresh_replayed"] += 1
                        if cfg.reuse_detection:
                            fam = provider.families.get(token)
                            if fam is not None:
                                provider.revoked_families.add(fam)
                    await _send_json(send, cfg.consume_then_status, {"error": "bad_gateway"})
                    return True
                return False
            if path == "/register":
                provider.registration_requests.append(json.loads(body or b"{}"))
                return False
            if path == "/revoke":
                provider.counters["revoke"] += 1
                form = {k: v[0] for k, v in parse_qs(body.decode()).items()}
                provider.revocation_requests.append(
                    {**form, **{"_auth": headers.get("authorization", "")}}
                )
                if cfg.revocation_status != 200:
                    await _send_json(send, cfg.revocation_status, {"error": "unavailable"})
                    return True
                # The SDK RevocationRequest requires client_secret; accept a
                # public client that sends client_id alone.
                client = provider.clients.get(form.get("client_id", ""))
                if client is not None and client.token_endpoint_auth_method == "none":
                    token = form.get("token", "")
                    loaded = provider.refresh_tokens.get(token) or provider.access_tokens.get(token)
                    if loaded is not None and loaded.client_id == client.client_id:
                        await provider.revoke_token(loaded)
                    await send({"type": "http.response.start", "status": 200, "headers": []})
                    await send({"type": "http.response.body", "body": b""})
                    return True
                return False
            if path == "/authorize":
                return False
            return False

        return _Middleware(inner, hook)

    def _rs_app(self) -> Any:
        from mcp.server.auth.provider import ProviderTokenVerifier
        from mcp.server.auth.settings import AuthSettings
        from mcp.server.mcpserver import MCPServer as SDKServer

        assert self.provider is not None
        provider = self.provider
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            server = SDKServer(
                name="oauth-rs",
                auth=AuthSettings(
                    issuer_url=self.as_url,  # type: ignore[arg-type]  # pydantic coerces the str
                    resource_server_url=self.rs_url,  # type: ignore[arg-type]  # pydantic coerces
                    required_scopes=["mcp"],
                    validate_token_resource=True,
                ),
                token_verifier=ProviderTokenVerifier(provider),  # type: ignore[arg-type]  # test provider
            )
        harness = self

        @server.tool()
        def echo(text: str) -> str:
            harness.rs_tool_calls += 1
            return text

        @server.tool()
        def admin_echo(text: str) -> str:
            harness.rs_tool_calls += 1
            return f"admin:{text}"

        inner = server.streamable_http_app()

        async def hook(scope: dict[str, Any], body: bytes, send: Any) -> bool:
            path = scope["path"]
            raw_headers = scope.get("headers", [])
            auths = [v.decode() for k, v in raw_headers if k.decode().lower() == "authorization"]
            if path == "/mcp":
                harness.rs_auth_headers.append(auths)
            if path.startswith("/.well-known/oauth-protected-resource"):
                provider.counters["prm"] += 1
                if harness.prm_status is not None:
                    await _send_json(send, harness.prm_status, {"error": "x"})
                    return True
                if path != "/.well-known/oauth-protected-resource/mcp":
                    await _send_json(send, 404, {"error": "not_found"})
                    return True
                prm: dict[str, Any] = {
                    "resource": harness.rs_url,
                    "authorization_servers": [harness.as_url],
                    "scopes_supported": ["mcp"],
                    "bearer_methods_supported": ["header"],
                }
                if harness.prm_override is not None:
                    prm.update(harness.prm_override)
                await _send_json(send, 200, prm)
                return True
            if (
                path == "/mcp"
                and scope["method"] == "POST"
                and harness.challenge_override
                and not auths
            ):
                await _send_json(
                    send,
                    401,
                    {"error": "invalid_token"},
                    [(b"www-authenticate", harness.challenge_override.encode())],
                )
                return True
            if path == "/mcp" and scope["method"] == "POST" and body:
                try:
                    message = json.loads(body)
                except ValueError:
                    return False
                if (
                    isinstance(message, dict)
                    and message.get("method") == "tools/call"
                    and message.get("params", {}).get("name") == "admin_echo"
                ):
                    token = auths[0][7:] if auths and auths[0].lower().startswith("bearer ") else ""
                    access = await provider.load_access_token(token)
                    if access is not None and "admin" not in access.scopes:
                        await _send_json(
                            send,
                            403,
                            {"error": "insufficient_scope"},
                            [
                                (
                                    b"www-authenticate",
                                    b'Bearer error="insufficient_scope", scope="admin", '
                                    b'error_description="needs admin"',
                                )
                            ],
                        )
                        return True
            return False

        return _Middleware(inner, hook)

    async def __aenter__(self) -> OAuthHarness:
        import uvicorn

        self.provider = InMemoryAS(self.config)
        socks = []
        for _ in range(2):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("127.0.0.1", 0))
            sock.listen(64)
            socks.append(sock)
        self.as_port = socks[0].getsockname()[1]
        self.rs_port = socks[1].getsockname()[1]
        self.provider.issuer = self.as_url
        for app, sock in ((self._as_app(), socks[0]), (self._rs_app(), socks[1])):
            server = uvicorn.Server(uvicorn.Config(app, log_level="warning", lifespan="on"))
            task = asyncio.create_task(server.serve(sockets=[sock]))
            self._servers.append((server, task, sock))
        for server, task, _ in self._servers:
            while not server.started:
                if task.done():
                    task.result()
                await asyncio.sleep(0.01)
        for client_id, spec in self.preregistered.items():
            self.preregister(client_id, *spec)
        return self

    async def __aexit__(self, *exc: object) -> None:
        for server, _, _ in self._servers:
            server.should_exit = True
            server.force_exit = True
        for _, task, sock in self._servers:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(task, timeout=5.0)
            sock.close()

    async def browser(self, authorization_url: str) -> dict[str, str | None]:
        """Follow the authorization URL like a consenting user's browser.

        Uses a bare transport (never ``httpx2.AsyncClient``) and returns the
        callback's ``code``, ``state``, ``iss``, and ``error`` parameters.
        """
        transport = httpx2.AsyncHTTPTransport()
        try:
            response = await transport.handle_async_request(
                httpx2.Request("GET", authorization_url)
            )
            await response.aread()
        finally:
            await transport.aclose()
        location = response.headers.get("location")
        assert response.status_code in (302, 303, 307), (
            response.status_code,
            response.content[:200],
        )
        assert location is not None
        query = parse_qs(urlsplit(location).query)

        def one(name: str) -> str | None:
            values = query.get(name)
            return values[0] if values else None

        return {name: one(name) for name in ("code", "state", "iss", "error")}


def basic_header(client_id: str, secret: str) -> str:
    return "Basic " + base64.b64encode(f"{client_id}:{secret}".encode()).decode()


@contextlib.asynccontextmanager
async def harness(**kwargs: Any) -> AsyncIterator[OAuthHarness]:
    async with OAuthHarness(**kwargs) as running:
        yield running


__all__ = [
    "ASConfig",
    "OAuthHarness",
    "RawHTTPServer",
    "RawRequest",
    "basic_header",
    "free_port",
    "harness",
    "raw_response",
    "scan_chain",
    "urlencode",
]
