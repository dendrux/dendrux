"""OAuth 2.1 sign-in and token lifecycle for remote MCP servers.

Dendrux owns the protocol (discovery, registration, PKCE, code exchange,
refresh, revocation) and the token lifecycle. The application owns durable
storage through :class:`MCPOAuthStore`, user identity through
:class:`MCPAuthKey`, and the browser redirect.

Every value read from a server is untrusted. Identity strings (issuer,
resource, endpoints) are read from the raw JSON, validated, and compared as
raw strings before any SDK model is built. SDK, HTTP, pydantic, and store
exceptions are translated outside their handlers so third-party text never
reaches an exception chain.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import math
import re
import secrets
import time

# Public annotations are resolved at runtime by documentation tooling.
from collections.abc import Awaitable, Callable, Coroutine, Mapping, Sequence  # noqa: TC003
from dataclasses import dataclass, replace
from functools import partial
from typing import Any, Literal, Protocol, TypeVar, runtime_checkable
from urllib.parse import quote, urlencode, urlsplit

from mcp.client.auth.oauth2 import PKCEParameters, check_registration_usable
from mcp.client.auth.utils import (
    build_oauth_authorization_server_metadata_discovery_urls,
    build_protected_resource_metadata_discovery_urls,
)
from mcp.shared.auth import (
    OAuthClientInformationFull,
    OAuthClientMetadata,
    OAuthMetadata,
    ProtectedResourceMetadata,
)
from mcp.shared.inbound import MCP_PROTOCOL_VERSION_HEADER
from mcp.types import LATEST_PROTOCOL_VERSION

from dendrux.mcp._challenge import challenge_from_headers
from dendrux.mcp._credentials import MCPAuthKey, MCPRefreshingCredentialProvider
from dendrux.mcp._destination import MCPDestinationPolicy  # noqa: TC001
from dendrux.mcp._errors import (
    MCPAuthChallenge,
    MCPAuthorizationReason,
    MCPAuthorizationRequired,
    MCPOAuthError,
    MCPOAuthFlowError,
    MCPOAuthFlowReason,
    MCPOAuthRefreshError,
)
from dendrux.mcp._oauth_http import (
    OAuthHTTP,
    OAuthResponse,
    OAuthTransportError,
    SendAttempt,
    URLRejectedError,
    canonical_url,
    origin_of,
    validate_url,
)
from dendrux.mcp._observability import (
    MCPOAuthRefreshed,
    MCPOAuthRefreshFailed,
    MCPRuntimeEvent,
)
from dendrux.mcp._source import MCPSource

logger = logging.getLogger(__name__)

_T = TypeVar("_T")

_AUTH_METHODS = ("none", "client_secret_post", "client_secret_basic")
_SECRET_METHODS = ("client_secret_post", "client_secret_basic")
_REJECTED_ERRORS = frozenset(
    {
        "invalid_grant",
        "invalid_client",
        "invalid_request",
        "invalid_scope",
        "unauthorized_client",
        "unsupported_grant_type",
    }
)
_SCOPE_TOKEN = re.compile(r"[\x21\x23-\x5b\x5d-\x7e]{1,128}")
_ACCESS_TOKEN = re.compile(r"[\x21-\x7e]{1,8192}")
_REFRESH_TOKEN = re.compile(r"[\x20-\x7e]{1,8192}")
_EXPIRES_IN = re.compile(r"[0-9]{1,11}")
_MAX_EXPIRES_IN = 10 * 365 * 24 * 3600
_CLAIM_POLL = 0.1
_PUBLISH_ATTEMPTS = 5
_METADATA_HEADERS = {MCP_PROTOCOL_VERSION_HEADER: LATEST_PROTOCOL_VERSION}
_PROBE_HEADERS = {
    "Accept": "application/json, text/event-stream",
    "Content-Type": "application/json",
    MCP_PROTOCOL_VERSION_HEADER: LATEST_PROTOCOL_VERSION,
}

AuthMethod = Literal["none", "client_secret_post", "client_secret_basic"]


def _new_id() -> str:
    return secrets.token_hex(16)


def _plain(value: object, *, allow_empty: bool = False) -> bool:
    return (
        isinstance(value, str)
        and (allow_empty or bool(value.strip()))
        and not any(ord(char) < 32 or ord(char) == 127 for char in value)
    )


# --------------------------------------------------------------------------
# Record model
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MCPOAuthVersion:
    """Exact version of one stored authorization row.

    ``generation`` is random and minted at sign-in and sign-out only;
    ``revision`` increments on every write within a generation.
    """

    generation: str
    revision: int


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class MCPOAuthClientRecord:
    """The OAuth client a credential was issued to.

    ``issuer`` is the raw issuer string of the authorization server the
    client belongs to. ``client_secret_expires_at`` of None or 0 means the
    secret never expires.
    """

    client_id: str
    client_secret: str | None = None
    client_secret_expires_at: float | None = None
    token_endpoint_auth_method: AuthMethod
    source: Literal["preregistered", "cimd", "dcr"]
    issuer: str

    def __repr__(self) -> str:
        secret = None if self.client_secret is None else "<redacted>"
        return (
            f"MCPOAuthClientRecord(source={self.source!r}, "
            f"token_endpoint_auth_method={self.token_endpoint_auth_method!r}, "
            f"client_secret={secret!r})"
        )

    def secret_expired(self, now: float) -> bool:
        """Whether the client secret has expired at Unix time ``now``."""
        expires = self.client_secret_expires_at
        return self.client_secret is not None and bool(expires) and now >= float(expires or 0)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class MCPOAuthCredential:
    """One published access token (and refresh token) for one resource.

    Times are Unix time. ``refresh_token`` of None means the credential
    cannot be refreshed. ``publication_id`` is random per sign-in or refresh.
    """

    access_token: str
    refresh_token: str | None
    token_type: str = "Bearer"
    expires_at: float | None
    scope: str | None
    issuer: str
    resource: str
    token_endpoint: str
    revocation_endpoint: str | None
    client: MCPOAuthClientRecord
    publication_id: str
    published_at: float

    def __repr__(self) -> str:
        refresh = None if self.refresh_token is None else "<redacted>"
        return (
            f"MCPOAuthCredential(access_token='<redacted>', refresh_token={refresh!r}, "
            f"expires_at={self.expires_at!r})"
        )

    @property
    def authorization(self) -> str:
        """The exact ``Authorization`` header value for this credential."""
        return f"{self.token_type} {self.access_token}"


@dataclass(frozen=True, slots=True, kw_only=True)
class MCPOAuthClaim:
    """A worker's exclusive right to redeem the row's refresh token."""

    id: str
    expires_at: float


@dataclass(frozen=True, slots=True, kw_only=True)
class MCPOAuthRecord:
    """One authorization row per :class:`MCPAuthKey`; never deleted by Dendrux.

    ``active``: refreshable credential. ``retired``: the refresh token is
    unusable; the access token may be used until it expires. ``signed_out``:
    tombstone with no credential.
    """

    version: MCPOAuthVersion
    status: Literal["active", "signed_out", "retired"]
    credential: MCPOAuthCredential | None
    refresh_claim: MCPOAuthClaim | None
    retired_claim_id: str | None
    updated_at: float


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class MCPOAuthFlow:
    """One pending sign-in, created by ``begin_authorization``.

    Holds the PKCE verifier and ``state``; the store must treat it as secret.
    """

    state: str
    key: MCPAuthKey
    resource: str
    code_verifier: str
    redirect_uri: str
    scope: str | None
    issuer: str
    authorization_endpoint: str
    token_endpoint: str
    revocation_endpoint: str | None
    client: MCPOAuthClientRecord
    iss_required: bool
    expires_at: float
    previous_generation: str | None

    def __repr__(self) -> str:
        return f"MCPOAuthFlow(expires_at={self.expires_at!r})"


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class MCPOAuthStatus:
    """Point-in-time sign-in state for one key.

    ``signed_out``: tombstone or absent. ``active``: usable now. ``expired``:
    the access token expired and a refresh token exists (the next use
    refreshes). ``refreshing``: a live refresh claim. ``retired``: no refresh
    token, the access token is still valid. ``needs_sign_in``: retired and
    expired. Issuer, resource, and scope are server-chosen: attributes only.
    """

    state: Literal["signed_out", "active", "expired", "refreshing", "retired", "needs_sign_in"]
    scope: str | None
    expires_at: float | None
    issuer: str | None
    resource: str | None

    def __repr__(self) -> str:
        return f"MCPOAuthStatus(state={self.state!r})"


@dataclass(frozen=True, slots=True, kw_only=True)
class MCPOAuthRevocation:
    """Outcome of a sign-out: the local tombstone and the remote revocation."""

    local: Literal["signed_out", "already_signed_out"]
    remote: Literal["confirmed", "failed", "unsupported", "skipped"]


# --------------------------------------------------------------------------
# Store
# --------------------------------------------------------------------------


@runtime_checkable
class MCPOAuthStore(Protocol):
    """Application-owned durable storage for authorization rows and flows.

    ``save(expected=None)`` inserts only when no row exists; otherwise it
    replaces only when the stored version equals ``expected`` in both fields.
    It returns True only after the write is durable. ``take_flow`` is an
    atomic one-shot read-and-delete across processes. A ``load`` from any
    client observes the latest acknowledged ``save`` for that key. Rows hold
    tokens and flows hold PKCE verifiers: encrypt both at rest.
    """

    async def load(self, key: MCPAuthKey) -> MCPOAuthRecord | None:
        """Return the row for ``key``, or None when none was ever written."""
        ...

    async def save(
        self,
        key: MCPAuthKey,
        record: MCPOAuthRecord,
        *,
        expected: MCPOAuthVersion | None,
    ) -> bool:
        """Compare-and-swap ``record`` against ``expected``."""
        ...

    async def save_flow(self, flow: MCPOAuthFlow) -> None:
        """Persist a pending flow under ``flow.state``."""
        ...

    async def take_flow(self, state: str) -> MCPOAuthFlow | None:
        """Atomically remove and return the flow for ``state``."""
        ...


class _MemoryBackend:
    __slots__ = ("flows", "rows")

    def __init__(self) -> None:
        self.rows: dict[MCPAuthKey, MCPOAuthRecord] = {}
        self.flows: dict[str, MCPOAuthFlow] = {}


class MCPOAuthMemoryStore:
    """Reference in-process store; every client shares one backend.

    For tests, demos, and single-process development only: contents are lost
    on restart and never shared across processes. ``client()`` returns a new
    client over the same backend, for :func:`check_oauth_store`.
    """

    def __init__(self) -> None:
        self._backend = _MemoryBackend()

    def client(self) -> MCPOAuthMemoryStore:
        """Return another client bound to this store's backend."""
        other = MCPOAuthMemoryStore()
        other._backend = self._backend
        return other

    async def load(self, key: MCPAuthKey) -> MCPOAuthRecord | None:
        """Return the row for ``key``."""
        return self._backend.rows.get(key)

    async def save(
        self,
        key: MCPAuthKey,
        record: MCPOAuthRecord,
        *,
        expected: MCPOAuthVersion | None,
    ) -> bool:
        """Insert or compare-and-swap one row."""
        current = self._backend.rows.get(key)
        if expected is None:
            if current is not None:
                return False
        elif current is None or current.version != expected:
            return False
        self._backend.rows[key] = record
        return True

    async def save_flow(self, flow: MCPOAuthFlow) -> None:
        """Store one pending flow."""
        self._backend.flows[flow.state] = flow

    async def take_flow(self, state: str) -> MCPOAuthFlow | None:
        """Remove and return one pending flow."""
        return self._backend.flows.pop(state, None)


def _conformance(condition: bool, check: str) -> None:
    if not condition:
        raise AssertionError(f"MCPOAuthStore conformance check failed: {check}.")


def _sample_record(generation: str, revision: int, marker: str) -> MCPOAuthRecord:
    client = MCPOAuthClientRecord(
        client_id=f"conformance-{marker}",
        token_endpoint_auth_method="none",
        source="preregistered",
        issuer="https://issuer.invalid",
    )
    credential = MCPOAuthCredential(
        access_token=f"conformance-access-{marker}",
        refresh_token=f"conformance-refresh-{marker}",
        expires_at=time.time() + 3600,
        scope="conformance",
        issuer="https://issuer.invalid",
        resource="https://resource.invalid/mcp",
        token_endpoint="https://issuer.invalid/token",
        revocation_endpoint=None,
        client=client,
        publication_id=_new_id(),
        published_at=time.time(),
    )
    return MCPOAuthRecord(
        version=MCPOAuthVersion(generation, revision),
        status="active",
        credential=credential,
        refresh_claim=None,
        retired_claim_id=None,
        updated_at=time.time(),
    )


def _sample_flow(key: MCPAuthKey, *, expired: bool) -> MCPOAuthFlow:
    record = _sample_record(_new_id(), 0, "flow")
    assert record.credential is not None
    return MCPOAuthFlow(
        state=secrets.token_urlsafe(32),
        key=key,
        resource="https://resource.invalid/mcp",
        code_verifier=secrets.token_urlsafe(48),
        redirect_uri="https://app.invalid/callback",
        scope=None,
        issuer="https://issuer.invalid",
        authorization_endpoint="https://issuer.invalid/authorize",
        token_endpoint="https://issuer.invalid/token",
        revocation_endpoint=None,
        client=record.credential.client,
        iss_required=False,
        expires_at=time.time() + (-60 if expired else 600),
        previous_generation=None,
    )


async def check_oauth_store(factory: Callable[[], MCPOAuthStore]) -> None:
    """Check an :class:`MCPOAuthStore` implementation's semantics.

    ``factory()`` must return a **new client** bound to the **same backend
    namespace** on every call; every competing operation (insert/insert,
    CAS/CAS, take/take) runs on two clients concurrently. Also checks
    exact-version semantics, read-after-write across clients, key isolation,
    and expired flows. Writes rows under random ``dendrux-conformance-*``
    keys, which are never deleted: point it at a test namespace. This is a
    semantics check, not a proof of multi-process atomicity. Raises
    ``AssertionError`` naming the first failed check.
    """
    first, second = factory(), factory()
    for client in (first, second):
        _conformance(isinstance(client, MCPOAuthStore), "factory returns an MCPOAuthStore")
    _conformance(first is not second, "factory returns a new client per call")
    suffix = secrets.token_hex(8)
    key = MCPAuthKey(f"dendrux-conformance-{suffix}", tenant_key=f"tenant-{suffix}")
    _conformance(await first.load(key) is None, "load of an unknown key returns None")

    generation = _new_id()
    a = _sample_record(generation, 0, "a")
    b = _sample_record(_new_id(), 0, "b")
    results = await asyncio.gather(
        first.save(key, a, expected=None), second.save(key, b, expected=None)
    )
    _conformance(sorted(results) == [False, True], "exactly one concurrent insert wins")
    winner = a if results[0] else b
    _conformance(await second.load(key) == winner, "read-after-write across clients")
    _conformance(await first.load(key) == winner, "read-after-write on the writing client")
    _conformance(not await first.save(key, a, expected=None), "insert is refused when a row exists")

    base = winner.version
    next_a = replace(a, version=MCPOAuthVersion(base.generation, base.revision + 1))
    next_b = replace(b, version=MCPOAuthVersion(base.generation, base.revision + 1))
    results = await asyncio.gather(
        first.save(key, next_a, expected=base), second.save(key, next_b, expected=base)
    )
    _conformance(sorted(results) == [False, True], "exactly one concurrent CAS wins")
    current = next_a if results[0] else next_b
    _conformance(await first.load(key) == current, "a CAS winner is visible to every client")
    _conformance(not await second.save(key, a, expected=base), "a stale version is refused")
    _conformance(
        not await second.save(
            key, a, expected=MCPOAuthVersion(_new_id(), current.version.revision)
        ),
        "a version with a different generation is refused",
    )
    _conformance(
        not await second.save(
            key,
            a,
            expected=MCPOAuthVersion(current.version.generation, current.version.revision + 7),
        ),
        "a version with a different revision is refused",
    )

    flow = _sample_flow(key, expired=False)
    await first.save_flow(flow)
    taken = await asyncio.gather(first.take_flow(flow.state), second.take_flow(flow.state))
    _conformance(
        sorted(item is None for item in taken) == [False, True], "exactly one take_flow wins"
    )
    _conformance(next(item for item in taken if item is not None) == flow, "take_flow round-trips")
    _conformance(await second.take_flow(flow.state) is None, "take_flow is one-shot")

    expired = _sample_flow(key, expired=True)
    await second.save_flow(expired)
    taken_expired = await first.take_flow(expired.state)
    _conformance(
        taken_expired is None or taken_expired == expired,
        "an expired flow is either collected or returned unchanged",
    )
    _conformance(await first.take_flow(expired.state) is None, "an expired flow is one-shot")

    for other in (
        MCPAuthKey(key.connection_key),
        MCPAuthKey(key.connection_key, tenant_key=f"{key.tenant_key}x"),
    ):
        _conformance(await second.load(other) is None, "keys are isolated")
    # Pairs that collide under naive concatenation with common separators.
    pairs: list[tuple[MCPAuthKey, MCPAuthKey]] = []
    for sep in (":", "/", "|", ".", "-", "_"):
        left, right = f"l{suffix}", f"r{suffix}"
        pairs.append(
            (MCPAuthKey("c", tenant_key=f"{left}{sep}{right}"), MCPAuthKey(f"{right}{sep}c", left))
        )
        pairs.append(
            (MCPAuthKey(f"{left}{sep}{right}", tenant_key="t"), MCPAuthKey(left, f"{right}{sep}t"))
        )
    pairs.append((MCPAuthKey(f"n{suffix}"), MCPAuthKey(f"n{suffix}", tenant_key="None")))
    for index, (left_key, right_key) in enumerate(pairs):
        marker = _sample_record(_new_id(), 0, f"iso{index}")
        _conformance(
            await first.save(left_key, marker, expected=None), "an unrelated key accepts an insert"
        )
        _conformance(
            await second.load(right_key) is None, "keys that concatenate alike are isolated"
        )
    _conformance(await first.load(key) == current, "other keys never overwrite a row")


# --------------------------------------------------------------------------
# Clients
# --------------------------------------------------------------------------


def _client_url(value: object, field_name: str) -> str:
    try:
        canonical_url(value)
    except URLRejectedError:
        raise ValueError(f"MCPOAuthClient {field_name} must be an absolute http(s) URL.") from None
    assert isinstance(value, str)
    return value


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class MCPOAuthClient:
    """How Dendrux identifies itself to an authorization server.

    Build with :meth:`preregistered` or :meth:`dynamic`.
    """

    kind: Literal["preregistered", "dynamic"]
    client_id: str | None = None
    client_secret: str | None = None
    issuer: str | None = None
    token_endpoint_auth_method: AuthMethod = "none"
    client_name: str | None = None
    redirect_uris: tuple[str, ...] = ()
    cimd_url: str | None = None
    application_type: Literal["web", "native"] = "web"
    client_uri: str | None = None
    logo_uri: str | None = None
    contacts: tuple[str, ...] = ()

    def __repr__(self) -> str:
        return f"MCPOAuthClient(kind={self.kind!r})"

    @classmethod
    def preregistered(
        cls,
        client_id: str,
        client_secret: str | None = None,
        *,
        issuer: str,
        token_endpoint_auth_method: AuthMethod | None = None,
    ) -> MCPOAuthClient:
        """A client registered out of band with one authorization server.

        Used only when the authorization server's raw metadata ``issuer``
        equals ``issuer`` exactly, so its secret is never sent elsewhere.
        """
        if not _plain(client_id):
            raise ValueError("MCPOAuthClient client_id must be a non-empty string.")
        if client_secret is not None and not _plain(client_secret):
            raise ValueError("MCPOAuthClient client_secret must be a non-empty string.")
        try:
            validate_url(issuer, allow_insecure=True, issuer=True)
        except URLRejectedError:
            raise ValueError(
                "MCPOAuthClient issuer must be an absolute http(s) URL without a query."
            ) from None
        method = token_endpoint_auth_method
        if method is None:
            method = "client_secret_post" if client_secret is not None else "none"
        if method not in _AUTH_METHODS:
            raise ValueError("MCPOAuthClient token_endpoint_auth_method is not supported.")
        if method == "none" and client_secret is not None:
            raise ValueError("A client secret requires a client_secret_* auth method.")
        if method in _SECRET_METHODS and client_secret is None:
            raise ValueError(f"token_endpoint_auth_method {method!r} requires a client secret.")
        return cls(
            kind="preregistered",
            client_id=client_id,
            client_secret=client_secret,
            issuer=issuer,
            token_endpoint_auth_method=method,
        )

    @classmethod
    def dynamic(
        cls,
        *,
        client_name: str,
        redirect_uris: Sequence[str],
        cimd_url: str | None = None,
        application_type: Literal["web", "native"] = "web",
        token_endpoint_auth_method: AuthMethod = "none",
        client_uri: str | None = None,
        logo_uri: str | None = None,
        contacts: Sequence[str] = (),
    ) -> MCPOAuthClient:
        """A client resolved per authorization server.

        With ``cimd_url`` and an authorization server that supports client ID
        metadata documents, the URL is the client ID; the application hosts
        that document with ``redirect_uris`` matching its callback (Dendrux
        validates the URL's shape only). Otherwise the client registers
        dynamically (RFC 7591) when the server offers registration.
        """
        if not _plain(client_name):
            raise ValueError("MCPOAuthClient client_name must be a non-empty string.")
        if isinstance(redirect_uris, str) or not redirect_uris:
            raise ValueError("MCPOAuthClient redirect_uris must be a non-empty sequence.")
        uris = tuple(_client_url(uri, "redirect_uris") for uri in redirect_uris)
        if cimd_url is not None:
            _client_url(cimd_url, "cimd_url")
            parts = urlsplit(cimd_url)
            if parts.scheme != "https" or parts.path in ("", "/") or parts.query or "?" in cimd_url:
                raise ValueError(
                    "MCPOAuthClient cimd_url must be an https URL with a non-root path "
                    "and no query."
                )
        if application_type not in ("web", "native"):
            raise ValueError("MCPOAuthClient application_type must be 'web' or 'native'.")
        if token_endpoint_auth_method not in _AUTH_METHODS:
            raise ValueError("MCPOAuthClient token_endpoint_auth_method is not supported.")
        if client_uri is not None:
            _client_url(client_uri, "client_uri")
        if logo_uri is not None:
            _client_url(logo_uri, "logo_uri")
        if isinstance(contacts, str) or not all(_plain(item) for item in contacts):
            raise ValueError("MCPOAuthClient contacts must be a sequence of strings.")
        return cls(
            kind="dynamic",
            client_name=client_name,
            redirect_uris=uris,
            cimd_url=cimd_url,
            application_type=application_type,
            token_endpoint_auth_method=token_endpoint_auth_method,
            client_uri=client_uri,
            logo_uri=logo_uri,
            contacts=tuple(contacts),
        )


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _json_object(body: bytes) -> dict[str, Any] | None:
    try:
        value = json.loads(body)
    except Exception:
        return None
    return value if isinstance(value, dict) else None


def _model(build: Callable[[], _T]) -> _T | None:
    """Build an SDK model; any failure is None, never third-party text."""
    try:
        return build()
    except Exception:
        return None


@dataclass(frozen=True, slots=True, repr=False)
class _Token:
    access_token: str
    token_type: str
    expires_in: int | None
    refresh_token: str | None
    scope: str | None

    def __repr__(self) -> str:
        refresh = None if self.refresh_token is None else "<set>"
        return f"_Token(expires_in={self.expires_in!r}, refresh_token={refresh})"


def _parse_token(body: bytes) -> _Token | None:
    """Parse a token response strictly; None when it is unusable. Never raises."""
    try:
        return _parse_token_fields(body)
    except Exception:
        return None


def _parse_token_fields(body: bytes) -> _Token | None:
    data = _json_object(body)
    if data is None:
        return None
    access = data.get("access_token")
    if not isinstance(access, str) or not _ACCESS_TOKEN.fullmatch(access):
        return None
    token_type = data.get("token_type", "Bearer")
    if not isinstance(token_type, str) or token_type.lower() != "bearer":
        return None
    expires_raw = data.get("expires_in")
    expires_in: int | None
    if expires_raw is None:
        expires_in = None
    elif isinstance(expires_raw, int) and not isinstance(expires_raw, bool):
        expires_in = expires_raw
    elif isinstance(expires_raw, str) and _EXPIRES_IN.fullmatch(expires_raw):
        expires_in = int(expires_raw)
    else:
        return None
    if expires_in is not None and not 0 <= expires_in <= _MAX_EXPIRES_IN:
        return None
    refresh = data.get("refresh_token")
    if refresh is not None and (
        not isinstance(refresh, str) or not _REFRESH_TOKEN.fullmatch(refresh)
    ):
        return None
    scope = data.get("scope")
    if scope is not None and (
        not isinstance(scope, str)
        or not all(_SCOPE_TOKEN.fullmatch(item) for item in scope.split(" ") if item)
    ):
        return None
    return _Token(access, "Bearer", expires_in, refresh or None, scope or None)


def _usable(info: OAuthClientInformationFull) -> bool:
    check_registration_usable(info)
    return True


def _scope_tokens(values: object) -> tuple[str, ...] | None:
    if not isinstance(values, (list, tuple)):
        return None
    return tuple(item for item in values if isinstance(item, str) and _SCOPE_TOKEN.fullmatch(item))


def _client_auth(
    client: MCPOAuthClientRecord,
    form: dict[str, str],
) -> dict[str, str]:
    """Apply token-endpoint client authentication; returns request headers."""
    form["client_id"] = client.client_id
    headers: dict[str, str] = {}
    method = client.token_endpoint_auth_method
    if method == "client_secret_post" and client.client_secret is not None:
        form["client_secret"] = client.client_secret
    elif method == "client_secret_basic" and client.client_secret is not None:
        pair = f"{quote(client.client_id, safe='')}:{quote(client.client_secret, safe='')}"
        headers["Authorization"] = "Basic " + base64.b64encode(pair.encode()).decode()
    return headers


def _next_version(record: MCPOAuthRecord) -> MCPOAuthVersion:
    return MCPOAuthVersion(record.version.generation, record.version.revision + 1)


def _flow_error(reason: MCPOAuthFlowReason, status: int | None = None) -> MCPOAuthFlowError:
    return MCPOAuthFlowError(reason, status_code=status)


_DELIVER = "deliver"
_NEEDS_REFRESH = "needs_refresh"


@dataclass(frozen=True, slots=True)
class _Discovery:
    issuer: str
    authorization_endpoint: str
    token_endpoint: str
    registration_endpoint: str | None
    revocation_endpoint: str | None
    iss_required: bool
    cimd_supported: bool
    scope: str | None


# --------------------------------------------------------------------------
# MCPOAuth
# --------------------------------------------------------------------------


def _positive(value: float, name: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"MCPOAuth {name} must be a number.")
    resolved = float(value)
    if not math.isfinite(resolved) or resolved < 0 or (resolved == 0 and not allow_zero):
        raise ValueError(f"MCPOAuth {name} must be a finite positive number.")
    return resolved


class MCPOAuth:
    """OAuth sign-in and token lifecycle for remote MCP servers.

    One instance per application (or per authorization configuration). It
    owns an HTTP transport for OAuth requests and runs every public
    operation as a tracked task: a caller cancelled while awaiting never
    cancels the operation, which runs to its natural outcome. ``aclose()``
    stops new operations, waits up to ``oauth_timeout`` for running ones,
    cancels what remains (a cancelled refresh retires its refresh token, it
    is never replayed), then closes the transport.
    """

    def __init__(
        self,
        *,
        store: MCPOAuthStore,
        client: MCPOAuthClient,
        destination_policy: MCPDestinationPolicy | None = None,
        oauth_timeout: float = 10.0,
        refresh_skew: float = 60.0,
        flow_ttl: float = 600.0,
        claim_ttl: float = 300.0,
        allow_insecure_issuer: bool = False,
        allow_legacy_discovery: bool = False,
    ) -> None:
        if not isinstance(store, MCPOAuthStore):
            raise ValueError("MCPOAuth store must implement MCPOAuthStore.")
        if not isinstance(client, MCPOAuthClient):
            raise ValueError("MCPOAuth client must be an MCPOAuthClient.")
        if destination_policy is not None and not callable(destination_policy):
            raise ValueError("MCPOAuth destination_policy must be callable.")
        self.oauth_timeout = _positive(oauth_timeout, "oauth_timeout")
        self.refresh_skew = _positive(refresh_skew, "refresh_skew", allow_zero=True)
        self.flow_ttl = _positive(flow_ttl, "flow_ttl")
        self.claim_ttl = _positive(claim_ttl, "claim_ttl")
        if self.claim_ttl <= self.oauth_timeout + 60:
            raise ValueError(
                "MCPOAuth claim_ttl must exceed oauth_timeout by more than 60 seconds."
            )
        if not isinstance(allow_insecure_issuer, bool) or not isinstance(
            allow_legacy_discovery, bool
        ):
            raise ValueError("MCPOAuth allow_* options must be bools.")
        self.store = store
        self.client = client
        self.destination_policy = destination_policy
        self.allow_insecure_issuer = allow_insecure_issuer
        self.allow_legacy_discovery = allow_legacy_discovery
        self._http = OAuthHTTP(policy=destination_policy, timeout=self.oauth_timeout)
        self._clock: Callable[[], float] = time.time
        self._operations: set[asyncio.Task[Any]] = set()
        self._children: set[asyncio.Task[Any]] = set()
        self._closed = False
        self._close_task: asyncio.Task[None] | None = None

    def __repr__(self) -> str:
        return f"MCPOAuth(client={self.client!r}, closed={self._closed!r})"

    # -- operation tracking ------------------------------------------------

    def _now(self) -> float:
        return self._clock()

    @staticmethod
    def _consume(task: asyncio.Task[Any]) -> None:
        if not task.cancelled():
            task.exception()

    def _operation_done(self, task: asyncio.Task[Any]) -> None:
        self._operations.discard(task)
        self._consume(task)

    def _child_done(self, task: asyncio.Task[Any]) -> None:
        self._children.discard(task)
        self._consume(task)

    async def _run(self, coro: Coroutine[Any, Any, _T]) -> _T:
        """Run one public operation as a tracked task the caller only awaits."""
        if self._closed:
            coro.close()
            raise _flow_error("closed")
        task = asyncio.get_running_loop().create_task(coro)
        self._operations.add(task)
        task.add_done_callback(self._operation_done)
        return await asyncio.shield(task)

    async def _write(self, coro: Awaitable[_T]) -> tuple[_T, bool]:
        """Run a store write to completion even if this operation is cancelled.

        Returns the result and whether cancellation was requested meanwhile;
        the caller finishes its cleanup and then re-raises.
        """

        async def run() -> _T:
            return await coro

        task = asyncio.get_running_loop().create_task(run())
        self._children.add(task)
        task.add_done_callback(self._child_done)
        cancelled = False
        while True:
            try:
                return await asyncio.shield(task), cancelled
            except asyncio.CancelledError:
                if task.cancelled():
                    raise
                cancelled = True
                if task.done():
                    # The write finished as cancellation arrived: its result
                    # is committed state the caller must still act on.
                    return task.result(), cancelled

    async def aclose(self) -> None:
        """Stop new operations, settle running ones, and close the transport.

        Running operations get ``oauth_timeout`` to finish; any still running
        are then cancelled (their shielded store writes complete first).
        Idempotent; concurrent callers share one close.
        """
        self._closed = True
        if self._close_task is None:
            self._close_task = asyncio.get_running_loop().create_task(self._close_impl())
        await asyncio.shield(self._close_task)

    async def _close_impl(self) -> None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.oauth_timeout
        while self._operations and loop.time() < deadline:
            await asyncio.wait(set(self._operations), timeout=deadline - loop.time())
        for task in set(self._operations):
            task.cancel()
        grace = loop.time() + self.oauth_timeout
        while (self._operations or self._children) and loop.time() < grace:
            await asyncio.wait(self._operations | self._children, timeout=grace - loop.time())
        if self._operations or self._children:
            logger.warning("MCP OAuth close left store writes running in the background.")
        await self._http.aclose()

    # -- store boundary ----------------------------------------------------

    async def _load(self, key: MCPAuthKey) -> MCPOAuthRecord | None:
        failure: str | None = None
        try:
            record = await self.store.load(key)
        except Exception as exc:
            failure = type(exc).__name__
        if failure is not None:
            raise MCPOAuthError(f"MCP OAuth store operation failed ({failure}).")
        if record is not None and not isinstance(record, MCPOAuthRecord):
            raise MCPOAuthError("MCP OAuth store returned an invalid record.")
        return record

    async def _save(
        self, key: MCPAuthKey, record: MCPOAuthRecord, expected: MCPOAuthVersion | None
    ) -> bool:
        failure: str | None = None
        try:
            saved = await self.store.save(key, record, expected=expected)
        except Exception as exc:
            failure = type(exc).__name__
        if failure is not None:
            raise MCPOAuthError(f"MCP OAuth store operation failed ({failure}).")
        return saved is True

    async def _save_flow(self, flow: MCPOAuthFlow) -> None:
        failure: str | None = None
        try:
            await self.store.save_flow(flow)
        except Exception as exc:
            failure = type(exc).__name__
        if failure is not None:
            raise MCPOAuthError(f"MCP OAuth store operation failed ({failure}).")

    async def _take_flow(self, state: str) -> MCPOAuthFlow | None:
        failure: str | None = None
        result: object = None
        try:
            result = await self.store.take_flow(state)
        except Exception as exc:
            failure = type(exc).__name__
        if failure is not None:
            raise MCPOAuthError(f"MCP OAuth store operation failed ({failure}).")
        if result is not None and not isinstance(result, MCPOAuthFlow):
            raise MCPOAuthError("MCP OAuth store returned an invalid flow.")
        return result

    # -- URL rules ---------------------------------------------------------

    def _url(
        self,
        url: object,
        *,
        issuer: bool = False,
        untrusted: bool = False,
    ) -> str:
        """Validate a URL before I/O; failures map to value-free flow errors."""
        insecure = False
        try:
            validate_url(url, allow_insecure=self.allow_insecure_issuer, issuer=issuer)
        except URLRejectedError as exc:
            insecure = exc.insecure
        else:
            assert isinstance(url, str)
            return url
        if insecure:
            raise _flow_error("insecure_url")
        raise _flow_error("discovery_failed" if untrusted else "invalid_argument")

    def _valid(self, url: object, *, issuer: bool = False) -> bool:
        try:
            validate_url(url, allow_insecure=self.allow_insecure_issuer, issuer=issuer)
        except URLRejectedError:
            return False
        return True

    # -- HTTP --------------------------------------------------------------

    async def _send(
        self,
        method: str,
        url: str,
        *,
        http: OAuthHTTP | None = None,
        form: Mapping[str, str] | None = None,
        json_body: object = None,
        headers: Mapping[str, str] | None = None,
        attempt: SendAttempt | None = None,
    ) -> OAuthResponse | OAuthTransportError:
        """Send one request; a transport failure is returned, never raised."""
        try:
            return await (http or self._http).send(
                method, url, form=form, json_body=json_body, headers=headers, attempt=attempt
            )
        except OAuthTransportError as exc:
            return exc

    async def _get_json(self, url: str) -> tuple[str, int | None, dict[str, Any] | None]:
        """Fetch one metadata candidate: ``ok``, ``absent``, or ``transient``."""
        result = await self._send("GET", url, headers=_METADATA_HEADERS)
        if isinstance(result, OAuthTransportError):
            if result.denied:
                raise _flow_error("destination_denied")
            return "transient", None, None
        status = result.status
        if status == 200:
            data = _json_object(result.body)
            return ("ok", status, data) if data is not None else ("absent", status, None)
        if 300 <= status < 400 or status == 429 or status >= 500:
            return "transient", status, None
        return "absent", status, None

    # -- begin ---------------------------------------------------------------

    async def begin_authorization(
        self,
        key: MCPAuthKey,
        source: MCPSource,
        *,
        redirect_uri: str,
        scopes: Sequence[str] | None = None,
        resource_metadata_url: str | None = None,
        authorization_server: str | None = None,
        client: MCPOAuthClient | None = None,
    ) -> str:
        """Start a sign-in and return the authorization URL for the browser.

        Discovers the resource's authorization server, resolves the client,
        and stores a one-shot flow bound to ``key``. Step-up callers pass the
        union of :meth:`status` ``scope`` and the challenged scopes.
        """
        return await self._run(
            self._begin(
                key,
                source,
                redirect_uri=redirect_uri,
                scopes=scopes,
                resource_metadata_url=resource_metadata_url,
                authorization_server=authorization_server,
                client=client,
            )
        )

    async def _probe(self, source: MCPSource) -> tuple[str | None, tuple[str, ...]]:
        """One bounded unauthenticated ``initialize``; a 401 challenge is the hint."""
        assert source.url is not None
        probe = OAuthHTTP(policy=source.destination_policy, timeout=source.connect_timeout)
        try:
            result = await self._send(
                "POST",
                source.url,
                http=probe,
                headers=_PROBE_HEADERS,
                json_body={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": LATEST_PROTOCOL_VERSION,
                        "capabilities": {},
                        "clientInfo": {"name": "dendrux", "version": "1"},
                    },
                },
            )
        finally:
            await probe.aclose()
        if isinstance(result, OAuthTransportError) or result.status != 401:
            return None, ()
        challenge: MCPAuthChallenge | None = _model(
            lambda: challenge_from_headers(result.headers.get_list("www-authenticate"))
        )
        if challenge is None:
            return None, ()
        return challenge.resource_metadata_url, challenge.scope

    async def _discover(
        self,
        source: MCPSource,
        resource: str,
        hint: str | None,
        authorization_server: str | None,
    ) -> tuple[_Discovery, ProtectedResourceMetadata | None]:
        assert source.url is not None
        candidates = _model(
            lambda: build_protected_resource_metadata_discovery_urls(hint, source.url or "")
        )
        if candidates is None:
            raise _flow_error("discovery_failed")
        prm: ProtectedResourceMetadata | None = None
        prm_data: dict[str, Any] | None = None
        transient_status: int | None = None
        saw_transient = False
        for url in candidates:
            if not self._valid(url):
                continue
            category, status, data = await self._get_json(url)
            if category == "transient":
                saw_transient = True
                transient_status = status
                continue
            if category != "ok" or data is None:
                continue
            raw_resource = data.get("resource")
            if not isinstance(raw_resource, str) or not self._valid(raw_resource):
                continue
            if raw_resource != resource:
                continue
            built = _model(partial(ProtectedResourceMetadata.model_validate, data))
            if built is None:
                continue
            prm, prm_data = built, data
            break
        legacy = False
        if prm is None or prm_data is None:
            if saw_transient or not self.allow_legacy_discovery:
                raise _flow_error("discovery_failed", transient_status)
            legacy = True
            as_url = origin_of(source.url)
        else:
            raw_servers = prm_data.get("authorization_servers")
            if not isinstance(raw_servers, list) or not raw_servers:
                raise _flow_error("discovery_failed")
            if authorization_server is not None:
                if authorization_server not in raw_servers:
                    raise _flow_error("discovery_failed")
                as_url = authorization_server
            else:
                as_url = raw_servers[0]
            as_url = self._url(as_url, issuer=True, untrusted=True)

        as_candidates = _model(
            lambda: build_oauth_authorization_server_metadata_discovery_urls(
                as_url, source.url or ""
            )
        )
        if as_candidates is None:
            raise _flow_error("discovery_failed")
        meta: OAuthMetadata | None = None
        meta_data: dict[str, Any] | None = None
        saw_transient = False
        transient_status = None
        for url in as_candidates:
            if not self._valid(url):
                continue
            category, status, data = await self._get_json(url)
            if category == "transient":
                saw_transient = True
                transient_status = status
                continue
            if category != "ok" or data is None:
                continue
            raw_issuer = data.get("issuer")
            if not isinstance(raw_issuer, str) or not self._valid(raw_issuer, issuer=True):
                continue
            if raw_issuer != as_url:
                continue
            built_meta = _model(partial(OAuthMetadata.model_validate, data))
            if built_meta is None:
                continue
            meta, meta_data = built_meta, data
            break
        if meta is None or meta_data is None:
            if saw_transient or not legacy:
                raise _flow_error("discovery_failed", transient_status)
            return (
                _Discovery(
                    issuer=as_url,
                    authorization_endpoint=f"{as_url}/authorize",
                    token_endpoint=f"{as_url}/token",
                    registration_endpoint=f"{as_url}/register",
                    revocation_endpoint=None,
                    iss_required=False,
                    cimd_supported=False,
                    scope=None,
                ),
                prm,
            )

        def endpoint(name: str, *, required: bool) -> str | None:
            assert meta_data is not None
            raw = meta_data.get(name)
            if raw is None and not required:
                return None
            if not isinstance(raw, str):
                raise _flow_error("discovery_failed")
            return self._url(raw, untrusted=True)

        authorization_endpoint = endpoint("authorization_endpoint", required=True)
        token_endpoint = endpoint("token_endpoint", required=True)
        assert authorization_endpoint is not None and token_endpoint is not None
        methods = meta_data.get("code_challenge_methods_supported")
        if not isinstance(methods, list) or "S256" not in methods:
            raise _flow_error("pkce_unsupported")
        return (
            _Discovery(
                issuer=as_url,
                authorization_endpoint=authorization_endpoint,
                token_endpoint=token_endpoint,
                registration_endpoint=endpoint("registration_endpoint", required=False),
                revocation_endpoint=endpoint("revocation_endpoint", required=False),
                iss_required=meta.authorization_response_iss_parameter_supported is True,
                cimd_supported=meta.client_id_metadata_document_supported is True,
                scope=None,
            ),
            prm,
        )

    async def _resolve_client(
        self,
        client: MCPOAuthClient,
        discovery: _Discovery,
        redirect_uri: str,
        scope: str | None,
    ) -> MCPOAuthClientRecord:
        if client.kind == "preregistered":
            assert client.client_id is not None and client.issuer is not None
            if client.issuer != discovery.issuer:
                raise _flow_error("registration_failed")
            return MCPOAuthClientRecord(
                client_id=client.client_id,
                client_secret=client.client_secret,
                token_endpoint_auth_method=client.token_endpoint_auth_method,
                source="preregistered",
                issuer=discovery.issuer,
            )
        if client.cimd_url is not None and discovery.cimd_supported:
            return MCPOAuthClientRecord(
                client_id=client.cimd_url,
                token_endpoint_auth_method="none",
                source="cimd",
                issuer=discovery.issuer,
            )
        if discovery.registration_endpoint is None:
            raise _flow_error("registration_failed")
        redirect_uris = [
            redirect_uri,
            *(uri for uri in client.redirect_uris if uri != redirect_uri),
        ]
        metadata = _model(
            lambda: OAuthClientMetadata(
                redirect_uris=redirect_uris,  # type: ignore[arg-type]  # pydantic validates the str URLs
                token_endpoint_auth_method=client.token_endpoint_auth_method,
                grant_types=["authorization_code", "refresh_token"],
                response_types=["code"],
                scope=scope,
                client_name=client.client_name,
                client_uri=client.client_uri,  # type: ignore[arg-type]  # pydantic validates the str URL
                logo_uri=client.logo_uri,  # type: ignore[arg-type]  # pydantic validates the str URL
                contacts=list(client.contacts) or None,
                application_type=client.application_type,
            ).model_dump(by_alias=True, mode="json", exclude_none=True)
        )
        if metadata is None:
            raise _flow_error("registration_failed")
        # Sent to the raw, validated endpoint: the SDK request builder would
        # use a pydantic-normalized form of it.
        result = await self._send(
            "POST",
            discovery.registration_endpoint,
            json_body=metadata,
            headers={"Content-Type": "application/json"},
        )
        if isinstance(result, OAuthTransportError):
            raise _flow_error("destination_denied" if result.denied else "registration_failed")
        if result.status not in (200, 201):
            raise _flow_error("registration_failed", result.status)
        data = _json_object(result.body)
        if data is None:
            raise _flow_error("registration_failed", result.status)
        data.pop("issuer", None)
        raw_redirects = data.get("redirect_uris")
        if not isinstance(raw_redirects, list) or redirect_uri not in raw_redirects:
            raise _flow_error("registration_failed", result.status)
        info = _model(lambda: OAuthClientInformationFull.model_validate(data))
        if info is None or _model(partial(_usable, info)) is None:
            raise _flow_error("registration_failed", result.status)
        method = info.token_endpoint_auth_method or "client_secret_basic"
        if method not in _AUTH_METHODS:
            raise _flow_error("registration_failed", result.status)
        secret = info.client_secret
        if method in _SECRET_METHODS and not _plain(secret):
            raise _flow_error("registration_failed", result.status)
        if not _plain(info.client_id):
            raise _flow_error("registration_failed", result.status)
        expires_at = info.client_secret_expires_at
        if secret is not None and expires_at and expires_at <= self._now():
            raise _flow_error("registration_failed", result.status)
        return MCPOAuthClientRecord(
            client_id=info.client_id,
            client_secret=secret if method in _SECRET_METHODS else None,
            client_secret_expires_at=float(expires_at) if expires_at else None,
            token_endpoint_auth_method=method,  # type: ignore[arg-type]  # narrowed by _AUTH_METHODS
            source="dcr",
            issuer=discovery.issuer,
        )

    async def _begin(
        self,
        key: MCPAuthKey,
        source: MCPSource,
        *,
        redirect_uri: str,
        scopes: Sequence[str] | None,
        resource_metadata_url: str | None,
        authorization_server: str | None,
        client: MCPOAuthClient | None,
    ) -> str:
        if not isinstance(key, MCPAuthKey) or not isinstance(source, MCPSource):
            raise _flow_error("invalid_argument")
        if source.url is None:
            raise _flow_error("invalid_argument")
        if client is not None and not isinstance(client, MCPOAuthClient):
            raise _flow_error("invalid_argument")
        resolved_client = client or self.client
        self._url(source.url)
        resource = canonical_url(source.url)
        self._url(redirect_uri)
        if resolved_client.kind == "preregistered":
            self._url(resolved_client.issuer, issuer=True)
        if resource_metadata_url is not None:
            self._url(resource_metadata_url)
        if authorization_server is not None:
            self._url(authorization_server, issuer=True)
        explicit: tuple[str, ...] | None = None
        if scopes is not None:
            if isinstance(scopes, str) or not all(
                isinstance(item, str) and _SCOPE_TOKEN.fullmatch(item) for item in scopes
            ):
                raise _flow_error("invalid_argument")
            explicit = tuple(dict.fromkeys(scopes))

        hint = resource_metadata_url
        challenged: tuple[str, ...] = ()
        if hint is None:
            hint, challenged = await self._probe(source)
        discovery, prm = await self._discover(source, resource, hint, authorization_server)

        selected: tuple[str, ...] | None
        if explicit is not None:
            selected = explicit
        elif challenged:
            selected = challenged
        elif prm is not None and prm.scopes_supported is not None:
            selected = _scope_tokens(prm.scopes_supported)
        else:
            selected = None
        scope = " ".join(selected) if selected else None

        record = await self._resolve_client(resolved_client, discovery, redirect_uri, scope)
        pkce = PKCEParameters.generate()
        state = secrets.token_urlsafe(32)
        row = await self._load(key)
        flow = MCPOAuthFlow(
            state=state,
            key=key,
            resource=resource,
            code_verifier=pkce.code_verifier,
            redirect_uri=redirect_uri,
            scope=scope,
            issuer=discovery.issuer,
            authorization_endpoint=discovery.authorization_endpoint,
            token_endpoint=discovery.token_endpoint,
            revocation_endpoint=discovery.revocation_endpoint,
            client=record,
            iss_required=discovery.iss_required,
            expires_at=self._now() + self.flow_ttl,
            previous_generation=row.version.generation if row is not None else None,
        )
        await self._write(self._save_flow(flow))
        params = {
            "response_type": "code",
            "client_id": record.client_id,
            "redirect_uri": redirect_uri,
            "state": state,
            "code_challenge": pkce.code_challenge,
            "code_challenge_method": "S256",
            "resource": resource,
        }
        if scope:
            params["scope"] = scope
        separator = "&" if urlsplit(discovery.authorization_endpoint).query else "?"
        return f"{discovery.authorization_endpoint}{separator}{urlencode(params)}"

    # -- complete ------------------------------------------------------------

    async def complete_authorization(
        self,
        *,
        key: MCPAuthKey,
        state: str,
        code: str | None,
        iss: str | None = None,
        error: str | None = None,
    ) -> MCPAuthKey:
        """Finish a sign-in from the authorization server's callback.

        ``key`` is the application's authenticated principal for this request;
        it must equal the key the flow was begun with. The flow is consumed
        before any other check, so a callback can never be replayed.
        """
        return await self._run(
            self._complete(key=key, state=state, code=code, iss=iss, error=error)
        )

    async def _complete(
        self,
        *,
        key: MCPAuthKey,
        state: str,
        code: str | None,
        iss: str | None,
        error: str | None,
    ) -> MCPAuthKey:
        if not isinstance(key, MCPAuthKey):
            raise _flow_error("invalid_argument")
        if not isinstance(state, str) or not state or len(state) > 512:
            raise _flow_error("unknown_state")
        flow, cancelled = await self._write(self._take_flow(state))
        if cancelled:
            raise asyncio.CancelledError
        if flow is None:
            raise _flow_error("unknown_state")
        if flow.expires_at <= self._now():
            raise _flow_error("expired_state")
        if flow.key != key:
            raise _flow_error("key_mismatch")
        if iss is not None:
            if iss != flow.issuer:
                raise _flow_error("issuer_mismatch")
        elif flow.iss_required:
            raise _flow_error("issuer_mismatch")
        if error is not None:
            raise _flow_error("access_denied" if error == "access_denied" else "invalid_callback")
        if not isinstance(code, str) or not code or len(code) > 4096 or not _plain(code):
            raise _flow_error("invalid_callback")

        row = await self._load(key)
        if (row.version.generation if row is not None else None) != flow.previous_generation:
            raise _flow_error("superseded")
        client = flow.client
        if client.secret_expired(self._now()):
            raise _flow_error("registration_failed")
        form = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": flow.redirect_uri,
            "code_verifier": flow.code_verifier,
            "resource": flow.resource,
        }
        headers = _client_auth(client, form)
        issued_at = self._now()
        result = await self._send("POST", flow.token_endpoint, form=form, headers=headers)
        if isinstance(result, OAuthTransportError):
            raise _flow_error("destination_denied" if result.denied else "token_exchange_failed")
        if result.status != 200:
            raise _flow_error("token_exchange_failed", result.status)
        token = _parse_token(result.body)
        if token is None:
            raise _flow_error("token_exchange_failed", result.status)

        now = self._now()
        credential = MCPOAuthCredential(
            access_token=token.access_token,
            refresh_token=token.refresh_token,
            token_type=token.token_type,
            expires_at=issued_at + token.expires_in if token.expires_in is not None else None,
            scope=token.scope if token.scope is not None else flow.scope,
            issuer=flow.issuer,
            resource=flow.resource,
            token_endpoint=flow.token_endpoint,
            revocation_endpoint=flow.revocation_endpoint,
            client=client,
            publication_id=_new_id(),
            published_at=now,
        )
        published = MCPOAuthRecord(
            version=MCPOAuthVersion(_new_id(), 0),
            status="active" if token.refresh_token is not None else "retired",
            credential=credential,
            refresh_claim=None,
            retired_claim_id=None,
            updated_at=now,
        )
        for _ in range(_PUBLISH_ATTEMPTS):
            row = await self._load(key)
            if (row.version.generation if row is not None else None) != flow.previous_generation:
                # A revoke or another sign-in won. The tokens are discarded,
                # not revoked: RFC 7009 revocation may be grant-wide.
                raise _flow_error("superseded")
            saved, cancelled = await self._write(
                self._save(key, published, row.version if row is not None else None)
            )
            if cancelled:
                raise asyncio.CancelledError
            if saved:
                return flow.key
        raise _flow_error("contention")

    # -- status ----------------------------------------------------------------

    async def status(self, key: MCPAuthKey) -> MCPOAuthStatus:
        """Return the sign-in state for ``key`` without any network call."""
        return await self._run(self._status(key))

    async def _status(self, key: MCPAuthKey) -> MCPOAuthStatus:
        if not isinstance(key, MCPAuthKey):
            raise _flow_error("invalid_argument")
        row = await self._load(key)
        if row is None or row.status == "signed_out":
            return MCPOAuthStatus(
                state="signed_out", scope=None, expires_at=None, issuer=None, resource=None
            )
        credential = row.credential
        if credential is None:
            return MCPOAuthStatus(
                state="needs_sign_in", scope=None, expires_at=None, issuer=None, resource=None
            )
        now = self._now()
        expired = credential.expires_at is not None and now >= credential.expires_at
        state: Literal["active", "expired", "refreshing", "retired", "needs_sign_in"]
        if row.status == "active" and credential.refresh_token is not None:
            claim = row.refresh_claim
            if claim is not None and claim.expires_at > now:
                state = "refreshing"
            else:
                state = "expired" if expired else "active"
        else:
            state = "needs_sign_in" if expired else "retired"
        return MCPOAuthStatus(
            state=state,
            scope=credential.scope,
            expires_at=credential.expires_at,
            issuer=credential.issuer,
            resource=credential.resource,
        )

    # -- revoke ------------------------------------------------------------------

    async def revoke(self, key: MCPAuthKey) -> MCPOAuthRevocation:
        """Sign ``key`` out: write a tombstone, then revoke at the server.

        Every call advances the row's generation, so any sign-in begun before
        it is superseded. The tombstone is written first and never undone;
        the remote revocation is best effort and its outcome is reported.
        """
        return await self._run(self._revoke(key))

    async def _revoke(self, key: MCPAuthKey) -> MCPOAuthRevocation:
        if not isinstance(key, MCPAuthKey):
            raise _flow_error("invalid_argument")
        replaced: MCPOAuthRecord | None = None
        written = False
        for _ in range(_PUBLISH_ATTEMPTS):
            row = await self._load(key)
            tombstone = MCPOAuthRecord(
                version=MCPOAuthVersion(_new_id(), 0),
                status="signed_out",
                credential=None,
                refresh_claim=None,
                retired_claim_id=None,
                updated_at=self._now(),
            )
            saved, cancelled = await self._write(
                self._save(key, tombstone, row.version if row is not None else None)
            )
            if cancelled:
                raise asyncio.CancelledError
            if saved:
                replaced, written = row, True
                break
        if not written:
            raise _flow_error("contention")
        local: Literal["signed_out", "already_signed_out"] = (
            "already_signed_out"
            if replaced is None or replaced.status == "signed_out"
            else "signed_out"
        )
        credential = replaced.credential if replaced is not None else None
        return MCPOAuthRevocation(local=local, remote=await self._remote_revoke(credential))

    async def _remote_revoke(
        self, credential: MCPOAuthCredential | None
    ) -> Literal["confirmed", "failed", "unsupported", "skipped"]:
        if credential is None:
            return "skipped"
        endpoint = credential.revocation_endpoint
        if endpoint is None:
            return "unsupported"
        if credential.client.secret_expired(self._now()):
            return "failed"
        if credential.refresh_token is not None:
            form = {"token": credential.refresh_token, "token_type_hint": "refresh_token"}
        else:
            form = {"token": credential.access_token, "token_type_hint": "access_token"}
        headers = _client_auth(credential.client, form)
        result = await self._send("POST", endpoint, form=form, headers=headers)
        if isinstance(result, OAuthTransportError) or result.status != 200:
            return "failed"
        return "confirmed"

    # -- credentials ---------------------------------------------------------------

    def credentials(self, key: MCPAuthKey, source: MCPSource) -> MCPRefreshingCredentialProvider:
        """Return a new refreshing credential provider for ``key`` and ``source``.

        The provider delivers an access token only when the stored credential
        was issued for exactly ``source``'s canonical URL.
        """
        return _OAuthCredentialProvider(self, key, source)


# --------------------------------------------------------------------------
# Provider
# --------------------------------------------------------------------------


def _verdict(
    row: MCPOAuthRecord | None,
    resource: str,
    now: float,
    *,
    started_active: bool,
) -> str:
    """Delivery verdict; evaluated in this exact order on every delivery."""
    if row is None or row.status == "signed_out":
        return "revoked" if started_active else "not_signed_in"
    credential = row.credential
    if credential is None:
        return "resource_mismatch"
    if credential.resource != resource:
        return "resource_mismatch"
    if credential.client.issuer != credential.issuer:
        return "resource_mismatch"
    if credential.expires_at is not None and now >= credential.expires_at:
        if credential.refresh_token is not None and row.status == "active":
            return _NEEDS_REFRESH
        return "expired"
    return _DELIVER


class _OAuthCredentialProvider:
    """The Dendrux OAuth credential: a refreshing provider over the store.

    Holds no issued-token state; every delivery re-reads and re-validates the
    stored row. Refresh is single-flight within this provider and serialized
    across processes by the store's refresh claim.
    """

    def __init__(self, oauth: MCPOAuth, key: MCPAuthKey, source: MCPSource) -> None:
        if not isinstance(key, MCPAuthKey):
            raise ValueError("MCPOAuth credentials require an MCPAuthKey.")
        if not isinstance(source, MCPSource) or source.url is None:
            raise ValueError("MCPOAuth credentials require an HTTP MCPSource.")
        try:
            self._resource = canonical_url(source.url)
        except URLRejectedError:
            raise ValueError("MCPOAuth credentials require a valid source URL.") from None
        self._oauth = oauth
        self.key = key
        self._source_name = source.name
        self._lock = asyncio.Lock()
        self._emit: Callable[[MCPRuntimeEvent], None] | None = None

    def __repr__(self) -> str:
        return "MCPOAuthCredentials()"

    @property
    def oauth(self) -> MCPOAuth:
        """The :class:`MCPOAuth` instance this provider belongs to."""
        return self._oauth

    def _event(self, event_type: Callable[..., MCPRuntimeEvent], **fields: Any) -> None:
        emit = self._emit
        if emit is None:
            return
        emit(
            event_type(
                tenant_key=self.key.tenant_key,
                connection_key=self.key.connection_key,
                source_name=self._source_name,
                **fields,
            )
        )

    def _required(self, reason: str, **fields: Any) -> MCPAuthorizationRequired:
        verdict: MCPAuthorizationReason = reason  # type: ignore[assignment]  # verdict strings are reasons
        return MCPAuthorizationRequired(self.key, verdict, **fields)

    async def get_auth(self) -> Mapping[str, str]:
        """Return ``{"Authorization": ...}`` for a valid, resource-bound token."""
        return await self._oauth._run(self._get_auth())

    async def on_auth_rejected(self) -> None:
        """Treat the current credential as rejected with HTTP 401."""
        await self._oauth._run(self._handle_rejection(401, None, None, compare=False))

    async def handle_rejection(
        self,
        status: int,
        challenge: MCPAuthChallenge | None,
        rejected_authorization: str | None,
    ) -> None:
        """React to a 401/403 for the exact ``Authorization`` value that was sent."""
        await self._oauth._run(
            self._handle_rejection(status, challenge, rejected_authorization, compare=True)
        )

    async def _get_auth(self) -> Mapping[str, str]:
        oauth = self._oauth
        start = await oauth._load(self.key)
        now = oauth._now()
        verdict = _verdict(start, self._resource, now, started_active=False)
        if verdict not in (_DELIVER, _NEEDS_REFRESH):
            raise self._required(verdict)
        assert start is not None and start.credential is not None
        credential = start.credential
        proactive = (
            start.status == "active"
            and credential.refresh_token is not None
            and credential.expires_at is not None
            and now >= credential.expires_at - oauth.refresh_skew
            and now - credential.published_at > oauth.refresh_skew
        )
        row: MCPOAuthRecord | None = start
        if verdict == _NEEDS_REFRESH:
            row = await self._refresh(start)
        elif proactive:
            failure: MCPOAuthRefreshError | None = None
            try:
                row = await self._refresh(start)
            except MCPOAuthRefreshError as exc:
                failure = exc
            if failure is not None:
                # A transient refresh failure must not take down a user whose
                # access token is still valid: deliver it if it still is.
                row = await oauth._load(self.key)
                if _verdict(row, self._resource, oauth._now(), started_active=True) != _DELIVER:
                    raise failure
        verdict = _verdict(row, self._resource, oauth._now(), started_active=True)
        if verdict == _NEEDS_REFRESH:
            verdict = "expired"
        if verdict != _DELIVER:
            raise self._required(verdict)
        assert row is not None and row.credential is not None
        return {"Authorization": row.credential.authorization}

    async def _handle_rejection(
        self,
        status: int,
        challenge: MCPAuthChallenge | None,
        rejected_authorization: str | None,
        *,
        compare: bool,
    ) -> None:
        oauth = self._oauth
        start = await oauth._load(self.key)
        verdict = _verdict(start, self._resource, oauth._now(), started_active=True)
        if verdict not in (_DELIVER, _NEEDS_REFRESH):
            raise self._required(verdict)
        assert start is not None and start.credential is not None
        if compare and start.credential.authorization != rejected_authorization:
            # Another worker already refreshed or stepped up.
            return
        if status == 401:
            if start.status != "active" or start.credential.refresh_token is None:
                raise self._required("expired")
            await self._refresh(start)
            return
        if status == 403 and challenge is not None and challenge.error == "insufficient_scope":
            raise self._required(
                "insufficient_scope",
                scopes=challenge.scope,
                resource_metadata_url=challenge.resource_metadata_url,
                status_code=403,
            )
        raise self._required("revoked", status_code=status)

    async def _refresh(self, start: MCPOAuthRecord) -> MCPOAuthRecord | None:
        async with self._lock:
            return await self._refresh_locked(start)

    def _retired(self, row: MCPOAuthRecord, claim_id: str) -> MCPOAuthRecord:
        assert row.credential is not None
        return replace(
            row,
            version=_next_version(row),
            status="retired",
            credential=replace(row.credential, refresh_token=None),
            refresh_claim=None,
            retired_claim_id=claim_id,
            updated_at=self._oauth._now(),
        )

    async def _cleanup(
        self, record: MCPOAuthRecord, expected: MCPOAuthVersion, outcome: str
    ) -> bool:
        """A shielded cleanup write; failures are logged with the outcome only."""
        oauth = self._oauth
        failed = False
        saved = False
        cancelled = False
        try:
            saved, cancelled = await oauth._write(oauth._save(self.key, record, expected))
        except MCPOAuthError:
            failed = True
        if failed or not saved:
            logger.warning("MCP OAuth refresh cleanup write did not apply (%s).", outcome)
        if cancelled:
            raise asyncio.CancelledError
        return saved

    async def _refresh_locked(self, start: MCPOAuthRecord) -> MCPOAuthRecord | None:
        oauth = self._oauth
        loop = asyncio.get_running_loop()
        deadline = loop.time() + oauth.oauth_timeout
        assert start.credential is not None
        claimed: MCPOAuthRecord | None = None
        claim: MCPOAuthClaim | None = None
        while True:
            row = await oauth._load(self.key)
            now = oauth._now()
            verdict = _verdict(row, self._resource, now, started_active=True)
            if verdict not in (_DELIVER, _NEEDS_REFRESH):
                raise self._required(verdict)
            assert row is not None and row.credential is not None
            if row.version.generation != start.version.generation:
                return row
            if row.credential.publication_id != start.credential.publication_id:
                return row
            if row.status == "retired":
                return row
            current = row.refresh_claim
            if current is not None and current.expires_at > now:
                if loop.time() >= deadline:
                    self._event(MCPOAuthRefreshFailed, outcome="contention", cause="claim")
                    raise MCPOAuthRefreshError("MCP OAuth refresh is held by another worker.")
                await asyncio.sleep(_CLAIM_POLL)
                continue
            if current is not None:
                # A dead or stalled claimant's refresh token is never replayed.
                retired = self._retired(row, current.id)
                saved, cancelled = await oauth._write(oauth._save(self.key, retired, row.version))
                if cancelled:
                    raise asyncio.CancelledError
                if not saved:
                    continue
                self._event(MCPOAuthRefreshFailed, outcome="expired_claim", cause="claim")
                return retired
            if loop.time() >= deadline:
                self._event(MCPOAuthRefreshFailed, outcome="contention", cause="claim")
                raise MCPOAuthRefreshError("MCP OAuth refresh is held by another worker.")
            claim = MCPOAuthClaim(id=_new_id(), expires_at=now + oauth.claim_ttl)
            claimed = replace(row, version=_next_version(row), refresh_claim=claim, updated_at=now)
            saved, cancelled = await oauth._write(oauth._save(self.key, claimed, row.version))
            if cancelled:
                if saved:
                    await self._cleanup(
                        replace(claimed, version=_next_version(claimed), refresh_claim=None),
                        claimed.version,
                        "unsent",
                    )
                raise asyncio.CancelledError
            if saved:
                break
        assert claimed is not None and claim is not None and claimed.credential is not None
        credential = claimed.credential
        assert credential.refresh_token is not None

        if credential.client.secret_expired(oauth._now()):
            await self._cleanup(self._retired(claimed, claim.id), claimed.version, "expired")
            raise self._required("expired")

        form = {
            "grant_type": "refresh_token",
            "refresh_token": credential.refresh_token,
            "resource": credential.resource,
        }
        headers = _client_auth(credential.client, form)
        attempt = SendAttempt()
        issued_at = oauth._now()
        try:
            result = await oauth._send(
                "POST", credential.token_endpoint, form=form, headers=headers, attempt=attempt
            )
        except asyncio.CancelledError:
            if attempt.started:
                await self._cleanup(self._retired(claimed, claim.id), claimed.version, "uncertain")
            else:
                await self._cleanup(
                    replace(claimed, version=_next_version(claimed), refresh_claim=None),
                    claimed.version,
                    "unsent",
                )
            raise

        if isinstance(result, OAuthTransportError) and result.outcome == "unsent":
            await self._cleanup(
                replace(claimed, version=_next_version(claimed), refresh_claim=None),
                claimed.version,
                "unsent",
            )
            self._event(MCPOAuthRefreshFailed, outcome="unsent", cause=result.cause)
            raise MCPOAuthRefreshError("MCP OAuth refresh request was not sent.")

        token: _Token | None = None
        cause: str
        if isinstance(result, OAuthTransportError):
            cause = result.cause
        else:
            cause = f"HTTP{result.status}"
            if result.status in (400, 401):
                data = _json_object(result.body)
                code = data.get("error") if data is not None else None
                if isinstance(code, str) and code in _REJECTED_ERRORS:
                    await self._cleanup(
                        self._retired(claimed, claim.id), claimed.version, "rejected"
                    )
                    self._event(MCPOAuthRefreshFailed, outcome="rejected", cause=cause)
                    raise self._required("revoked", status_code=result.status)
            elif result.status == 200:
                token = _parse_token(result.body)
                if token is None:
                    cause = "InvalidTokenResponse"
        if token is None:
            await self._cleanup(self._retired(claimed, claim.id), claimed.version, "uncertain")
            self._event(MCPOAuthRefreshFailed, outcome="uncertain", cause=cause)
            raise MCPOAuthRefreshError("MCP OAuth refresh outcome is uncertain.")

        now = oauth._now()
        effective_refresh = token.refresh_token or credential.refresh_token
        refreshed = replace(
            credential,
            access_token=token.access_token,
            refresh_token=effective_refresh,
            token_type=token.token_type,
            expires_at=issued_at + token.expires_in if token.expires_in is not None else None,
            scope=token.scope if token.scope is not None else credential.scope,
            publication_id=_new_id(),
            published_at=now,
        )
        success = MCPOAuthRecord(
            version=_next_version(claimed),
            status="active" if effective_refresh is not None else "retired",
            credential=refreshed,
            refresh_claim=None,
            retired_claim_id=None,
            updated_at=now,
        )
        saved, cancelled = await oauth._write(oauth._save(self.key, success, claimed.version))
        if cancelled:
            raise asyncio.CancelledError
        if saved:
            self._event(
                MCPOAuthRefreshed,
                expires_in=float(token.expires_in) if token.expires_in is not None else None,
            )
            return success
        row = await oauth._load(self.key)
        if (
            row is not None
            and row.status == "retired"
            and row.retired_claim_id == claim.id
            and row.version.generation == claimed.version.generation
        ):
            # Our claim expired and a reader retired the row on our behalf.
            # Nobody redeemed in between (no-replay rule), so our rotated
            # token is the live one: publish it with claim provenance.
            late = replace(success, version=_next_version(row))
            saved, cancelled = await oauth._write(oauth._save(self.key, late, row.version))
            if cancelled:
                raise asyncio.CancelledError
            if saved:
                self._event(
                    MCPOAuthRefreshed,
                    expires_in=float(token.expires_in) if token.expires_in is not None else None,
                )
                return late
            row = await oauth._load(self.key)
        # Our tokens belong to a superseded operation; discard them.
        return row
