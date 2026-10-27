"""Typed errors raised by the Dendrux MCP integration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# Public error attributes are resolved at runtime by documentation tooling.
from dendrux.mcp._credentials import MCPAuthKey  # noqa: TC001
from dendrux.tools._limits import ToolResultTooLargeError


@dataclass(frozen=True, slots=True, repr=False)
class MCPOrigin:
    """Explicit diagnostic origin, without userinfo, path, query, or fragment.

    Hostnames may be private. These fields are opt-in diagnostics and are not
    included in error messages or the default representation.
    """

    scheme: str
    host: str
    port: int

    def __repr__(self) -> str:
        return "MCPOrigin()"


DiscoveryLimitReason = Literal[
    "repeated_cursor", "page_cap", "deadline", "tool_cap", "byte_cap", "cursor_length"
]


class MCPError(RuntimeError):
    """Base class for Dendrux MCP runtime errors."""


class MCPConnectionError(MCPError):
    """An MCP source could not be connected to or discovered.

    ``transport_detail`` carries the underlying failure with the source's
    configured credentials removed. It is deliberately kept out of ``str()``:
    that rendering becomes ``last_error``, which is emitted as a governance
    event and persisted.
    """

    transport_detail: str | None = None
    origin: MCPOrigin | None = None
    destination: MCPOrigin | None = None
    redirect_target: MCPOrigin | None = None

    @property
    def host(self) -> str | None:
        """Return the attempted hostname for opt-in diagnostics."""
        target = self.destination or self.origin
        return target.host if target is not None else None


class MCPDestinationDeniedError(MCPConnectionError):
    """A destination was denied before connecting or sending credentials."""


class MCPDiscoveryLimitError(MCPConnectionError):
    """``tools/list`` passed one of its bounds, so discovery stopped.

    ``reason`` is ``repeated_cursor`` (the server returned a page cursor it
    had already returned, so paging would never end), ``page_cap`` (more than
    ``MCPSource.max_discovery_pages`` pages) or ``deadline`` (discovery took
    longer than ``MCPSource.discovery_timeout``), ``tool_cap`` (more than
    ``MCPSource.max_discovery_tools`` tools), ``byte_cap`` (the responses were
    larger than ``MCPSource.max_discovery_bytes`` on the wire; HTTP sources
    only) or ``cursor_length`` (a cursor longer than 4096 characters).
    ``str()`` names the source
    and the reason only, never a cursor. A subclass of
    :class:`MCPConnectionError`, so existing handlers keep catching it.
    """

    reason: DiscoveryLimitReason = "deadline"


class MCPCredentialError(MCPConnectionError):
    """Credentials for an MCP source could not be resolved or applied.

    Raised before anything is sent: the provider failed, timed out, or
    returned a value the transport cannot use, so no connection attempt was
    made. Provider exception text is application-authored and may embed the
    very secret being refreshed, so ``str()`` names only the failure class
    and the original exception is detached from the chain entirely.
    """


_KNOWN_CHALLENGE_ERRORS = frozenset({"invalid_request", "invalid_token", "insufficient_scope"})


@dataclass(frozen=True, slots=True, repr=False)
class MCPAuthChallenge:
    """Validated ``Bearer`` challenge from a 401/403 ``WWW-Authenticate``.

    Safe by construction: free text (``error_description``, ``realm``) is
    never stored, and every kept field matches a narrow grammar. ``error`` is
    an RFC 6750 error code; ``scope`` holds RFC 6749 scope tokens (at most
    32); ``resource_metadata_url`` is the RFC 9728 protected-resource
    metadata hint, an absolute http(s) URL without userinfo or fragment.
    A field that fails validation is empty, never an exception.

    The values are still server-chosen, so the default representation shows
    only the well-known scheme and error codes, a scope count, and whether a
    metadata URL is present.
    """

    scheme: str | None = None
    error: str | None = None
    scope: tuple[str, ...] = ()
    resource_metadata_url: str | None = None

    def __repr__(self) -> str:
        scheme = self.scheme
        if scheme is not None and scheme.lower() != "bearer":
            scheme = "<other>"
        error = self.error
        if error is not None and error not in _KNOWN_CHALLENGE_ERRORS:
            error = "<other>"
        metadata = None if self.resource_metadata_url is None else "<set>"
        return (
            f"MCPAuthChallenge(scheme={scheme!r}, error={error!r}, "
            f"scope=<{len(self.scope)} items>, resource_metadata_url={metadata!r})"
        )


class MCPAuthenticationError(MCPConnectionError):
    """The MCP server rejected the source's credentials (HTTP 401/403).

    Typed separately from transport failure so applications can refresh or
    rotate credentials and evict, instead of retrying blindly. Raised from
    connect, discovery, and tool calls; ``status_code`` carries the rejecting
    status. ``request_rejected`` is True only when the rejecting response
    came from the source's own origin. ``challenge`` is the parsed
    ``WWW-Authenticate`` header of that response, or None when it sent none.
    """

    status_code: int | None = None
    request_rejected: bool = False
    challenge: MCPAuthChallenge | None = None


MCPAuthorizationReason = Literal[
    "not_signed_in", "expired", "revoked", "insufficient_scope", "resource_mismatch"
]


class MCPAuthorizationRequired(MCPConnectionError):  # noqa: N818 - name fixed by the public API
    """The user must sign in (again) before this MCP connection can be used.

    A verdict about the user's authorization, not about the server: it never
    counts toward the circuit breaker. ``reason`` says why:

    * ``not_signed_in`` — no credential, or the user signed out.
    * ``expired`` — the access token expired and cannot be refreshed.
    * ``revoked`` — the authorization server or resource server rejected the
      grant, or it was revoked while in use.
    * ``insufficient_scope`` — the server requires more scopes (step-up);
      ``scopes`` names the challenged scopes to add to the next sign-in.
    * ``resource_mismatch`` — the stored credential was issued for a different
      resource or issuer than the bound source; it is never delivered.

    ``scopes`` and ``resource_metadata_url`` are server-chosen values from a
    validated challenge. They are attributes only and never appear in the
    message or representation.
    """

    def __init__(
        self,
        key: MCPAuthKey,
        reason: MCPAuthorizationReason,
        *,
        scopes: tuple[str, ...] = (),
        resource_metadata_url: str | None = None,
        status_code: int | None = None,
    ) -> None:
        self.key = key
        self.reason: MCPAuthorizationReason = reason
        self.scopes = tuple(scopes)
        self.resource_metadata_url = resource_metadata_url
        self.status_code = status_code
        super().__init__(f"MCP authorization required ({reason}).")


class MCPOAuthError(MCPError):
    """Base class for MCP OAuth failures. Messages never carry values."""


MCPOAuthFlowReason = Literal[
    "invalid_argument",
    "unknown_state",
    "expired_state",
    "key_mismatch",
    "access_denied",
    "issuer_mismatch",
    "invalid_callback",
    "superseded",
    "contention",
    "token_exchange_failed",
    "discovery_failed",
    "registration_failed",
    "destination_denied",
    "insecure_url",
    "pkce_unsupported",
    "closed",
]


class MCPOAuthFlowError(MCPOAuthError):
    """A sign-in, sign-out, or OAuth lifecycle operation failed.

    ``reason`` is a fixed code; ``status_code`` is the HTTP status of the
    failing authorization-server response when there was one. Server text
    (``error_description``, reason phrases, bodies) is never kept.
    """

    def __init__(self, reason: MCPOAuthFlowReason, *, status_code: int | None = None) -> None:
        self.reason: MCPOAuthFlowReason = reason
        self.status_code = status_code
        super().__init__(f"MCP OAuth operation failed ({reason}).")


class MCPOAuthRefreshError(MCPOAuthError, MCPCredentialError):
    """A token refresh could not complete right now.

    Transient and value-free: the refresh was contended, never sent, or its
    outcome was uncertain (in which case the refresh token was retired and the
    remaining access token is still used until it expires). It never marks
    the connection's credentials as failed and never counts toward the
    circuit breaker; the next rejection may recover.
    """


class MCPBindingConflictError(MCPError):
    """A runtime connection identity was rebound to a conflicting source."""

    def __init__(
        self,
        identity: tuple[str | None, str],
        *,
        mismatch: Literal["transport", "endpoint", "configuration"],
    ) -> None:
        self.identity = identity
        self.mismatch = mismatch
        if mismatch == "configuration":
            message = (
                f"MCP binding {identity!r} has a live connection with a different "
                "configuration. Use MCPRuntime.rebind() to replace it, or evict first."
            )
        else:
            message = (
                f"MCP binding {identity!r} is already registered for a different "
                f"physical target ({mismatch} mismatch)."
            )
        super().__init__(message)


class MCPRuntimeClosedError(MCPError):
    """The managed runtime is shutting down or closed.

    Binding and connecting are refused for the rest of the process lifetime.
    Applications that recycle runtimes must create a new one.
    """


class MCPCapacityError(MCPError):
    """Base for runtime saturation failures.

    Always transient, and always raised before anything is sent: the request
    never reached the MCP server, so backing off and retrying is safe.
    """


class MCPConnectionStateError(MCPError):
    """Base for connection-identity lifecycle failures.

    Carries the ``(tenant_key, connection_key)`` identity so callers can log
    and branch without parsing messages.
    """

    def __init__(self, identity: tuple[str | None, str], message: str) -> None:
        self.identity = identity
        super().__init__(message)


class MCPConnectionEvictingError(MCPConnectionStateError):
    """The connection identity is mid-eviction.

    Transient: the same key may be bound again once the eviction completes.
    """


class MCPConnectionCapacityError(MCPConnectionStateError, MCPCapacityError):
    """The runtime had no free connection slot within the wait budget.

    Transient: the identity is still registered and the same handle succeeds
    once another connection is released, retired, or evicted.
    """

    def __init__(
        self,
        identity: tuple[str | None, str],
        *,
        limit: int,
        timeout: float,
    ) -> None:
        self.limit = limit
        self.timeout = timeout
        super().__init__(
            identity,
            f"MCP connection {identity!r} could not be opened: the runtime is at its "
            f"{limit}-connection limit and no slot became free within {timeout}s.",
        )


class MCPStaleConnectionError(MCPConnectionStateError):
    """The connection handle no longer matches its registered identity.

    Raised when the identity was evicted, was never bound, or was superseded
    by a rebind with different configuration. Bind again and use the new
    handle; retrying with this one can never succeed.
    """


class MCPConnectionLostError(MCPConnectionStateError):
    """The physical transport behind a managed connection died.

    Raised only for calls that were *never sent*: they were queued for, or
    routed to, an entry another call had already proven broken. Nothing
    reached the server, so acquiring tools again — which opens a replacement
    connection with freshly resolved credentials — and retrying is safe.
    The identity stays registered; only the physical transport is gone.
    """


class MCPCircuitOpenError(MCPConnectionStateError):
    """The connection's circuit breaker is open after repeated failures.

    Raised before anything is sent: new physical connections for this
    identity are rejected until the cooldown elapses, so an unavailable
    server is not hammered by a reconnect storm. ``retry_after`` is the
    remaining cooldown in seconds and is safe to surface to end users.
    After the cooldown one probe connection is attempted and concurrent
    callers share its outcome. Evicting the identity, or rebinding it with
    changed source configuration or ``credential_identity``, resets the
    circuit immediately.
    """

    def __init__(
        self,
        identity: tuple[str | None, str],
        *,
        retry_after: float,
        failure_count: int,
        last_failure: str | None,
    ) -> None:
        self.retry_after = retry_after
        self.failure_count = failure_count
        self.last_failure = last_failure
        detail = f" (last failure: {last_failure})" if last_failure else ""
        super().__init__(
            identity,
            f"MCP connection {identity!r} circuit is open after {failure_count} "
            f"consecutive connection failures{detail}. New connections are "
            f"rejected for another {retry_after:.1f}s; evict, or rebind with changed "
            "source configuration or credential_identity, to reset immediately.",
        )


class MCPCallCapacityError(MCPCapacityError):
    """No MCP tool-call slot became free within the wait budget.

    Load shedding, not failure: the call was never started, so the server
    state is untouched and the same call may simply be retried. It is
    deliberately not an :class:`MCPToolCallError` subclass, because nothing
    was actually attempted against the server.
    """

    def __init__(
        self,
        identity: tuple[str | None, str],
        *,
        tool: str,
        limit: int,
        timeout: float,
    ) -> None:
        self.identity = identity
        self.tool = tool
        self.limit = limit
        self.timeout = timeout
        super().__init__(
            f"MCP tool '{tool}' on connection {identity!r} was not started: the "
            f"runtime is at its {limit} in-flight call limit and no slot became "
            f"free within {timeout}s."
        )


class MCPToolCallError(MCPError):
    """An MCP server returned an unsuccessful tool result.

    ``connection_lost`` is True when the failure tree shows the transport or
    session died mid-call, rather than the tool merely failing over a working
    connection. The managed runtime consults it to fence the shared physical
    connection; the message itself is identical either way.
    """

    connection_lost: bool = False


class MCPSessionExpiredError(MCPToolCallError):
    """The server answered ``tools/call`` with HTTP 404 for the call's session.

    The MCP spec says a 404 to a request carrying ``Mcp-Session-Id`` means
    the server no longer knows that session (it restarted, a load balancer
    moved the client, or the session expired). The request was rejected
    before the tool ran, so it was **not executed**. The managed runtime
    fences the connection, opens a new session and retries the call once.
    Stateless servers (no session id) never raise it.
    """

    connection_lost = True


class MCPToolNotFoundError(MCPError):
    """``MCPConnection.call_tool`` named a tool the server does not list.

    No ``tools/call`` request was sent (connecting and discovery may have
    happened). ``tool_name`` is the caller-supplied name with configured
    credentials scrubbed, since a name can echo one; use
    ``MCPConnection.discover()`` to see the advertised catalog.
    """

    tool_name: str

    def __init__(self, tool_name: str) -> None:
        super().__init__(
            f"MCP server does not list a tool named {tool_name!r}. "
            "No tools/call request was sent. Use discover() to see the advertised tools."
        )
        self.tool_name = tool_name


class MCPOutcomeUnknownError(MCPError):
    """A tool call was interrupted with an unknown outcome.

    Raised when a forced eviction or shutdown tears the transport out from
    under a running call, and when the transport itself dies mid-call. The
    server may already have applied the effect, so the call must never be
    retried automatically. It is deliberately not an :class:`MCPToolCallError`
    subclass: handlers that retry failed tool calls must not catch this.
    """


class MCPResultTooLargeError(MCPError, ToolResultTooLargeError):
    """An MCP result exceeded the configured output boundary.

    In an Agent run the tool result becomes the ``size_limit`` error payload.
    """


class MCPInvalidToolSchemaError(ValueError):
    """A strict view found a tool whose argument schema cannot be validated against.

    Raised at discovery with ``strict=True`` (best-effort views skip the tool
    with reason ``invalid_schema``). A ``ValueError`` subclass, so existing
    handlers keep catching it; ``error_code_for`` gives ``invalid_tool_schema``.
    The message names the source, tool and problem kind, never schema text.
    """


class MCPToolNameCollisionError(ValueError):
    """An MCP tool's exposed name clashes with another tool on the Agent.

    Raised at discovery when the name equals a local tool, a tool from
    another MCP source, or the reserved ``use_skill``. Rename one side with
    ``connection.tools(rename=...)``. A ``ValueError`` subclass, so existing
    handlers keep catching it.
    """


# Most specific first: the first match wins.
_ERROR_CODES: tuple[tuple[tuple[type[BaseException], ...], str], ...] = (
    ((MCPAuthorizationRequired,), "authorization_required"),
    ((MCPAuthenticationError,), "authentication_failed"),
    ((MCPCredentialError,), "credential_error"),
    ((MCPDestinationDeniedError,), "destination_denied"),
    ((MCPDiscoveryLimitError,), "discovery_limit"),
    ((MCPCircuitOpenError,), "circuit_open"),
    ((MCPCapacityError,), "capacity"),
    ((MCPConnectionLostError, MCPConnectionError), "connection_failed"),
    ((MCPConnectionStateError,), "connection_state"),
    ((MCPToolNameCollisionError,), "tool_name_collision"),
    ((MCPToolNotFoundError,), "tool_not_found"),
    ((MCPInvalidToolSchemaError,), "invalid_tool_schema"),
    # Strict view naming raises plain ValueError; MCP failures never do.
    ((ValueError,), "invalid_tool_names"),
)


def error_code_for(exc: BaseException) -> str:
    """Return the stable machine code for an MCP discovery failure.

    Hosts branch on this instead of matching error text. The code depends on
    the exception class alone, most specific first, and is one of
    ``authorization_required``, ``authentication_failed``, ``credential_error``,
    ``destination_denied``, ``discovery_limit``, ``circuit_open``, ``capacity``,
    ``connection_failed``, ``connection_state``, ``tool_name_collision``,
    ``tool_not_found``, ``invalid_tool_schema``, ``invalid_tool_names`` or ``unknown``.
    """
    for kinds, code in _ERROR_CODES:
        if isinstance(exc, kinds):
            return code
    return "unknown"
