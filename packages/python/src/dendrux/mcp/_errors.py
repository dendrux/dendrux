"""Typed errors raised by the Dendrux MCP integration."""

from __future__ import annotations

from typing import Literal


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


class MCPCredentialError(MCPConnectionError):
    """Credentials for an MCP source could not be resolved or applied.

    Raised before anything is sent: the provider failed, timed out, or
    returned a value the transport cannot use, so no connection attempt was
    made. Provider exception text is application-authored and may embed the
    very secret being refreshed, so ``str()`` names only the failure class
    and the original exception is detached from the chain entirely.
    """


class MCPAuthenticationError(MCPConnectionError):
    """The MCP server rejected the source's credentials (HTTP 401/403).

    Typed separately from transport failure so applications can refresh or
    rotate credentials and evict, instead of retrying blindly. Raised from
    connect and discovery; ``status_code`` carries the rejecting status.
    """

    status_code: int | None = None


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
                "configuration. Evict or close it before rebinding with new settings."
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


class MCPOutcomeUnknownError(MCPError):
    """A tool call was interrupted with an unknown outcome.

    Raised when a forced eviction or shutdown tears the transport out from
    under a running call, and when the transport itself dies mid-call. The
    server may already have applied the effect, so the call must never be
    retried automatically. It is deliberately not an :class:`MCPToolCallError`
    subclass: handlers that retry failed tool calls must not catch this.
    """


class MCPResultTooLargeError(MCPError):
    """An MCP result exceeded the configured output boundary."""
