"""Typed errors raised by the Dendrux MCP integration."""

from __future__ import annotations

from typing import Literal


class MCPError(RuntimeError):
    """Base class for Dendrux MCP runtime errors."""


class MCPConnectionError(MCPError):
    """An MCP source could not be connected to or discovered."""


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


class MCPConnectionCapacityError(MCPConnectionStateError):
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


class MCPToolCallError(MCPError):
    """An MCP server returned an unsuccessful tool result."""


class MCPOutcomeUnknownError(MCPError):
    """A tool call was interrupted by forced eviction with an unknown outcome.

    The server may already have applied the effect, so the call must never be
    retried automatically. It is deliberately not an :class:`MCPToolCallError`
    subclass: handlers that retry failed tool calls must not catch this.
    """


class MCPResultTooLargeError(MCPError):
    """An MCP result exceeded the configured output boundary."""
