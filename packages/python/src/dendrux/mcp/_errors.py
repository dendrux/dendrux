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


class MCPToolCallError(MCPError):
    """An MCP server returned an unsuccessful tool result."""


class MCPResultTooLargeError(MCPError):
    """An MCP result exceeded the configured output boundary."""
