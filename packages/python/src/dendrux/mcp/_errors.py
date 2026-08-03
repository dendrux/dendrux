"""Typed errors raised by the Dendrux MCP integration."""

from __future__ import annotations


class MCPError(RuntimeError):
    """Base class for Dendrux MCP runtime errors."""


class MCPConnectionError(MCPError):
    """An MCP source could not be connected to or discovered."""


class MCPToolCallError(MCPError):
    """An MCP server returned an unsuccessful tool result."""


class MCPResultTooLargeError(MCPError):
    """An MCP result exceeded the configured output boundary."""

