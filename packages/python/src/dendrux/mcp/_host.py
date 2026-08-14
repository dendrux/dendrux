"""Shared MCP runtime ownership for multi-agent applications."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence  # noqa: TC003
from typing import TYPE_CHECKING, Any

from dendrux.mcp._server import MCPServer
from dendrux.mcp._source import MCPSource

if TYPE_CHECKING:
    from dendrux.types import ToolDef


class _HostedMCPServer(MCPServer):
    """Agent-facing lease whose close operation does not close the host."""

    def __init__(self, host: MCPHost, name: str) -> None:
        self._host = host
        runtime = host._servers[name]
        self.source = runtime.source
        self.name = runtime.name
        self.failure_mode = runtime.failure_mode
        self._url = runtime._url
        self._command = runtime._command

    @property
    def last_error(self) -> str | None:
        return self._host._servers[self.name].last_error

    @last_error.setter
    def last_error(self, value: str | None) -> None:
        self._host._servers[self.name].last_error = value

    async def _discover(self) -> list[ToolDef]:
        return await self._host._discover(self.name)

    def _create_executor(self, mcp_tool_name: str) -> Callable[..., Any]:
        return self._host._create_executor(self.name, mcp_tool_name)

    async def close(self) -> None:
        """Release this agent view; the application-owned host stays open."""


class MCPHost:
    """Own and share MCP connections across agents in one runtime/tenant.

    Pass a host directly inside ``Agent(tool_sources=[host])``. Agents receive
    lightweight leases: ``Agent.close()`` clears their run catalog but cannot
    terminate connections used by other agents. The application must close the
    host at shutdown.

    Create separate hosts for different tenants or credential identities. A
    host intentionally never merges authentication contexts.
    """

    def __init__(self, sources: Sequence[MCPSource | MCPServer]) -> None:
        if not sources:
            raise ValueError("MCPHost requires at least one MCP source.")

        self._servers: dict[str, MCPServer] = {}
        for configured in sources:
            if isinstance(configured, MCPSource):
                server = MCPServer.from_source(configured)
            elif isinstance(configured, MCPServer):
                server = configured
            else:
                raise ValueError(
                    f"MCPHost source is {type(configured).__name__}, "
                    "not an MCPSource or MCPServer instance."
                )
            if server.name in self._servers:
                raise ValueError(f"MCPHost has duplicate source name '{server.name}'.")
            self._servers[server.name] = server

        self._catalogs: dict[str, tuple[ToolDef, ...]] = {}
        self._locks = {name: asyncio.Lock() for name in self._servers}
        self._closed = False
        self._tool_sources = tuple(_HostedMCPServer(self, name) for name in self._servers)

    @property
    def tool_sources(self) -> tuple[MCPServer, ...]:
        """Agent-facing source leases managed by this host."""
        return self._tool_sources

    async def _discover(self, source_name: str) -> list[ToolDef]:
        if self._closed:
            raise RuntimeError("MCPHost is closed and cannot be reused.")
        cached = self._catalogs.get(source_name)
        if cached is not None:
            return list(cached)

        async with self._locks[source_name]:
            cached = self._catalogs.get(source_name)
            if cached is None:
                discovered = await self._servers[source_name]._discover()
                cached = tuple(discovered)
                self._catalogs[source_name] = cached
            return list(cached)

    def _create_executor(self, source_name: str, mcp_tool_name: str) -> Callable[..., Any]:
        if source_name not in self._catalogs:
            raise RuntimeError(f"MCP source '{source_name}' has not been discovered.")
        return self._servers[source_name]._create_executor(mcp_tool_name)

    async def refresh(self) -> None:
        """Close connections and clear catalogs for the next agent run."""
        if self._closed:
            raise RuntimeError("MCPHost is closed and cannot be refreshed.")
        await self._close_servers()
        self._catalogs.clear()

    async def close(self) -> None:
        """Close every hosted connection. Idempotent."""
        if self._closed:
            return
        self._closed = True
        await self._close_servers()
        self._catalogs.clear()

    async def _close_servers(self) -> None:
        for server in self._servers.values():
            await server.close()

    async def __aenter__(self) -> MCPHost:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()
