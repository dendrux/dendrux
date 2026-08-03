"""Narrow adapter over the official MCP Python SDK v2 client."""

from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import httpx2
from mcp import Client, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client

from dendrux.mcp._errors import MCPConnectionError

if TYPE_CHECKING:
    from dendrux.mcp._source import MCPSource


@dataclass(frozen=True, slots=True)
class MCPConnectionInfo:
    """Negotiated server facts captured for audit metadata."""

    protocol_version: str | None
    server_name: str | None
    server_version: str | None
    instructions: str | None


class MCPClientAdapter:
    """Own one SDK ``Client`` lifecycle without leaking SDK API publicly."""

    def __init__(self, source: MCPSource) -> None:
        self.source = source
        self._stack: AsyncExitStack | None = None
        self._client: Client | None = None
        self.info: MCPConnectionInfo | None = None

    @property
    def connected(self) -> bool:
        return self._client is not None

    async def connect(self) -> None:
        if self._client is not None:
            raise RuntimeError(f"MCP source '{self.source.name}' is already connected.")

        stack = AsyncExitStack()
        await stack.__aenter__()
        try:
            if self.source.url is not None:
                timeout = httpx2.Timeout(
                    self.source.connect_timeout,
                    read=self.source.call_timeout,
                )
                http_client = await stack.enter_async_context(
                    httpx2.AsyncClient(
                        headers=dict(self.source.headers),
                        auth=self.source.auth,
                        timeout=timeout,
                        follow_redirects=True,
                    )
                )
                transport = streamable_http_client(
                    self.source.url,
                    http_client=http_client,
                )
            else:
                assert self.source.command is not None
                params = StdioServerParameters(
                    command=self.source.command[0],
                    args=list(self.source.command[1:]),
                    env=dict(self.source.env) or None,
                    cwd=self.source.cwd,
                )
                transport = stdio_client(params)

            client = Client(transport, read_timeout_seconds=self.source.call_timeout)
            async with asyncio.timeout(self.source.connect_timeout):
                await stack.enter_async_context(client)

            server_info = client.server_info
            self.info = MCPConnectionInfo(
                protocol_version=str(client.protocol_version),
                server_name=getattr(server_info, "name", None),
                server_version=getattr(server_info, "version", None),
                instructions=client.instructions,
            )
            self._stack = stack
            self._client = client
        except BaseException as exc:
            await stack.aclose()
            if isinstance(exc, asyncio.CancelledError):
                raise
            raise MCPConnectionError(
                f"Failed to connect to MCP source '{self.source.name}': {exc}"
            ) from exc

    async def list_tools(self) -> list[Any]:
        client = self._require_client()
        tools: list[Any] = []
        cursor: str | None = None
        while True:
            page = await client.list_tools(cursor=cursor, cache_mode="refresh")
            tools.extend(page.tools)
            cursor = page.next_cursor
            if cursor is None:
                return tools

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        client = self._require_client()
        return await client.call_tool(
            name,
            arguments,
            read_timeout_seconds=self.source.call_timeout,
        )

    async def close(self) -> None:
        stack = self._stack
        self._stack = None
        self._client = None
        self.info = None
        if stack is not None:
            await stack.aclose()

    def _require_client(self) -> Client:
        if self._client is None:
            raise MCPConnectionError(f"MCP source '{self.source.name}' is not connected.")
        return self._client
