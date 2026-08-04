"""Production MCP source, adapter facade, and safety-boundary tests."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from mcp.types import CallToolResult, TextContent, Tool, ToolAnnotations

from dendrux.agent import Agent
from dendrux.mcp import (
    MCPHost,
    MCPResultTooLargeError,
    MCPServer,
    MCPSource,
    MCPToolCallError,
)
from dendrux.mcp._client import MCPConnectionInfo
from dendrux.mcp._result import normalize_mcp_result
from dendrux.types import ToolDef, ToolTarget


class _FakeClientAdapter:
    tools: list[Tool] = []
    result: CallToolResult = CallToolResult(content=[TextContent(text="ok")])

    def __init__(self, source: MCPSource) -> None:
        self.source = source
        self._stack = object()
        self._client = object()
        self.info = MCPConnectionInfo(
            protocol_version="2026-07-28",
            server_name="fake-server",
            server_version="1.0.0",
            instructions=None,
        )
        self.closed = False

    async def connect(self) -> None:
        return None

    async def list_tools(self) -> list[Tool]:
        return list(self.tools)

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> CallToolResult:
        return self.result

    async def close(self) -> None:
        self.closed = True


class TestMCPSource:
    def test_http_configuration(self) -> None:
        source = MCPSource.http(
            "github",
            "https://mcp.example.com",
            headers={"Authorization": "Bearer secret"},
            allowed_tools=["search_code"],
            failure_mode="best_effort",
        )

        assert source.transport == "http"
        assert source.allowed_tools == frozenset({"search_code"})
        assert source.failure_mode == "best_effort"

    def test_stdio_configuration(self) -> None:
        source = MCPSource.stdio(
            "filesystem",
            ["server", "--root", "/workspace"],
            env={"LOG_LEVEL": "warning"},
            cwd="/workspace",
        )

        assert source.transport == "stdio"
        assert source.command == ("server", "--root", "/workspace")
        assert source.env == {"LOG_LEVEL": "warning"}

    def test_transport_specific_options_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="only valid for HTTP"):
            MCPSource(
                name="fs",
                command=("server",),
                headers={"Authorization": "secret"},
            )

        with pytest.raises(ValueError, match="only valid for stdio"):
            MCPSource(name="remote", url="https://example.com/mcp", env={"TOKEN": "x"})

    def test_agent_accepts_source_and_builds_compatibility_facade(self) -> None:
        source = MCPSource.http("github", "https://mcp.example.com")
        agent = Agent(prompt="test", tool_sources=[source])

        assert len(agent._tool_sources) == 1
        assert isinstance(agent._tool_sources[0], MCPServer)
        assert agent._tool_sources[0].source is source


class TestMCPToolAdaptation:
    @pytest.mark.asyncio
    async def test_real_sdk_v2_stdio_discovery_and_call(self) -> None:
        fixture = Path(__file__).parents[1] / "fixtures" / "mcp_echo_server.py"
        server = MCPServer(
            "echo",
            command=[sys.executable, str(fixture)],
            connect_timeout=10.0,
            call_timeout=10.0,
        )

        try:
            tool_defs = await server._discover()
            assert [tool.name for tool in tool_defs] == ["echo__echo"]
            assert tool_defs[0].meta["server_name"] == "dendrux-test-server"

            executor = server._create_executor("echo")
            assert await executor(message="hello through MCP") == {"result": "hello through MCP"}
        finally:
            await server.close()

    @pytest.mark.asyncio
    async def test_discovery_filters_tools_and_sets_safe_parallelism(self) -> None:
        _FakeClientAdapter.tools = [
            Tool(
                name="read",
                description="Read data",
                input_schema={"type": "object"},
                annotations=ToolAnnotations(read_only_hint=True, destructive_hint=False),
            ),
            Tool(
                name="delete",
                description="Delete data",
                input_schema={"type": "object"},
                annotations=ToolAnnotations(destructive_hint=True),
            ),
        ]
        source = MCPSource.http(
            "store",
            "https://mcp.example.com",
            allowed_tools=["read"],
            call_timeout=45.0,
        )
        server = MCPServer.from_source(source)

        with patch("dendrux.mcp._server.MCPClientAdapter", _FakeClientAdapter):
            tool_defs = await server._discover()

        assert [tool.name for tool in tool_defs] == ["store__read"]
        assert tool_defs[0].parallel is True
        assert tool_defs[0].timeout_seconds == 45.0
        assert tool_defs[0].meta["protocol_version"] == "2026-07-28"
        await server.close()

    @pytest.mark.asyncio
    async def test_tools_are_serial_by_default(self) -> None:
        _FakeClientAdapter.tools = [
            Tool(name="unknown", description="", input_schema={"type": "object"})
        ]
        server = MCPServer("store", url="https://mcp.example.com")

        with patch("dendrux.mcp._server.MCPClientAdapter", _FakeClientAdapter):
            tool_defs = await server._discover()

        assert tool_defs[0].parallel is False
        await server.close()

    @pytest.mark.asyncio
    async def test_best_effort_source_does_not_block_other_tools(self) -> None:
        optional = MCPServer(
            "optional",
            url="https://unavailable.example.com",
            failure_mode="best_effort",
        )

        async def fail_discovery() -> list[Any]:
            optional.last_error = "connection refused"
            raise ConnectionError("connection refused")

        optional._discover = fail_discovery  # type: ignore[method-assign]
        optional.close = AsyncMock()  # type: ignore[method-assign]
        agent = Agent(prompt="test", tool_sources=[optional])

        lookups = await agent.get_tool_lookups()

        assert not lookups.fn
        assert optional.last_error == "connection refused"


class TestMCPHost:
    @pytest.mark.asyncio
    async def test_host_shares_discovery_and_owns_shutdown(self) -> None:
        runtime = MCPServer("shared", url="https://mcp.example.com")
        discovered = [
            ToolDef(
                name="shared__read",
                description="Read",
                parameters={"type": "object"},
                target=ToolTarget.SERVER,
                parallel=False,
                meta={"source_name": "shared", "mcp_tool_name": "read"},
            )
        ]
        runtime._discover = AsyncMock(return_value=discovered)  # type: ignore[method-assign]

        async def execute(**params: Any) -> dict[str, Any]:
            return params

        def create_executor(name: str) -> Any:
            return execute

        runtime._create_executor = create_executor  # type: ignore[assignment]
        runtime.close = AsyncMock()  # type: ignore[method-assign]
        host = MCPHost([runtime])
        first = Agent(prompt="first", tool_sources=[host])
        second = Agent(prompt="second", tool_sources=[host])

        first_lookups = await first.get_tool_lookups()
        second_lookups = await second.get_tool_lookups()

        runtime._discover.assert_awaited_once()
        assert await first_lookups.fn["shared__read"](value=1) == {"value": 1}
        assert "shared__read" in second_lookups.fn

        await first.close()
        runtime.close.assert_not_awaited()

        await host.close()
        runtime.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_host_refresh_reconnects_on_next_discovery(self) -> None:
        runtime = MCPServer("shared", url="https://mcp.example.com")
        runtime._discover = AsyncMock(return_value=[])  # type: ignore[method-assign]
        runtime.close = AsyncMock()  # type: ignore[method-assign]
        host = MCPHost([runtime])

        await host.tool_sources[0]._discover()
        await host.refresh()
        await host.tool_sources[0]._discover()

        assert runtime._discover.await_count == 2
        runtime.close.assert_awaited_once()
        await host.close()


class TestMCPResultBoundaries:
    def test_large_result_is_rejected(self) -> None:
        result = CallToolResult(content=[TextContent(text="x" * 100)])

        with pytest.raises(MCPResultTooLargeError, match="configured limit"):
            normalize_mcp_result(result, max_result_bytes=20)

    @pytest.mark.asyncio
    async def test_mcp_error_result_becomes_typed_exception(self) -> None:
        server = MCPServer("store", url="https://mcp.example.com")
        adapter = _FakeClientAdapter(server.source)
        adapter.result = CallToolResult(
            content=[TextContent(text="permission denied")],
            is_error=True,
        )
        server._client = adapter  # type: ignore[assignment]

        executor = server._create_executor("write")
        with pytest.raises(MCPToolCallError, match="permission denied"):
            await executor(value="x")
