"""Credential-safe error contracts for the MCP SDK adapter."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest
from mcp.types import CallToolResult, TextContent

from dendrux.mcp import MCPServer
from dendrux.mcp._client import MCPClientAdapter
from dendrux.mcp._errors import MCPConnectionError, MCPToolCallError
from dendrux.mcp._runtime import MCPRuntime, _ViewToolSource
from dendrux.mcp._server import create_mcp_executor
from dendrux.mcp._source import MCPSource


class _FailingClient:
    error_detail = ""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    async def __aenter__(self) -> _FailingClient:
        raise RuntimeError(type(self).error_detail)

    async def __aexit__(self, *exc: Any) -> None:
        return None


def _rendered_chain(error: BaseException) -> str:
    """Every message a monitoring SDK would collect from the chain."""
    parts: list[str] = []
    node: BaseException | None = error
    while node is not None:
        parts.append(str(node))
        node = node.__cause__ or node.__context__
    return "\n".join(parts)


def _failing_sdk(detail: str) -> Any:
    _FailingClient.error_detail = detail
    return (
        patch("dendrux.mcp._client.Client", _FailingClient),
        patch("dendrux.mcp._client.streamable_http_client", return_value=object()),
        patch("dendrux.mcp._client.stdio_client", return_value=object()),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("source", "secret", "detail"),
    [
        (
            MCPSource.http(
                "remote",
                "https://mcp.example.com/connect?access_token=SIGNED-SECRET",
            ),
            "SIGNED-SECRET",
            "Client error for url 'https://mcp.example.com/connect?access_token=SIGNED-SECRET'",
        ),
        (
            MCPSource.stdio(
                "local",
                ["mcp-server", "--token", "ARGV-SECRET"],
            ),
            "ARGV-SECRET",
            "process failed: mcp-server --token ARGV-SECRET",
        ),
    ],
)
async def test_connect_error_never_contains_transport_detail(
    source: MCPSource,
    secret: str,
    detail: str,
) -> None:
    adapter = MCPClientAdapter(source)
    client_patch, http_patch, stdio_patch = _failing_sdk(detail)

    with client_patch, http_patch, stdio_patch, pytest.raises(MCPConnectionError) as excinfo:
        await adapter.connect()

    rendered = str(excinfo.value)
    assert rendered == f"Failed to connect to MCP source '{source.name}' (RuntimeError)."
    assert secret not in rendered
    assert detail not in rendered
    assert excinfo.value.__cause__ is None
    assert excinfo.value.__suppress_context__ is True
    # __suppress_context__ only stops traceback rendering. Error-monitoring
    # SDKs walk __context__ regardless, so the original must be detached.
    assert excinfo.value.__context__ is None
    assert secret not in _rendered_chain(excinfo.value)
    # Transport detail stays available for operators, minus the credential.
    assert secret not in (excinfo.value.transport_detail or "")


@pytest.mark.asyncio
async def test_safe_error_propagates_to_legacy_and_managed_last_error() -> None:
    secret = "PERSISTED-SECRET"
    source = MCPSource.http(
        "remote",
        f"https://mcp.example.com/connect?access_token={secret}",
    )
    detail = f"401 Unauthorized for url '{source.url}'"
    client_patch, http_patch, stdio_patch = _failing_sdk(detail)

    with client_patch, http_patch, stdio_patch:
        legacy = MCPServer.from_source(source)
        with pytest.raises(MCPConnectionError):
            await legacy._discover()

        runtime = MCPRuntime(shutdown_timeout=0.05)
        connection = runtime.bind(connection_key="remote", source=source)
        managed = _ViewToolSource(connection.tools())
        with pytest.raises(MCPConnectionError):
            await managed._discover()

    expected = "Failed to connect to MCP source 'remote' (RuntimeError)."
    assert legacy.last_error == expected
    assert managed.last_error == expected
    assert secret not in legacy.last_error
    assert secret not in managed.last_error
    await runtime.close()


class _CleanupFailingStack:
    """Enter the HTTP client, fail on MCP Client, then fail cleanup too."""

    cleanup_detail = ""

    def __init__(self) -> None:
        self.enter_calls = 0

    async def __aenter__(self) -> _CleanupFailingStack:
        return self

    async def enter_async_context(self, context: Any) -> Any:
        self.enter_calls += 1
        if self.enter_calls == 1:
            return context
        return await context.__aenter__()

    async def aclose(self) -> None:
        raise RuntimeError(type(self).cleanup_detail)


@pytest.mark.asyncio
async def test_cleanup_failure_cannot_replace_or_rechain_safe_connect_error() -> None:
    query_secret = "CONNECT-SECRET"
    cleanup_secret = "CLEANUP-SECRET"
    source = MCPSource.http(
        "remote",
        f"https://mcp.example.com/c?access_token={query_secret}",
        headers={"X-Cleanup-Token": cleanup_secret},
    )
    _FailingClient.error_detail = f"connect failed for {source.url}"
    _CleanupFailingStack.cleanup_detail = f"cleanup echoed {cleanup_secret}"

    with (
        patch("dendrux.mcp._client.AsyncExitStack", _CleanupFailingStack),
        patch("dendrux.mcp._client.Client", _FailingClient),
        patch("dendrux.mcp._client.streamable_http_client", return_value=object()),
        pytest.raises(MCPConnectionError) as excinfo,
    ):
        await MCPClientAdapter(source).connect()

    error = excinfo.value
    assert str(error) == "Failed to connect to MCP source 'remote' (RuntimeError)."
    assert error.__context__ is None
    assert query_secret not in _rendered_chain(error)
    assert cleanup_secret not in _rendered_chain(error)


class _RaisingClient:
    """Stands in for a connected SDK client whose requests fail."""

    def __init__(self, detail: str) -> None:
        self.detail = detail

    async def list_tools(self, cursor: Any = None, cache_mode: Any = None) -> Any:
        raise RuntimeError(self.detail)

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        read_timeout_seconds: Any = None,
    ) -> Any:
        raise RuntimeError(self.detail)


def _connected_adapter(source: MCPSource, detail: str) -> MCPClientAdapter:
    adapter = MCPClientAdapter(source)
    adapter._client = _RaisingClient(detail)  # type: ignore[assignment]
    return adapter


@pytest.mark.asyncio
async def test_discovery_failure_is_sanitised_like_connect() -> None:
    """A token can expire between connect and tools/list."""
    secret = "DISCOVERY-SECRET"
    source = MCPSource.http("remote", f"https://mcp.example.com/c?access_token={secret}")
    adapter = _connected_adapter(source, f"401 Unauthorized for url '{source.url}'")

    with pytest.raises(MCPConnectionError) as excinfo:
        await adapter.list_tools()

    error = excinfo.value
    assert str(error) == "Failed to list tools from MCP source 'remote' (RuntimeError)."
    assert secret not in _rendered_chain(error)
    assert error.__context__ is None
    assert error.transport_detail == "401 Unauthorized for url 'https://mcp.example.com/c'"


@pytest.mark.asyncio
async def test_tool_call_failure_keeps_diagnostics_but_drops_credentials() -> None:
    """This message becomes a ToolResult: persisted and replayed to the model."""
    secret = "CALL-SECRET"
    source = MCPSource.http(
        "remote",
        f"https://mcp.example.com/c?access_token={secret}",
        headers={"Authorization": "Bearer HEADER-SECRET"},
    )
    adapter = _connected_adapter(
        source,
        f"401 for url '{source.url}' with header Bearer HEADER-SECRET",
    )
    executor = create_mcp_executor(
        adapter,
        namespace="remote",
        mcp_tool_name="write",
        max_result_bytes=10_000,
    )

    with pytest.raises(MCPToolCallError) as excinfo:
        await executor(value=1)

    rendered = _rendered_chain(excinfo.value)
    assert secret not in rendered
    assert "HEADER-SECRET" not in rendered
    assert excinfo.value.__context__ is None
    # Unlike connect, the detail is retained: the model reads it to recover.
    assert "401 for url 'https://mcp.example.com/c'" in str(excinfo.value)


@pytest.mark.asyncio
async def test_server_reported_tool_error_is_also_scrubbed() -> None:
    """A server that echoes a credential back must not reach model context."""
    source = MCPSource.http(
        "remote",
        "https://mcp.example.com/c",
        headers={"Authorization": "Bearer ECHOED-SECRET"},
    )

    class _EchoingClient:
        async def call_tool(self, name: str, arguments: dict[str, Any], **kwargs: Any) -> Any:
            return CallToolResult(
                content=[TextContent(text="rejected token Bearer ECHOED-SECRET")],
                isError=True,
            )

    adapter = MCPClientAdapter(source)
    adapter._client = _EchoingClient()  # type: ignore[assignment]
    executor = create_mcp_executor(
        adapter,
        namespace="remote",
        mcp_tool_name="write",
        max_result_bytes=10_000,
    )

    with pytest.raises(MCPToolCallError) as excinfo:
        await executor(value=1)

    assert "ECHOED-SECRET" not in str(excinfo.value)


@pytest.mark.asyncio
async def test_successful_structured_result_is_recursively_scrubbed() -> None:
    source = MCPSource.http(
        "remote",
        "https://mcp.example.com/c?access_token=QUERY-SECRET",
        headers={"Authorization": "Bearer HEADER-SECRET"},
    )

    class _EchoingSuccessClient:
        async def call_tool(self, name: str, arguments: dict[str, Any], **kwargs: Any) -> Any:
            return CallToolResult(
                content=[],
                structuredContent={
                    "nested": [
                        "Bearer HEADER-SECRET",
                        {"QUERY-SECRET": "QUERY-SECRET"},
                    ],
                    "safe": 42,
                },
            )

    adapter = MCPClientAdapter(source)
    adapter._client = _EchoingSuccessClient()  # type: ignore[assignment]
    executor = create_mcp_executor(
        adapter,
        namespace="remote",
        mcp_tool_name="read",
        max_result_bytes=10_000,
    )

    result = await executor()

    assert result == {
        "nested": ["[redacted]", {"[redacted]": "[redacted]"}],
        "safe": 42,
    }
