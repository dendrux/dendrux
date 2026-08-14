"""Credential-safe error contracts for the MCP SDK adapter."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import anyio
import httpx2
import pytest
from mcp.shared.exceptions import MCPError as SDKMCPError
from mcp.types import CONNECTION_CLOSED, REQUEST_TIMEOUT, CallToolResult, TextContent

from dendrux.mcp import MCPServer
from dendrux.mcp._client import MCPClientAdapter, is_connection_loss
from dendrux.mcp._errors import (
    MCPAuthenticationError,
    MCPConnectionError,
    MCPToolCallError,
)
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


@pytest.mark.asyncio
async def test_opaque_auth_suppresses_connect_cleanup_diagnostics(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Cleanup is part of the credential boundary too; an opaque auth
    object's secret must not reappear through a secondary debug log."""

    class _OpaqueAuth:
        token = "OPAQUE-CLEANUP-SECRET"

    source = MCPSource.http(
        "remote",
        "https://mcp.example.com/c",
        auth=_OpaqueAuth(),
    )
    _FailingClient.error_detail = "connect failed"
    _CleanupFailingStack.cleanup_detail = "cleanup echoed OPAQUE-CLEANUP-SECRET"

    with (
        caplog.at_level("DEBUG", logger="dendrux.mcp._client"),
        patch("dendrux.mcp._client.AsyncExitStack", _CleanupFailingStack),
        patch("dendrux.mcp._client.Client", _FailingClient),
        patch("dendrux.mcp._client.streamable_http_client", return_value=object()),
        pytest.raises(MCPConnectionError),
    ):
        await MCPClientAdapter(source).connect()

    assert "OPAQUE-CLEANUP-SECRET" not in caplog.text
    assert "detail suppressed" in caplog.text


@pytest.mark.asyncio
async def test_resolved_credentials_are_redacted_from_connect_failures() -> None:
    """Provider-resolved values join the source for one connect, so the same
    exact-substring scrub covers them when the transport echoes them back."""

    class _Provider:
        async def get_auth(self) -> Any:
            return {"Authorization": "Bearer RESOLVED-SECRET"}

    source = MCPSource.http("remote", "https://mcp.example.com/c")
    detail = "401 for url 'https://mcp.example.com/c' with header Bearer RESOLVED-SECRET"
    client_patch, http_patch, stdio_patch = _failing_sdk(detail)
    runtime = MCPRuntime(shutdown_timeout=0.05)
    connection = runtime.bind(connection_key="remote", source=source, credentials=_Provider())
    managed = _ViewToolSource(connection.tools())

    with client_patch, http_patch, stdio_patch, pytest.raises(MCPConnectionError) as excinfo:
        await managed._discover()

    error = excinfo.value
    assert "RESOLVED-SECRET" not in _rendered_chain(error)
    assert "RESOLVED-SECRET" not in (error.transport_detail or "")
    assert "[redacted]" in (error.transport_detail or "")
    assert "RESOLVED-SECRET" not in (managed.last_error or "")
    await runtime.close()


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


class _Response:
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code


class _HTTPStatusError(RuntimeError):
    """Shape-compatible stand-in for httpx.HTTPStatusError."""

    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.response = _Response(status_code)


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [401, 403])
async def test_server_credential_rejection_is_typed(status: int) -> None:
    """Applications must be able to branch on revoked credentials without
    parsing transport text."""
    secret = "REJECTED-SECRET"
    source = MCPSource.http("remote", f"https://mcp.example.com/c?access_token={secret}")
    adapter = MCPClientAdapter(source)

    class _RejectingClient:
        async def list_tools(self, cursor: Any = None, cache_mode: Any = None) -> Any:
            raise _HTTPStatusError(f"{status} rejected for url '{source.url}'", status)

    adapter._client = _RejectingClient()  # type: ignore[assignment]

    with pytest.raises(MCPAuthenticationError) as excinfo:
        await adapter.list_tools()

    error = excinfo.value
    assert isinstance(error, MCPConnectionError)  # existing handlers still catch it
    assert str(error) == (
        f"Failed to list tools from MCP source 'remote': "
        f"the server rejected its credentials (HTTP {status})."
    )
    assert error.status_code == status
    assert secret not in _rendered_chain(error)
    assert secret not in (error.transport_detail or "")


@pytest.mark.asyncio
async def test_credential_rejection_is_detected_through_the_cause_chain() -> None:
    """SDK layers wrap transport errors; the status must be found anyway."""
    source = MCPSource.http("remote", "https://mcp.example.com/c")
    adapter = MCPClientAdapter(source)

    class _WrappingClient:
        async def list_tools(self, cursor: Any = None, cache_mode: Any = None) -> Any:
            try:
                raise _HTTPStatusError("401 Unauthorized", 401)
            except _HTTPStatusError as exc:
                raise RuntimeError("transport request failed") from exc

    adapter._client = _WrappingClient()  # type: ignore[assignment]

    with pytest.raises(MCPAuthenticationError) as excinfo:
        await adapter.list_tools()

    assert excinfo.value.status_code == 401
    assert excinfo.value.__context__ is None


class _OpaqueAuth:
    """Stands in for an httpx Auth implementation carrying a secret."""

    token = "OPAQUE-SECRET"


@pytest.mark.asyncio
async def test_opaque_configured_auth_suppresses_transport_detail() -> None:
    """An auth object's secrets cannot be enumerated, so its transport text
    cannot be proven clean; suppression is the only sound treatment."""
    source = MCPSource.http("remote", "https://mcp.example.com/c", auth=_OpaqueAuth())
    adapter = _connected_adapter(source, "401 rejected header Bearer OPAQUE-SECRET")

    with pytest.raises(MCPConnectionError) as excinfo:
        await adapter.list_tools()

    assert excinfo.value.transport_detail is None
    assert "OPAQUE-SECRET" not in _rendered_chain(excinfo.value)


@pytest.mark.asyncio
async def test_tuple_configured_auth_is_redacted() -> None:
    """Basic-auth tuples carry exact strings, so they redact instead of
    costing the transport detail."""
    source = MCPSource.http(
        "remote",
        "https://mcp.example.com/c",
        auth=("svc-user", "svc-password-123"),
    )
    adapter = _connected_adapter(source, "401 for svc-user:svc-password-123")

    with pytest.raises(MCPConnectionError) as excinfo:
        await adapter.list_tools()

    detail = excinfo.value.transport_detail or ""
    assert "svc-password-123" not in detail
    assert detail == "401 for [redacted]:[redacted]"


@pytest.mark.asyncio
async def test_opaque_auth_suppresses_tool_error_detail() -> None:
    """Tool-call errors reach model context and the run store; with opaque
    auth the transport text cannot be proven clean there either."""
    source = MCPSource.http("remote", "https://mcp.example.com/c", auth=_OpaqueAuth())
    adapter = _connected_adapter(source, "boom with OPAQUE-SECRET inside")
    executor = create_mcp_executor(
        adapter,
        namespace="remote",
        mcp_tool_name="write",
        max_result_bytes=10_000,
    )

    with pytest.raises(MCPToolCallError) as excinfo:
        await executor(value=1)

    assert "OPAQUE-SECRET" not in _rendered_chain(excinfo.value)
    assert "RuntimeError" in str(excinfo.value)  # the class survives as a clue


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
async def test_server_reported_tool_error_is_suppressed_for_opaque_auth() -> None:
    """With an opaque auth object even server-authored error text could carry
    an echoed credential the runtime cannot recognise."""
    source = MCPSource.http("remote", "https://mcp.example.com/c", auth=_OpaqueAuth())

    class _EchoingClient:
        async def call_tool(self, name: str, arguments: dict[str, Any], **kwargs: Any) -> Any:
            return CallToolResult(
                content=[TextContent(text="rejected token OPAQUE-SECRET")],
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

    assert "OPAQUE-SECRET" not in str(excinfo.value)


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


@pytest.mark.parametrize(
    "error",
    [
        ConnectionResetError("peer reset"),
        BrokenPipeError("pipe closed"),
        ConnectionError("connection refused"),
        EOFError(),
        httpx2.ReadError("socket closed"),
        httpx2.ConnectError("no route to host"),
        httpx2.CloseError("close failed"),
        httpx2.WriteError("send failed"),
        httpx2.RemoteProtocolError("server disconnected without response"),
        anyio.BrokenResourceError(),
        anyio.ClosedResourceError(),
        anyio.EndOfStream(),
    ],
)
def test_transport_death_is_classified_as_connection_loss(error: BaseException) -> None:
    assert is_connection_loss(error)


@pytest.mark.parametrize(
    "error",
    [
        ValueError("bad arguments"),
        RuntimeError("tool exploded"),
        TimeoutError(),
        httpx2.ReadTimeout("slow server"),
        httpx2.ConnectTimeout("slow handshake"),
        httpx2.PoolTimeout("pool exhausted"),
        httpx2.LocalProtocolError("client-side bug"),
        _HTTPStatusError("401 Unauthorized", 401),
        MCPToolCallError("tool failed"),
    ],
)
def test_ordinary_failures_are_not_connection_loss(error: BaseException) -> None:
    assert not is_connection_loss(error)


def test_connection_loss_is_found_through_chains_and_groups() -> None:
    wrapper = RuntimeError("sdk wrapper")
    wrapper.__cause__ = httpx2.ReadError("socket closed")
    assert is_connection_loss(wrapper)

    handling = RuntimeError("raised while handling")
    handling.__context__ = ConnectionResetError("peer reset")
    assert is_connection_loss(handling)

    group = BaseExceptionGroup(
        "task group",
        [ValueError("unrelated"), ExceptionGroup("inner", [BrokenPipeError("pipe")])],
    )
    assert is_connection_loss(group)
    assert not is_connection_loss(ExceptionGroup("clean", [ValueError("still ordinary")]))


def test_official_sdk_connection_closed_is_classified_but_timeout_is_not() -> None:
    """The SDK normalizes a dead dispatcher to MCPError(CONNECTION_CLOSED),
    so recovery cannot depend only on its lower-level AnyIO cause."""
    assert is_connection_loss(SDKMCPError(code=CONNECTION_CLOSED, message="Connection closed"))
    assert not is_connection_loss(SDKMCPError(code=REQUEST_TIMEOUT, message="Request timed out"))

    wrapped = RuntimeError("SDK wrapper")
    wrapped.__cause__ = SDKMCPError(code=CONNECTION_CLOSED, message="Connection closed")
    assert is_connection_loss(wrapped)


@pytest.mark.asyncio
async def test_executor_tags_connection_loss_without_changing_the_error() -> None:
    source = MCPSource.http("github", "https://mcp.example.com")

    class _DyingClient:
        error: Exception = httpx2.ReadError("socket closed")

        async def call_tool(self, name: str, arguments: dict[str, Any], **kwargs: Any) -> Any:
            raise type(self).error

    adapter = MCPClientAdapter(source)
    adapter._client = _DyingClient()  # type: ignore[assignment]
    executor = create_mcp_executor(
        adapter,
        namespace="github",
        mcp_tool_name="write",
        max_result_bytes=10_000,
    )

    with pytest.raises(MCPToolCallError) as lost:
        await executor()
    assert lost.value.connection_lost is True
    assert str(lost.value) == "MCP tool 'github__write' call failed: socket closed"
    assert lost.value.__cause__ is None and lost.value.__context__ is None

    _DyingClient.error = RuntimeError("tool exploded")
    with pytest.raises(MCPToolCallError) as ordinary:
        await executor()
    assert ordinary.value.connection_lost is False
