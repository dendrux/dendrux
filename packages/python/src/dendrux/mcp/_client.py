"""Narrow adapter over the official MCP Python SDK v2 client."""

from __future__ import annotations

import asyncio
import logging
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import httpx2
from mcp import Client, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client

from dendrux.mcp._errors import MCPAuthenticationError, MCPConnectionError
from dendrux.mcp._source import safe_source_exception_detail, source_has_opaque_credentials

if TYPE_CHECKING:
    from dendrux.mcp._source import MCPSource

logger = logging.getLogger(__name__)

_AUTH_REJECTION_STATUSES = (401, 403)


def _authentication_status(exc: BaseException) -> int | None:
    """Find a credential-rejection HTTP status anywhere in an error tree.

    SDK and transport layers wrap the original ``httpx`` status error, so the
    cause/context chain and exception-group branches are all searched.
    """
    stack: list[BaseException] = [exc]
    seen: set[int] = set()
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        response = getattr(node, "response", None)
        status = getattr(response, "status_code", None)
        if status is None:
            status = getattr(node, "status_code", None)
        if status in _AUTH_REJECTION_STATUSES:
            return int(status)
        if isinstance(node, BaseExceptionGroup):
            stack.extend(node.exceptions)
        if node.__cause__ is not None:
            stack.append(node.__cause__)
        if node.__context__ is not None:
            stack.append(node.__context__)
    return None


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
            failure: BaseException = exc
        else:
            return
        cleanup_failure: BaseException | None = None
        try:
            await stack.aclose()
        except BaseException as exc:
            cleanup_failure = exc
        # Cleanup must never replace the primary transport failure with an
        # unsafe exception chain. A cancellation or process-level signal still
        # wins; ordinary cleanup errors are retained only as redacted debug
        # diagnostics.
        if isinstance(failure, asyncio.CancelledError):
            raise failure from None
        if cleanup_failure is not None and not isinstance(cleanup_failure, Exception):
            raise cleanup_failure from None
        if cleanup_failure is not None:
            logger.debug(
                "MCP source '%s' cleanup after connect failure also failed: %s (%s)",
                self.source.name,
                safe_source_exception_detail(self.source, cleanup_failure),
                type(cleanup_failure).__name__,
            )
        # Raised after the handler has exited. Inside it, Python attaches the
        # original exception as __context__ even with `from None`, and
        # error-monitoring SDKs walk __context__ regardless of
        # __suppress_context__ — which would republish the endpoint.
        raise self._safe_error("connect to", failure) from None

    async def list_tools(self) -> list[Any]:
        client = self._require_client()
        try:
            tools: list[Any] = []
            cursor: str | None = None
            while True:
                page = await client.list_tools(cursor=cursor, cache_mode="refresh")
                tools.extend(page.tools)
                cursor = page.next_cursor
                if cursor is None:
                    return tools
        except Exception as exc:
            # Discovery fails on its own request, not just at connect: a token
            # can expire between the two. Its text reaches last_error and the
            # persisted MCP_ERROR governance event, so it needs the same care.
            failure: Exception = exc
        raise self._safe_error("list tools from", failure) from None

    def _safe_error(self, action: str, exc: BaseException) -> MCPConnectionError:
        """Build a transport error that cannot carry configured credentials.

        Only the exception class is rendered, because ``str()`` of this error
        becomes ``last_error`` and is persisted. The redacted detail stays
        reachable on the error and through debug logging so operators are not
        left with an undiagnosable failure — except when the source carries an
        opaque auth object, whose secrets cannot be enumerated, so its
        transport text cannot be proven clean and is suppressed outright.
        A recognisable 401/403 becomes a typed authentication failure.
        """
        detail: str | None
        if source_has_opaque_credentials(self.source):
            detail = None
        else:
            detail = safe_source_exception_detail(self.source, exc)
        logger.debug(
            "MCP source '%s' failed to %s: %s (%s)",
            self.source.name,
            action,
            detail if detail is not None else "[detail suppressed: opaque auth]",
            type(exc).__name__,
        )
        status = _authentication_status(exc)
        error: MCPConnectionError
        if status is not None:
            auth_error = MCPAuthenticationError(
                f"Failed to {action} MCP source '{self.source.name}': "
                f"the server rejected its credentials (HTTP {status})."
            )
            auth_error.status_code = status
            error = auth_error
        else:
            error = MCPConnectionError(
                f"Failed to {action} MCP source '{self.source.name}' ({type(exc).__name__})."
            )
        error.transport_detail = detail
        return error

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
