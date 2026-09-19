"""Narrow adapter over the official MCP Python SDK v2 client."""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import AsyncExitStack, suppress
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import anyio
import httpx2
from mcp import Client, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.exceptions import MCPError as SDKMCPError
from mcp.types import CONNECTION_CLOSED

from dendrux.mcp._errors import (
    MCPAuthenticationError,
    MCPConnectionError,
    MCPDestinationDeniedError,
    MCPOrigin,
)
from dendrux.mcp._http import create_http_client
from dendrux.mcp._source import (
    exception_class_name,
    safe_source_exception_detail,
    source_has_opaque_credentials,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from dendrux.mcp._source import MCPSource

logger = logging.getLogger(__name__)

_AUTH_REJECTION_STATUSES = (401, 403)

# Failures that mean the transport or session itself died. Timeouts are
# deliberately absent (a slow server is not a dead one), as are HTTP status
# errors (a rejection arrives over a working connection) and local protocol
# errors (a client-side bug does not invalidate the transport).
_CONNECTION_LOSS_TYPES: tuple[type[BaseException], ...] = (
    ConnectionError,
    EOFError,
    httpx2.NetworkError,
    httpx2.RemoteProtocolError,
    anyio.BrokenResourceError,
    anyio.ClosedResourceError,
    anyio.EndOfStream,
)


def _iter_error_tree(exc: BaseException) -> Iterator[BaseException]:
    """Yield every distinct exception reachable from one failure.

    SDK and transport layers wrap the original error, so the cause/context
    chains and exception-group branches are all walked.
    """
    stack: list[BaseException] = [exc]
    seen: set[int] = set()
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        yield node
        if isinstance(node, BaseExceptionGroup):
            stack.extend(node.exceptions)
        if node.__cause__ is not None:
            stack.append(node.__cause__)
        if node.__context__ is not None:
            stack.append(node.__context__)


def _authentication_status(exc: BaseException) -> int | None:
    """Find a credential-rejection HTTP status anywhere in an error tree."""
    for node in _iter_error_tree(exc):
        response = getattr(node, "response", None)
        status = getattr(response, "status_code", None)
        if status is None:
            status = getattr(node, "status_code", None)
        if status in _AUTH_REJECTION_STATUSES:
            return int(status)
    return None


def is_connection_loss(exc: BaseException) -> bool:
    """Whether an error tree shows the physical transport or session died.

    Used by the managed runtime to distinguish a dead connection — which
    must be fenced and replaced — from a tool that merely failed over a
    working one, which must never poison the shared connection.
    """
    return any(
        isinstance(node, _CONNECTION_LOSS_TYPES)
        or (isinstance(node, SDKMCPError) and node.code == CONNECTION_CLOSED)
        for node in _iter_error_tree(exc)
    )


@dataclass(frozen=True, slots=True)
class MCPConnectionInfo:
    """Negotiated server facts captured for audit metadata."""

    protocol_version: str | None
    server_name: str | None
    server_version: str | None
    instructions: str | None


def _diagnostic_origin(url: Any) -> MCPOrigin | None:
    try:
        parsed = httpx2.URL(url)
        if parsed.scheme not in ("http", "https") or not parsed.host:
            return None
        return MCPOrigin(
            parsed.scheme,
            parsed.host,
            parsed.port if parsed.port is not None else (443 if parsed.scheme == "https" else 80),
        )
    except (TypeError, ValueError, httpx2.InvalidURL):
        return None


@dataclass
class _ToolHTTPAttempt:
    adapter: MCPClientAdapter
    tool: str
    status: int | None = None
    destination: MCPOrigin | None = None


_tool_http_attempt: ContextVar[_ToolHTTPAttempt | None] = ContextVar(
    "mcp_tool_http_attempt", default=None
)


class MCPClientAdapter:
    """Own one SDK ``Client`` lifecycle without leaking SDK API publicly."""

    def __init__(self, source: MCPSource) -> None:
        self.source = source
        self._close_requested = asyncio.Event()
        self._connect_error: BaseException | None = None
        self._owner_task: asyncio.Task[None] | None = None
        self._ready = asyncio.Event()
        self._stack: AsyncExitStack | None = None
        self._client: Client | None = None
        self.info: MCPConnectionInfo | None = None
        self._last_status: int | None = None
        self._destination_denied = False
        self._last_destination = _diagnostic_origin(source.url) if source.url else None

    @property
    def connected(self) -> bool:
        return self._client is not None

    async def connect(self) -> None:
        if self._owner_task is not None:
            raise RuntimeError(f"MCP source '{self.source.name}' is already connected.")

        self._owner_task = asyncio.create_task(
            self._run_lifecycle(),
            name=f"dendrux-mcp-{self.source.name}",
        )
        try:
            await asyncio.shield(self._ready.wait())
        except asyncio.CancelledError:
            self._close_requested.set()
            self._owner_task.cancel()
            await asyncio.gather(self._owner_task, return_exceptions=True)
            raise
        if self._connect_error is not None:
            raise self._connect_error from None

    async def _run_lifecycle(self) -> None:
        """Enter and exit SDK contexts in this one owning task."""
        try:
            failure = await self._open()
        except BaseException as exc:
            failure = self._safe_error("connect to", exc) if isinstance(exc, Exception) else exc
        if failure is not None:
            self._connect_error = failure
            self._ready.set()
            return

        self._ready.set()
        try:
            await self._close_requested.wait()
        finally:
            await self._close_owned_stack()

    async def _observe_request(self, request: httpx2.Request) -> None:
        self._last_destination = _diagnostic_origin(request.url)

    def _observe_destination_denied(self) -> None:
        self._destination_denied = True

    async def _observe_response(self, response: Any) -> None:
        """Remember the status of the latest HTTP response.

        The SDK turns a non-2xx reply to the initialize POST into a JSON-RPC
        error that carries no status, so the exception tree alone cannot show
        a 401/403. The status recorded here is what lets ``_safe_error``
        still classify that failure as a credential rejection.
        """
        self._last_status = getattr(response, "status_code", None)
        attempt = _tool_http_attempt.get()
        if attempt is None or attempt.adapter is not self:
            return
        request = response.request
        if request.method != "POST":
            return
        try:
            body = json.loads(request.content)
        except (ValueError, httpx2.RequestNotRead):
            return
        if not isinstance(body, dict) or body.get("method") != "tools/call":
            return
        if body.get("params", {}).get("name") != attempt.tool:
            return
        attempt.destination = _diagnostic_origin(request.url)
        attempt.status = response.status_code

    async def _open(self) -> BaseException | None:
        """Open the transport, returning a detached safe failure if needed."""

        self._last_status = None
        self._destination_denied = False
        stack = AsyncExitStack()
        await stack.__aenter__()
        try:
            if self.source.url is not None:
                http_client = await stack.enter_async_context(
                    create_http_client(
                        self.source,
                        response_hook=self._observe_response,
                        request_hook=self._observe_request,
                        on_denied=self._observe_destination_denied,
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
            return None
        failure_destination = self._last_destination
        failure_status = self._last_status
        failure_denied = self._destination_denied
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
            return failure
        if cleanup_failure is not None and not isinstance(cleanup_failure, Exception):
            return cleanup_failure
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
        self._last_destination = failure_destination
        self._last_status = failure_status
        self._destination_denied = failure_denied
        return self._safe_error("connect to", failure)

    async def list_tools(self) -> list[Any]:
        client = self._require_client()
        self._last_status = None
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
        A recognisable 401/403 becomes a typed authentication failure, whether
        it is carried by the exception tree or was the last response seen.
        """
        detail: str | None
        if source_has_opaque_credentials(self.source):
            detail = None
        else:
            detail = safe_source_exception_detail(self.source, exc)
        failure_class = exception_class_name(exc)
        logger.debug(
            "MCP source '%s' failed to %s: %s (%s)",
            self.source.name,
            action,
            detail if detail is not None else "[detail suppressed: opaque auth]",
            failure_class,
        )
        status = _authentication_status(exc)
        if status is None and self._last_status in _AUTH_REJECTION_STATUSES:
            status = self._last_status
        error: MCPConnectionError
        if self._destination_denied or any(
            isinstance(node, MCPDestinationDeniedError) for node in _iter_error_tree(exc)
        ):
            error = MCPDestinationDeniedError(
                f"Failed to {action} MCP source '{self.source.name}': destination denied."
            )
            detail = None
        elif status is not None:
            auth_error = MCPAuthenticationError(
                f"Failed to {action} MCP source '{self.source.name}': "
                f"the server rejected its credentials (HTTP {status})."
            )
            auth_error.status_code = status
            error = auth_error
        else:
            error = MCPConnectionError(
                f"Failed to {action} MCP source '{self.source.name}' ({failure_class})."
            )
        error.transport_detail = detail
        error.origin = _diagnostic_origin(self.source.url) if self.source.url else None
        error.destination = self._last_destination
        for node in _iter_error_tree(exc):
            request = None
            with suppress(RuntimeError):
                request = getattr(node, "request", None)
            if request is not None:
                error.destination = _diagnostic_origin(request.url)
                break
        if error.destination != error.origin:
            error.redirect_target = error.destination
        return error

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        client = self._require_client()
        attempt = _ToolHTTPAttempt(self, name)
        token = _tool_http_attempt.set(attempt)
        try:
            return await client.call_tool(
                name,
                arguments,
                read_timeout_seconds=self.source.call_timeout,
            )
        except Exception as exc:
            failure = exc
        finally:
            _tool_http_attempt.reset(token)
        if attempt.status in _AUTH_REJECTION_STATUSES:
            error = MCPAuthenticationError(
                f"MCP source '{self.source.name}' rejected tool-call credentials "
                f"(HTTP {attempt.status})."
            )
            error.status_code = attempt.status
            error.request_rejected = attempt.destination == _diagnostic_origin(self.source.url)
            error.origin = _diagnostic_origin(self.source.url)
            error.destination = attempt.destination
            if error.destination != error.origin:
                error.redirect_target = error.destination
            raise error from None
        raise failure

    async def close(self) -> None:
        owner_task = self._owner_task
        if owner_task is None:
            self._client = None
            self.info = None
            return
        self._close_requested.set()
        if not self._ready.is_set():
            owner_task.cancel()
        await asyncio.shield(owner_task)

    async def _close_owned_stack(self) -> None:
        """Close the SDK stack from the same task that entered it."""
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
