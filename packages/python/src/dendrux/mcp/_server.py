"""MCPServer — declarative MCP server configuration.

Connects lazily on first use. Manages transport lifecycle via
AsyncExitStack so the MCP session stays open for the agent's lifetime.
"""

from __future__ import annotations

import logging
import re
from contextlib import AsyncExitStack
from typing import TYPE_CHECKING, Any

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamable_http_client

from dendrux.types import ToolDef, ToolTarget

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

# Provider-safe tool name: Anthropic and OpenAI allow [a-zA-Z0-9_-], max 64 chars.
_PROVIDER_SAFE_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")

# Source name: alphanumeric + underscore + hyphen, no double underscore.
_SOURCE_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]+$")


def _validate_canonical_name(canonical: str, source_name: str, mcp_name: str) -> None:
    """Validate namespaced name is safe for all LLM providers."""
    if not _PROVIDER_SAFE_RE.match(canonical):
        raise ValueError(
            f"MCP tool '{mcp_name}' from source '{source_name}' produces "
            f"canonical name '{canonical}' which is not provider-safe. "
            f"Tool names must match [a-zA-Z0-9_-] and be <= 64 chars."
        )


def _sanitize_tool_name(name: str) -> str:
    """Replace non-provider-safe chars with underscores.

    Preserves alphanumeric, underscore, and hyphen. Everything else
    becomes underscore.
    """
    return re.sub(r"[^a-zA-Z0-9_-]", "_", name)


def _normalize_mcp_result(result: Any) -> Any:
    """Normalize MCP CallToolResult to a JSON-serializable Python object.

    Returns a Python object (dict, list, or str) — NOT a JSON string.
    The loop's _execute_tool() already calls json.dumps(result) on the
    return value, so returning a pre-serialized JSON string would
    double-encode it.

    Priority:
    1. structuredContent (if present) — return as dict/list
    2. text Content blocks — join with newlines, return as str
    3. non-text Content blocks — clear unsupported marker
    """
    if result.structuredContent is not None:
        return result.structuredContent

    texts = []
    for content in result.content:
        if content.type == "text":
            texts.append(content.text)
        else:
            texts.append(f"[unsupported content type: {content.type}]")
    return "\n".join(texts) if texts else ""


class MCPServer:
    """Declarative MCP server configuration.

    Connects lazily on first use via ``_discover()``. The MCP session
    stays open for the server's lifetime (agent-lifetime cache).
    Call ``close()`` to tear down transport and session.

    **Security:** stdio MCP servers run as local subprocesses with full
    environment access. Only use trusted MCP server implementations.

    Args:
        name: Source name for namespacing (e.g. ``"filesystem"``).
            Tool names become ``name__tool_name``.
        url: Streamable HTTP endpoint URL. Mutually exclusive with ``command``.
        command: stdio subprocess command as a list of args.
            Mutually exclusive with ``url``.
    """

    def __init__(
        self,
        name: str,
        *,
        url: str | None = None,
        command: list[str] | None = None,
    ) -> None:
        # Validate name
        if not name or not _SOURCE_NAME_RE.match(name):
            raise ValueError(
                f"MCPServer name '{name}' is not a valid identifier. "
                f"Must match [a-zA-Z0-9_-] and be non-empty."
            )
        if "__" in name:
            raise ValueError(
                f"MCPServer name '{name}' cannot contain '__'. "
                f"Double underscore is reserved as the namespace separator."
            )

        # Validate exactly one transport
        if (url is None) == (command is None):
            raise ValueError(
                "MCPServer requires exactly one transport: "
                "url='...' for HTTP or command=[...] for stdio."
            )

        # Validate command
        if command is not None:
            if not isinstance(command, list) or not command:
                raise ValueError("MCPServer command must be a non-empty list of strings.")
            if not all(isinstance(arg, str) for arg in command):
                raise ValueError("MCPServer command must contain only strings.")

        self.name = name
        self._url = url
        self._command = command
        self._exit_stack: AsyncExitStack | None = None
        self._session: ClientSession | None = None

    async def _discover(self) -> list[ToolDef]:
        """Connect to MCP server and discover tools.

        Opens transport + session via AsyncExitStack. Contexts stay
        open until close() is called. Idempotent — raises if already
        connected (agent-level caching prevents double calls, but
        the class itself should be safe).

        Returns adapted ToolDefs with namespaced names and meta.
        """
        if self._session is not None:
            raise RuntimeError(
                f"MCPServer '{self.name}' is already connected. Call close() before re-discovering."
            )
        self._exit_stack = AsyncExitStack()
        await self._exit_stack.__aenter__()

        try:
            if self._command:
                params = StdioServerParameters(
                    command=self._command[0],
                    args=self._command[1:],
                )
                read_stream, write_stream = await self._exit_stack.enter_async_context(
                    stdio_client(params)
                )
            else:
                assert self._url is not None
                read_stream, write_stream, _ = await self._exit_stack.enter_async_context(
                    streamable_http_client(self._url)
                )

            self._session = await self._exit_stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )
            await self._session.initialize()

            # Paginate through all tools — some servers split across pages
            all_tools = []
            cursor: str | None = None
            while True:
                tools_result = await self._session.list_tools(cursor=cursor)
                all_tools.extend(tools_result.tools)
                cursor = getattr(tools_result, "nextCursor", None)
                if not cursor:
                    break

            if not all_tools:
                logger.warning(
                    "MCP source '%s' discovered zero tools. "
                    "This may indicate a configuration problem.",
                    self.name,
                )

            tool_defs: list[ToolDef] = []
            seen_names: set[str] = set()

            for tool in all_tools:
                sanitized = _sanitize_tool_name(tool.name)
                canonical = f"{self.name}__{sanitized}"

                _validate_canonical_name(canonical, self.name, tool.name)

                if canonical in seen_names:
                    raise ValueError(
                        f"MCP source '{self.name}' has duplicate tool names after "
                        f"sanitization: '{tool.name}' → '{canonical}'. "
                        f"Two tools cannot share the same canonical name."
                    )
                seen_names.add(canonical)

                # Convert annotations to plain dict for JSON safety
                annotations: dict[str, Any] | None = None
                if tool.annotations is not None:
                    annotations = tool.annotations.model_dump(exclude_none=True)

                td = ToolDef(
                    name=canonical,
                    description=tool.description or "",
                    parameters=tool.inputSchema,
                    target=ToolTarget.SERVER,
                    meta={
                        "source_name": self.name,
                        "mcp_tool_name": tool.name,
                        "transport": "stdio" if self._command else "http",
                        "annotations": annotations,
                    },
                )
                tool_defs.append(td)

            return tool_defs

        except BaseException:
            await self.close()
            raise

    def _create_executor(self, mcp_tool_name: str) -> Callable[..., Any]:
        """Create a callable that executes an MCP tool via the session.

        The returned coroutine function has the same shape as a local
        @tool function: ``async def executor(**params) -> Any``.
        """
        session = self._session
        assert session is not None, "Cannot create executor before discovery"

        async def executor(**params: Any) -> Any:
            result = await session.call_tool(mcp_tool_name, params)
            normalized = _normalize_mcp_result(result)
            if result.isError:
                raise RuntimeError(normalized or "MCP tool returned an error")
            return normalized

        return executor

    async def close(self) -> None:
        """Close session + transport. Kills subprocess for stdio.

        Idempotent — safe to call multiple times.
        """
        if self._exit_stack is not None:
            try:
                await self._exit_stack.aclose()
            except Exception:
                logger.warning("MCPServer '%s' cleanup failed", self.name, exc_info=True)
            self._exit_stack = None
            self._session = None
