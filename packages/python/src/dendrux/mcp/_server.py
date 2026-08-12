"""Backwards-compatible MCPServer facade over the production MCP adapter."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from dendrux.mcp._client import MCPClientAdapter
from dendrux.mcp._errors import MCPToolCallError
from dendrux.mcp._result import normalize_mcp_result
from dendrux.mcp._source import (
    MCPFailureMode,
    MCPSource,
    redact_source_text,
    redact_source_value,
    safe_source_exception_detail,
    source_has_opaque_credentials,
)
from dendrux.types import ToolDef, ToolTarget

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

logger = logging.getLogger(__name__)

# Anthropic and OpenAI accept this portable subset, capped at 64 chars.
_PROVIDER_SAFE_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


def _validate_canonical_name(canonical: str, source_name: str, mcp_name: str) -> None:
    """Validate a namespaced tool name against provider restrictions."""
    if not _PROVIDER_SAFE_RE.fullmatch(canonical):
        raise ValueError(
            f"MCP tool '{mcp_name}' from source '{source_name}' produces "
            f"canonical name '{canonical}' which is not provider-safe. "
            "Tool names must match [a-zA-Z0-9_-] and be <= 64 chars."
        )


def _sanitize_tool_name(name: str) -> str:
    """Replace non-provider-safe characters with underscores."""
    return re.sub(r"[^a-zA-Z0-9_-]", "_", name)


def _normalize_mcp_result(result: Any, *, max_result_bytes: int = 1_000_000) -> Any:
    """Compatibility export for the result normalizer."""
    return normalize_mcp_result(result, max_result_bytes=max_result_bytes)


def _annotation_dict(annotations: Any) -> dict[str, Any] | None:
    if annotations is None:
        return None
    return cast(
        "dict[str, Any]",
        annotations.model_dump(mode="json", by_alias=True, exclude_none=True),
    )


def _tool_is_parallel_safe(annotations: Any) -> bool:
    """Only explicitly read-only, non-destructive MCP tools run in parallel."""
    if annotations is None:
        return False
    return bool(
        getattr(annotations, "read_only_hint", False)
        and not getattr(annotations, "destructive_hint", False)
    )


def _tool_meta(tool: Any) -> dict[str, Any] | None:
    value = getattr(tool, "meta", None)
    return dict(value) if value else None


def build_mcp_tool_defs(
    *,
    source: MCPSource,
    namespace: str,
    tools: Sequence[Any],
    connection_info: Any,
    force_serial_tools: frozenset[str] = frozenset(),
) -> list[ToolDef]:
    """Adapt raw MCP tools to namespaced ToolDefs for one Agent view."""
    tool_defs: list[ToolDef] = []
    seen_names: set[str] = set()
    for tool in tools:
        sanitized = _sanitize_tool_name(tool.name)
        canonical = f"{namespace}__{sanitized}"
        _validate_canonical_name(canonical, source.name, tool.name)
        if canonical in seen_names:
            raise ValueError(
                f"MCP source '{source.name}' has duplicate tool names after "
                f"sanitization: '{tool.name}' → '{canonical}'. "
                "Two tools cannot share the same canonical name."
            )
        seen_names.add(canonical)

        annotations = _annotation_dict(tool.annotations)
        meta: dict[str, Any] = {
            "source_name": source.name,
            "namespace": namespace,
            "mcp_tool_name": tool.name,
            "transport": source.transport,
            "annotations": annotations,
            "title": getattr(tool, "title", None),
            "output_schema": getattr(tool, "output_schema", None),
            "mcp_meta": _tool_meta(tool),
        }
        if connection_info is not None:
            meta.update(
                {
                    "protocol_version": connection_info.protocol_version,
                    "server_name": connection_info.server_name,
                    "server_version": connection_info.server_version,
                }
            )

        tool_defs.append(
            ToolDef(
                name=canonical,
                description=tool.description or "",
                parameters=tool.input_schema,
                target=ToolTarget.SERVER,
                parallel=(
                    _tool_is_parallel_safe(tool.annotations) and tool.name not in force_serial_tools
                ),
                timeout_seconds=source.call_timeout,
                has_explicit_timeout=True,
                meta=meta,
            )
        )
    return tool_defs


def create_mcp_executor(
    adapter: MCPClientAdapter,
    *,
    namespace: str,
    mcp_tool_name: str,
    max_result_bytes: int,
) -> Callable[..., Any]:
    """Create an async Dendrux executor bound to one live MCP adapter."""

    async def executor(**params: Any) -> Any:
        # With an opaque auth object even server-authored error text could
        # carry an echoed credential the runtime cannot recognise, so error
        # detail cannot be proven clean and is suppressed; mapping, string,
        # and tuple credentials keep their exact-substring scrub in full.
        opaque = source_has_opaque_credentials(adapter.source)
        try:
            result = await adapter.call_tool(mcp_tool_name, params)
        except Exception as exc:
            failure: Exception = exc
        else:
            normalized = normalize_mcp_result(result, max_result_bytes=max_result_bytes)
            redacted = redact_source_value(adapter.source, normalized)
            is_error = bool(getattr(result, "is_error", getattr(result, "isError", False)))
            if is_error:
                # Server-authored text lands in the same sink, so it gets the
                # same scrub in case the server echoes a credential back.
                message = (
                    "MCP tool returned an error (detail suppressed: opaque auth)."
                    if opaque
                    else str(redacted or "MCP tool returned an error")
                )
                raise MCPToolCallError(message)
            return redacted
        # Unlike a connect failure, this text is worth keeping: the model reads
        # it to correct itself. So it is redacted rather than suppressed — and
        # raised outside the handler, because both the message and __context__
        # would otherwise reach the run store and the model's context window.
        detail = (
            type(failure).__name__
            if opaque
            else redact_source_text(adapter.source, str(failure)) or type(failure).__name__
        )
        raise MCPToolCallError(
            f"MCP tool '{namespace}__{mcp_tool_name}' call failed: {detail}"
        ) from None

    return executor


class MCPServer:
    """One MCP tool source managed through the official SDK v2 client.

    ``MCPServer(name, url=... | command=[...])`` remains supported. New code
    may configure an :class:`MCPSource` and pass it directly to ``Agent``.

    HTTP sources support headers and any authentication object accepted by
    ``httpx2.AsyncClient`` (including the SDK's OAuth providers). Stdio
    sources support explicit environment additions and a working directory.

    Stdio processes execute with the user's privileges. Use only trusted
    implementations, or place them behind an isolated MCP gateway/runtime.
    """

    def __init__(
        self,
        name: str,
        *,
        url: str | None = None,
        command: list[str] | None = None,
        headers: Mapping[str, str] | None = None,
        auth: Any = None,
        env: Mapping[str, str] | None = None,
        cwd: str | Path | None = None,
        connect_timeout: float = 30.0,
        call_timeout: float = 120.0,
        max_result_bytes: int = 1_000_000,
        failure_mode: MCPFailureMode = "strict",
    ) -> None:
        # Keep the legacy error wording and private transport attributes while
        # all real configuration lives in the immutable MCPSource.
        if command is not None and (not isinstance(command, list) or not command):
            raise ValueError("MCPServer command must be a non-empty list of strings.")
        if command is not None and not all(isinstance(arg, str) for arg in command):
            raise ValueError("MCPServer command must contain only strings.")

        source = MCPSource(
            name=name,
            url=url,
            command=tuple(command) if command is not None else None,
            headers=dict(headers or {}),
            auth=auth,
            env=dict(env or {}),
            cwd=Path(cwd) if cwd is not None else None,
            connect_timeout=connect_timeout,
            call_timeout=call_timeout,
            max_result_bytes=max_result_bytes,
            failure_mode=failure_mode,
        )
        self._configure(source)

    @classmethod
    def from_source(cls, source: MCPSource) -> MCPServer:
        """Create the runtime facade for an immutable source configuration."""
        server = cls.__new__(cls)
        server._configure(source)
        return server

    def _configure(self, source: MCPSource) -> None:
        self.source = source
        self.name = source.name
        self.failure_mode = source.failure_mode
        self._url = source.url
        self._command = list(source.command) if source.command is not None else None
        self._client: MCPClientAdapter | None = None
        # Retained as compatibility/debugging views; lifecycle belongs to _client.
        self._exit_stack: Any = None
        self._session: Any = None
        self.last_error: str | None = None

    async def _discover(self) -> list[ToolDef]:
        """Connect once, discover all allowed tools, and adapt them to ToolDef."""
        if self._client is not None or self._session is not None:
            raise RuntimeError(
                f"MCPServer '{self.name}' is already connected. Call close() before re-discovering."
            )

        adapter = MCPClientAdapter(self.source)
        self.last_error = None
        try:
            await adapter.connect()
            self._client = adapter
            self._exit_stack = adapter._stack
            self._session = adapter._client
            all_tools = await adapter.list_tools()

            if not all_tools:
                logger.warning(
                    "MCP source '%s' discovered zero tools. "
                    "This may indicate a configuration problem.",
                    self.name,
                )

            return build_mcp_tool_defs(
                source=self.source,
                namespace=self.name,
                tools=all_tools,
                connection_info=adapter.info,
            )
        except BaseException as exc:
            self.last_error = str(exc)
            await self.close()
            raise

    def _create_executor(self, mcp_tool_name: str) -> Callable[..., Any]:
        """Create an async Dendrux executor bound to this MCP connection."""
        adapter = self._client
        if adapter is None:
            raise RuntimeError("Cannot create an MCP executor before discovery.")
        return create_mcp_executor(
            adapter,
            namespace=self.name,
            mcp_tool_name=mcp_tool_name,
            max_result_bytes=self.source.max_result_bytes,
        )

    async def close(self) -> None:
        """Close the SDK client and underlying HTTP/subprocess transport."""
        client = self._client
        legacy_stack = self._exit_stack if client is None else None
        self._client = None
        self._exit_stack = None
        self._session = None
        try:
            if client is not None:
                await client.close()
            elif legacy_stack is not None:
                await legacy_stack.aclose()
        except Exception as exc:
            source = client.source if client is not None else self.source
            logger.warning(
                "MCPServer '%s' cleanup failed: %s (%s)",
                self.name,
                safe_source_exception_detail(source, exc),
                type(exc).__name__,
            )
