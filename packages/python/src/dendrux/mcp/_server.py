"""Backwards-compatible MCPServer facade over the production MCP adapter."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Collection, Mapping, Sequence  # noqa: TC003
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from dendrux.mcp._client import MCPClientAdapter, is_connection_loss
from dendrux.mcp._destination import MCPDestinationPolicy  # noqa: TC001
from dendrux.mcp._errors import (
    MCPAuthenticationError,
    MCPAuthorizationRequired,
    MCPDestinationDeniedError,
    MCPInvalidToolSchemaError,
    MCPResultTooLargeError,
    MCPSessionExpiredError,
    MCPToolCallError,
)
from dendrux.mcp._result import error_text, has_image_blocks, normalize_mcp_result
from dendrux.mcp._source import (
    MCPFailureMode,
    MCPSource,
    redact_source_text,
    redact_source_value,
    redact_transport_text,
    safe_source_exception_detail,
    source_has_opaque_credentials,
)
from dendrux.tools._validate import schema_problem
from dendrux.types import ToolDef, ToolTarget

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


MCPSkipReason = Literal["name_unsafe", "duplicate_name", "unknown_tool", "invalid_schema"]


@dataclass(frozen=True, slots=True)
class MCPSkippedTool:
    """One server tool a view left out instead of failing its whole source.

    ``reason`` is ``name_unsafe`` (the exposed name is not provider-safe, or
    its base name is empty),
    ``duplicate_name`` (an earlier tool in server order already produced the
    exposed name), ``invalid_schema`` (the tool's argument schema has a shape
    that cannot be validated against) or ``unknown_tool`` (the view's
    ``allowed_tools``, ``force_serial_tools`` or ``rename`` names a tool the
    server does not list). ``tool`` is the server's tool name, or the requested name for
    ``unknown_tool``.
    """

    source_name: str
    namespace: str
    tool: str
    reason: MCPSkipReason


def exposed_tool_name(
    mcp_name: str,
    *,
    namespace: str,
    prefix: bool = True,
    rename: Mapping[str, str] | None = None,
) -> str:
    """Return the Agent-visible name of one server tool under a view's policy."""
    base = (rename or {}).get(mcp_name) or _sanitize_tool_name(mcp_name)
    return f"{namespace}__{base}" if prefix else base


def _schema_unusable(source: MCPSource, tool: Any, *, strict: bool) -> bool:
    """Whether ``tool``'s argument schema cannot be validated against (it is then skipped)."""
    issue = schema_problem(tool.input_schema)
    if issue is None:
        return False
    if strict:
        raise MCPInvalidToolSchemaError(
            f"MCP source '{source.name}' tool '{tool.name}' has a malformed "
            f"argument schema ({issue})."
        )
    logger.warning(
        "MCP tool skipped: code=invalid_schema source=%s tool=%s", source.name, tool.name
    )
    return True


def _build_tool_def(
    source: MCPSource,
    namespace: str,
    tool: Any,
    canonical: str,
    *,
    connection_info: Any,
    serial: bool,
) -> ToolDef:
    meta: dict[str, Any] = {
        "source_name": source.name,
        "namespace": namespace,
        "mcp_tool_name": tool.name,
        "exposed_name": canonical,
        "transport": source.transport,
        "annotations": _annotation_dict(tool.annotations),
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
    return ToolDef(
        name=canonical,
        description=tool.description or "",
        parameters=tool.input_schema,
        target=ToolTarget.SERVER,
        parallel=_tool_is_parallel_safe(tool.annotations) and not serial,
        timeout_seconds=source.call_timeout,
        has_explicit_timeout=True,
        meta=meta,
    )


def build_mcp_tool_defs(
    *,
    source: MCPSource,
    namespace: str,
    tools: Sequence[Any],
    connection_info: Any,
    force_serial_tools: frozenset[str] = frozenset(),
    allowed_tools: Collection[str] | None = None,
    prefix: bool = True,
    rename: Mapping[str, str] | None = None,
    strict: bool = False,
) -> tuple[list[ToolDef], list[MCPSkippedTool]]:
    """Adapt raw MCP tools to ToolDefs for one Agent view.

    Returns ``(tool_defs, skipped)``. Names in ``allowed_tools``,
    ``force_serial_tools`` or ``rename`` that the server does not list are
    reported first; then tools outside ``allowed_tools`` are dropped; then each
    remaining tool whose exposed name is unsafe or already taken is skipped.
    With ``strict=True`` the first such problem raises ``ValueError`` instead.
    """
    rename = rename or {}
    skipped: list[MCPSkippedTool] = []

    def skip(tool_name: str, reason: MCPSkipReason) -> None:
        skipped.append(
            MCPSkippedTool(
                source_name=source.name, namespace=namespace, tool=tool_name, reason=reason
            )
        )

    available = {tool.name for tool in tools}
    reported: set[str] = set()
    for field_name, requested in (
        ("allowed_tools", allowed_tools),
        ("force_serial_tools", force_serial_tools),
        ("rename", rename.keys()),
    ):
        unknown = sorted(set(requested or ()) - available)
        if unknown and strict:
            raise ValueError(
                f"MCP tool view '{namespace}' {field_name} references tools "
                f"unknown to source '{source.name}': {', '.join(unknown)}."
            )
        for name in unknown:
            if name not in reported:
                reported.add(name)
                skip(name, "unknown_tool")

    if allowed_tools is not None:
        tools = [tool for tool in tools if tool.name in allowed_tools]

    tool_defs: list[ToolDef] = []
    seen_names: set[str] = set()
    for tool in tools:
        canonical = exposed_tool_name(tool.name, namespace=namespace, prefix=prefix, rename=rename)
        # An empty base is never a real name, even though "ns__" alone would
        # pass the provider-safe pattern.
        empty_base = not exposed_tool_name(
            tool.name, namespace=namespace, prefix=False, rename=rename
        )
        if empty_base or not _PROVIDER_SAFE_RE.fullmatch(canonical):
            if strict:
                if empty_base:
                    raise ValueError(
                        f"MCP source '{source.name}' has a tool with an empty name, "
                        "which cannot be exposed."
                    )
                _validate_canonical_name(canonical, source.name, tool.name)
            skip(tool.name, "name_unsafe")
            continue
        if canonical in seen_names:
            if strict:
                raise ValueError(
                    f"MCP source '{source.name}' has duplicate tool names after "
                    f"sanitization: '{tool.name}' → '{canonical}'. "
                    "Two tools cannot share the same canonical name."
                )
            skip(tool.name, "duplicate_name")
            continue
        if _schema_unusable(source, tool, strict=strict):
            skip(tool.name, "invalid_schema")
            continue
        seen_names.add(canonical)
        tool_defs.append(
            _build_tool_def(
                source,
                namespace,
                tool,
                canonical,
                connection_info=connection_info,
                serial=tool.name in force_serial_tools,
            )
        )
    return tool_defs, skipped


def create_mcp_executor(
    adapter: MCPClientAdapter,
    *,
    namespace: str,
    mcp_tool_name: str,
    max_result_bytes: int,
    exposed_name: str | None = None,
    images: Literal["parts", "inline"] = "parts",
) -> Callable[..., Any]:
    """Create an async Dendrux executor bound to one live MCP adapter.

    ``exposed_name`` is the Agent-visible name used in error text; it defaults
    to the prefixed, sanitized name. The server is always called with
    ``mcp_tool_name``. ``images="parts"`` (Agent runs) returns image content
    as a ``ToolOutput``; ``"inline"`` keeps the legacy base64 envelope
    (``MCPConnection.call_tool``).
    """
    display_name = exposed_name or exposed_tool_name(mcp_tool_name, namespace=namespace)

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
            normalized = normalize_mcp_result(
                result, max_result_bytes=max_result_bytes, images=images
            )
            redacted = redact_source_value(adapter.source, normalized)
            is_error = bool(getattr(result, "is_error", getattr(result, "isError", False)))
            if is_error:
                # Server-authored text lands in the same sink, so it gets the
                # same scrub in case the server echoes a credential back.
                if opaque:
                    message = "MCP tool returned an error (detail suppressed: opaque auth)."
                elif images == "parts" and has_image_blocks(result):
                    # Never str() a ToolOutput: text blocks only, images omitted.
                    message = (
                        redact_source_text(adapter.source, error_text(result))
                        or "MCP tool returned an error"
                    )
                else:
                    message = str(redacted or "MCP tool returned an error")
                raise MCPToolCallError(message)
            return redacted
        # Unlike a connect failure, this text is worth keeping: the model reads
        # it to correct itself. So it is redacted rather than suppressed — and
        # raised outside the handler, because both the message and __context__
        # would otherwise reach the run store and the model's context window.
        if isinstance(
            failure,
            (
                MCPAuthenticationError,
                MCPAuthorizationRequired,
                MCPDestinationDeniedError,
                MCPResultTooLargeError,
                MCPSessionExpiredError,
            ),
        ):
            raise failure from None
        detail = (
            type(failure).__name__
            if opaque
            else redact_transport_text(adapter.source, str(failure)) or type(failure).__name__
        )
        error = MCPToolCallError(f"MCP tool '{display_name}' call failed: {detail}")
        # The managed runtime consults this to fence a dead transport; a tool
        # that merely failed must never poison the shared connection.
        error.connection_lost = is_connection_loss(failure)
        raise error from None

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
        follow_redirects: bool = True,
        max_redirects: int = 20,
        destination_policy: MCPDestinationPolicy | None = None,
        max_discovery_pages: int = 100,
        discovery_timeout: float = 60.0,
        max_discovery_tools: int = 2000,
        max_discovery_bytes: int = 16 * 1024 * 1024,
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
            follow_redirects=follow_redirects,
            max_redirects=max_redirects,
            destination_policy=destination_policy,
            max_discovery_pages=max_discovery_pages,
            discovery_timeout=discovery_timeout,
            max_discovery_tools=max_discovery_tools,
            max_discovery_bytes=max_discovery_bytes,
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
        self._skipped_tools: list[MCPSkippedTool] = []

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

            # The legacy facade has no view options and keeps failing loudly.
            tool_defs, _ = build_mcp_tool_defs(
                source=self.source,
                namespace=self.name,
                tools=all_tools,
                connection_info=adapter.info,
                strict=True,
            )
            return tool_defs
        except BaseException as exc:
            self.last_error = str(exc)
            await self.close()
            raise

    def _create_executor(
        self, mcp_tool_name: str, exposed_name: str | None = None
    ) -> Callable[..., Any]:
        """Create an async Dendrux executor bound to this MCP connection."""
        adapter = self._client
        if adapter is None:
            raise RuntimeError("Cannot create an MCP executor before discovery.")
        return create_mcp_executor(
            adapter,
            namespace=self.name,
            mcp_tool_name=mcp_tool_name,
            max_result_bytes=self.source.max_result_bytes,
            exposed_name=exposed_name,
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
