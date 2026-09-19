"""Catalog snapshot returned by ``MCPConnection.discover()``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from dendrux.mcp._server import _annotation_dict, _tool_meta


@dataclass(frozen=True, slots=True)
class MCPToolInfo:
    """One tool exactly as the server advertises it.

    Raw server names, before Dendrux namespacing and before any
    :class:`MCPToolPolicy` is applied, so an application can show them
    and let a user choose an allowlist. Schemas are plain JSON-compatible
    dicts.
    """

    name: str
    description: str | None
    input_schema: dict[str, Any]
    title: str | None = None
    output_schema: dict[str, Any] | None = None
    annotations: dict[str, Any] | None = None
    meta: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class MCPDiscovery:
    """What one managed connection learned from its server.

    Server facts come from the initialize handshake; ``tools`` is the
    complete paginated ``tools/list`` catalog of the live physical
    connection, the same catalog every Agent leasing it sees.
    """

    server_name: str | None
    server_version: str | None
    protocol_version: str | None
    instructions: str | None
    tools: tuple[MCPToolInfo, ...]

    @property
    def tool_names(self) -> tuple[str, ...]:
        """Raw tool names, in server order."""
        return tuple(tool.name for tool in self.tools)


def build_discovery(connection_info: Any, raw_tools: list[Any]) -> MCPDiscovery:
    """Adapt negotiated server facts and SDK tool objects to public values."""
    tools = tuple(
        MCPToolInfo(
            name=tool.name,
            description=tool.description,
            input_schema=dict(tool.input_schema or {}),
            title=getattr(tool, "title", None),
            output_schema=(
                dict(tool.output_schema) if getattr(tool, "output_schema", None) else None
            ),
            annotations=_annotation_dict(tool.annotations),
            meta=_tool_meta(tool),
        )
        for tool in raw_tools
    )
    return MCPDiscovery(
        server_name=getattr(connection_info, "server_name", None),
        server_version=getattr(connection_info, "server_version", None),
        protocol_version=getattr(connection_info, "protocol_version", None),
        instructions=getattr(connection_info, "instructions", None),
        tools=tools,
    )
