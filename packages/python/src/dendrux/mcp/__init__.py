"""MCP (Model Context Protocol) integration for Dendrux.

Provides a production client wrapper over the official MCP Python SDK.

Usage:
    from dendrux.mcp import MCPHost, MCPSource

    host = MCPHost([
        MCPSource.stdio("filesystem", command=[
            "npx", "-y", "@modelcontextprotocol/server-filesystem", "/tmp",
        ]),
    ])

    agent = Agent(
        provider=provider,
        tool_sources=[host],
    )

Requires the ``mcp`` package: ``pip install dendrux[mcp]``
"""

try:
    import mcp  # noqa: F401
except ModuleNotFoundError as err:
    if err.name == "mcp":
        raise ImportError(
            "MCP support requires the 'mcp' package. Install with: pip install dendrux[mcp]"
        ) from None
    raise  # Real import error from within the mcp package

from dendrux.mcp._errors import (  # noqa: E402
    MCPConnectionError,
    MCPError,
    MCPResultTooLargeError,
    MCPToolCallError,
)
from dendrux.mcp._host import MCPHost  # noqa: E402
from dendrux.mcp._server import MCPServer  # noqa: E402
from dendrux.mcp._source import MCPFailureMode, MCPSource  # noqa: E402

__all__ = [
    "MCPConnectionError",
    "MCPError",
    "MCPFailureMode",
    "MCPHost",
    "MCPResultTooLargeError",
    "MCPServer",
    "MCPSource",
    "MCPToolCallError",
]
