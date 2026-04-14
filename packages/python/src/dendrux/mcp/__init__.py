"""MCP (Model Context Protocol) integration for Dendrux.

Provides MCPServer — a declarative configuration for connecting
to external MCP servers and discovering tools at runtime.

Usage:
    from dendrux.mcp import MCPServer

    agent = Agent(
        provider=provider,
        tool_sources=[
            MCPServer("filesystem", command=[
                "npx", "-y", "@modelcontextprotocol/server-filesystem", "/tmp",
            ]),
        ],
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

from dendrux.mcp._server import MCPServer  # noqa: E402

__all__ = ["MCPServer"]
