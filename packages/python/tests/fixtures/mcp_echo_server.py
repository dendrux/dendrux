"""Minimal SDK v2 stdio server used by the Dendrux MCP client test."""

from mcp.server import MCPServer

server = MCPServer("dendrux-test-server", version="1.0.0")


@server.tool()
def echo(message: str) -> str:
    """Return the supplied message."""
    return message


if __name__ == "__main__":
    server.run()

