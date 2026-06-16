"""Example 24: MCP against a current server + the provider recipe form.

A "does it still work" check using the newer pieces together:

  - provider recipe string: ``provider="anthropic:claude-haiku-4-5"``
    (dendrux builds the provider; mirrors database_url)
  - a different MCP server than the filesystem one: the official
    ``@modelcontextprotocol/server-everything`` conformance server, which
    exposes simple tools (echo, get-sum, get-env, ...) for exercising an
    MCP client. Note the current server names its add tool ``get-sum`` —
    dendrux discovers whatever the server advertises, so it just works.

Prerequisites:
    - Node.js + npx
    - ANTHROPIC_API_KEY in .env

Run:
    ANTHROPIC_API_KEY=sk-... python examples/24_mcp_recipe_provider.py
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from dotenv import load_dotenv

from dendrux import Agent
from dendrux.mcp import MCPServer

load_dotenv(Path(__file__).resolve().parents[3] / ".env")


async def main() -> None:
    server = MCPServer(
        "everything",
        command=["npx", "-y", "@modelcontextprotocol/server-everything"],
    )
    async with Agent(
        name="EverythingAgent",
        provider="anthropic:claude-haiku-4-5",  # recipe-form provider (new)
        prompt=(
            "You can call tools exposed by the 'everything' MCP server. "
            "Use the available arithmetic tool (named get-sum) to add numbers, "
            "then confirm the result."
        ),
        tool_sources=[server],
        max_iterations=6,
    ) as agent:
        # Prove discovery worked against the current server structure.
        lookups = await agent.get_tool_lookups()
        tool_names = sorted(lookups.fn.keys())
        print(f"Discovered {len(tool_names)} MCP tool(s): {tool_names[:8]}")

        result = await agent.run("Compute 25 + 17 using the arithmetic tool.")
        print(f"\nStatus: {result.status.value}")
        print(f"Answer: {result.answer}")
        print(f"Iterations: {result.iteration_count}")


if __name__ == "__main__":
    asyncio.run(main())
