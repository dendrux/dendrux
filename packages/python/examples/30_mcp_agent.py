"""MCP + local tools — one agent, one shared production lifecycle.

The agent combines a regular Dendrux ``@tool`` with ``echo`` from the
official MCP conformance server. ``MCPSource`` describes the external tool
source; ``MCPHost`` owns its connection and discovered catalog.

Prerequisites:
    - Node.js + npx
    - ANTHROPIC_API_KEY in the repository-root .env or environment
    - ``pip install "dendrux[anthropic,mcp]"``

Run from ``packages/python``:
    python examples/30_mcp_agent.py
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from dotenv import load_dotenv

from dendrux import Agent, tool
from dendrux.mcp import MCPHost, MCPSource

load_dotenv(Path(__file__).resolve().parents[3] / ".env")


@tool()
async def count_words(text: str) -> dict[str, int]:
    """Count the words in a piece of text."""
    return {"word_count": len(text.split())}


async def main() -> None:
    # Configuration is declarative: nothing starts until the first discovery.
    everything = MCPSource.stdio(
        name="everything",
        command=["npx", "-y", "@modelcontextprotocol/server-everything@2026.7.4"],
        allowed_tools=("echo",),
        # Leave room for a first-run npx download and protocol negotiation.
        connect_timeout=60,
        call_timeout=30,
        max_result_bytes=100_000,
        failure_mode="strict",
    )

    # A host is application-owned. Multiple agents can safely share it; an
    # Agent.close() only releases its view and does not close the host.
    async with (
        MCPHost([everything]) as mcp_host,
        Agent(
            name="MCPExampleAgent",
            provider="anthropic:claude-haiku-4-5",
            prompt=(
                "You have an MCP echo tool and a local count_words tool. "
                "Always use both tools when the user asks you to echo and count text."
            ),
            tools=[count_words],
            tool_sources=[mcp_host],
            max_iterations=6,
        ) as agent,
    ):
        # Discovery is normally triggered by run(). Calling this explicitly
        # is useful in startup health checks and makes the example visible.
        lookups = await agent.get_tool_lookups()
        print(f"Available tools: {sorted(lookups.fn)}")

        result = await agent.run(
            "Use the MCP echo tool to echo 'Dendrux uses MCP tools safely'. "
            "Then use count_words on that exact sentence and report both results."
        )

        print(f"Status: {result.status.value}")
        print(f"Answer: {result.answer}")
        print(f"Iterations: {result.iteration_count}")


if __name__ == "__main__":
    asyncio.run(main())
