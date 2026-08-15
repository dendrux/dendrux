"""Let a Dendrux Agent use GitHub's hosted MCP server.

The Anthropic model must select the read-only ``github__get_me`` tool before
answering. Lifecycle output proves that the MCP call started and completed.

Prerequisites:
    - ``ANTHROPIC_API_KEY`` in the repository-root .env or environment
    - A narrowly scoped GitHub PAT in ``GITHUB_MCP_PAT``
    - ``pip install "dendrux[anthropic,mcp]" python-dotenv``

Run from ``packages/python``:

    export GITHUB_MCP_PAT="github_pat_..."
    python examples/32_mcp_github_remote.py

The token is read only from the process environment and is never persisted by
Dendrux. For a multi-user application, resolve each user's OAuth token through
``credentials=`` instead of placing it in the source configuration.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv

from dendrux import Agent
from dendrux.llm.anthropic import AnthropicProvider
from dendrux.mcp import (
    MCPRuntime,
    MCPRuntimeEvent,
    MCPSource,
    MCPToolCallCompleted,
    MCPToolCallStarted,
)

GITHUB_MCP_URL = "https://api.githubcopilot.com/mcp/"
load_dotenv(Path(__file__).resolve().parents[3] / ".env")


class ToolCallReporter:
    """Print value-free proof of MCP tool execution."""

    def __init__(self) -> None:
        self.completed_tools: list[str] = []

    def on_event(self, event: MCPRuntimeEvent) -> None:
        """Report MCP tool-call lifecycle events."""
        if isinstance(event, MCPToolCallStarted):
            print(f"MCP started:   {event.tool}")
        elif isinstance(event, MCPToolCallCompleted):
            self.completed_tools.append(event.tool)
            print(f"MCP completed: {event.tool} ({event.duration:.2f}s)")


async def run_github_agent(token: str) -> None:
    """Run an Anthropic Agent that must call GitHub before answering."""
    source = MCPSource.http(
        "github",
        GITHUB_MCP_URL,
        headers={
            "Authorization": f"Bearer {token}",
            "X-MCP-Readonly": "true",
            "X-MCP-Tools": "get_me",
        },
        connect_timeout=15.0,
        call_timeout=30.0,
    )
    reporter = ToolCallReporter()

    async with MCPRuntime(
        max_connections=1,
        max_in_flight_calls=1,
        idle_timeout=None,
        shutdown_timeout=5.0,
        observer=reporter,
    ) as runtime:
        connection = runtime.bind(
            tenant_key="local-example-user",
            connection_key="github-account",
            source=source,
        )
        async with Agent(
            name="GitHubProfileAgent",
            provider=AnthropicProvider(model=os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5")),
            prompt=(
                "You identify the connected GitHub user. You must call "
                "github__get_me before answering; never guess profile details."
            ),
            tool_sources=[connection.tools(allowed_tools=["get_me"])],
            max_iterations=4,
        ) as agent:
            result = await agent.run(
                "Use the GitHub tool to identify my account, then summarize "
                "the profile in two concise sentences."
            )

    if "github__get_me" not in reporter.completed_tools:
        raise RuntimeError("The Agent finished without completing github__get_me.")

    print(f"Run status: {result.status.value}")
    print(f"Agent answer: {result.answer}")


def main() -> None:
    token = os.environ.get("GITHUB_MCP_PAT", "").strip()
    if not token:
        raise SystemExit(
            "GITHUB_MCP_PAT is not set. Export a narrowly scoped GitHub PAT, "
            "then run this example again."
        )
    if not os.environ.get("ANTHROPIC_API_KEY", "").strip():
        raise SystemExit(
            "ANTHROPIC_API_KEY is not set. Add it to the repository-root .env "
            "or export it, then run this example again."
        )

    asyncio.run(run_github_agent(token))


if __name__ == "__main__":
    main()
