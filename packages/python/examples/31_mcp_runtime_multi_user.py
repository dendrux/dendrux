"""Multi-user MCP — one process-wide runtime serving isolated tenants.

Two users share one ``MCPRuntime``. Each gets a private tenant partition
keyed by ``(tenant_key, connection_key)``: their own filesystem server
subprocess rooted at their own workspace, established with their own
lazily resolved credentials. The runtime owns every physical connection;
agents only lease them. Two request rounds demonstrate that fresh agents
reuse those connections. A telemetry observer and post-request snapshot
show what the runtime is doing, and one ``close()`` drains everything.

Prerequisites:
    - Node.js + npx
    - ANTHROPIC_API_KEY in the repository-root .env or environment
    - ``pip install "dendrux[anthropic,mcp]"``

Run from ``packages/python``:
    python examples/31_mcp_runtime_multi_user.py
"""

from __future__ import annotations

import asyncio
import tempfile
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

from dendrux import Agent
from dendrux.mcp import MCPRuntime, MCPRuntimeEvent, MCPSource

load_dotenv(Path(__file__).resolve().parents[3] / ".env")


class EventCounter:
    """Bridge runtime lifecycle events to your metrics system.

    Events are typed, immutable, and value-free: identities, class names,
    counts, and durations — never credentials, arguments, or results.
    """

    def __init__(self) -> None:
        self.counts: Counter[str] = Counter()

    def on_event(self, event: MCPRuntimeEvent) -> None:
        self.counts[type(event).__name__] += 1


class TenantEnvironment:
    """Per-tenant credentials, resolved once per physical connection.

    In production this would fetch a token from your secret store. The
    resolved mapping only establishes the transport (environment variables
    for stdio, headers for HTTP) and is never persisted or logged.
    """

    def __init__(self, user_id: str) -> None:
        self.user_id = user_id

    async def get_auth(self) -> dict[str, str]:
        return {"DENDRUX_EXAMPLE_USER": self.user_id}


def seed_workspace(workspace: Path, note: str) -> None:
    (workspace / "notes.txt").write_text(note)


async def serve_user(runtime: MCPRuntime, user_id: str, workspace: Path) -> str:
    """One request handler: bind, lease through an agent, answer."""
    # bind() is synchronous, performs no I/O, and is cheap to repeat per
    # request. The physical connection opens on first tool discovery and
    # is reused by every later agent for this tenant.
    connection = runtime.bind(
        connection_key="filesystem",
        tenant_key=user_id,
        source=MCPSource.stdio(
            name="filesystem",
            command=[
                "npx",
                "-y",
                "@modelcontextprotocol/server-filesystem@2026.7.4",
                str(workspace),
            ],
            connect_timeout=60,  # allows a cold first-run npx download
            call_timeout=30,
        ),
        credentials=TenantEnvironment(user_id),
    )

    async with Agent(
        provider="anthropic:claude-haiku-4-5",
        prompt=(
            "You are this user's private filesystem assistant. "
            "Use the filesystem tools to answer from their workspace."
        ),
        # Allowlisting tools is an explicit governance decision.
        tool_sources=[connection.tools(allowed_tools=["list_directory", "read_text_file"])],
        max_iterations=6,
    ) as agent:
        result = await agent.run("Read notes.txt and repeat its exact contents.")
        return result.answer or ""


async def main() -> None:
    observer = EventCounter()
    with (
        tempfile.TemporaryDirectory(prefix="dendrux_user_a_") as alice_directory,
        tempfile.TemporaryDirectory(prefix="dendrux_user_b_") as bob_directory,
    ):
        workspaces = {
            "user-a": Path(alice_directory),
            "user-b": Path(bob_directory),
        }
        seed_workspace(workspaces["user-a"], "Alice's roadmap: ship the runtime docs.")
        seed_workspace(workspaces["user-b"], "Bob's reminder: rotate the API tokens.")

        async with MCPRuntime(
            max_connections=10,
            max_in_flight_calls=8,
            observer=observer,
        ) as runtime:
            for request_number in (1, 2):
                answers = await asyncio.gather(
                    *(serve_user(runtime, user_id, path) for user_id, path in workspaces.items())
                )
                for user_id, answer in zip(workspaces, answers, strict=True):
                    print(f"\n[request {request_number}, {user_id}] {answer}")

            # snapshot() is synchronous and thread-safe: poll it from a metrics
            # scraper on an interval. Both request rounds have released their
            # leases, while the two tenant connections remain warm.
            snapshot = runtime.snapshot()
            print(f"\nruntime state: {snapshot.state.value}")
            for connection_snapshot in snapshot.connections:
                print(
                    f"  {connection_snapshot.tenant_key}/{connection_snapshot.connection_key}: "
                    f"{connection_snapshot.status.value} "
                    f"(leases={connection_snapshot.leases}, "
                    f"active_calls={connection_snapshot.active_calls})"
                )

            # Four fresh Agents used only two physical MCP connections.
            print(f"physical connections opened: {observer.counts['MCPConnectionOpening']}")

        # The context manager drains leases and closes every transport once.
        print(f"after close: {runtime.snapshot().state.value}")

        print("\nlifecycle events observed:")
        for event_name, count in sorted(observer.counts.items()):
            print(f"  {event_name}: {count}")


asyncio.run(main())
