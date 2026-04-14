"""MCP Filesystem Agent — use external tools from an MCP server.

Demonstrates Dendrux's MCP integration: the agent discovers tools from
an external MCP server (filesystem) and uses them alongside governance
(deny dangerous tools) and persistence (everything recorded in SQLite).

Prerequisites:
    - Node.js + npx installed
    - ANTHROPIC_API_KEY in .env

Run with:
    ANTHROPIC_API_KEY=sk-... python examples/13_mcp_filesystem.py

After running, inspect the data with:
    dendrux runs
    dendrux traces <run_id> --tools
    dendrux dashboard --db ~/.dendrux/dendrux.db
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

from dendrux import Agent
from dendrux.llm.anthropic import AnthropicProvider
from dendrux.mcp import MCPServer
from dendrux.notifiers.console import ConsoleNotifier

load_dotenv(Path(__file__).resolve().parents[3] / ".env")

console = Console()


async def main() -> None:
    # Use a temp directory so the example is self-contained
    work_dir = Path(tempfile.mkdtemp(prefix="dendrux_mcp_"))
    console.print(
        Panel(f"[bold]MCP Filesystem Agent[/bold]\nWork dir: {work_dir}", border_style="cyan")
    )

    # Seed a file so the agent has something to read
    (work_dir / "hello.txt").write_text(
        "Hello from Dendrux! This file was created before the agent ran."
    )

    db_path = Path.home() / ".dendrux" / "dendrux.db"

    async with Agent(
        name="FilesystemAgent",
        provider=AnthropicProvider(model="claude-sonnet-4-6"),
        prompt=(
            f"You are a file assistant. You can read and write files in {work_dir}. "
            "Use the filesystem tools to help the user. "
            "Always confirm what you did after completing a task."
        ),
        # MCP tools — discovered at runtime from the filesystem server
        tool_sources=[
            MCPServer(
                "filesystem",
                command=[
                    "npx",
                    "-y",
                    "@modelcontextprotocol/server-filesystem",
                    str(work_dir),
                ],
            ),
        ],
        # Governance — deny destructive tools
        deny=[
            "filesystem__move_file",
            "filesystem__edit_file",
        ],
        # Persistence + observability
        database_url=f"sqlite+aiosqlite:///{db_path}",
        max_iterations=10,
    ) as agent:
        console.print()
        console.print("[bold]Task 1:[/bold] Read the existing file")
        console.print()

        result = await agent.run(
            "Read the file hello.txt and tell me what it says.",
            notifier=ConsoleNotifier(),
        )
        console.print(f"\n[bold green]Answer:[/bold green] {result.answer}")
        console.print(
            f"[dim]Run {result.run_id} — {result.iteration_count} iterations, "
            f"{result.usage.total_tokens} tokens[/dim]"
        )

        console.print()
        console.print("[bold]Task 2:[/bold] Create a new file")
        console.print()

        result2 = await agent.run(
            "Create a file called notes.txt with the content: 'Meeting at 3pm with the team.'",
            notifier=ConsoleNotifier(),
        )
        console.print(f"\n[bold green]Answer:[/bold green] {result2.answer}")
        console.print(
            f"[dim]Run {result2.run_id} — {result2.iteration_count} iterations, "
            f"{result2.usage.total_tokens} tokens[/dim]"
        )

        # Verify the file was actually created
        notes_path = work_dir / "notes.txt"
        if notes_path.exists():
            console.print(f"\n[bold cyan]Verification:[/bold cyan] {notes_path} exists!")
            console.print(f"  Content: {notes_path.read_text()}")
        else:
            console.print(f"\n[bold red]File not created:[/bold red] {notes_path}")

        console.print()
        console.print("[bold]Task 3:[/bold] List directory contents")
        console.print()

        result3 = await agent.run(
            "List all files in the working directory.",
            notifier=ConsoleNotifier(),
        )
        console.print(f"\n[bold green]Answer:[/bold green] {result3.answer}")
        console.print(
            f"[dim]Run {result3.run_id} — {result3.iteration_count} iterations, "
            f"{result3.usage.total_tokens} tokens[/dim]"
        )

    console.print()
    console.print("[bold]Inspect persisted data with:[/bold]")
    console.print("  dendrux runs")
    console.print(f"  dendrux traces {result.run_id} --tools")
    console.print(f"  dendrux dashboard --db {db_path}")


if __name__ == "__main__":
    asyncio.run(main())
