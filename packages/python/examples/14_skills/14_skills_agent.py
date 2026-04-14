"""Skills + MCP Agent — composable capabilities with governance.

Demonstrates skills (instruction packages) working alongside MCP tools
and governance. The agent has two skills but one is denied by policy.

Prerequisites:
    - Node.js + npx installed
    - ANTHROPIC_API_KEY in .env

Run with:
    cd examples/14_skills
    ANTHROPIC_API_KEY=sk-... python 14_skills_agent.py

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

load_dotenv(Path(__file__).resolve().parents[4] / ".env")

console = Console()

SKILLS_DIR = Path(__file__).parent / "skills"


async def main() -> None:
    work_dir = Path(tempfile.mkdtemp(prefix="dendrux_skills_"))

    # Seed files for the agent to work with
    (work_dir / "meeting-notes.txt").write_text(
        "Q2 Planning Meeting - April 14, 2026\n"
        "Attendees: Alice, Bob, Charlie\n\n"
        "Key decisions:\n"
        "- Launch new API by June 1\n"
        "- Hire two more engineers\n"
        "- Move to weekly standups instead of daily\n\n"
        "Action items:\n"
        "- Alice: Draft API spec by April 21\n"
        "- Bob: Post job listings by April 18\n"
        "- Charlie: Update sprint cadence in Jira\n\n"
        "Budget approved: $150K for Q2 infrastructure.\n"
    )

    (work_dir / "project-ideas.txt").write_text(
        "Ideas for Q3:\n"
        "1. Build a customer dashboard with real-time metrics\n"
        "2. Automate the onboarding flow with agent workflows\n"
        "3. Add multi-language support for the API docs\n"
        "4. Explore partnership with Acme Corp for distribution\n"
    )

    db_path = Path.home() / ".dendrux" / "dendrux.db"

    # Show what skills are available
    from dendrux.skills import Skill

    all_skills = Skill.scan_dir(SKILLS_DIR)
    console.print(
        Panel(
            f"[bold]Skills + MCP Agent Demo[/bold]\n"
            f"Work dir: {work_dir}\n"
            f"Skills: {', '.join(s.name for s in all_skills)}\n"
            f"Denied: organize-notes",
            border_style="cyan",
        )
    )

    async with Agent(
        name="SkillsAgent",
        provider=AnthropicProvider(model="claude-sonnet-4-6"),
        prompt=(
            f"You are a document assistant. You can read and write files in {work_dir}. "
            "Use your skills and filesystem tools to help the user. "
            "Always confirm what you did after completing a task."
        ),
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
        skills_dir=str(SKILLS_DIR),
        deny_skills=["organize-notes"],
        deny=[
            "filesystem__move_file",
            "filesystem__edit_file",
        ],
        database_url=f"sqlite+aiosqlite:///{db_path}",
        max_iterations=10,
    ) as agent:
        # Task 1: Use the summarize-file skill
        console.print()
        console.print("[bold]Task 1:[/bold] Summarize meeting notes (uses summarize-file skill)")
        console.print()

        result = await agent.run(
            "Summarize the meeting-notes.txt file.",
            notifier=ConsoleNotifier(),
        )
        console.print(f"\n[bold green]Answer:[/bold green] {result.answer}")
        console.print(
            f"[dim]Run {result.run_id} — {result.iteration_count} iterations, "
            f"{result.usage.total_tokens} tokens[/dim]"
        )

        # Task 2: Try to use the denied skill
        console.print()
        console.print(
            "[bold]Task 2:[/bold] Organize notes (skill denied — agent handles gracefully)"
        )
        console.print()

        result2 = await agent.run(
            "Organize all the notes in the working directory into a single structured document.",
            notifier=ConsoleNotifier(),
        )
        console.print(f"\n[bold green]Answer:[/bold green] {result2.answer}")
        console.print(
            f"[dim]Run {result2.run_id} — {result2.iteration_count} iterations, "
            f"{result2.usage.total_tokens} tokens[/dim]"
        )

        # Task 3: Use filesystem tools directly (no skill needed)
        console.print()
        console.print("[bold]Task 3:[/bold] List files in working directory")
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
