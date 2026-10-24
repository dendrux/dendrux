"""Skills + MCP Agent — instructions, files, scripts and governance.

Demonstrates skills working alongside MCP tools:

  * ``summarize-file`` has a reference file (read with ``read_skill_resource``)
    and a script (run with ``run_skill_script`` through the local executor).
  * ``organize-notes`` is denied by policy.
  * Task 1 lets the model activate the skill; task 3 activates it up front
    with ``activate_skills=``.

The local executor runs scripts on THIS machine. That is fine for the
skills bundled here; use a container or hosted sandbox for anything else.

Prerequisites:
    - Node.js + npx installed
    - ANTHROPIC_API_KEY in .env

Run with:
    cd examples/14_skills
    ANTHROPIC_API_KEY=sk-... python 14_skills_agent.py

After running, inspect the data with:
    dendrux runs
    dendrux traces <run_id> --tools
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from local_executor import LocalSubprocessExecutor
from rich.console import Console
from rich.panel import Panel

from dendrux import Agent
from dendrux.llm.anthropic import AnthropicProvider
from dendrux.mcp import MCPSource
from dendrux.notifiers.console import ConsoleNotifier
from dendrux.skills import Skill
from dendrux.store import RunStore

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
            MCPSource.stdio(
                "filesystem",
                [
                    "npx",
                    "-y",
                    "@modelcontextprotocol/server-filesystem@2026.7.4",
                    str(work_dir),
                ],
            ),
        ],
        skills_dir=str(SKILLS_DIR),
        deny_skills=["organize-notes"],
        skill_executor=LocalSubprocessExecutor(),
        deny=[
            "filesystem__move_file",
            "filesystem__edit_file",
        ],
        database_url=f"sqlite+aiosqlite:///{db_path}",
        max_iterations=10,
    ) as agent:
        # Task 1: the model activates the skill, reads its style guide, runs its script
        console.print()
        console.print("[bold]Task 1:[/bold] Summarize meeting notes (model activates the skill)")
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

        # Task 3: activate the skill up front, so the model starts with its instructions
        console.print()
        console.print("[bold]Task 3:[/bold] Summarize project ideas (skill activated explicitly)")
        console.print()

        result3 = await agent.run(
            "Summarize the project-ideas.txt file.",
            activate_skills=["summarize-file"],
            notifier=ConsoleNotifier(),
        )
        console.print(f"\n[bold green]Answer:[/bold green] {result3.answer}")
        console.print(
            f"[dim]Run {result3.run_id} — {result3.iteration_count} iterations, "
            f"{result3.usage.total_tokens} tokens[/dim]"
        )

    # The skill events: which skills were resolved, activated, read and run
    store = RunStore.from_database_url(f"sqlite+aiosqlite:///{db_path}")
    async with store:
        for run_id in (result.run_id, result2.run_id, result3.run_id):
            names = [e.event_type for e in await store.get_events(run_id)]
            console.print(f"[dim]{run_id}: {[n for n in names if n.startswith('skill.')]}[/dim]")

    console.print()
    console.print("[bold]Inspect persisted data with:[/bold]")
    console.print("  dendrux runs")
    console.print(f"  dendrux traces {result.run_id} --tools")


if __name__ == "__main__":
    asyncio.run(main())
