"""CLI commands for querying agent runs."""

from __future__ import annotations

import asyncio

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(name="runs", help="Query agent runs.", invoke_without_command=True)
console = Console()


@app.callback(invoke_without_command=True)
def list_runs(
    limit: int = typer.Option(20, "--limit", "-n", help="Maximum number of runs to show."),
    offset: int = typer.Option(0, "--offset", help="Skip this many runs."),
    status: str = typer.Option("", "--status", "-s", help="Filter by status."),
    tenant: str = typer.Option("", "--tenant", "-t", help="Filter by tenant ID."),
) -> None:
    """List recent agent runs."""
    asyncio.run(_list_runs(limit, offset, status or None, tenant or None))


async def _list_runs(limit: int, offset: int, status: str | None, tenant_id: str | None) -> None:
    try:
        from dendrux.db.session import get_engine, reset_engine
        from dendrux.runtime.state import SQLAlchemyStateStore
    except ImportError:
        console.print("[red]Database support not installed.[/red] Run: pip install dendrux[db]")
        raise typer.Exit(1) from None

    try:
        engine = await get_engine()
    except Exception as e:
        console.print(f"[red]Cannot connect to database:[/red] {e}")
        raise typer.Exit(1) from None

    try:
        store = SQLAlchemyStateStore(engine)
        runs = await store.list_runs(
            limit=min(limit, 1000),
            offset=max(0, offset),
            tenant_id=tenant_id,
            status=status,
        )
    finally:
        await reset_engine()

    if not runs:
        console.print("[dim]No runs found.[/dim]")
        return

    table = Table(title="Agent Runs", show_lines=False)
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Agent", style="bold")
    table.add_column("Status", no_wrap=True)
    table.add_column("Iters", justify="right")
    table.add_column("Tokens", justify="right")
    table.add_column("Model")
    table.add_column("Created")

    for run in runs:
        status_style = {
            "running": "yellow",
            "success": "green",
            "error": "red",
            "max_iterations": "magenta",
            "cancelled": "dim",
            "pending": "dim",
        }.get(run.status, "")

        total_tokens = run.total_input_tokens + run.total_output_tokens
        created = str(run.created_at)[:19] if run.created_at else "—"

        table.add_row(
            run.id,
            run.agent_name,
            f"[{status_style}]{run.status}[/{status_style}]",
            str(run.iteration_count),
            f"{total_tokens:,}" if total_tokens else "—",
            run.model or "—",
            created,
        )

    console.print(table)
