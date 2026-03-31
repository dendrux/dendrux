"""CLI commands for inspecting run traces and tool calls."""

from __future__ import annotations

import asyncio

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

app = typer.Typer(name="traces", help="Inspect traces for a run.", invoke_without_command=True)
console = Console()


@app.callback(invoke_without_command=True)
def show_traces(
    run_id: str = typer.Argument(..., help="Run ID to inspect."),
    tool_calls: bool = typer.Option(False, "--tools", "-t", help="Also show tool call records."),
) -> None:
    """Show the conversation trace for an agent run."""
    asyncio.run(_show_traces(run_id, tool_calls))


async def _show_traces(run_id: str, show_tool_calls: bool) -> None:
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

    store = SQLAlchemyStateStore(engine)

    try:
        # Fetch run info
        run = await store.get_run(run_id)
        if run is None:
            console.print(f"[red]Run not found:[/red] {run_id}")
            raise typer.Exit(1)

        # Print run header
        console.print(
            Panel(
                f"[bold]{run.agent_name}[/bold]  •  {run.status}  •  "
                f"{run.iteration_count} iterations  •  model: {run.model or '—'}",
                title=f"Run {run.id}",
                border_style="cyan",
            )
        )

        # Fetch and display traces
        traces = await store.get_traces(run_id)
        if not traces:
            console.print("[dim]No traces recorded.[/dim]")
        else:
            console.print()
            for trace in traces:
                role_style = {
                    "user": "bold green",
                    "assistant": "bold blue",
                    "tool": "bold yellow",
                }.get(trace.role, "bold")

                header = Text(f"[{trace.order_index}] {trace.role}", style=role_style)
                content = trace.content
                # Truncate very long content for display
                if len(content) > 500:
                    content = content[:500] + f"\n… ({len(trace.content)} chars total)"

                console.print(header)
                console.print(content)
                console.print()

        # Optionally show tool calls
        if show_tool_calls:
            records = await store.get_tool_calls(run_id)
            if not records:
                console.print("[dim]No tool calls recorded.[/dim]")
            else:
                table = Table(title="Tool Calls", show_lines=True)
                table.add_column("Tool", style="bold")
                table.add_column("Success", no_wrap=True)
                table.add_column("Duration", justify="right")
                table.add_column("Iter", justify="right")
                table.add_column("Error")

                for r in records:
                    success_str = "[green]✓[/green]" if r.success else "[red]✗[/red]"
                    duration = f"{r.duration_ms}ms" if r.duration_ms else "—"
                    table.add_row(
                        r.tool_name,
                        success_str,
                        duration,
                        str(r.iteration_index) if r.iteration_index is not None else "—",
                        r.error_message or "",
                    )

                console.print(table)
    finally:
        await reset_engine()
