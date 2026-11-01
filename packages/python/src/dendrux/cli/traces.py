"""CLI commands for inspecting run traces and tool calls."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from dendrux._redact import mask_database_urls_in

if TYPE_CHECKING:
    from dendrux.runtime.state import TraceRecord

MAX_ROWS = 1000

console = Console()


def show_traces(
    run_id: str = typer.Argument(..., help="Run ID to inspect."),
    tool_calls: bool = typer.Option(False, "--tools", "-t", help="Also show tool call records."),
    limit: int = typer.Option(
        200, "--limit", "-n", help=f"Maximum rows per section (at most {MAX_ROWS})."
    ),
) -> None:
    """Show the conversation trace for an agent run."""
    asyncio.run(_show_traces(run_id, tool_calls, limit))


async def _show_traces(run_id: str, show_tool_calls: bool, limit: int = 200) -> None:
    limit = max(1, min(limit, MAX_ROWS))
    try:
        from dendrux.db.session import get_engine, reset_engine
        from dendrux.runtime.stores._factory import store_for_engine
    except ImportError:
        console.print("[red]Database support not installed.[/red] Run: pip install dendrux[db]")
        raise typer.Exit(1) from None

    try:
        engine = await get_engine()
    except Exception as e:
        console.print(
            f"[red]Cannot connect to database:[/red] {escape(mask_database_urls_in(str(e)))}"
        )
        raise typer.Exit(1) from None

    store = store_for_engine(engine)

    try:
        # Fetch run info
        run = await store.get_run(run_id)
        if run is None:
            console.print(f"[red]Run not found:[/red] {escape(run_id)}")
            raise typer.Exit(1)

        # Print run header
        console.print(
            Panel(
                f"[bold]{escape(run.agent_name)}[/bold]  •  {escape(run.status)}  •  "
                f"{run.iteration_count} iterations  •  model: {escape(run.model or '—')}",
                title=f"Run {escape(run.id)}",
                border_style="cyan",
            )
        )

        # Fetch and display traces
        from dendrux.runtime.replay import conversation_records

        async def read_traces(limit: int, offset: int = 0) -> list[TraceRecord]:
            if run.storage_format == 2:
                return await conversation_records(store, run_id, limit=limit, offset=offset)
            return await store.get_traces(run_id, limit=limit, offset=offset)

        traces = await read_traces(limit)
        # Probe one row past the page in a separate read: the store clamps
        # ``limit`` to 1000, so asking for limit + 1 cannot detect overflow at the maximum.
        more_traces = len(traces) == limit and bool(await read_traces(1, offset=limit))
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
                console.print(content, markup=False, highlight=False)
                console.print()
            if more_traces:
                console.print(f"[dim]… more traces not shown (limit {limit}).[/dim]")

        # Optionally show tool calls
        if show_tool_calls:
            records = await store.get_tool_calls(run_id, limit=limit)
            more_calls = len(records) == limit and bool(
                await store.get_tool_calls(run_id, limit=1, offset=limit)
            )
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
                        escape(r.tool_name),
                        success_str,
                        duration,
                        str(r.iteration_index) if r.iteration_index is not None else "—",
                        escape(r.error_message or ""),
                    )

                console.print(table)
                if more_calls:
                    console.print(f"[dim]… more tool calls not shown (limit {limit}).[/dim]")
    finally:
        await reset_engine()
