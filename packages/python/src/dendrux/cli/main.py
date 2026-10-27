"""Dendrux CLI entry point.

Provides ops commands for inspecting agent runs, traces, and database.
The ``dendrux run`` command was removed in Sprint 4 — developers start
agents programmatically via ``agent.run()``.
"""

from __future__ import annotations

import typer
from rich.console import Console

app = typer.Typer(
    name="dendrux",
    help="Dendrux — The runtime for agents that act in the real world.",
    no_args_is_help=True,
    # Older Typer versions print frame locals in tracebacks, which would put the
    # unmasked database URL on stderr.
    pretty_exceptions_show_locals=False,
)
console = Console()


def _register_subcommands() -> None:
    """Register subcommand groups. Called after app is created."""
    from dendrux.cli.db import app as db_app
    from dendrux.cli.runs import app as runs_app
    from dendrux.cli.traces import show_traces

    app.add_typer(db_app)
    app.add_typer(runs_app)
    # A plain command (not a sub-group) so options may follow the run id:
    # ``dendrux traces <run_id> --tools``.
    app.command(name="traces", help="Inspect traces for a run.")(show_traces)


_register_subcommands()


@app.callback(invoke_without_command=True)
def main(
    version: bool = typer.Option(False, "--version", "-v", help="Show version and exit."),
) -> None:
    """Dendrux — The runtime for agents that act in the real world."""
    if version:
        from dendrux import __version__

        typer.echo(f"dendrux {__version__}")
        raise typer.Exit()
