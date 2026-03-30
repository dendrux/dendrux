"""Dendrite CLI entry point.

Provides ops commands for inspecting agent runs, traces, and database.
The ``dendrite run`` command was removed in Sprint 4 — developers start
agents programmatically via ``agent.run()`` or mount the bridge.
"""

from __future__ import annotations

import asyncio

import typer
from rich.console import Console

app = typer.Typer(
    name="dendrite",
    help="Dendrite — The runtime for agents that act in the real world.",
    no_args_is_help=True,
)
console = Console()


def _register_subcommands() -> None:
    """Register subcommand groups. Called after app is created."""
    from dendrite.cli.db import app as db_app
    from dendrite.cli.runs import app as runs_app
    from dendrite.cli.traces import app as traces_app

    app.add_typer(db_app)
    app.add_typer(runs_app)
    app.add_typer(traces_app)


_register_subcommands()


def _resolve_db_url(db: str | None) -> str | None:
    """Resolve --db flag to a full database URL.

    Accepts:
      - None → returns None (use default)
      - Full database URL (contains ://) → pass through
      - Bare file path (./my.db, /path/to/db) → convert to sqlite+aiosqlite:///abs/path
    """
    if db is None:
        return None
    if "://" in db:
        return db
    from pathlib import Path

    return f"sqlite+aiosqlite:///{Path(db).resolve()}"


@app.callback(invoke_without_command=True)
def main(
    version: bool = typer.Option(False, "--version", "-v", help="Show version and exit."),
) -> None:
    """Dendrite — The runtime for agents that act in the real world."""
    if version:
        from dendrite import __version__

        typer.echo(f"dendrite {__version__}")
        raise typer.Exit()


@app.command()
def dashboard(
    port: int = typer.Option(8001, "--port", "-p", help="Port to serve on."),
    db: str | None = typer.Option(None, "--db", help="Database URL or SQLite file path."),
    open_browser: bool = typer.Option(False, "--open", help="Open browser on start."),
) -> None:
    """Launch the Dendrite observability dashboard.

    Starts a local server that reads from the same database as
    your agent runs. All data is read-only.

    Examples:
        dendrite dashboard
        dendrite dashboard --db ./my-agent.db
        dendrite dashboard --db sqlite+aiosqlite:///path/to/dendrite.db
    """
    try:
        import uvicorn  # noqa: F401

        from dendrite.dashboard.api import create_dashboard_api
    except ImportError:
        console.print(
            "[red]Dashboard requires FastAPI + uvicorn.[/red] Run: pip install dendrite[bridge]"
        )
        raise typer.Exit(1) from None

    async def _start() -> None:
        from dendrite.db.session import get_engine
        from dendrite.runtime.state import SQLAlchemyStateStore

        db_url = _resolve_db_url(db)
        engine = await get_engine(url=db_url)
        store = SQLAlchemyStateStore(engine)
        api = create_dashboard_api(store)

        from dendrite.db.session import get_database_url

        resolved = db_url or get_database_url()
        console.print(
            f"[bold green]Dendrite Dashboard[/bold green] running at "
            f"[link=http://localhost:{port}]http://localhost:{port}[/link]"
        )
        console.print(f"[dim]Database: {resolved}[/dim]")
        console.print("[dim]Press Ctrl+C to stop.[/dim]\n")

        if open_browser:
            import webbrowser

            webbrowser.open(f"http://localhost:{port}")

        config = uvicorn.Config(api, host="0.0.0.0", port=port, log_level="warning")
        server = uvicorn.Server(config)
        await server.serve()

    asyncio.run(_start())
