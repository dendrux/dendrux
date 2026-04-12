"""Dendrux CLI entry point.

Provides ops commands for inspecting agent runs, traces, and database.
The ``dendrux run`` command was removed in Sprint 4 — developers start
agents programmatically via ``agent.run()`` or mount the bridge.
"""

from __future__ import annotations

import asyncio

import typer
from rich.console import Console

app = typer.Typer(
    name="dendrux",
    help="Dendrux — The runtime for agents that act in the real world.",
    no_args_is_help=True,
)
console = Console()


def _register_subcommands() -> None:
    """Register subcommand groups. Called after app is created."""
    from dendrux.cli.db import app as db_app
    from dendrux.cli.runs import app as runs_app
    from dendrux.cli.traces import app as traces_app

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
    """Dendrux — The runtime for agents that act in the real world."""
    if version:
        from dendrux import __version__

        typer.echo(f"dendrux {__version__}")
        raise typer.Exit()


@app.command()
def dashboard(
    port: int = typer.Option(8001, "--port", "-p", help="Port to serve on."),
    host: str = typer.Option(
        "127.0.0.1", "--host", help="Host to bind to. Default localhost only."
    ),
    db: str | None = typer.Option(None, "--db", help="Database URL or SQLite file path."),
    open_browser: bool = typer.Option(False, "--open", help="Open browser on start."),
    auth_token: str | None = typer.Option(
        None,
        "--auth-token",
        help="Bearer token for API access. When set, all API requests require auth.",
    ),
) -> None:
    """Launch the Dendrux observability dashboard.

    Starts a local server that reads from the same database as
    your agent runs. All data is read-only.

    Examples:
        dendrux dashboard
        dendrux dashboard --db ./my-agent.db
        dendrux dashboard --db sqlite+aiosqlite:///path/to/dendrux.db
    """
    try:
        import uvicorn  # noqa: F401

        from dendrux.dashboard.api import create_dashboard_api
    except ImportError:
        console.print(
            "[red]Dashboard requires FastAPI + uvicorn.[/red] Run: pip install dendrux[bridge]"
        )
        raise typer.Exit(1) from None

    async def _start() -> None:
        from dendrux.db.session import get_engine
        from dendrux.runtime.state import SQLAlchemyStateStore

        db_url = _resolve_db_url(db)
        engine = await get_engine(url=db_url)
        store = SQLAlchemyStateStore(engine)
        api = create_dashboard_api(store, auth_token=auth_token)

        from dendrux.db.session import get_database_url

        resolved = db_url or get_database_url()
        console.print(
            f"[bold green]Dendrux Dashboard[/bold green] running at "
            f"[link=http://localhost:{port}]http://localhost:{port}[/link]"
        )
        console.print(f"[dim]Database: {resolved}[/dim]")
        console.print("[dim]Press Ctrl+C to stop.[/dim]\n")

        if open_browser:
            import webbrowser

            webbrowser.open(f"http://localhost:{port}")

        config = uvicorn.Config(api, host=host, port=port, log_level="warning")
        server = uvicorn.Server(config)
        await server.serve()

    asyncio.run(_start())
