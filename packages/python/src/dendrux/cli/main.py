"""Dendrux CLI entry point.

Provides ops commands for inspecting agent runs, traces, and database.
The ``dendrux run`` command was removed in Sprint 4 — developers start
agents programmatically via ``agent.run()`` or mount ``make_read_router``.
"""

from __future__ import annotations

import asyncio
from typing import Annotated

import typer
from rich.console import Console
from rich.markup import escape

from dendrux._redact import mask_database_url, mask_database_urls_in

app = typer.Typer(
    name="dendrux",
    help="Dendrux — The runtime for agents that act in the real world.",
    no_args_is_help=True,
    # Older Typer versions print frame locals in tracebacks, which would put the
    # dashboard token and the unmasked database URL on stderr.
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


_TOKEN_ARGV_WARNING = (
    "[yellow]Warning:[/yellow] a token passed on the command line is visible in process "
    "listings and shell history. Set {env} instead."
)


@app.command()
def dashboard(
    port: int = typer.Option(8001, "--port", "-p", help="Port to serve on."),
    host: str = typer.Option(
        "127.0.0.1",
        "--host",
        help="Host to bind to. Default localhost only; any other address requires a token.",
    ),
    db: str | None = typer.Option(None, "--db", help="Database URL or SQLite file path."),
    open_browser: bool = typer.Option(False, "--open", help="Open browser on start."),
    token: str | None = typer.Option(
        None,
        "--token",
        help="Bearer token for API access. Prefer the DENDRUX_DASHBOARD_TOKEN environment "
        "variable: an argument shows up in process listings.",
    ),
    auth_token: str | None = typer.Option(None, "--auth-token", hidden=True),
    allow_origin: Annotated[
        list[str] | None,
        typer.Option(
            "--allow-origin",
            help="Browser origin allowed to call the API cross-origin, e.g. "
            "http://localhost:5173 for the Vite dev server. Repeatable. Default: none.",
        ),
    ] = None,
    allowed_host: Annotated[
        list[str] | None,
        typer.Option(
            "--allowed-host",
            help="Extra Host header name to accept (needed when serving under a DNS name). "
            "Repeatable.",
        ),
    ] = None,
) -> None:
    """Launch the Dendrux observability dashboard.

    Starts a local server that reads from the same database as
    your agent runs. All data is read-only.

    Binding to anything but localhost requires a token
    (DENDRUX_DASHBOARD_TOKEN), and cross-origin browser access is off unless
    you pass --allow-origin.

    Examples:
        dendrux dashboard
        dendrux dashboard --db ./my-agent.db
        dendrux dashboard --db sqlite+aiosqlite:///path/to/dendrux.db
        DENDRUX_DASHBOARD_TOKEN=... dendrux dashboard --host 0.0.0.0
    """
    try:
        import uvicorn  # noqa: F401

        from dendrux.dashboard.api import create_dashboard_api
    except ImportError:
        console.print(
            "[red]Dashboard requires FastAPI + uvicorn.[/red] Run: pip install dendrux[http]"
        )
        raise typer.Exit(1) from None

    from dendrux.dashboard.security import TOKEN_ENV, is_loopback_host, resolve_token

    if token or auth_token:
        console.print(_TOKEN_ARGV_WARNING.format(env=TOKEN_ENV))
    if auth_token:
        console.print("[yellow]--auth-token is deprecated; use --token.[/yellow]")
    resolved_token = resolve_token(token or auth_token)

    if resolved_token is None and not is_loopback_host(host):
        console.print(
            f"[red]Refusing to bind {host!r} without a token.[/red] The dashboard serves run "
            f"payloads. Set {TOKEN_ENV}, or bind to 127.0.0.1."
        )
        raise typer.Exit(1)

    if any(h.strip() == "*" for h in allowed_host or []):
        console.print(
            "[yellow]Warning:[/yellow] --allowed-host '*' turns the Host check off, which "
            "removes the DNS-rebinding protection."
        )

    try:
        from dendrux.dashboard.security import normalize_origins

        origins = normalize_origins(allow_origin or [])
    except ValueError as exc:
        console.print(f"[red]{escape(str(exc))}[/red]")
        raise typer.Exit(1) from None

    async def _start() -> None:
        from dendrux.db.session import get_database_url, get_engine, reset_engine
        from dendrux.runtime.state import SQLAlchemyStateStore

        db_url = _resolve_db_url(db)
        try:
            engine = await get_engine(url=db_url)
        except Exception as exc:
            console.print(
                f"[red]Cannot connect to database:[/red] {escape(mask_database_urls_in(str(exc)))}"
            )
            raise typer.Exit(1) from None
        try:
            store = SQLAlchemyStateStore(engine)
            api = create_dashboard_api(
                store,
                auth_token=resolved_token,
                bind_host=host,
                allowed_hosts=allowed_host or [],
                allow_origins=origins,
            )

            resolved = db_url or get_database_url()
            shown_host = "localhost" if host in {"0.0.0.0", "::", ""} else host
            if ":" in shown_host and not shown_host.startswith("["):
                shown_host = f"[{shown_host}]"
            url = f"http://{shown_host}:{port}"
            # A bracketed IPv6 host cannot sit inside rich's [link=...] tag.
            shown_url = escape(url) if "[" in url else f"[link={url}]{url}[/link]"
            console.print(f"[bold green]Dendrux Dashboard[/bold green] running at {shown_url}")
            console.print(f"[dim]Database: {escape(mask_database_url(resolved))}[/dim]")
            if resolved_token is not None:
                console.print(
                    "[dim]API token required (you will be prompted in the browser).[/dim]"
                )
            if not is_loopback_host(host):
                console.print(
                    "[yellow]Serving plain HTTP beyond localhost: the token and run data are not "
                    "encrypted. Put a TLS-terminating proxy in front.[/yellow]"
                )
            console.print("[dim]Press Ctrl+C to stop.[/dim]\n")

            if open_browser:
                import webbrowser

                webbrowser.open(url)

            config = uvicorn.Config(api, host=host, port=port, log_level="warning")
            server = uvicorn.Server(config)
            await server.serve()
        except typer.Exit:
            raise
        except Exception as exc:
            # No traceback: its locals would include the token and the database URL.
            console.print(
                f"[red]Dashboard stopped:[/red] {type(exc).__name__}: "
                f"{escape(mask_database_urls_in(str(exc)))}"
            )
            raise typer.Exit(1) from None
        finally:
            await reset_engine()

    asyncio.run(_start())
