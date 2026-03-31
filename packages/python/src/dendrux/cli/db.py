"""CLI commands for database management."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import typer
from rich.console import Console

app = typer.Typer(name="db", help="Database management commands.", no_args_is_help=True)
console = Console()


def _build_alembic_config() -> Any:
    """Build an Alembic Config programmatically.

    Works both from a repo checkout (finds alembic.ini) and from a
    pip-installed package (uses the migrations directory inside the
    dendrux.db.migrations package as the script_location).

    Database URL is resolved from DENDRUX_DATABASE_URL env var, falling
    back to async SQLite. CLI flags are intentionally not used for URLs
    to avoid leaking credentials in shell history and process listings.
    """
    try:
        from alembic.config import Config
    except ImportError:
        console.print("[red]Alembic not installed.[/red] Run: pip install dendrux[db]")
        raise typer.Exit(1) from None

    # Try to find alembic.ini on disk (repo checkout)
    ini_path = _find_alembic_ini()
    cfg = Config(str(ini_path)) if ini_path is not None else Config()

    # Always set script_location to the absolute path — alembic.ini's
    # relative path only works from packages/python/, but the CLI can
    # be run from any directory.
    migrations_dir = str(Path(__file__).resolve().parent.parent / "db" / "migrations")
    cfg.set_main_option("script_location", migrations_dir)

    # Resolve database URL from env var only (no CLI arg — security)
    resolved_url = os.environ.get("DENDRUX_DATABASE_URL", "sqlite+aiosqlite:///./dendrux.db")
    cfg.set_main_option("sqlalchemy.url", resolved_url)

    return cfg


@app.command()
def migrate() -> None:
    """Run database migrations (alembic upgrade head).

    Set DENDRUX_DATABASE_URL to override the default SQLite path.
    """
    try:
        from alembic import command
    except ImportError:
        console.print("[red]Alembic not installed.[/red] Run: pip install dendrux[db]")
        raise typer.Exit(1) from None

    alembic_cfg = _build_alembic_config()

    console.print("[bold]Running migrations...[/bold]")
    try:
        command.upgrade(alembic_cfg, "head")
    except Exception as e:
        console.print(f"[red]Migration failed:[/red] {e}")
        raise typer.Exit(1) from e

    console.print("[green]✓[/green] Migrations complete.")


@app.command()
def status() -> None:
    """Show current migration revision.

    Set DENDRUX_DATABASE_URL to override the default SQLite path.
    """
    try:
        from alembic import command
    except ImportError:
        console.print("[red]Alembic not installed.[/red] Run: pip install dendrux[db]")
        raise typer.Exit(1) from None

    alembic_cfg = _build_alembic_config()
    command.current(alembic_cfg, verbose=True)


def _find_alembic_ini() -> Path | None:
    """Find alembic.ini by walking up from CWD, or use package-relative path."""
    # Try CWD and parents
    cwd = Path.cwd()
    for parent in [cwd, *cwd.parents]:
        candidate = parent / "alembic.ini"
        if candidate.exists():
            return candidate

    # Fallback: relative to this package
    pkg_root = Path(__file__).resolve().parents[3]  # cli/ → dendrux/ → src/ → python/
    candidate = pkg_root / "alembic.ini"
    if candidate.exists():
        return candidate

    return None
