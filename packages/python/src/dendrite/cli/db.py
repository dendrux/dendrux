"""CLI commands for database management."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

app = typer.Typer(name="db", help="Database management commands.", no_args_is_help=True)
console = Console()


@app.command()
def migrate(
    url: str = typer.Option(
        "", "--url", "-u", help="Database URL. Falls back to DENDRITE_DATABASE_URL or SQLite."
    ),
) -> None:
    """Run database migrations (alembic upgrade head)."""
    try:
        from alembic import command
        from alembic.config import Config
    except ImportError:
        console.print("[red]Alembic not installed.[/red] Run: pip install dendrite[db]")
        raise typer.Exit(1) from None

    # Locate alembic.ini relative to package root
    ini_path = _find_alembic_ini()
    if ini_path is None:
        console.print("[red]Cannot find alembic.ini.[/red] Run from the project root.")
        raise typer.Exit(1)

    alembic_cfg = Config(str(ini_path))

    # Override URL if provided (via config, not os.environ — no side effects)
    if url:
        alembic_cfg.set_main_option("sqlalchemy.url", url)

    console.print("[bold]Running migrations...[/bold]")
    try:
        command.upgrade(alembic_cfg, "head")
    except Exception as e:
        console.print(f"[red]Migration failed:[/red] {e}")
        raise typer.Exit(1) from e

    console.print("[green]✓[/green] Migrations complete.")


@app.command()
def status(
    url: str = typer.Option(
        "", "--url", "-u", help="Database URL. Falls back to DENDRITE_DATABASE_URL or SQLite."
    ),
) -> None:
    """Show current migration revision."""
    try:
        from alembic import command
        from alembic.config import Config
    except ImportError:
        console.print("[red]Alembic not installed.[/red] Run: pip install dendrite[db]")
        raise typer.Exit(1) from None

    ini_path = _find_alembic_ini()
    if ini_path is None:
        console.print("[red]Cannot find alembic.ini.[/red] Run from the project root.")
        raise typer.Exit(1)

    alembic_cfg = Config(str(ini_path))

    if url:
        alembic_cfg.set_main_option("sqlalchemy.url", url)

    command.current(alembic_cfg, verbose=True)


def _find_alembic_ini() -> Path | None:
    """Find alembic.ini by walking up from CWD, or use package-relative path.

    TODO(post-alpha): alembic.ini isn't included in the wheel (hatch only packages
    src/dendrite/). For pip-installed users, either ship it as package data via
    importlib.resources or generate the Alembic config programmatically.
    """
    # First, try CWD and parents
    cwd = Path.cwd()
    for parent in [cwd, *cwd.parents]:
        candidate = parent / "alembic.ini"
        if candidate.exists():
            return candidate

    # Fallback: relative to this package
    pkg_root = Path(__file__).resolve().parents[3]  # cli/ → dendrite/ → src/ → python/
    candidate = pkg_root / "alembic.ini"
    if candidate.exists():
        return candidate

    return None
