"""CLI commands for database management."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import typer
from rich.console import Console

from dendrux._redact import mask_database_url, mask_database_urls_in

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

    # Resolve database URL from env var only (no CLI arg — security).
    # Alembic's config is interpolated ConfigParser: escape '%' in the URL.
    cfg.set_main_option("sqlalchemy.url", _database_url().replace("%", "%%"))

    return cfg


def _database_url() -> str:
    return os.environ.get("DENDRUX_DATABASE_URL", "sqlite+aiosqlite:///./dendrux.db")


@app.command()
def migrate() -> None:
    """Run database migrations (alembic upgrade head).

    On MySQL/MariaDB an empty database is bootstrapped straight to the current schema
    (an interrupted bootstrap resumes); a database holding anything else is refused.
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
        console.print(f"[red]Migration failed:[/red] {mask_database_urls_in(str(e))}")
        raise typer.Exit(1) from e

    console.print("[green]✓[/green] Migrations complete.")


@app.command()
def status() -> None:
    """Show the Alembic revision, storage format and resumable activation step.

    On MySQL/MariaDB an unfinished bootstrap is shown as ``bootstrap_in_progress=<rev>``.
    """
    try:
        revision, storage_format, step, bootstrap = asyncio.run(
            _read_activation_status(_database_url())
        )
    except Exception:
        console.print("[red]Could not read database status.[/red] Run `dendrux db check`.")
        raise typer.Exit(2) from None
    line = f"revision={revision} storage_format={storage_format} activate_step={step}"
    if bootstrap is not None:
        line += f" bootstrap_in_progress={bootstrap}"
    console.print(line, markup=False)


async def _read_activation_status(url: str) -> tuple[str, str, str, str | None]:
    from sqlalchemy import inspect, select, text
    from sqlalchemy.ext.asyncio import create_async_engine
    from sqlalchemy.pool import NullPool

    from dendrux.db.models import DendruxSetting

    engine = create_async_engine(url, poolclass=NullPool)
    try:
        async with engine.connect() as conn:
            revision = None
            if await conn.run_sync(lambda c: inspect(c).has_table("alembic_version")):
                revision = (
                    await conn.execute(text("SELECT version_num FROM alembic_version"))
                ).scalar_one_or_none()
            revision = revision or "unstamped"
            if not await conn.run_sync(lambda c: inspect(c).has_table("dendrux_settings")):
                return revision, "1", "not-started", None
            keys = ("storage_format", "activate_step", "bootstrap_in_progress")
            settings = dict(
                (
                    await conn.execute(
                        select(DendruxSetting.key, DendruxSetting.value).where(
                            DendruxSetting.key.in_(keys)
                        )
                    )
                )
                .tuples()
                .all()
            )
            return (
                revision,
                settings.get("storage_format", "1"),
                settings.get("activate_step", "not-started"),
                settings.get("bootstrap_in_progress"),
            )
    finally:
        await engine.dispose()


@app.command()
def activate(
    batch_size: int = typer.Option(5000, "--batch-size", min=1, max=5000),
) -> None:
    """Dedupe and activate format 2 after every writer and reader has been upgraded.

    Stop every other process first when using SQLite. Take a pre-activation backup:
    activation is one-way. Use a direct PostgreSQL connection for concurrent DDL.
    """
    _run_activation(batch_size, dedupe_only=False)


@app.command()
def dedupe(
    batch_size: int = typer.Option(5000, "--batch-size", min=1, max=5000),
) -> None:
    """Only dedupe identical rows, with the same resumable cursors as activate."""
    _run_activation(batch_size, dedupe_only=True)


def _run_activation(batch_size: int, *, dedupe_only: bool) -> None:
    from dendrux.db._activate import ActivationConflictError, ActivationIndexConflictError

    try:
        asyncio.run(_activate_database(_database_url(), batch_size, dedupe_only))
    except (ActivationConflictError, ActivationIndexConflictError) as error:
        console.print(str(error), markup=False)
        raise typer.Exit(3) from None
    except RuntimeError as error:
        console.print(str(error), markup=False)
        raise typer.Exit(1) from None
    except Exception:
        # Driver exceptions can contain tool/trace content in SQL parameters.
        console.print("[red]Activation failed.[/red] Check database availability and retry.")
        raise typer.Exit(1) from None
    console.print(
        "[green]Dedupe complete.[/green]" if dedupe_only else "[green]Activation complete.[/green]"
    )


async def _activate_database(url: str, batch_size: int, dedupe_only: bool) -> None:
    from sqlalchemy.ext.asyncio import create_async_engine
    from sqlalchemy.pool import NullPool

    from dendrux.db._activate import activate_database

    engine = create_async_engine(url, poolclass=NullPool)
    try:
        await activate_database(engine, batch_size=batch_size, dedupe_only=dedupe_only)
    finally:
        await engine.dispose()


EXIT_CURRENT = 0
EXIT_BEHIND = 1
EXIT_UNKNOWN = 2


@app.command()
def check(
    allow_unstamped: bool = typer.Option(
        False,
        "--allow-unstamped",
        help=(
            "For schemas built with create_all (no alembic_version table): check the "
            "tables, columns and indexes instead of the revision."
        ),
    ),
) -> None:
    """Check that the database schema is current (no changes are made).

    Compares the database's Alembic revision with this version's head. Exit
    codes: 0 current, 1 behind (run `dendrux db migrate`), 2 unknown (no
    alembic_version table, or a revision this version does not know).
    With --allow-unstamped, an unstamped database is checked by its tables,
    columns and (Postgres) indexes instead: 0 when complete, 1 when not.

    Set DENDRUX_DATABASE_URL to the database to check.
    """
    try:
        from alembic.script import ScriptDirectory
    except ImportError:
        console.print("[red]Alembic not installed.[/red] Run: pip install dendrux[db]")
        raise typer.Exit(1) from None

    url = _database_url()
    script = ScriptDirectory.from_config(_build_alembic_config())
    head = script.get_current_head()
    try:
        state = asyncio.run(_read_schema_state(url))
    except Exception as e:
        console.print(
            f"[red]Could not read the schema of {mask_database_url(url)}:[/red] "
            f"{mask_database_urls_in(str(e))}"
        )
        raise typer.Exit(EXIT_UNKNOWN) from e

    revision, missing = state
    target = mask_database_url(url)
    if revision is None:
        if not allow_unstamped:
            console.print(
                f"[yellow]Unknown:[/yellow] {target} has no alembic_version table. "
                "Run `dendrux db migrate`, or pass --allow-unstamped if the app builds "
                "the schema with create_all."
            )
            raise typer.Exit(EXIT_UNKNOWN)
        if missing:
            console.print(f"[red]Behind:[/red] {target} is missing: {', '.join(missing)}")
            raise typer.Exit(EXIT_BEHIND)
        console.print(f"[green]✓[/green] {target} has every table, column and index (unstamped).")
        raise typer.Exit(EXIT_CURRENT)
    if revision == head:
        if missing:
            # Stamped current by an earlier build of this revision that lacked some steps.
            console.print(
                f"[red]Behind:[/red] {target} is stamped {head} but is missing: "
                f"{', '.join(missing)}. Run `alembic downgrade -1` then `dendrux db migrate`."
            )
            raise typer.Exit(EXIT_BEHIND)
        console.print(f"[green]✓[/green] {target} is at the current revision ({head}).")
        raise typer.Exit(EXIT_CURRENT)
    known = {rev.revision for rev in script.walk_revisions()}
    if revision not in known:
        console.print(
            f"[yellow]Unknown:[/yellow] {target} is at revision {revision}, which this "
            "version of dendrux does not know (is it newer?)."
        )
        raise typer.Exit(EXIT_UNKNOWN)
    console.print(
        f"[red]Behind:[/red] {target} is at {revision}; this version needs {head}. "
        "Run `dendrux db migrate`."
    )
    raise typer.Exit(EXIT_BEHIND)


async def _read_schema_state(url: str) -> tuple[str | None, list[str]]:
    """``(alembic revision or None, missing schema objects)`` for ``url``. Read-only."""
    from sqlalchemy import inspect, text
    from sqlalchemy.ext.asyncio import create_async_engine
    from sqlalchemy.pool import NullPool

    from dendrux.db._schema_check import (
        schema_problems,
    )

    engine = create_async_engine(url, poolclass=NullPool)
    try:
        async with engine.connect() as conn:
            stamped = await conn.run_sync(lambda c: inspect(c).has_table("alembic_version"))
            revision = None
            if stamped:
                row = (await conn.execute(text("SELECT version_num FROM alembic_version"))).first()
                revision = row[0] if row is not None else None
            missing = await conn.run_sync(schema_problems)
            return revision, missing
    finally:
        await engine.dispose()


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
