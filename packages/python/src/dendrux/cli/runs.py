"""CLI commands for querying agent runs."""

from __future__ import annotations

import asyncio
from enum import StrEnum
from typing import TYPE_CHECKING, Annotated, Any

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from dendrux._lifecycle import STORAGE_FORMAT_CHECKPOINTED
from dendrux._redact import mask_database_urls_in
from dendrux.runtime._bounds import MAX_LIST_OFFSET, MAX_PAGE
from dendrux.types import RunStatus

if TYPE_CHECKING:
    from sqlalchemy import ColumnElement, Select

DEFAULT_RUN_LIMIT = 20

app = typer.Typer(
    name="runs",
    help="Query agent runs.",
    invoke_without_command=True,
    pretty_exceptions_show_locals=False,
)
console = Console()


class StorageFormat(StrEnum):
    """Storage formats accepted by the runs command."""

    legacy = "1"
    checkpointed = str(STORAGE_FORMAT_CHECKPOINTED)


@app.callback(invoke_without_command=True)
def list_runs(
    limit: int = typer.Option(
        DEFAULT_RUN_LIMIT, "--limit", "-n", help="Maximum number of runs to show."
    ),
    offset: int = typer.Option(0, "--offset", help="Skip this many runs."),
    status: str = typer.Option("", "--status", "-s", help="Filter by status."),
    tenant: str = typer.Option("", "--tenant", "-t", help="Filter by tenant ID."),
    exclude_waiting: bool = typer.Option(
        False, "--exclude-waiting", help="Leave out runs waiting on subagents."
    ),
    storage_format: Annotated[
        StorageFormat | None, typer.Option("--format", help="Filter by storage format.")
    ] = None,
) -> None:
    """List recent agent runs."""
    asyncio.run(
        _list_runs(
            limit,
            offset,
            status or None,
            tenant or None,
            int(storage_format.value) if storage_format else None,
            include_waiting=not exclude_waiting,
        )
    )


async def _list_runs(
    limit: int,
    offset: int,
    status: str | None,
    tenant_id: str | None,
    storage_format: int | None = None,
    *,
    include_waiting: bool = True,
) -> None:
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

    try:
        store = store_for_engine(engine)
        # No --tenant means every tenant: omit the argument (the store's default
        # scope) instead of passing None, which means "untenanted runs only".
        scope: dict[str, Any] = {} if tenant_id is None else {"tenant_id": tenant_id}
        runs: list[Any]
        if storage_format is None:
            runs = await store.list_runs(
                limit=max(1, min(limit, MAX_PAGE)),
                offset=max(0, offset),
                status=status,
                include_waiting=include_waiting,
                **scope,
            )
        else:
            from dendrux.runtime.stores.sql._sql_reads import _waiting_for_expression

            statement = _format_statement(
                storage_format=storage_format,
                status=status,
                tenant_id=tenant_id,
                limit=limit,
                offset=offset,
                waiting_for=_waiting_for_expression(store),
                include_waiting=include_waiting,
            )
            async with engine.connect() as conn:
                runs = list(await conn.execute(statement))
    finally:
        await reset_engine()

    if not runs:
        console.print("[dim]No runs found.[/dim]")
        return

    table = Table(title="Agent Runs", show_lines=False)
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Agent", style="bold", no_wrap=True)
    table.add_column("Status", no_wrap=True)
    table.add_column("Waiting for", no_wrap=True)
    table.add_column("Iters", justify="right")
    table.add_column("Tokens", justify="right")
    table.add_column("Model")
    table.add_column("Created")

    for run in runs:
        status_style = {
            RunStatus.RUNNING.value: "yellow",
            RunStatus.SUCCESS.value: "green",
            RunStatus.ERROR.value: "red",
            RunStatus.MAX_ITERATIONS.value: "magenta",
            RunStatus.CANCELLED.value: "dim",
            RunStatus.PENDING.value: "dim",
        }.get(run.status, "")

        total_tokens = run.total_input_tokens + run.total_output_tokens
        created = str(run.created_at)[:19] if run.created_at else "—"

        table.add_row(
            escape(run.id),
            escape(run.agent_name),
            f"[{status_style}]{run.status}[/{status_style}]" if status_style else run.status,
            escape(run.waiting_for) if run.waiting_for else "—",
            str(run.iteration_count),
            f"{total_tokens:,}" if total_tokens else "—",
            run.model or "—",
            created,
        )

    console.print(table)


def _format_statement(
    *,
    storage_format: int,
    status: str | None,
    tenant_id: str | None,
    limit: int,
    offset: int,
    waiting_for: ColumnElement[str | None],
    include_waiting: bool,
) -> Select[Any]:
    from dendrux.db.models import AgentRun

    statement = (
        AgentRun.__table__.select()
        .with_only_columns(
            AgentRun.id,
            AgentRun.agent_name,
            AgentRun.status,
            AgentRun.iteration_count,
            AgentRun.total_input_tokens,
            AgentRun.total_output_tokens,
            AgentRun.model,
            AgentRun.created_at,
            waiting_for.label("waiting_for"),
        )
        .where(AgentRun.storage_format == storage_format)
    )
    if status:
        statement = statement.where(AgentRun.status == status)
    if tenant_id:
        statement = statement.where(AgentRun.tenant_id == tenant_id)
    if not include_waiting:
        statement = statement.where(waiting_for.is_(None))
    return (
        statement.order_by(AgentRun.created_at.desc(), AgentRun.id.desc())
        .limit(max(1, min(limit, MAX_PAGE)))
        .offset(max(0, min(offset, MAX_LIST_OFFSET)))
    )
