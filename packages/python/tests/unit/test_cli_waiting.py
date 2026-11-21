"""Waiting labels and exclusions in both CLI read paths."""

from __future__ import annotations

import asyncio

import pytest
from sqlalchemy import update
from typer.testing import CliRunner

from dendrux.cli.main import app
from dendrux.db.models import AgentRun
from dendrux.db.session import get_engine, reset_engine
from dendrux.runtime.stores._factory import store_for_engine


@pytest.mark.parametrize("format_args", [[], ["--format", "1"], ["--format", "2"]])
@pytest.mark.parametrize("exclude", [False, True])
def test_cli_waiting_display_and_filter(tmp_path, monkeypatch, format_args, exclude) -> None:
    url = f"sqlite+aiosqlite:///{tmp_path / 'waiting.db'}"
    monkeypatch.setenv("DENDRUX_DATABASE_URL", url)

    async def seed() -> None:
        engine = await get_engine(url)
        store = store_for_engine(engine)
        for run_id, reason in [("managed", "managed_wait"), ("ordinary", "input")]:
            await store.create_run(run_id, "Test", tenant_id="t")
            await store.pause_run(
                run_id, status="paused", pause_data={"v": 4, "pause_reason": reason}
            )
        async with engine.begin() as conn:
            await conn.execute(
                update(AgentRun).values(storage_format=2 if format_args == ["--format", "2"] else 1)
            )
        await reset_engine()

    asyncio.run(seed())
    args = ["runs", "--status", "paused", "--tenant", "t", *format_args]
    if exclude:
        args += ["--exclude-waiting"]
    result = CliRunner().invoke(app, args, terminal_width=240)
    assert result.exit_code == 0, result.output
    assert "ordinary" in result.output
    assert "Waiting for" in result.output
    if exclude:
        assert "managed" not in result.output
    else:
        assert "managed" in result.output and "subagents" in result.output
