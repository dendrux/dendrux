"""CLI commands: runs/traces output, URL masking."""

from __future__ import annotations

import asyncio
import re
from typing import TYPE_CHECKING, Any

import pytest
from typer.testing import CliRunner

from dendrux.cli.main import app

if TYPE_CHECKING:
    from pathlib import Path

runner = CliRunner()
PASSWORD = "hunter2-pw"


def _plain(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


@pytest.fixture
def seeded_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """A throwaway SQLite database with two runs, pointed at by the env var."""
    from dendrux.db.session import get_engine, reset_engine
    from dendrux.runtime.state import SQLAlchemyStateStore

    url = f"sqlite+aiosqlite:///{tmp_path / 'cli.db'}"
    monkeypatch.setenv("DENDRUX_DATABASE_URL", url)

    async def seed() -> None:
        engine = await get_engine(url)
        store = SQLAlchemyStateStore(engine)
        await store.create_run("run-alpha", "AgentA", tenant_id="acme", model="m1")
        await store.create_run("run-beta", "AgentB", model="m2")
        await store.save_trace("run-alpha", "user", "hello [red]markup[/red]", order_index=0)
        for i in range(1, 6):
            await store.save_trace("run-alpha", "assistant", f"reply {i}", order_index=i)
        await store.save_tool_call(
            "run-alpha",
            tool_call_id="tc1",
            provider_tool_call_id=None,
            tool_name="lookup",
            target="server",
            params={"q": 1},
            result_payload='"ok"',
            success=False,
            duration_ms=12,
            iteration_index=1,
            error_message="boom [bold]x[/bold]",
        )
        await reset_engine()

    asyncio.run(seed())
    return url


class TestVersionAndHelp:
    def test_version(self) -> None:
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert result.output.startswith("dendrux ")


class TestNoSecretsInTracebacks:
    def test_root_app_never_shows_locals(self) -> None:
        from dendrux.cli.runs import app as runs_app

        assert app.pretty_exceptions_show_locals is False
        assert runs_app.pretty_exceptions_show_locals is False

    @pytest.mark.parametrize("show_locals", [False, True], ids=["default", "control"])
    def test_escaped_runs_exception_hides_database_url_locals(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capfd: pytest.CaptureFixture[str],
        show_locals: bool,
    ) -> None:
        import os
        import sys

        from dendrux.cli import runs

        url = f"postgresql+asyncpg://user:{PASSWORD}@db.example/app"
        monkeypatch.setenv("DENDRUX_DATABASE_URL", url)

        async def boom(*_args: Any, **_kwargs: Any) -> None:
            database_url = os.environ["DENDRUX_DATABASE_URL"]
            assert database_url
            raise ValueError("command failed")

        monkeypatch.setattr(runs, "_list_runs", boom)
        monkeypatch.setattr(sys, "argv", ["dendrux", "runs"])
        monkeypatch.setattr(sys, "excepthook", sys.excepthook)
        if show_locals:
            monkeypatch.setattr(app, "pretty_exceptions_show_locals", True)
            monkeypatch.setattr(runs.app, "pretty_exceptions_show_locals", True)
        try:
            app()
        except ValueError:
            sys.excepthook(*sys.exc_info())
        else:
            pytest.fail("expected an escaped command exception")
        captured = capfd.readouterr()
        output = captured.out + captured.err
        assert "command failed" in output
        assert (PASSWORD in output) is show_locals


class TestRunsCommand:
    def test_lists_all_tenants_when_no_filter(self, seeded_db: str) -> None:
        out = _plain(runner.invoke(app, ["runs"], terminal_width=200).output)
        assert "run-alpha" in out
        assert "run-beta" in out

    def test_tenant_filter_narrows(self, seeded_db: str) -> None:
        out = _plain(runner.invoke(app, ["runs", "--tenant", "acme"], terminal_width=200).output)
        assert "run-alpha" in out
        assert "run-beta" not in out

    def test_status_filter_and_empty_result(self, seeded_db: str) -> None:
        result = runner.invoke(app, ["runs", "--status", "success"])
        assert "No runs found" in _plain(result.output)

    def test_limit_is_bounded_below(self, seeded_db: str) -> None:
        result = runner.invoke(app, ["runs", "--limit", "0"], terminal_width=200)
        assert result.exit_code == 0
        assert "run-" in _plain(result.output)

    def test_connection_error_masks_the_password(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from dendrux.db import session

        async def boom(url: str | None = None) -> Any:
            raise RuntimeError(f"cannot reach postgresql://user:{PASSWORD}@db.example:5432/app")

        monkeypatch.setattr(session, "get_engine", boom)
        result = runner.invoke(app, ["runs"])
        assert result.exit_code == 1
        assert PASSWORD not in result.output
        assert "Cannot connect to database" in result.output


class TestTracesCommand:
    def test_shows_run_header_and_traces_without_markup_interpretation(
        self, seeded_db: str
    ) -> None:
        out = _plain(runner.invoke(app, ["traces", "run-alpha"], terminal_width=200).output)
        assert "AgentA" in out
        assert "hello [red]markup[/red]" in out
        assert "reply 5" in out

    def test_limit_truncates_with_a_notice(self, seeded_db: str) -> None:
        out = _plain(runner.invoke(app, ["traces", "run-alpha", "--limit", "2"]).output)
        assert "reply 1" in out
        assert "reply 3" not in out
        assert "more traces not shown" in out

    def test_tools_table(self, seeded_db: str) -> None:
        out = _plain(
            runner.invoke(app, ["traces", "run-alpha", "--tools"], terminal_width=200).output
        )
        assert "lookup" in out
        assert "boom [bold]x[/bold]" in out

    def test_tools_limit_notice(self, seeded_db: str) -> None:
        out = _plain(
            runner.invoke(
                app, ["traces", "run-alpha", "--tools", "--limit", "1"], terminal_width=200
            ).output
        )
        assert "lookup" in out
        assert "more tool calls" not in out  # exactly one record, nothing hidden

    def test_unknown_run(self, seeded_db: str) -> None:
        result = runner.invoke(app, ["traces", "missing"])
        assert result.exit_code == 1
        assert "Run not found" in _plain(result.output)

    def test_connection_error_masks_the_password(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from dendrux.db import session

        async def boom(url: str | None = None) -> Any:
            raise RuntimeError(
                f"failed: postgresql+asyncpg://u:{PASSWORD}@h/db?password={PASSWORD}"
            )

        monkeypatch.setattr(session, "get_engine", boom)
        result = runner.invoke(app, ["traces", "x"])
        assert result.exit_code == 1
        assert PASSWORD not in result.output


class TestTraceLimitNotices:
    """The overflow probe must work at the store's own maximum, not just below it."""

    @staticmethod
    def _seed(url: str, traces: int, tool_calls: int) -> None:
        from dendrux.db.session import get_engine, reset_engine
        from dendrux.runtime.state import SQLAlchemyStateStore

        async def seed() -> None:
            engine = await get_engine(url)
            store = SQLAlchemyStateStore(engine)
            await store.create_run("big", "A")
            for i in range(traces):
                await store.save_trace("big", "assistant", f"t{i}", order_index=i)
            for i in range(tool_calls):
                await store.save_tool_call(
                    "big",
                    tool_call_id=f"tc{i}",
                    provider_tool_call_id=None,
                    tool_name="tool",
                    target="server",
                    params=None,
                    result_payload='"ok"',
                    success=True,
                    duration_ms=1,
                    iteration_index=0,
                )
            await reset_engine()

        asyncio.run(seed())

    def test_notice_when_one_more_row_exists_exactly_at_the_limit(self, seeded_db: str) -> None:
        self._seed(seeded_db, traces=4, tool_calls=4)
        out = _plain(runner.invoke(app, ["traces", "big", "--tools", "--limit", "3"]).output)
        assert "more traces not shown" in out
        assert "more tool calls not shown" in out

    def test_no_notice_when_exactly_the_limit_exists(self, seeded_db: str) -> None:
        self._seed(seeded_db, traces=3, tool_calls=3)
        out = _plain(runner.invoke(app, ["traces", "big", "--tools", "--limit", "3"]).output)
        assert "not shown" not in out

    def test_notice_at_the_store_maximum(self, seeded_db: str) -> None:
        from dendrux.cli.traces import MAX_ROWS

        self._seed(seeded_db, traces=MAX_ROWS + 1, tool_calls=MAX_ROWS + 1)
        out = _plain(runner.invoke(app, ["traces", "big", "--tools", "--limit", "999999"]).output)
        assert "more traces not shown" in out
        assert "more tool calls not shown" in out
