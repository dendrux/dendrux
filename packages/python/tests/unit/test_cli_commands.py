"""CLI commands: dashboard launch rules, runs/traces output, URL masking."""

from __future__ import annotations

import asyncio
import re
from typing import TYPE_CHECKING, Any

import pytest
from typer.testing import CliRunner

from dendrux.cli.main import app
from dendrux.dashboard.security import TOKEN_ENV

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

    def test_dashboard_help_lists_new_options_not_the_old_alias(self) -> None:
        out = _plain(runner.invoke(app, ["dashboard", "--help"]).output)
        for flag in ("--token", "--allow-origin", "--allowed-host", "DENDRUX_DASHBOARD_TOKEN"):
            assert flag in out
        assert "--auth-token" not in out


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


class _FakeServer:
    """Stands in for uvicorn.Server so the command can run without binding a port."""

    configs: list[Any] = []

    def __init__(self, config: Any) -> None:
        self.config = config
        _FakeServer.configs.append(config)

    async def serve(self) -> None:
        return None


@pytest.fixture
def fake_uvicorn(monkeypatch: pytest.MonkeyPatch) -> type[_FakeServer]:
    import uvicorn

    _FakeServer.configs = []
    monkeypatch.setattr(uvicorn, "Server", _FakeServer)
    monkeypatch.delenv(TOKEN_ENV, raising=False)
    return _FakeServer


class TestDashboardCommand:
    def test_refuses_non_loopback_bind_without_token(
        self, fake_uvicorn: type[_FakeServer], seeded_db: str
    ) -> None:
        result = runner.invoke(app, ["dashboard", "--host", "0.0.0.0"])
        assert result.exit_code == 1
        assert "without a token" in _plain(result.output)
        assert fake_uvicorn.configs == []

    def test_non_loopback_with_env_token_starts_and_never_prints_it(
        self, fake_uvicorn: type[_FakeServer], seeded_db: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(TOKEN_ENV, "env-token-value")
        result = runner.invoke(app, ["dashboard", "--host", "0.0.0.0", "--port", "9123"])
        assert result.exit_code == 0, result.output
        out = _plain(result.output)
        assert "env-token-value" not in out
        assert "token required" in out
        assert "TLS" in out
        assert "localhost:9123" in out
        assert fake_uvicorn.configs[0].host == "0.0.0.0"

    def test_loopback_without_token_starts(
        self, fake_uvicorn: type[_FakeServer], seeded_db: str
    ) -> None:
        result = runner.invoke(app, ["dashboard"])
        assert result.exit_code == 0, result.output
        assert "token required" not in _plain(result.output)
        assert fake_uvicorn.configs[0].host == "127.0.0.1"

    def test_token_on_argv_warns_and_is_never_echoed(
        self, fake_uvicorn: type[_FakeServer], seeded_db: str
    ) -> None:
        result = runner.invoke(app, ["dashboard", "--token", "argv-token-value"])
        out = _plain(result.output)
        assert result.exit_code == 0, out
        assert "process listings" in out
        assert TOKEN_ENV in out
        assert "argv-token-value" not in out

    def test_env_token_does_not_warn(
        self, fake_uvicorn: type[_FakeServer], seeded_db: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(TOKEN_ENV, "env-token-value")
        out = _plain(runner.invoke(app, ["dashboard"]).output)
        assert "process listings" not in out

    def test_old_auth_token_flag_still_works_with_a_deprecation_note(
        self, fake_uvicorn: type[_FakeServer], seeded_db: str
    ) -> None:
        result = runner.invoke(app, ["dashboard", "--auth-token", "old-flag-token"])
        out = _plain(result.output)
        assert result.exit_code == 0, out
        assert "deprecated" in out
        assert "old-flag-token" not in out

    def test_bad_allow_origin_is_rejected(
        self, fake_uvicorn: type[_FakeServer], seeded_db: str
    ) -> None:
        result = runner.invoke(app, ["dashboard", "--allow-origin", "*"])
        assert result.exit_code == 1
        assert fake_uvicorn.configs == []

    def test_database_password_is_masked_in_the_banner(
        self, fake_uvicorn: type[_FakeServer], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from dendrux.db import session

        async def fake_engine(url: str | None = None) -> Any:
            from sqlalchemy.ext.asyncio import create_async_engine

            return create_async_engine("sqlite+aiosqlite://")

        monkeypatch.setattr(session, "get_engine", fake_engine)
        url = f"postgresql+asyncpg://user:{PASSWORD}@db.example:5432/app?sslpassword={PASSWORD}"
        result = runner.invoke(app, ["dashboard", "--db", url])
        out = _plain(result.output)
        assert result.exit_code == 0, out
        assert PASSWORD not in out
        assert "***" in out

    def test_connection_failure_masks_the_password(
        self, fake_uvicorn: type[_FakeServer], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from dendrux.db import session

        async def boom(url: str | None = None) -> Any:
            raise RuntimeError(f"initialization failed for {url!r}")

        monkeypatch.setattr(session, "get_engine", boom)
        url = f"postgresql+asyncpg://user:{PASSWORD}@db.example:5432/app"
        result = runner.invoke(app, ["dashboard", "--db", url])
        assert result.exit_code == 1
        assert PASSWORD not in result.output
        assert "Cannot connect to database" in _plain(result.output)
        assert fake_uvicorn.configs == []

    def test_api_is_built_with_the_resolved_security_settings(
        self,
        fake_uvicorn: type[_FakeServer],
        seeded_db: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv(TOKEN_ENV, "env-token-value")
        result = runner.invoke(
            app,
            [
                "dashboard",
                "--host",
                "dash.internal",
                "--allow-origin",
                "http://localhost:5173",
                "--allowed-host",
                "alias.internal",
            ],
        )
        assert result.exit_code == 0, result.output
        api = fake_uvicorn.configs[0].app

        async def probe() -> list[int]:
            from httpx import ASGITransport, AsyncClient

            codes = []
            headers = {"Authorization": "Bearer env-token-value"}
            for host in ("dash.internal", "alias.internal", "evil.example"):
                async with AsyncClient(
                    transport=ASGITransport(app=api), base_url=f"http://{host}"
                ) as c:
                    codes.append((await c.get("/api/health", headers=headers)).status_code)
            return codes

        assert asyncio.run(probe()) == [200, 200, 403]


class TestNoSecretsInTracebacks:
    """An exception that escapes must not print the token or the database URL."""

    TOKEN = "SECRET-TOKEN-XYZ"

    @staticmethod
    def _run_escaping(monkeypatch: pytest.MonkeyPatch, *, show_locals: bool | None) -> None:
        import sys

        import dendrux.dashboard.security as security
        from dendrux.cli import main as cli_main

        def boom(*args: Any, **kwargs: Any) -> None:
            raise ValueError("kaboom")

        monkeypatch.setattr(security, "resolve_token", boom)
        monkeypatch.setattr(
            sys,
            "argv",
            ["dendrux", "dashboard", "--token", TestNoSecretsInTracebacks.TOKEN],
        )
        if show_locals is not None:
            monkeypatch.setattr(cli_main.app, "pretty_exceptions_show_locals", show_locals)
        # Typer installs sys.excepthook and lets the exception escape; the
        # interpreter then calls the hook. Emulate that here.
        monkeypatch.setattr(sys, "excepthook", sys.excepthook)
        try:
            cli_main.app()
        except ValueError:
            sys.excepthook(*sys.exc_info())

    def test_root_app_never_shows_locals(self) -> None:
        assert app.pretty_exceptions_show_locals is False
        from dendrux.cli.runs import app as runs_app

        assert runs_app.pretty_exceptions_show_locals is False

    def test_escaped_exception_does_not_print_the_token(
        self, monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
    ) -> None:
        self._run_escaping(monkeypatch, show_locals=None)
        out, err = capfd.readouterr()
        assert "kaboom" in out + err
        assert self.TOKEN not in out + err

    def test_the_check_would_catch_locals_being_shown(
        self, monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
    ) -> None:
        """Control: with locals on, the same crash does print the token."""
        self._run_escaping(monkeypatch, show_locals=True)
        out, err = capfd.readouterr()
        assert self.TOKEN in out + err

    def test_server_failure_is_reported_without_traceback_or_secrets(
        self, fake_uvicorn: type[_FakeServer], seeded_db: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def failing_serve(self: Any) -> None:
            raise OSError(f"cannot bind postgresql://u:{PASSWORD}@h/db")

        monkeypatch.setattr(_FakeServer, "serve", failing_serve)
        result = runner.invoke(app, ["dashboard", "--token", self.TOKEN])
        assert result.exit_code == 1
        out = _plain(result.output)
        assert "Dashboard stopped" in out
        assert "OSError" in out
        assert self.TOKEN not in out
        assert PASSWORD not in out
        assert "Traceback" not in out


class TestDashboardBannerAndWarnings:
    def test_ipv6_host_is_bracketed_in_the_url(
        self, fake_uvicorn: type[_FakeServer], seeded_db: str
    ) -> None:
        result = runner.invoke(app, ["dashboard", "--host", "::1", "--port", "9124"])
        assert result.exit_code == 0, result.output
        assert "http://[::1]:9124" in _plain(result.output)

    def test_star_allowed_host_warns(self, fake_uvicorn: type[_FakeServer], seeded_db: str) -> None:
        out = _plain(runner.invoke(app, ["dashboard", "--allowed-host", "*"]).output)
        assert "turns the Host check off" in out

    def test_named_allowed_host_does_not_warn(
        self, fake_uvicorn: type[_FakeServer], seeded_db: str
    ) -> None:
        out = _plain(runner.invoke(app, ["dashboard", "--allowed-host", "dash.example"]).output)
        assert "Host check off" not in out


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
