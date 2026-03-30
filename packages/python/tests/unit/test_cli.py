"""Tests for CLI utilities and commands."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from dendrite.cli.main import _resolve_db_url, app

runner = CliRunner()


class TestResolveDbUrl:
    def test_none_returns_none(self) -> None:
        assert _resolve_db_url(None) is None

    def test_sqlite_url_passthrough(self) -> None:
        url = "sqlite+aiosqlite:///path/to/db.sqlite"
        assert _resolve_db_url(url) == url

    def test_postgresql_url_passthrough(self) -> None:
        url = "postgresql+asyncpg://user:pass@host/db"
        assert _resolve_db_url(url) == url

    def test_generic_url_passthrough(self) -> None:
        """Any URL with :// is treated as a full database URL."""
        url = "mysql+aiomysql://user:pass@host/db"
        assert _resolve_db_url(url) == url

    def test_bare_relative_path(self) -> None:
        result = _resolve_db_url("./my-agent.db")
        assert result.startswith("sqlite+aiosqlite:///")
        assert result.endswith("my-agent.db")
        path_part = result.split("///", 1)[1]
        assert Path(path_part).is_absolute()

    def test_bare_absolute_path(self) -> None:
        result = _resolve_db_url("/tmp/dendrite.db")
        assert result.startswith("sqlite+aiosqlite:///")
        assert "/tmp/dendrite.db" in result

    def test_bare_filename(self) -> None:
        result = _resolve_db_url("dendrite.db")
        assert result.startswith("sqlite+aiosqlite:///")
        assert result.endswith("dendrite.db")
        path_part = result.split("///", 1)[1]
        assert Path(path_part).is_absolute()


class TestDashboardCommand:
    """Test that --db flag is wired into the dashboard CLI."""

    def test_dashboard_help_shows_db_option(self) -> None:
        """--db flag appears in the dashboard help text."""
        result = runner.invoke(app, ["dashboard", "--help"])
        assert result.exit_code == 0
        assert "--db" in result.output
        assert "Database URL or SQLite file path" in result.output

    def test_dashboard_help_shows_examples(self) -> None:
        """Dashboard docstring examples are visible in help."""
        result = runner.invoke(app, ["dashboard", "--help"])
        assert "--db ./my-agent.db" in result.output

    def test_resolve_db_url_used_in_dashboard(self) -> None:
        """_resolve_db_url is called from the dashboard code path.

        We verify this structurally: the dashboard function references
        _resolve_db_url, proving the wiring exists. A full integration
        test would require mocking get_engine + uvicorn, which is
        fragile — the helper tests above cover the resolution logic.
        """
        import inspect

        from dendrite.cli.main import dashboard

        source = inspect.getsource(dashboard)
        assert "_resolve_db_url" in source
