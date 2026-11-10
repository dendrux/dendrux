"""Schema checks preserve exit codes and guidance for each history state."""

import builtins
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import typer
from alembic.script import ScriptDirectory
from rich.console import Console
from typer.testing import CliRunner

from dendrux.cli import db
from dendrux.cli.main import app
from dendrux.db._version_table import MigrationHistory, MigrationVersionTableConflictError

URL = "sqlite+aiosqlite:///check.db"
ADOPTION = "Behind: Run `dendrux db migrate` to adopt verified legacy Dendrux history."
UNSTAMPED = (
    f"Unknown: {URL} has no dendrux_alembic_version table. "
    "Run `dendrux db migrate`, or pass --allow-unstamped if the app builds "
    "the schema with create_all."
)


@pytest.fixture(autouse=True)
def wide_console(monkeypatch):
    monkeypatch.setattr(db, "console", Console(width=240, color_system=None))


@pytest.fixture
def schema_state(monkeypatch):
    monkeypatch.setattr(db, "_database_url", lambda: URL)
    monkeypatch.setattr(db, "_build_alembic_config", Mock())
    script = Mock()
    script.get_current_head.return_value = "head"
    script.walk_revisions.return_value = [
        SimpleNamespace(revision="head"),
        SimpleNamespace(revision="prior"),
    ]
    monkeypatch.setattr(ScriptDirectory, "from_config", Mock(return_value=script))
    read = AsyncMock()
    monkeypatch.setattr(db, "_read_schema_state", read)
    return read


@pytest.mark.parametrize("allow_unstamped", [False, True])
@pytest.mark.parametrize(
    "history,missing,code,message",
    [
        (MigrationHistory("prior", needs_adoption=True), [], 1, ADOPTION),
        (MigrationHistory(None), [], 0, f"✓ {URL} has every table, column and index (unstamped)."),
        (MigrationHistory(None), ["agent_runs"], 1, f"Behind: {URL} is missing: agent_runs"),
        (MigrationHistory("head"), [], 0, f"✓ {URL} is at the current revision (head)."),
        (
            MigrationHistory("head"),
            ["agent_runs", "agent_runs.id"],
            1,
            f"Behind: {URL} is stamped head but is missing: agent_runs, agent_runs.id. "
            "Run `alembic downgrade -1` then `dendrux db migrate`.",
        ),
        (
            MigrationHistory("future"),
            [],
            2,
            f"Unknown: {URL} is at revision future, which this version of dendrux "
            "does not know (is it newer?).",
        ),
        (
            MigrationHistory("prior"),
            [],
            1,
            f"Behind: {URL} is at prior; this version needs head. Run `dendrux db migrate`.",
        ),
    ],
    ids=[
        "adoption",
        "complete-unstamped",
        "incomplete-unstamped",
        "current",
        "incomplete-current",
        "unknown",
        "behind",
    ],
)
def test_check_history_outcomes(schema_state, allow_unstamped, history, missing, code, message):
    schema_state.return_value = history, missing
    args = ["db", "check"] + (["--allow-unstamped"] if allow_unstamped else [])
    result = CliRunner().invoke(app, args)
    if history.revision is None and not allow_unstamped:
        code, message = 2, UNSTAMPED
    assert result.exit_code == code, result.output
    assert result.output == message + "\n"
    schema_state.assert_awaited_once_with(URL)


@pytest.mark.parametrize("conflict", [False, True])
def test_check_schema_read_errors(schema_state, monkeypatch, conflict):
    url = "postgresql+asyncpg://user:secret-check-password@localhost/check"
    safe = "postgresql+asyncpg://user:***@localhost/check"
    monkeypatch.setattr(db, "_database_url", lambda: url)
    error = (
        MigrationVersionTableConflictError() if conflict else RuntimeError(f"unavailable: {url}")
    )
    schema_state.side_effect = error
    result = CliRunner().invoke(app, ["db", "check", "--allow-unstamped"])
    assert result.exit_code == 2, result.output
    expected = (
        str(error) if conflict else f"Could not read the schema of {safe}: unavailable: {safe}"
    )
    assert " ".join(result.output.split()) == expected
    assert "secret-check-password" not in result.output
    schema_state.assert_awaited_once_with(url)


def test_check_without_alembic(monkeypatch):
    original_import = builtins.__import__

    def import_without_script(name, *args, **kwargs):
        if name == "alembic.script":
            raise ImportError("alembic is unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_script)
    result = CliRunner().invoke(app, ["db", "check"])
    assert result.exit_code == 1
    assert result.output == "Alembic not installed. Run: pip install dendrux\n"


@pytest.mark.parametrize("allow_unstamped", [False, True])
@pytest.mark.parametrize(
    "history,missing,code,message",
    [
        (MigrationHistory("prior", needs_adoption=True), [], 1, ADOPTION),
        (MigrationHistory(None, needs_adoption=True), ["agent_runs"], 1, ADOPTION),
        (MigrationHistory(None), [], 0, f"✓ {URL} has every table, column and index (unstamped)."),
        (MigrationHistory(None), ["agent_runs"], 1, f"Behind: {URL} is missing: agent_runs"),
    ],
    ids=["adoption", "adoption-before-unstamped", "complete", "incomplete"],
)
def test_check_history_helper_exits(capsys, allow_unstamped, history, missing, code, message):
    if history.revision is None and not history.needs_adoption and not allow_unstamped:
        code, message = 2, UNSTAMPED
    with pytest.raises(typer.Exit) as caught:
        db._check_history(URL, history, missing, allow_unstamped=allow_unstamped)
    assert caught.value.exit_code == code
    assert capsys.readouterr().out == message + "\n"


@pytest.mark.parametrize("allow_unstamped", [False, True])
@pytest.mark.parametrize("missing", [[], ["agent_runs"]])
def test_check_history_helper_leaves_stamped_revision_to_check(capsys, allow_unstamped, missing):
    assert (
        db._check_history(URL, MigrationHistory("head"), missing, allow_unstamped=allow_unstamped)
        is None
    )
    assert capsys.readouterr().out == ""
