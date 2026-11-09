"""Status preserves operator guidance and optional bootstrap markers."""

from unittest.mock import AsyncMock

import pytest
from typer.testing import CliRunner

from dendrux.cli import db
from dendrux.cli.main import app
from dendrux.db._version_table import MigrationVersionTableConflictError


@pytest.mark.parametrize("bootstrap", [None, "revision-to-resume"])
def test_status_formats_optional_bootstrap(monkeypatch, bootstrap):
    monkeypatch.setattr(db, "_database_url", lambda: "unused")
    read = AsyncMock(return_value=("head", "2", "done", bootstrap))
    monkeypatch.setattr(db, "_read_activation_status", read)
    result = CliRunner().invoke(app, ["db", "status"])
    assert result.exit_code == 0, result.output
    expected = "revision=head storage_format=2 activate_step=done"
    if bootstrap is not None:
        expected += f" bootstrap_in_progress={bootstrap}"
    assert " ".join(result.output.split()) == expected
    read.assert_awaited_once_with("unused")


@pytest.mark.parametrize("conflict", [False, True])
def test_status_reports_safe_failure_guidance(monkeypatch, conflict):
    error = (
        MigrationVersionTableConflictError()
        if conflict
        else RuntimeError("private-database-sentinel")
    )
    monkeypatch.setattr(db, "_database_url", lambda: "unused")
    monkeypatch.setattr(db, "_read_activation_status", AsyncMock(side_effect=error))
    result = CliRunner().invoke(app, ["db", "status"])
    assert result.exit_code == 2
    output = " ".join(result.output.split())
    if conflict:
        assert output == str(error)
    else:
        assert output == "Could not read database status. Run `dendrux db check`."
    assert "private-database-sentinel" not in output
