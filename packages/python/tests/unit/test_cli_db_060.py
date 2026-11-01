"""Storage format CLI filtering happens before pagination."""

import datetime as dt

import pytest
from sqlalchemy import create_engine
from typer.testing import CliRunner

from dendrux.cli.main import app
from dendrux.db.models import AgentRun, Base


@pytest.mark.parametrize("format", ["1", "2"])
def test_runs_format_filters_before_limit(tmp_path, monkeypatch, format):
    path = tmp_path / "runs.db"
    engine = create_engine(f"sqlite:///{path}")
    try:
        Base.metadata.create_all(engine)
        with engine.begin() as conn:
            conn.execute(
                AgentRun.__table__.insert(),
                [
                    {
                        "id": "legacy",
                        "agent_name": "LegacyAgent",
                        "storage_format": 1,
                        "created_at": dt.datetime(2025, 1, 1, tzinfo=dt.UTC),
                    },
                    {
                        "id": "new",
                        "agent_name": "NewAgent",
                        "storage_format": 2,
                        "created_at": dt.datetime(2026, 1, 1, tzinfo=dt.UTC),
                    },
                ],
            )
    finally:
        engine.dispose()
    monkeypatch.setenv("DENDRUX_DATABASE_URL", f"sqlite+aiosqlite:///{path}")
    result = CliRunner().invoke(app, ["runs", "--format", format, "--limit", "1"])
    assert result.exit_code == 0, result.output
    assert ("LegacyAgent" if format == "1" else "NewAgent") in result.output
    assert ("NewAgent" if format == "1" else "LegacyAgent") not in result.output
    assert CliRunner().invoke(app, ["runs", "--format", "3"]).exit_code == 2
