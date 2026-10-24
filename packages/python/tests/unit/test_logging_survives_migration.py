"""Running a migration must not switch off the host application's loggers."""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config


@pytest.mark.xfail(
    strict=True,
    reason=(
        "contract gap: src/dendrux/db/migrations/env.py calls fileConfig() with the default "
        "disable_existing_loggers=True, so a migration run disables every existing dendrux.* "
        "logger in the host process. Fix: fileConfig(..., disable_existing_loggers=False), "
        "then delete this xfail."
    ),
)
def test_migration_run_leaves_dendrux_loggers_enabled(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    import dendrux.llm.anthropic  # noqa: F401  (creates the logger before the migration)

    package = Path(dendrux.__file__).parent
    cfg = Config(str(package.parent.parent / "alembic.ini"))
    cfg.set_main_option("script_location", str(package / "db" / "migrations"))
    monkeypatch.setenv("DENDRUX_DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'x.db'}")
    # SQLite cannot run the whole chain; env.py has already run by the time it fails.
    with contextlib.suppress(Exception):
        command.upgrade(cfg, "head")
    assert not logging.getLogger("dendrux.llm.anthropic").disabled
