"""Running a migration must not switch off the host application's loggers."""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path

from alembic import command
from alembic.config import Config


def test_migration_run_leaves_every_existing_logger_enabled(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    import dendrux.llm.anthropic  # noqa: F401  (creates the logger before the migration)

    host = logging.getLogger("unrelated_host_app.worker")
    package = Path(dendrux.__file__).parent
    cfg = Config(str(package.parent.parent / "alembic.ini"))
    cfg.set_main_option("script_location", str(package / "db" / "migrations"))
    monkeypatch.setenv("DENDRUX_DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'x.db'}")
    # SQLite cannot run the whole chain; env.py has already run by the time it fails.
    with contextlib.suppress(Exception):
        command.upgrade(cfg, "head")
    assert not logging.getLogger("dendrux.llm.anthropic").disabled
    assert not host.disabled
