"""Running a migration must not switch off the host application's loggers."""

from __future__ import annotations

import contextlib
import logging
import logging.config
from pathlib import Path
from typing import Any

from alembic import command
from alembic.config import Config


def _migrate(tmp_path: Path, monkeypatch: Any) -> list[dict[str, Any]]:
    """Run `alembic upgrade head` on SQLite; return the fileConfig calls env.py made."""
    import dendrux.llm.anthropic  # noqa: F401  (creates the logger before the migration)

    calls: list[dict[str, Any]] = []
    real = logging.config.fileConfig

    def spy(*args: Any, **kwargs: Any) -> None:
        calls.append({"args": args, "kwargs": kwargs})
        real(*args, **kwargs)

    # env.py does `from logging.config import fileConfig` each time Alembic loads it.
    monkeypatch.setattr(logging.config, "fileConfig", spy)
    package = Path(dendrux.__file__).parent
    cfg = Config(str(package.parent.parent / "alembic.ini"))
    cfg.set_main_option("script_location", str(package / "db" / "migrations"))
    monkeypatch.setenv("DENDRUX_DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'x.db'}")
    # SQLite cannot run the whole chain; env.py has already run by the time it fails.
    with contextlib.suppress(Exception):
        command.upgrade(cfg, "head")
    return calls


def test_migration_run_leaves_every_existing_logger_enabled(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    host = logging.getLogger("unrelated_host_app.worker")
    calls = _migrate(tmp_path, monkeypatch)
    # env.py's logging configuration must really have run, and without disabling loggers.
    assert len(calls) == 1, "env.py did not configure logging: the check below proves nothing"
    assert calls[0]["kwargs"].get("disable_existing_loggers") is False
    assert not logging.getLogger("dendrux.llm.anthropic").disabled
    assert not host.disabled


def test_the_check_fails_when_alembic_never_reaches_env_py(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    """Guards the guard: an early Alembic failure must not look like a pass."""

    def explode(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("alembic failed before loading env.py")

    monkeypatch.setattr(command, "upgrade", explode)
    assert _migrate(tmp_path, monkeypatch) == []
