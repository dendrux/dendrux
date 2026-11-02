"""Local database resolution agrees across runtime, CLI and migrations."""

import logging
from pathlib import Path

import pytest

from dendrux.cli.db import _build_alembic_config, _database_url
from dendrux.db.session import get_database_url, get_engine, reset_engine


@pytest.fixture
def local_databases(monkeypatch, tmp_path):
    monkeypatch.delenv("DENDRUX_DATABASE_URL", raising=False)
    monkeypatch.chdir(tmp_path)
    home = tmp_path / "private-home-sentinel"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    return home


def test_runtime_cli_and_migrations_share_local_default(local_databases):
    expected = "sqlite+aiosqlite:///./dendrux.db"
    assert get_database_url() == expected
    assert _database_url() == expected
    assert _build_alembic_config().get_main_option("sqlalchemy.url") == expected


@pytest.mark.parametrize(
    "local_exists,old_exists,explicit",
    [
        (False, True, False),
        (True, True, False),
        (False, False, False),
        (False, True, True),
    ],
)
def test_legacy_file_warning_is_coded_and_sanitized(
    local_databases, monkeypatch, caplog, local_exists, old_exists, explicit
):
    old_file = local_databases / ".dendrux/dendrux.db"
    if old_exists:
        old_file.parent.mkdir()
        old_file.touch()
    if local_exists:
        Path("dendrux.db").touch()
    if explicit:
        monkeypatch.setenv("DENDRUX_DATABASE_URL", "sqlite+aiosqlite:///chosen.db")
    with caplog.at_level(logging.WARNING, logger="dendrux.db.session"):
        get_database_url()
    records = [
        r for r in caplog.records if getattr(r, "code", None) == "local_database_default_changed"
    ]
    assert len(records) == int(old_exists and not local_exists and not explicit)
    if records:
        message = records[0].getMessage()
        assert "~/.dendrux/dendrux.db" in message
        assert "./dendrux.db" in message
        assert "DENDRUX_DATABASE_URL" in message
        assert str(local_databases) not in message


async def test_default_engine_creates_file_in_working_directory(local_databases, monkeypatch):
    from dendrux.db import session

    if session.DEFAULT_SQLITE_URL != "sqlite+aiosqlite:///./dendrux.db":
        monkeypatch.setattr(
            session, "DEFAULT_SQLITE_URL", f"sqlite+aiosqlite:///{local_databases / 'old.db'}"
        )
    await reset_engine()
    try:
        engine = await get_engine()
        assert Path(engine.url.database).resolve() == Path("dendrux.db").resolve()
        assert Path("dendrux.db").is_file()
        assert not (local_databases / ".dendrux").exists()
    finally:
        await reset_engine()


@pytest.mark.parametrize("memory", [False, True])
@pytest.mark.parametrize("rollback", [False, True])
@pytest.mark.parametrize("explicit", [False, True])
async def test_resolved_engine_sessions_commit_and_rollback(
    tmp_path, monkeypatch, memory, rollback, explicit
):
    from sqlalchemy import text

    from dendrux.db.session import get_session

    await reset_engine()
    url = (
        "sqlite+aiosqlite:///:memory:"
        if memory
        else f"sqlite+aiosqlite:///{tmp_path / 'sessions.db'}"
    )
    monkeypatch.setenv("DENDRUX_DATABASE_URL", url)
    try:
        engine = await get_engine(url)
        async with engine.begin() as conn:
            await conn.execute(text("CREATE TABLE session_probe (value INTEGER)"))

        async def write():
            async with get_session(engine if explicit else None) as session:
                await session.execute(text("INSERT INTO session_probe VALUES (1)"))
                if rollback:
                    raise LookupError("rollback probe")

        if rollback:
            with pytest.raises(LookupError):
                await write()
        else:
            await write()
        async with engine.connect() as conn:
            assert (
                await conn.execute(text("SELECT COUNT(*) FROM session_probe"))
            ).scalar_one() == int(not rollback)
    finally:
        await reset_engine()
