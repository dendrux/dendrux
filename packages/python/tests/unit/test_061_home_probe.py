"""An optional legacy-home check cannot prevent use of the local database."""

import logging
from pathlib import Path

import pytest
from sqlalchemy import inspect
from sqlalchemy.exc import DatabaseError

from dendrux.cli.db import _build_alembic_config, _database_url
from dendrux.db import session
from dendrux.db.models import Base


@pytest.fixture
def local_home(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DENDRUX_DATABASE_URL", raising=False)
    home = tmp_path / "private-home-sentinel"
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    return home


@pytest.mark.parametrize("failure", [RuntimeError, PermissionError, OSError])
async def test_061_unavailable_home_keeps_local_default(local_home, monkeypatch, caplog, failure):
    exists = Path.exists

    def unavailable():
        raise failure("private-home-error-sentinel")

    if failure is RuntimeError:
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: unavailable()))
    else:

        def probe(path):
            if path == local_home / ".dendrux/dendrux.db":
                return unavailable()
            return exists(path)

        monkeypatch.setattr(Path, "exists", probe)
    await session.reset_engine()
    try:
        with caplog.at_level(logging.WARNING, logger="dendrux.db.session"):
            assert session.get_database_url() == session.DEFAULT_SQLITE_URL
            assert _database_url() == session.DEFAULT_SQLITE_URL
            assert (
                _build_alembic_config().get_main_option("sqlalchemy.url")
                == session.DEFAULT_SQLITE_URL
            )
            engine = await session.get_engine()
        assert Path(engine.url.database).resolve() == Path("dendrux.db").resolve()
        async with engine.connect() as conn:
            names = await conn.run_sync(lambda c: inspect(c).get_table_names())
        assert set(Base.metadata.tables) <= set(names)
        assert not any("alembic_version" in name for name in names)
    finally:
        await session.reset_engine()
    assert not [r for r in caplog.records if r.name == "dendrux.db.session"]
    assert "private-home-error-sentinel" not in caplog.text


@pytest.mark.parametrize("source", ["environment", "argument"])
async def test_061_explicit_url_bypasses_home_probe(local_home, monkeypatch, source):
    def forbidden(cls):
        pytest.fail("explicit URL must bypass the home probe")

    monkeypatch.setattr(Path, "home", classmethod(forbidden))
    url = "sqlite+aiosqlite:///:memory:"
    if source == "environment":
        monkeypatch.setenv("DENDRUX_DATABASE_URL", url)
        assert session.get_database_url() == url
        assert _database_url() == url
        assert _build_alembic_config().get_main_option("sqlalchemy.url") == url
    await session.reset_engine()
    try:
        engine = await session.get_engine(url if source == "argument" else None)
        assert engine.url.database == ":memory:"
    finally:
        await session.reset_engine()


@pytest.mark.parametrize("old_exists,local_exists", [(False, False), (True, False), (True, True)])
def test_061_legacy_warning_requires_found_file(local_home, caplog, old_exists, local_exists):
    if old_exists:
        old = local_home / ".dendrux/dendrux.db"
        old.parent.mkdir(parents=True)
        old.touch()
    if local_exists:
        Path("dendrux.db").touch()
    with caplog.at_level(logging.WARNING, logger="dendrux.db.session"):
        assert session.get_database_url() == session.DEFAULT_SQLITE_URL
    warnings = [
        r for r in caplog.records if getattr(r, "code", None) == "local_database_default_changed"
    ]
    assert len(warnings) == int(old_exists and not local_exists)
    assert str(local_home) not in caplog.text


async def test_061_selected_database_error_still_fails(local_home):
    Path("dendrux.db").write_bytes(b"not a SQLite database")
    await session.reset_engine()
    try:
        with pytest.raises(DatabaseError):
            await session.get_engine()
        assert session._engine is None
    finally:
        await session.reset_engine()


def test_061_local_path_error_is_not_an_optional_home_error(local_home, monkeypatch):
    def denied(path):
        raise PermissionError("selected-local-path")

    monkeypatch.setattr(Path, "exists", denied)
    with pytest.raises(PermissionError, match="selected-local-path"):
        session.get_database_url()
