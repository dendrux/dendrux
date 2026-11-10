"""Independent-review regressions for migration ownership and repair."""

import asyncio
import logging

import pytest
from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import inspect, text

from dendrux.db.models import Base
from tests.integration.test_061_migrations import app_config, app_upgrade, migrate
from tests.integration.test_061_migrations import migration_database as migration_database
from tests.integration.test_061_migrations_repair import invoke
from tests.migrations.test_060_activate import HEAD, MIGRATIONS, PRIOR, alembic, prior_schema
from tests.unit.test_061_migrations import DEDICATED, LEGACY, snapshot, stamp


def real_legacy(conn):
    prior_schema(conn)
    conn.execute(text(f"DROP TABLE {LEGACY}"))
    MigrationContext.configure(conn)._ensure_version_table()
    conn.execute(text(f"INSERT INTO {LEGACY} VALUES (:revision)"), {"revision": PRIOR})


async def test_061_legacy_pk_leaves_app_alembic_namespace_free(migration_database, tmp_path):
    engine = migration_database
    async with engine.begin() as conn:
        await conn.run_sync(real_legacy)
        assert (
            await conn.run_sync(lambda c: inspect(c).get_pk_constraint(LEGACY)["name"])
            == "alembic_version_pkc"
        )
    result = await migrate()
    assert result.exit_code == 0, result.output
    cfg = app_config(tmp_path / "app")
    async with engine.begin() as conn:
        assert await conn.run_sync(lambda c: app_upgrade(c, cfg, "app_two")) == ("app_two",)
        assert await conn.scalar(text(f"SELECT version_num FROM {DEDICATED}")) == HEAD
        if engine.dialect.name == "postgresql":
            assert (
                await conn.run_sync(lambda c: inspect(c).get_pk_constraint(DEDICATED)["name"])
                == "dendrux_alembic_version_pkc"
            )


async def test_061_postgres_pk_rename_rolls_back_with_table(migration_database, monkeypatch):
    if migration_database.dialect.name != "postgresql":
        pytest.skip("PostgreSQL transactional constraint rename")
    from dendrux.db import _migration_ddl

    engine = migration_database
    async with engine.begin() as conn:
        await conn.run_sync(real_legacy)
        before = await conn.run_sync(snapshot)
    original = _migration_ddl.bounded_step
    reached = []

    def fail_constraint(conn, statement, **kw):
        if "RENAME CONSTRAINT" in statement:
            reached.append(statement)
            raise RuntimeError("constraint rename interruption")
        return original(conn, statement, **kw)

    monkeypatch.setattr(_migration_ddl, "bounded_step", fail_constraint)
    result = await migrate()
    assert result.exit_code == 1 and reached
    async with engine.connect() as conn:
        assert await conn.run_sync(snapshot) == before
        assert (
            await conn.run_sync(lambda c: inspect(c).get_pk_constraint(LEGACY)["name"])
            == "alembic_version_pkc"
        )
    monkeypatch.setattr(_migration_ddl, "bounded_step", original)
    assert (await migrate()).exit_code == 0


@pytest.mark.parametrize("revision", [PRIOR, HEAD])
async def test_061_verified_legacy_readers_need_migrate_without_conflict(
    migration_database, revision, caplog
):
    engine = migration_database
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.run_sync(lambda c: stamp(c, LEGACY, [revision]))
        before = await conn.run_sync(snapshot)
    with caplog.at_level(logging.WARNING):
        for args in (("check",), ("check", "--allow-unstamped")):
            result = await invoke(*args)
            assert result.exit_code == 1, result.output
            assert "dendrux db migrate" in " ".join(result.output.split())
            assert "conflict" not in result.output
        result = await invoke("status")
        assert result.exit_code == 0 and f"revision={revision}" in result.output
        result = await invoke("activate")
        assert result.exit_code == 1, result.output
        assert "dendrux db migrate" in " ".join(result.output.split())
    assert not any(
        getattr(r, "code", None) == "migration_version_table_conflict" for r in caplog.records
    )
    async with engine.connect() as conn:
        assert await conn.run_sync(snapshot) == before


@pytest.mark.parametrize("operation", ["upgrade", "stamp"])
async def test_061_empty_history_is_unstamped(migration_database, operation):
    engine = migration_database
    target = (
        HEAD if engine.dialect.name == "postgresql" else ScriptDirectory(str(MIGRATIONS)).get_base()
    )
    async with engine.begin() as conn:
        await conn.run_sync(lambda c: stamp(c, DEDICATED, []))
    url = engine.url.render_as_string(hide_password=False)
    await asyncio.to_thread(alembic, url, operation, target)
    async with engine.connect() as conn:
        assert await conn.scalar(text(f"SELECT version_num FROM {DEDICATED}")) == target


async def test_061_downgrade_base_can_upgrade_again(migration_database, tmp_path):
    # The packaged initial revision deliberately refuses destructive downgrade-to-base.
    # A reversible graph proves the production environment handles Alembic's empty stamp.
    directory = tmp_path / "reversible"
    (directory / "versions").mkdir(parents=True)
    (directory / "env.py").write_text((MIGRATIONS / "env.py").read_text())
    (directory / "versions" / "first.py").write_text(
        "from alembic import op\nimport sqlalchemy as sa\n"
        "revision='test_reversible'\ndown_revision=None\n"
        "def upgrade():\n    op.create_table('agent_runs', sa.Column('id', sa.Integer))\n"
        "def downgrade():\n    op.drop_table('agent_runs')\n"
    )
    engine = migration_database
    cfg = Config()
    cfg.set_main_option("script_location", str(directory))
    cfg.set_main_option(
        "sqlalchemy.url", engine.url.render_as_string(hide_password=False).replace("%", "%%")
    )
    for operation, revision in (
        (command.upgrade, "head"),
        (command.downgrade, "base"),
        (command.upgrade, "head"),
    ):
        await asyncio.to_thread(operation, cfg, revision)
        async with engine.connect() as conn:
            assert await conn.run_sync(lambda c: inspect(c).has_table(DEDICATED))
            assert await conn.scalar(text(f"SELECT version_num FROM {DEDICATED}")) == (
                None if revision == "base" else "test_reversible"
            )
