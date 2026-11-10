"""Explicit Alembic stamp is an operator repair, never implicit adoption."""

import asyncio

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import text

from dendrux.db._version_table import MigrationVersionTableConflictError
from dendrux.db.models import Base
from tests.integration.test_061_migrations import migration_database as migration_database
from tests.integration.test_061_migrations_repair import invoke
from tests.migrations.test_060_activate import HEAD, MIGRATIONS, PRIOR, alembic
from tests.unit.test_061_migrations import DEDICATED, LEGACY, snapshot, stamp


@pytest.mark.parametrize("app_revisions", [None, [], ["app_one", "app_two"]])
async def test_061_explicit_stamp_repairs_unstamped_schema(migration_database, app_revisions):
    engine = migration_database
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        if app_revisions is not None:
            await conn.run_sync(lambda c: stamp(c, LEGACY, app_revisions))
        before = await conn.run_sync(snapshot)
    await asyncio.to_thread(
        alembic, engine.url.render_as_string(hide_password=False), "stamp", HEAD
    )
    async with engine.connect() as conn:
        after = await conn.run_sync(snapshot)
    assert after.pop(DEDICATED) == [(HEAD,)]
    assert after == before
    for action in ("migrate", "check", "status"):
        result = await invoke(action)
        assert result.exit_code == 0, result.output


@pytest.mark.parametrize(
    "dedicated,legacy",
    [
        (None, [PRIOR]),
        (None, [PRIOR, "app_one"]),
        ([HEAD], [PRIOR]),
        (["unknown_revision"], None),
        ([HEAD, PRIOR], None),
    ],
)
@pytest.mark.parametrize("purge", [False, True])
async def test_061_explicit_stamp_refuses_conflicting_history(
    migration_database, dedicated, legacy, purge
):
    engine = migration_database
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        if dedicated is not None:
            await conn.run_sync(lambda c: stamp(c, DEDICATED, dedicated))
        if legacy is not None:
            await conn.run_sync(lambda c: stamp(c, LEGACY, legacy))
        before = await conn.run_sync(snapshot)
    cfg = Config()
    cfg.set_main_option("script_location", str(MIGRATIONS))
    cfg.set_main_option(
        "sqlalchemy.url", engine.url.render_as_string(hide_password=False).replace("%", "%%")
    )
    with pytest.raises(MigrationVersionTableConflictError):
        await asyncio.to_thread(command.stamp, cfg, HEAD, purge=purge)
    async with engine.connect() as conn:
        assert await conn.run_sync(snapshot) == before


async def test_061_explicit_stamp_updates_valid_dedicated_history(migration_database):
    engine = migration_database
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.run_sync(lambda c: stamp(c, DEDICATED, [PRIOR]))
    await asyncio.to_thread(
        alembic, engine.url.render_as_string(hide_password=False), "stamp", HEAD
    )
    async with engine.connect() as conn:
        assert await conn.scalar(text(f"SELECT version_num FROM {DEDICATED}")) == HEAD
