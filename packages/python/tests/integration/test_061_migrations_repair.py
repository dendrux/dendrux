"""The frozen manual repair recipes executed against SQLite and Postgres."""

import asyncio

import pytest
from alembic.runtime.migration import MigrationContext
from sqlalchemy import text
from typer.testing import CliRunner

from dendrux.cli.main import app
from dendrux.db.models import Base
from tests.integration.test_061_migrations import migration_database as migration_database
from tests.migrations.test_060_activate import HEAD, PRIOR, prior_schema
from tests.unit.test_061_migrations import DEDICATED, LEGACY, snapshot, stamp


async def invoke(*args):
    return await asyncio.to_thread(CliRunner().invoke, app, ["db", *args])


@pytest.mark.parametrize("mixed", [False, True])
async def test_061_manual_repair_recipe(migration_database, mixed):
    engine = migration_database
    async with engine.begin() as conn:
        if mixed:
            await conn.run_sync(prior_schema)
            await conn.execute(text(f"INSERT INTO {LEGACY} VALUES ('app_one')"))
        else:
            await conn.run_sync(Base.metadata.create_all)
            await conn.run_sync(lambda c: stamp(c, LEGACY, ["app_one"]))
        before = await conn.run_sync(snapshot)
    check = await invoke("check", "--allow-unstamped")
    assert check.exit_code == (2 if mixed else 0), check.output
    assert (await invoke("migrate")).exit_code == 1
    async with engine.connect() as conn:
        assert await conn.run_sync(snapshot) == before
    # The non-mixed schema was created with this installed version's model definition.
    # The mixed row has the exact known revision created by the prior-schema fixture.
    async with engine.begin() as conn:
        await conn.execute(
            text(f"CREATE TABLE {DEDICATED} (version_num VARCHAR(32) NOT NULL PRIMARY KEY)")
        )
        await conn.execute(
            text(f"INSERT INTO {DEDICATED} VALUES (:revision)"),
            {"revision": PRIOR if mixed else HEAD},
        )
        if mixed:
            removed = await conn.execute(
                text(f"DELETE FROM {LEGACY} WHERE version_num = :verified_dendrux_revision"),
                {"verified_dendrux_revision": PRIOR},
            )
            assert removed.rowcount == 1
    for action in ("migrate", "check", "status"):
        result = await invoke(action)
        assert result.exit_code == 0, result.output
    async with engine.connect() as conn:
        assert await conn.scalar(text(f"SELECT version_num FROM {DEDICATED}")) == HEAD
        assert await conn.run_sync(lambda c: MigrationContext.configure(c).get_current_heads()) == (
            "app_one",
        )
        assert await conn.scalar(
            text("SELECT value FROM dendrux_settings WHERE key='storage_format'")
        ) == ("1" if mixed else "2")


@pytest.mark.parametrize("action", ["check", "status", "activate"])
async def test_061_conflicting_histories_never_mutate_in_readers(migration_database, action):
    engine = migration_database
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.run_sync(lambda c: stamp(c, DEDICATED, [HEAD]))
        await conn.run_sync(lambda c: stamp(c, LEGACY, [HEAD]))
        before = await conn.run_sync(snapshot)
    result = await invoke(action)
    assert result.exit_code == (1 if action == "activate" else 2), result.output
    assert "migration_version_table_conflict" in result.output
    async with engine.connect() as conn:
        assert await conn.run_sync(snapshot) == before
