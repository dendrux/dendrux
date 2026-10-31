"""T-BOOT: ``dendrux db migrate`` on MySQL 8.4 / MariaDB 10.11 (fresh baseline, D2).

An empty database is bootstrapped: the settings table and a ``bootstrap_in_progress``
marker first, then every table, the generated tenant scope and every index in a fixed
order (an existing object must match exactly), then the seed, the Alembic stamp and the
marker's removal. A crash before any step resumes on the next run. Anything that is not
empty, a bootstrap in progress or a known Dendrux revision is refused. Maintenance holds a
``GET_LOCK`` on one dedicated connection and always releases it.
"""

from __future__ import annotations

from typing import Any

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool
from typer.testing import CliRunner

from dendrux.cli.main import app
from dendrux.db._dialects import mysql
from dendrux.db.models import Base

from .conftest import run_sync, sessions_left

HEAD = "f9a0b1c2d3e4"


APP_TABLE = (
    "CREATE TABLE customers (id INT PRIMARY KEY, name VARCHAR(20), UNIQUE KEY uq_name (name))"
)


class CrashError(Exception):
    """A simulated process crash before a bootstrap step."""


def migrate(url: str, monkeypatch: pytest.MonkeyPatch, *args: str) -> Any:
    monkeypatch.setenv("DENDRUX_DATABASE_URL", url)
    return CliRunner().invoke(app, ["db", *(args or ("migrate",))])


def sql(url: str, *statements: str) -> list[Any]:
    """Run statements in one connection; return the last statement's rows (if any)."""

    async def go() -> list[Any]:
        engine = create_async_engine(url, poolclass=NullPool)
        try:
            async with engine.begin() as conn:
                result = None
                for statement in statements:
                    result = await conn.execute(text(statement))
                return list(result.all()) if result is not None and result.returns_rows else []
        finally:
            await engine.dispose()

    return list(run_sync(go()))


def settings(url: str) -> dict[str, str]:
    return dict(sql(url, "SELECT `key`, value FROM dendrux_settings"))


def tables(url: str) -> set[str]:
    rows = sql(
        url,
        "SELECT table_name FROM information_schema.tables WHERE table_schema = DATABASE()",
    )
    return {row[0] for row in rows}


def problems(url: str) -> list[str]:
    from dendrux.db._schema_check import schema_problems

    async def go() -> list[str]:
        engine = create_async_engine(url, poolclass=NullPool)
        try:
            async with engine.connect() as conn:
                return list(await conn.run_sync(schema_problems))
        finally:
            await engine.dispose()

    return list(run_sync(go()))


def lock_is_free(url: str) -> bool:
    """The maintenance lock is free and no maintenance session was left behind."""
    name = sql(url, "SELECT CONCAT('dendrux_maint_', SHA1(DATABASE()))")[0][0]
    free = sql(url, f"SELECT IS_FREE_LOCK('{name}')")[0][0] == 1
    return free and sessions_left(url) == 0


def assert_at_head(url: str) -> None:
    assert sql(url, "SELECT version_num FROM alembic_version") == [(HEAD,)]
    assert settings(url) == {"storage_format": "2", "activate_step": "done"}
    assert problems(url) == []
    assert lock_is_free(url)


def test_fresh_database_bootstraps_to_head_and_reruns_are_noops(
    mysql_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = migrate(mysql_url, monkeypatch)
    assert result.exit_code == 0, result.output
    assert_at_head(mysql_url)
    assert tables(mysql_url) == set(Base.metadata.tables) | {"alembic_version"}
    again = migrate(mysql_url, monkeypatch)
    assert again.exit_code == 0, again.output
    assert_at_head(mysql_url)


def test_crash_before_every_step_resumes_to_the_same_schema(
    mysql_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    crashed: list[str] = []

    def crash_once(step: str) -> None:
        if step not in crashed:
            crashed.append(step)
            raise CrashError(step)

    monkeypatch.setattr(mysql, "_before_step", crash_once)
    for _ in range(200):
        result = migrate(mysql_url, monkeypatch)
        assert lock_is_free(mysql_url), crashed[-1]
        if result.exit_code == 0:
            break
        assert isinstance(result.exception, SystemExit) or "Migration failed" in result.output
    else:
        pytest.fail("bootstrap never completed")
    monkeypatch.setattr(mysql, "_before_step", None)
    steps = mysql.bootstrap_steps(_dialect(mysql_url))
    assert crashed == steps
    assert steps[:2] == ["create:dendrux_settings", "marker"]
    assert steps[-3:] == ["seed", "stamp", "clear_marker"]
    assert_at_head(mysql_url)


def _dialect(url: str) -> Any:
    async def go() -> Any:
        engine = create_async_engine(url, poolclass=NullPool)
        try:
            async with engine.connect():
                return engine.dialect
        finally:
            await engine.dispose()

    return run_sync(go())


def test_crash_between_settings_table_and_marker_resumes(
    mysql_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(mysql, "_before_step", _crash_at("marker"))
    assert migrate(mysql_url, monkeypatch).exit_code == 1
    assert tables(mysql_url) == {"dendrux_settings"}
    assert settings(mysql_url) == {}
    monkeypatch.setattr(mysql, "_before_step", None)
    result = migrate(mysql_url, monkeypatch)
    assert result.exit_code == 0, result.output
    assert_at_head(mysql_url)


def test_crash_before_stamp_resumes(mysql_url: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mysql, "_before_step", _crash_at("stamp"))
    assert migrate(mysql_url, monkeypatch).exit_code == 1
    assert "alembic_version" not in tables(mysql_url)
    assert settings(mysql_url)["bootstrap_in_progress"] == HEAD
    status = migrate(mysql_url, monkeypatch, "status")
    assert "bootstrap_in_progress=" + HEAD in status.output
    assert migrate(mysql_url, monkeypatch, "check", "--allow-unstamped").exit_code == 1
    monkeypatch.setattr(mysql, "_before_step", None)
    result = migrate(mysql_url, monkeypatch)
    assert result.exit_code == 0, result.output
    assert_at_head(mysql_url)


def _crash_at(name: str):  # noqa: ANN202
    def hook(step: str) -> None:
        if step == name:
            raise CrashError(step)

    return hook


@pytest.mark.parametrize(
    ("setup", "message"),
    [
        # A Dendrux-named table in a shape this version did not create, no marker.
        (
            [APP_TABLE, "CREATE TABLE agent_runs (id INT PRIMARY KEY)"],
            "unrecognised",
        ),
        (
            [
                APP_TABLE,
                "CREATE TABLE dendrux_settings (`key` VARCHAR(64) PRIMARY KEY, "
                "value VARCHAR(255) NOT NULL, updated_at DATETIME(6) NOT NULL)",
                "INSERT INTO dendrux_settings VALUES ('storage_format', '2', UTC_TIMESTAMP(6))",
            ],
            "unrecognised",
        ),
        (
            [
                "CREATE TABLE alembic_version (version_num VARCHAR(32) PRIMARY KEY)",
                "INSERT INTO alembic_version VALUES ('0123456789ab')",
            ],
            "0123456789ab",
        ),
        (
            [
                "CREATE TABLE alembic_version (version_num VARCHAR(32) PRIMARY KEY)",
                "INSERT INTO alembic_version VALUES ('2ce3bb57b0c5')",
            ],
            "2ce3bb57b0c5",
        ),
    ],
)
def test_foreign_or_unknown_schemas_are_refused(
    mysql_url: str, monkeypatch: pytest.MonkeyPatch, setup: list[str], message: str
) -> None:
    sql(mysql_url, *setup)
    before = tables(mysql_url)
    result = migrate(mysql_url, monkeypatch)
    assert result.exit_code == 1
    assert message in result.output
    assert tables(mysql_url) == before
    assert lock_is_free(mysql_url)


def test_existing_object_that_differs_is_refused_on_resume(
    mysql_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(mysql, "_before_step", _crash_at("create:agent_runs"))
    assert migrate(mysql_url, monkeypatch).exit_code == 1
    # Someone creates agent_runs by hand, with the padding collation on its key.
    sql(
        mysql_url,
        "CREATE TABLE agent_runs (id VARCHAR(26) COLLATE utf8mb4_bin PRIMARY KEY) ENGINE=InnoDB",
    )
    monkeypatch.setattr(mysql, "_before_step", None)
    result = migrate(mysql_url, monkeypatch)
    assert result.exit_code == 1
    assert "agent_runs" in result.output
    assert "alembic_version" not in tables(mysql_url)
    assert lock_is_free(mysql_url)


def test_a_marker_from_another_revision_is_refused(
    mysql_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(mysql, "_before_step", _crash_at("create:agent_runs"))
    assert migrate(mysql_url, monkeypatch).exit_code == 1
    sql(
        mysql_url,
        "UPDATE dendrux_settings SET value = 'aaaaaaaaaaaa' WHERE `key` = 'bootstrap_in_progress'",
    )
    monkeypatch.setattr(mysql, "_before_step", None)
    result = migrate(mysql_url, monkeypatch)
    assert result.exit_code == 1
    assert "aaaaaaaaaaaa" in result.output


def test_a_held_maintenance_lock_refuses_and_the_holder_is_unaffected(
    mysql_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def hold_and_migrate() -> tuple[int, str]:
        engine = create_async_engine(mysql_url, poolclass=NullPool)
        try:
            async with engine.connect() as holder:
                got = (
                    await holder.execute(
                        text("SELECT GET_LOCK(CONCAT('dendrux_maint_', SHA1(DATABASE())), 0)")
                    )
                ).scalar_one()
                assert got == 1
                import asyncio

                result = await asyncio.to_thread(migrate, mysql_url, monkeypatch)
                return result.exit_code, result.output
        finally:
            await engine.dispose()

    code, output = run_sync(hold_and_migrate())
    assert code == 1
    assert "Another Dendrux database migration" in output
    assert tables(mysql_url) == set()
    assert lock_is_free(mysql_url)  # the holder's connection closed
    assert migrate(mysql_url, monkeypatch).exit_code == 0
    assert_at_head(mysql_url)


def test_lock_and_connection_are_dropped_when_maintenance_fails(mysql_url: str) -> None:
    """``activate_database`` on a pooled engine: an error invalidates the connection."""
    from dendrux.db._activate import activate_database

    async def go() -> tuple[set[int], set[int]]:
        engine = create_async_engine(mysql_url, pool_size=1, max_overflow=0)
        try:
            async with engine.connect() as conn:
                before = {(await conn.execute(text("SELECT CONNECTION_ID()"))).scalar_one()}
            with pytest.raises(RuntimeError, match="not current"):
                await activate_database(engine)  # empty database: not current
            async with engine.connect() as conn:
                after = {(await conn.execute(text("SELECT CONNECTION_ID()"))).scalar_one()}
            return before, after
        finally:
            await engine.dispose()

    before, after = run_sync(go())
    assert before.isdisjoint(after)  # the pool discarded the maintenance connection
    assert lock_is_free(mysql_url)


def test_cli_commands_on_a_bootstrapped_database(
    mysql_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert migrate(mysql_url, monkeypatch, "check").exit_code == 2  # unstamped, empty
    assert migrate(mysql_url, monkeypatch).exit_code == 0
    status = migrate(mysql_url, monkeypatch, "status")
    assert status.exit_code == 0
    assert f"revision={HEAD} storage_format=2 activate_step=done" in status.output
    assert "bootstrap" not in status.output
    check = migrate(mysql_url, monkeypatch, "check")
    assert check.exit_code == 0, check.output
    for command in ("activate", "dedupe"):
        result = migrate(mysql_url, monkeypatch, command)
        assert result.exit_code == 0, result.output
    assert_at_head(mysql_url)


def test_activation_refuses_a_mysql_schema_that_is_not_active(
    mysql_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert migrate(mysql_url, monkeypatch).exit_code == 0
    sql(mysql_url, "UPDATE dendrux_settings SET value = '1' WHERE `key` = 'storage_format'")
    result = migrate(mysql_url, monkeypatch, "activate")
    assert result.exit_code == 1
    assert "MySQL" in result.output
    assert lock_is_free(mysql_url)


def test_create_all_and_bootstrap_build_the_same_catalog(
    mysql_url: str, mysql_backend: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from .conftest import mysql_database

    assert migrate(mysql_url, monkeypatch).exit_code == 0
    other = mysql_database(mysql_backend)
    second = next(other)
    try:

        async def build() -> None:
            engine = create_async_engine(second, poolclass=NullPool)
            try:
                async with engine.begin() as conn:
                    await conn.run_sync(Base.metadata.create_all)
            finally:
                await engine.dispose()

        run_sync(build())
        catalog = (
            "SELECT table_name, column_name, column_type, is_nullable, column_default, "
            "collation_name, extra, generation_expression FROM information_schema.columns "
            "WHERE table_schema = DATABASE() AND table_name <> 'alembic_version' "
            "ORDER BY table_name, ordinal_position"
        )
        indexes = (
            "SELECT table_name, index_name, non_unique, seq_in_index, column_name, collation "
            "FROM information_schema.statistics WHERE table_schema = DATABASE() "
            "AND table_name <> 'alembic_version' ORDER BY table_name, index_name, seq_in_index"
        )
        assert sql(mysql_url, catalog) == sql(second, catalog)
        assert sql(mysql_url, indexes) == sql(second, indexes)
    finally:
        next(other, None)


def test_read_only_alembic_runs_and_other_targets_are_refused(
    mysql_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    import io

    from alembic import command

    from dendrux.cli.db import _build_alembic_config

    assert migrate(mysql_url, monkeypatch).exit_code == 0
    config = _build_alembic_config()
    config.stdout = io.StringIO()
    command.current(config)
    assert HEAD in config.stdout.getvalue()
    with pytest.raises(RuntimeError, match="only up to head"):
        command.downgrade(config, "-1")
    with pytest.raises(RuntimeError, match="live connection"):
        command.upgrade(config, "head", sql=True)
    assert_at_head(mysql_url)


# --- Shared database: the app's own tables live next to Dendrux's (owner, 2026-10-31) ---


def app_snapshot(url: str) -> tuple[Any, ...]:
    create = sql(url, "SHOW CREATE TABLE customers")[0][1]
    rows = sql(url, "SELECT id, name FROM customers ORDER BY id")
    return create, tuple(tuple(r) for r in rows)


def with_app_table(url: str) -> tuple[Any, ...]:
    sql(url, APP_TABLE, "INSERT INTO customers VALUES (1, 'ada'), (2, 'grace')")
    return app_snapshot(url)


def test_shared_database_bootstraps_and_leaves_app_tables_untouched(
    mysql_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    before = with_app_table(mysql_url)
    result = migrate(mysql_url, monkeypatch)
    assert result.exit_code == 0, result.output
    assert_at_head(mysql_url)
    assert tables(mysql_url) == set(Base.metadata.tables) | {"alembic_version", "customers"}
    assert app_snapshot(mysql_url) == before
    assert migrate(mysql_url, monkeypatch, "check").exit_code == 0
    assert migrate(mysql_url, monkeypatch).exit_code == 0  # rerun: still a no-op
    assert app_snapshot(mysql_url) == before


@pytest.mark.parametrize("cut", ["marker", "create:run_events", "stamp", "clear_marker"])
def test_shared_database_crash_recovery(
    mysql_url: str, monkeypatch: pytest.MonkeyPatch, cut: str
) -> None:
    before = with_app_table(mysql_url)
    monkeypatch.setattr(mysql, "_before_step", _crash_at(cut))
    assert migrate(mysql_url, monkeypatch).exit_code == 1
    if cut == "marker":  # only the empty settings table exists next to the app's table
        assert tables(mysql_url) == {"dendrux_settings", "customers"}
    monkeypatch.setattr(mysql, "_before_step", None)
    result = migrate(mysql_url, monkeypatch)
    assert result.exit_code == 0, result.output
    assert_at_head(mysql_url)
    assert app_snapshot(mysql_url) == before


def test_an_unrecognised_dendrux_table_is_refused_and_nothing_is_touched(
    mysql_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    before = with_app_table(mysql_url)
    sql(mysql_url, "CREATE TABLE run_events (id INT PRIMARY KEY)")
    result = migrate(mysql_url, monkeypatch)
    assert result.exit_code == 1
    assert "run_events" in result.output
    assert tables(mysql_url) == {"customers", "run_events"}
    assert app_snapshot(mysql_url) == before
    assert lock_is_free(mysql_url)


# --- Maintenance lock: a fault right after GET_LOCK succeeds ---


def test_a_fault_after_acquiring_the_lock_releases_it(mysql_url: str) -> None:
    from sqlalchemy import event

    from dendrux.db._activate import activate_database

    async def go() -> tuple[set[int], set[int]]:
        engine = create_async_engine(mysql_url, pool_size=1, max_overflow=0)
        state = {"locked": False, "fired": False}

        def after_execute(conn, cursor, statement, *args):  # noqa: ANN001, ANN002
            if "GET_LOCK" in statement:
                state["locked"] = True

        def on_commit(conn) -> None:  # noqa: ANN001
            if state["locked"] and not state["fired"]:
                state["fired"] = True
                raise RuntimeError("injected commit fault after GET_LOCK")

        event.listen(engine.sync_engine, "after_cursor_execute", after_execute)
        event.listen(engine.sync_engine, "commit", on_commit)
        try:
            async with engine.connect() as conn:
                before = {(await conn.execute(text("SELECT CONNECTION_ID()"))).scalar_one()}
            with pytest.raises(RuntimeError, match="injected commit fault"):
                await activate_database(engine)
            assert state["fired"]
            async with engine.connect() as conn:
                after = {(await conn.execute(text("SELECT CONNECTION_ID()"))).scalar_one()}
            return before, after
        finally:
            await engine.dispose()

    before, after = run_sync(go())
    assert lock_is_free(mysql_url)
    assert before.isdisjoint(after)


# --- Upgrades after the baseline: a throwaway later revision ---


LATER = "aa0000000001"
LATER_REVISION = f'''"""Throwaway revision after the MySQL baseline (test only)."""
from alembic import op

revision = "{LATER}"
down_revision = "{HEAD}"
branch_labels = None
depends_on = None


def upgrade():
    op.execute("CREATE TABLE dendrux_probe_later (id INT PRIMARY KEY)")


def downgrade():
    op.execute("DROP TABLE dendrux_probe_later")
'''


def test_a_database_at_the_baseline_upgrades_to_a_later_revision(
    mysql_url: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    import io
    import shutil
    from pathlib import Path

    from alembic import command

    from dendrux.cli.db import _build_alembic_config
    from dendrux.db import migrations

    assert migrate(mysql_url, monkeypatch).exit_code == 0  # bootstrapped at the baseline
    scripts = tmp_path / "migrations"
    shutil.copytree(Path(migrations.__file__).parent, scripts)
    (scripts / "versions" / "zz_later.py").write_text(LATER_REVISION)
    config = _build_alembic_config()
    config.set_main_option("script_location", str(scripts))
    config.stdout = io.StringIO()
    command.upgrade(config, "head")
    assert sql(mysql_url, "SELECT version_num FROM alembic_version") == [(LATER,)]
    assert "dendrux_probe_later" in tables(mysql_url)
    assert lock_is_free(mysql_url)


def test_upgradeable_revisions_include_the_baseline() -> None:
    from alembic.script import ScriptDirectory

    from dendrux.cli.db import _build_alembic_config

    script = ScriptDirectory.from_config(_build_alembic_config())
    head = script.get_current_head()
    assert head == mysql.MYSQL_BASELINE_REVISION
    assert mysql.upgradeable_revisions(script, head) == set()


def test_every_bootstrap_ddl_step_is_a_registered_log_step() -> None:
    from sqlalchemy.dialects import mysql as mysql_dialect

    from dendrux import _codes

    steps = [mysql.log_step(s) for s in mysql.bootstrap_steps(mysql_dialect.dialect())]
    ddl = [s for s in steps if s is not None]
    assert len(ddl) == 34  # 10 tables, the generated column, 22 indexes, the stamp table
    assert all(_codes._valid_field("step", s) for s in ddl), [
        s for s in ddl if not _codes._valid_field("step", s)
    ]
