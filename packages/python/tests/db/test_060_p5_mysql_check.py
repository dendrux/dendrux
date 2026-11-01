"""Schema and index checks on MySQL 8.4 / MariaDB 10.11 (runtime startup and the CLI).

The check compares the live catalog with the head definition: every model table and
column with its type, NO PAD collation, nullability and default; primary and foreign
keys; every named index (including the MySQL-only generated tenant scope and its unique
index); and the settings (active format, no unfinished bootstrap).
"""

from __future__ import annotations

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool

from dendrux.db._schema_check import ensure_schema_current, schema_problems
from dendrux.db.models import Base
from dendrux.errors import SchemaNotInitializedError, SchemaOutOfDateError


async def _create_all(url: str):  # noqa: ANN202
    engine = create_async_engine(url, poolclass=NullPool)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    return engine


async def _problems(engine) -> list[str]:  # noqa: ANN001
    async with engine.connect() as conn:
        return list(await conn.run_sync(schema_problems))


async def test_create_all_schema_passes_every_check(mysql_url: str) -> None:
    engine = await _create_all(mysql_url)
    try:
        assert await _problems(engine) == []
        await ensure_schema_current(engine)
    finally:
        await engine.dispose()


async def test_empty_database_is_not_initialized(mysql_url: str) -> None:
    engine = create_async_engine(mysql_url, poolclass=NullPool)
    try:
        with pytest.raises(SchemaNotInitializedError):
            await ensure_schema_current(engine)
    finally:
        await engine.dispose()


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("ALTER TABLE agent_runs MODIFY tenant_id VARCHAR(255) COLLATE utf8mb4_bin", "tenant_id"),
        ("ALTER TABLE agent_runs DROP INDEX uq_agent_runs_tenant_idempotency_key", "uq_agent_runs"),
        (
            "ALTER TABLE agent_runs DROP INDEX uq_agent_runs_tenant_idempotency_key, "
            "DROP COLUMN tenant_scope",
            "tenant_scope",
        ),
        ("ALTER TABLE react_traces MODIFY content TEXT NOT NULL", "content"),
        ("ALTER TABLE run_events DROP INDEX ix_run_events_started", "ix_run_events_started"),
        (
            "ALTER TABLE run_events DROP INDEX ix_run_events_started, "
            "ADD INDEX ix_run_events_started (agent_run_id)",
            "ix_run_events_started",
        ),
        ("ALTER TABLE agent_runs MODIFY created_at DATETIME NOT NULL", "created_at"),
        ("ALTER TABLE dendrux_settings ENGINE=MyISAM", "dendrux_settings (engine"),
        (
            "INSERT INTO dendrux_settings (`key`, value, updated_at) "
            "VALUES ('bootstrap_in_progress', 'x', UTC_TIMESTAMP(6))",
            "bootstrap",
        ),
        ("UPDATE dendrux_settings SET value = '1' WHERE `key` = 'storage_format'", "format"),
        ("CREATE UNIQUE INDEX uq_extra ON run_inbox (text(10))", "uq_extra"),
    ],
)
async def test_any_drift_from_the_head_definition_is_reported(
    mysql_url: str, mutation: str, expected: str
) -> None:
    engine = await _create_all(mysql_url)
    try:
        async with engine.begin() as conn:
            await conn.execute(text(mutation))
        problems = await _problems(engine)
        assert any(expected in problem for problem in problems), problems
        with pytest.raises(SchemaOutOfDateError):
            await ensure_schema_current(engine)
    finally:
        await engine.dispose()


async def test_index_helpers_read_mysql_indexes(mysql_url: str) -> None:
    from dendrux.db._index_check import index_exists, index_matches

    engine = await _create_all(mysql_url)
    try:
        async with engine.connect() as conn:

            def check(sync_conn):  # noqa: ANN001, ANN202
                return (
                    index_exists(sync_conn, "uq_run_events_run_seq"),
                    index_exists(sync_conn, "ix_missing"),
                    index_matches(
                        sync_conn,
                        "uq_run_events_run_seq",
                        "run_events",
                        "CREATE UNIQUE INDEX uq_run_events_run_seq "
                        "ON run_events (agent_run_id, sequence_index)",
                    ),
                    index_matches(
                        sync_conn,
                        "uq_run_events_run_seq",
                        "run_events",
                        "CREATE INDEX uq_run_events_run_seq ON run_events (agent_run_id)",
                    ),
                    index_matches(
                        sync_conn,
                        "uq_run_events_run_seq",
                        "react_traces",
                        "CREATE UNIQUE INDEX uq_run_events_run_seq "
                        "ON run_events (agent_run_id, sequence_index)",
                    ),
                )

            assert await conn.run_sync(check) == (True, False, True, False, False)
    finally:
        await engine.dispose()


async def test_bounded_ddl_gives_up_on_a_held_metadata_lock(
    mysql_url: str, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    import asyncio
    import logging

    from dendrux.db import _migration_ddl

    monkeypatch.setattr(_migration_ddl, "LOCK_TIMEOUT_SECONDS", 1)
    engine = await _create_all(mysql_url)
    try:
        async with engine.connect() as holder, engine.connect() as runner:
            await holder.execute(text("SELECT COUNT(*) FROM run_inbox"))  # holds the MDL
            before = (await runner.execute(text("SELECT @@SESSION.lock_wait_timeout"))).scalar()

            def alter(sync_conn) -> None:  # noqa: ANN001
                _migration_ddl.bounded_step(
                    sync_conn,
                    "ALTER TABLE run_inbox ADD COLUMN extra INT",
                    revision="f9a0b1c2d3e4",
                    step="create.unregistered_step_name",
                )

            with (
                caplog.at_level(logging.WARNING, logger="dendrux.db._migration_ddl"),
                pytest.raises(RuntimeError, match="Timed out acquiring"),
            ):
                await asyncio.wait_for(runner.run_sync(alter), timeout=20)
            after = (await runner.execute(text("SELECT @@SESSION.lock_wait_timeout"))).scalar()
            await holder.rollback()
        assert before == after  # the session's own wait is restored
        records = [
            r for r in caplog.records if getattr(r, "code", None) == "migration_lock_timeout"
        ]
        assert len(records) == 1
        assert records[0].revision == "f9a0b1c2d3e4"
        assert not hasattr(records[0], "step")  # unregistered values are left out
    finally:
        await engine.dispose()


async def test_settings_upsert_and_insert_if_absent(mysql_url: str) -> None:
    from dendrux.db._settings import put_setting

    engine = await _create_all(mysql_url)
    try:
        async with engine.begin() as conn:

            def write(sync_conn) -> None:  # noqa: ANN001
                put_setting(sync_conn, "probe", "1")
                put_setting(sync_conn, "probe", "2")  # overwrite
                put_setting(sync_conn, "probe", "3", overwrite=False)  # kept
                put_setting(sync_conn, "storage_format", "1", overwrite=False)  # kept

            await conn.run_sync(write)
        async with engine.connect() as conn:
            rows = dict(
                (await conn.execute(text("SELECT `key`, value FROM dendrux_settings"))).all()
            )
        assert rows["probe"] == "2"
        assert rows["storage_format"] == "2"
    finally:
        await engine.dispose()


@pytest.mark.parametrize(
    ("index", "table", "column"),
    [
        # Sorted before the real owner: the real index would overwrite it in a name-keyed map.
        ("uq_run_events_run_seq", "react_traces", "role"),
        # Sorted after the real owner.
        ("ix_run_events_event_type", "run_inbox", "run_id"),
    ],
)
async def test_same_named_unique_index_on_another_table_is_reported(
    mysql_url: str, index: str, table: str, column: str
) -> None:
    """MySQL index names are per table: one table's index cannot stand in for another's."""
    from dendrux.db._dialects import mysql

    engine = await _create_all(mysql_url)
    try:
        async with engine.begin() as conn:
            await conn.execute(text(f"CREATE UNIQUE INDEX {index} ON {table} ({column}, id)"))
        problems = await _problems(engine)
        assert f"{table}.{index} (unexpected unique index)" in problems, problems
        assert not any(p.startswith(f"run_events.{index}") for p in problems), problems
        async with engine.connect() as conn:
            valid, existing = await conn.run_sync(mysql.index_names)
        assert index in valid  # the real run_events index still matches
    finally:
        await engine.dispose()
