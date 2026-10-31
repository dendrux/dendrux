"""0.6 rollout barrier, activation and migration rehearsals (B1–B5)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, text
from typer.testing import CliRunner

from dendrux.cli.main import app
from dendrux.db.models import Base

HEAD = "f9a0b1c2d3e4"
PRIOR = "e8f9a0b1c2d3"
MIGRATIONS = Path(__file__).resolve().parents[2] / "src/dendrux/db/migrations"
NEW_COLUMNS = {
    "agent_runs": (
        "storage_format",
        "reclaim_attempts",
        "reclaim_limit",
        "lease_expired_at",
        "last_reclaimed_at",
        "loop_state",
        "trace_link",
    ),
    "tool_calls": ("outcome", "idempotent", "replayed_at", "content_digest"),
    "react_traces": ("trace_kind", "content_digest"),
    "run_events": ("content_digest",),
}
INDEXES = {
    "uq_react_traces_run_order": (
        "react_traces",
        "order_index",
        "ix_react_traces_agent_run_id_order",
    ),
    "uq_tool_calls_run_call": ("tool_calls", "tool_call_id", None),
    "uq_run_events_run_seq": ("run_events", "sequence_index", "ix_run_events_agent_run_id_seq"),
}
runner = CliRunner()


def alembic(url, operation, revision):
    cfg = Config()
    cfg.set_main_option("script_location", str(MIGRATIONS))
    cfg.set_main_option("sqlalchemy.url", url.replace("%", "%%"))
    getattr(command, operation)(cfg, revision)


def prior_schema(conn):
    """Build the released model schema independently of the current model metadata."""
    from tests.migrations._models_v050 import Base as LegacyBase

    LegacyBase.metadata.create_all(conn)
    conn.execute(text("CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL)"))
    conn.execute(text("INSERT INTO alembic_version VALUES (:revision)"), {"revision": PRIOR})


def test_frozen_v050_models_checksum():
    import hashlib

    fixture = Path(__file__).with_name("_models_v050.py")
    assert (
        hashlib.sha256(fixture.read_bytes()).hexdigest()
        == "8649e1452911d0cf8ab73329eccd546025b60ff7d43e97023a7f56d73b970901"
    )


def test_legacy_schema_does_not_require_git(tmp_path, monkeypatch):
    import subprocess

    def no_git(*args, **kwargs):
        pytest.fail("legacy schema fixtures must work in a shallow, untagged checkout")

    monkeypatch.setattr(subprocess, "run", no_git)
    engine = create_engine(f"sqlite:///{tmp_path / 'no-tags.db'}")
    try:
        with engine.begin() as conn:
            prior_schema(conn)
            assert "dendrux_settings" not in inspect(conn).get_table_names()
            assert "storage_format" not in {
                column["name"] for column in inspect(conn).get_columns("agent_runs")
            }
    finally:
        engine.dispose()


@pytest.fixture
def legacy(tmp_path, monkeypatch):
    path = tmp_path / "legacy.db"
    engine = create_engine(f"sqlite:///{path}")
    with engine.begin() as conn:
        prior_schema(conn)
        conn.execute(
            text("""
            INSERT INTO agent_runs (id, agent_name, status, iteration_count, delegation_level,
                total_input_tokens, total_output_tokens, total_cache_read_tokens,
                total_cache_creation_tokens, created_at, updated_at)
            VALUES ('00000000000000000000999999', 'a', 'success', 0, 0, 0, 0, 0, 0,
                CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        """)
        )
    url = f"sqlite+aiosqlite:///{path}"
    monkeypatch.setenv("DENDRUX_DATABASE_URL", url)
    yield engine, url
    engine.dispose()


def add_trace(conn, id, content="same", digest=None, order=0, run_id="00000000000000000000999999"):
    conn.execute(
        text("""
        INSERT INTO react_traces
            (id, agent_run_id, role, content, order_index, content_digest, created_at, updated_at)
        VALUES (:id, :run, 'assistant', :content, :order, :digest,
                CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
    """),
        {"id": id, "run": run_id, "content": content, "order": order, "digest": digest},
    )


def legacy_trace(conn, id, *, run_id="00000000000000000000000000", order=0, content="same"):
    conn.execute(
        text("""
        INSERT INTO react_traces (id, agent_run_id, role, content, order_index,
            created_at, updated_at)
        VALUES (:id, :run, 'assistant', :content, :order, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
    """),
        {"id": id, "run": run_id, "order": order, "content": content},
    )


def migrate_legacy(legacy):
    engine, url = legacy
    alembic(url, "upgrade", "head")
    with engine.connect() as conn:
        columns = {c["name"] for c in inspect(conn).get_columns("react_traces")}
    assert "content_digest" in columns, "the additive migration must add digest columns"
    return engine


def test_additive_migration_and_downgrade_barrier(legacy):
    engine = migrate_legacy(legacy)
    with engine.begin() as conn:
        assert dict(conn.execute(text("SELECT key, value FROM dendrux_settings")).all()) == {
            "storage_format": "1"
        }
        assert not any(i["unique"] for i in inspect(conn).get_indexes("react_traces"))
        for table, columns in NEW_COLUMNS.items():
            assert set(columns) <= {c["name"] for c in inspect(conn).get_columns(table)}
        add_trace(conn, "a")
        add_trace(conn, "b")  # a 0.5 writer's lost-ack retry remains legal
    assert runner.invoke(app, ["db", "check"]).exit_code == 0
    alembic(legacy[1], "downgrade", PRIOR)
    with engine.connect() as conn:
        assert "storage_format" not in {c["name"] for c in inspect(conn).get_columns("agent_runs")}
    alembic(legacy[1], "upgrade", "head")
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO dendrux_settings VALUES "
                "('activate_step', 'dedupe:react_traces', CURRENT_TIMESTAMP)"
            )
        )
    with pytest.raises(RuntimeError, match="activation"):
        alembic(legacy[1], "downgrade", PRIOR)
    with engine.connect() as conn:
        assert "loop_state" in {c["name"] for c in inspect(conn).get_columns("agent_runs")}


def test_create_all_is_active(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'fresh.db'}")
    try:
        Base.metadata.create_all(engine)
        with engine.connect() as conn:
            assert "dendrux_settings" in inspect(conn).get_table_names()
            assert dict(conn.execute(text("SELECT key, value FROM dendrux_settings")).all()) == {
                "storage_format": "2",
                "activate_step": "done",
            }
            for name, (table, _, old) in INDEXES.items():
                indexes = {i["name"]: i for i in inspect(conn).get_indexes(table)}
                assert indexes[name]["unique"]
                assert old not in indexes
    finally:
        engine.dispose()


def test_b1_conflicts_drop_nothing_then_dedupe_and_activate(legacy):
    engine = migrate_legacy(legacy)
    with engine.begin() as conn:
        add_trace(conn, "a")
        add_trace(conn, "b")
        add_trace(conn, "c", content="confidential-conflict", order=1)
        add_trace(conn, "d", content="different-confidential", order=1)
    result = runner.invoke(app, ["db", "activate"])
    assert result.exit_code == 3, result.output
    assert "react_traces" in result.output and "00000000000000000000999999" in result.output
    assert "confidential" not in result.output
    with engine.begin() as conn:
        assert conn.execute(text("SELECT count(*) FROM react_traces")).scalar_one() == 4
        conn.execute(text("DELETE FROM react_traces WHERE id='d'"))
    assert runner.invoke(app, ["db", "check"]).exit_code == 0
    result = runner.invoke(app, ["db", "activate", "--batch-size", "1"])
    assert result.exit_code == 0, (result.output, result.exception)
    with engine.connect() as conn:
        assert conn.execute(text("SELECT id FROM react_traces ORDER BY id")).scalars().all() == [
            "a",
            "c",
        ]
        settings = dict(conn.execute(text("SELECT key, value FROM dendrux_settings")).all())
        assert settings["storage_format"] == "2" and settings["activate_step"] == "done"
    assert runner.invoke(app, ["db", "check"]).exit_code == 0
    assert runner.invoke(app, ["db", "activate"]).exit_code == 0


def test_b5_sqlite_live_writer_refuses(legacy):
    engine = migrate_legacy(legacy)
    with engine.connect() as writer:
        writer.exec_driver_sql("BEGIN IMMEDIATE")
        writer.exec_driver_sql("UPDATE agent_runs SET iteration_count=1")
        result = runner.invoke(app, ["db", "activate"])
        assert result.exit_code == 1, result.output
        assert "transaction" in result.output.lower()
        writer.rollback()
    assert runner.invoke(app, ["db", "activate"]).exit_code == 0


def test_cli_status_and_dedupe_only(legacy):
    engine = migrate_legacy(legacy)
    result = runner.invoke(app, ["db", "status"])
    assert result.exit_code == 0, result.output
    assert f"revision={HEAD}" in result.output
    assert "storage_format=1" in result.output and "activate_step=" in result.output
    with engine.begin() as conn:
        add_trace(conn, "a")
        add_trace(conn, "b")
    result = runner.invoke(app, ["db", "dedupe"])
    assert result.exit_code == 0, (result.output, result.exception)
    with engine.connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM react_traces")).scalar_one() == 1
        assert (
            conn.execute(
                text("SELECT value FROM dendrux_settings WHERE key='storage_format'")
            ).scalar_one()
            == "1"
        )
        assert not any(i["unique"] for i in inspect(conn).get_indexes("react_traces"))


def test_status_on_legacy_and_unstamped_database(legacy):
    engine, _ = legacy
    result = runner.invoke(app, ["db", "status"])
    assert result.exit_code == 0, result.output
    assert f"revision={PRIOR}" in result.output
    assert "storage_format=1 activate_step=not-started" in result.output
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE alembic_version"))
    result = runner.invoke(app, ["db", "status"])
    assert result.exit_code == 0, result.output
    assert "revision=unstamped storage_format=1 activate_step=not-started" in result.output


def test_migration_statements_are_bounded():
    files = list(MIGRATIONS.glob("versions/*060*"))
    assert files, "0.6 must ship an additive migration"
    spec = importlib.util.spec_from_file_location("migration060", files[0])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.revision == HEAD and module.down_revision == PRIOR


def test_b2_interruption_resumes_committed_cursor(legacy, monkeypatch):
    from dendrux.db import _activate

    engine = migrate_legacy(legacy)
    with engine.begin() as conn:
        for i in range(7):
            add_trace(conn, f"t{i}", order=i // 2)
    original = _activate._checkpoint
    interrupted = False

    def stop(conn, table, cursor):
        nonlocal interrupted
        original(conn, table, cursor)
        if table == "react_traces" and cursor == "t2":
            interrupted = True
            raise RuntimeError("simulated interruption")

    monkeypatch.setattr(_activate, "_checkpoint", stop)
    result = runner.invoke(app, ["db", "activate", "--batch-size", "3"])
    assert result.exit_code == 1 and interrupted, result.output
    with engine.connect() as conn:
        assert (
            conn.execute(
                text("SELECT value FROM dendrux_settings WHERE key='dedupe_cursor_react_traces'")
            ).scalar_one()
            == "t2"
        )
        assert conn.execute(text("SELECT count(*) FROM react_traces")).scalar_one() == 6
    assert runner.invoke(app, ["db", "check"]).exit_code == 0
    seen = []
    page = _activate._page

    def capture(conn, table, column, after, end, size):
        seen.append((table.name, after))
        return page(conn, table, column, after, end, size)

    monkeypatch.setattr(_activate, "_checkpoint", original)
    monkeypatch.setattr(_activate, "_page", capture)
    assert runner.invoke(app, ["db", "activate", "--batch-size", "3"]).exit_code == 0
    assert ("react_traces", "t2") in seen
    with engine.connect() as conn:
        assert conn.execute(text("SELECT id FROM react_traces ORDER BY id")).scalars().all() == [
            "t0",
            "t2",
            "t4",
            "t6",
        ]


@pytest.mark.parametrize("table", ["react_traces", "tool_calls", "run_events"])
@pytest.mark.parametrize(
    "digests", [(None, None), ("a" * 64, "a" * 64), (None, "a" * 64), ("a" * 64, "b" * 64)]
)
def test_b1_digest_and_legacy_content_equality(legacy, table, digests):
    import datetime as dt

    engine = migrate_legacy(legacy)
    base = {"agent_run_id": "00000000000000000000999999", "created_at": dt.datetime.now(dt.UTC)}
    if table == "react_traces":
        base.update(role="assistant", content="same", order_index=0)
    elif table == "tool_calls":
        base.update(
            tool_call_id="call",
            tool_name="t",
            target="server",
            params={"a": 1},
            result={"x": 2},
            success=True,
        )
    else:
        base.update(event_type="run.started", sequence_index=0, iteration_index=0, data={"a": 1})
    with engine.begin() as conn:
        for id, digest in zip(["a", "b"], digests, strict=True):
            conn.execute(
                Base.metadata.tables[table].insert(), {**base, "id": id, "content_digest": digest}
            )
    result = runner.invoke(app, ["db", "dedupe"])
    conflict = digests[0] != digests[1]
    assert result.exit_code == (3 if conflict else 0), result.output
    with engine.connect() as conn:
        assert conn.execute(text(f"SELECT id FROM {table} ORDER BY id")).scalars().all() == (
            ["a", "b"] if conflict else ["a"]
        )


def test_b1_bounded_diagnostic(legacy):
    engine = migrate_legacy(legacy)
    with engine.begin() as conn:
        for i in range(25):
            run_id = f"{i:026d}"
            conn.execute(
                text("""
                INSERT INTO agent_runs (id, agent_name, status, iteration_count, delegation_level,
                    total_input_tokens, total_output_tokens, total_cache_read_tokens,
                    total_cache_creation_tokens, created_at, updated_at)
                SELECT :id, agent_name, status, iteration_count, delegation_level,
                    total_input_tokens, total_output_tokens, total_cache_read_tokens,
                    total_cache_creation_tokens, created_at, updated_at
                FROM agent_runs WHERE id='00000000000000000000999999'
            """),
                {"id": run_id},
            )
            add_trace(conn, f"a{i:02}", content="secret-one", run_id=run_id)
            add_trace(conn, f"b{i:02}", content="secret-two", run_id=run_id)
    result = runner.invoke(app, ["db", "activate", "--batch-size", "3"])
    assert result.exit_code == 3, result.output
    assert "count=25" in result.output and "secret" not in result.output
    assert (
        "00000000000000000000000019" in result.output
        and "00000000000000000000000020" not in result.output
    )


def test_activation_refuses_incomplete_schema(legacy):
    engine = migrate_legacy(legacy)
    with engine.begin() as conn:
        conn.execute(text("DROP INDEX ix_run_events_agent_run_id_seq"))
    result = runner.invoke(app, ["db", "activate"])
    assert result.exit_code == 1 and "not current" in result.output
    with engine.connect() as conn:
        assert (
            conn.execute(
                text("SELECT count(*) FROM dendrux_settings WHERE key='activate_step'")
            ).scalar_one()
            == 0
        )


@pytest.fixture(params=["sqlite", "postgres"])
async def rollout_database(request, tmp_path, pg_database):
    from sqlalchemy.ext.asyncio import create_async_engine
    from sqlalchemy.pool import NullPool

    from tests.integration._migration_schema import migration_schema

    if request.param == "postgres":
        if pg_database is None:
            pytest.skip("DENDRUX_TEST_PG_URL not set")
        async with migration_schema(pg_database, "activation_060") as engine:
            yield engine
        return
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{tmp_path / 'rehearsal.db'}", poolclass=NullPool
    )
    try:
        yield engine
    finally:
        await engine.dispose()


async def _rollout_prior(engine):
    async with engine.begin() as conn:
        await conn.run_sync(prior_schema)


async def _migrate(engine):
    import asyncio

    await asyncio.to_thread(
        alembic, engine.url.render_as_string(hide_password=False), "upgrade", "head"
    )


async def _seed_rehearsal(engine):
    run_sql = text("""
        INSERT INTO agent_runs (id, agent_name, status, iteration_count, delegation_level,
            total_input_tokens, total_output_tokens, total_cache_read_tokens,
            total_cache_creation_tokens, created_at, updated_at)
        VALUES (:id, 'rehearsal', 'success', 0, 0, 0, 0, 0, 0, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
    """)
    event_sql = text("""
        INSERT INTO run_events (id, agent_run_id, event_type, sequence_index,
            iteration_index, created_at)
        VALUES (:id, :run, 'tool.completed', :seq, 0, CURRENT_TIMESTAMP)
    """)
    async with engine.begin() as conn:
        for start in range(0, 5000, 500):
            await conn.execute(run_sql, [{"id": f"{i:026d}"} for i in range(start, start + 500)])
        for start in range(0, 200000, 5000):
            await conn.execute(
                event_sql,
                [
                    {"id": f"event-{i:06}", "run": f"{i // 40:026d}", "seq": i % 40}
                    for i in range(start, start + 5000)
                ],
            )


async def test_b3_b4_rehearsal_5000_runs_200k_events(
    rollout_database, record_property, monkeypatch
):
    import asyncio
    import time

    from dendrux.db._activate import activate_database
    from dendrux.db._schema_check import ensure_schema_current, read_schema_phase, schema_problems

    engine = rollout_database
    await _rollout_prior(engine)
    await _seed_rehearsal(engine)
    async with engine.begin() as conn:
        await conn.run_sync(lambda c: legacy_trace(c, "identical-a"))
        await conn.run_sync(lambda c: legacy_trace(c, "identical-b"))
        await conn.run_sync(lambda c: legacy_trace(c, "conflict-a", order=1, content="private-a"))
        await conn.run_sync(lambda c: legacy_trace(c, "conflict-b", order=1, content="private-b"))

    files_before = None
    if engine.dialect.name == "postgresql":
        async with engine.connect() as conn:
            files_before = (
                await conn.execute(
                    text(
                        "SELECT relname, relfilenode FROM pg_class "
                        "WHERE relnamespace=current_schema()::regnamespace "
                        "AND relname IN ('agent_runs', 'tool_calls', 'react_traces', 'run_events') "
                        "ORDER BY relname"
                    )
                )
            ).all()
    start = time.perf_counter()
    await _migrate(engine)
    elapsed = time.perf_counter() - start
    record_property("migration_seconds", elapsed)
    print(f"rehearsal dialect={engine.dialect.name} migration_seconds={elapsed:.4f}")
    assert elapsed < 2, "additive migration must complete in under two seconds"
    if files_before is not None:
        async with engine.connect() as conn:
            files_after = (
                await conn.execute(
                    text(
                        "SELECT relname, relfilenode FROM pg_class "
                        "WHERE relnamespace=current_schema()::regnamespace "
                        "AND relname IN ('agent_runs', 'tool_calls', 'react_traces', 'run_events') "
                        "ORDER BY relname"
                    )
                )
            ).all()
            assert files_after == files_before, "constant defaults must not rewrite existing tables"
    async with engine.begin() as conn:
        assert await conn.run_sync(read_schema_phase) == "pre-activation"
        assert await conn.run_sync(schema_problems) == []
        # Released-shape SQL omits every new column. Retry on a separate connection.
        await conn.run_sync(lambda c: legacy_trace(c, "lost-ack-a", order=2))
    async with engine.begin() as conn:
        await conn.run_sync(lambda c: legacy_trace(c, "lost-ack-b", order=2))
    await ensure_schema_current(engine)
    url = engine.url.render_as_string(hide_password=False)
    monkeypatch.setenv("DENDRUX_DATABASE_URL", url)
    assert (await asyncio.to_thread(runner.invoke, app, ["db", "check"])).exit_code == 0
    conflict = await asyncio.to_thread(runner.invoke, app, ["db", "activate"])
    assert conflict.exit_code == 3, conflict.output
    async with engine.begin() as conn:
        assert await conn.run_sync(read_schema_phase) == "activating"
        assert (await conn.execute(text("SELECT count(*) FROM react_traces"))).scalar_one() == 6
        await conn.execute(text("DELETE FROM react_traces WHERE id='conflict-b'"))
    await ensure_schema_current(engine)
    assert (await asyncio.to_thread(runner.invoke, app, ["db", "check"])).exit_code == 0
    from dendrux.db import _activate

    checkpoint = _activate._checkpoint
    interrupted = False

    def stop_large_batch(conn, table, cursor):
        nonlocal interrupted
        checkpoint(conn, table, cursor)
        if table == "run_events" and cursor == "event-004999" and not interrupted:
            interrupted = True
            raise RuntimeError("interrupted large event batch")

    monkeypatch.setattr(_activate, "_checkpoint", stop_large_batch)
    start = time.perf_counter()
    with pytest.raises(RuntimeError, match="interrupted large"):
        await activate_database(engine)
    async with engine.connect() as conn:
        assert (
            await conn.execute(
                text("SELECT value FROM dendrux_settings WHERE key='dedupe_cursor_run_events'")
            )
        ).scalar_one() == "event-004999"
    monkeypatch.setattr(_activate, "_checkpoint", checkpoint)
    await activate_database(engine)
    activation_seconds = time.perf_counter() - start
    record_property("activation_seconds", activation_seconds)
    print(f"rehearsal dialect={engine.dialect.name} activation_seconds={activation_seconds:.4f}")
    async with engine.connect() as conn:
        assert await conn.run_sync(read_schema_phase) == "active"
        assert await conn.run_sync(schema_problems) == []
        assert (await conn.execute(text("SELECT count(*) FROM run_events"))).scalar_one() == 200000
        assert (
            await conn.execute(text("SELECT id FROM react_traces ORDER BY id"))
        ).scalars().all() == ["conflict-a", "identical-a", "lost-ack-a"]
    await ensure_schema_current(engine)
    assert (await asyncio.to_thread(runner.invoke, app, ["db", "check"])).exit_code == 0


async def test_b2_matrix_resume_and_exclusive_batches(rollout_database, monkeypatch):
    from dendrux.db import _activate
    from dendrux.db._schema_check import schema_problems

    engine = rollout_database
    await _rollout_prior(engine)
    await _seed_rehearsal_runs_only(engine)
    await _migrate(engine)
    async with engine.begin() as conn:
        for i in range(8):
            await conn.run_sync(
                lambda c, i=i: add_trace(
                    c, f"t{i}", order=i // 2, run_id="00000000000000000000000000"
                )
            )
    original = _activate._checkpoint
    cursors = []

    def stop(conn, table, cursor):
        original(conn, table, cursor)
        if table == "react_traces" and cursor == "t3":
            # A committed cursor must not release SQLite's exclusive file lock.
            if engine.dialect.name == "sqlite":
                from sqlalchemy.exc import OperationalError

                other = create_engine(
                    str(engine.url).replace("+aiosqlite", ""), connect_args={"timeout": 0}
                )
                try:
                    with other.connect() as live, pytest.raises(OperationalError):
                        live.exec_driver_sql("BEGIN IMMEDIATE")
                finally:
                    other.dispose()
            raise RuntimeError("interrupt after committed cursor")

    monkeypatch.setattr(_activate, "_checkpoint", stop)
    with pytest.raises(RuntimeError, match="interrupt"):
        await _activate.activate_database(engine, batch_size=2)
    async with engine.connect() as conn:
        assert (
            await conn.execute(
                text("SELECT value FROM dendrux_settings WHERE key='dedupe_cursor_react_traces'")
            )
        ).scalar_one() == "t3"
        assert await conn.run_sync(schema_problems) == []
    monkeypatch.setattr(_activate, "_checkpoint", original)
    page = _activate._page

    def capture(conn, table, column, after, end, size):
        cursors.append((table.name, after))
        return page(conn, table, column, after, end, size)

    monkeypatch.setattr(_activate, "_page", capture)
    await _activate.activate_database(engine, batch_size=2)
    assert ("react_traces", "t3") in cursors
    async with engine.connect() as conn:
        assert (
            await conn.execute(text("SELECT id FROM react_traces ORDER BY id"))
        ).scalars().all() == ["t0", "t2", "t4", "t6"]


async def _seed_rehearsal_runs_only(engine):
    async with engine.begin() as conn:
        await conn.execute(
            text("""
            INSERT INTO agent_runs (id, agent_name, status, iteration_count, delegation_level,
                total_input_tokens, total_output_tokens, total_cache_read_tokens,
                total_cache_creation_tokens, created_at, updated_at)
            VALUES ('00000000000000000000000000', 'rehearsal', 'success', 0, 0, 0, 0, 0, 0,
                CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        """)
        )


@pytest.mark.parametrize("table_name", ["react_traces", "tool_calls", "run_events"])
@pytest.mark.parametrize(
    ("first", "second"),
    [(True, 1), (1, 1.0), ("1", 1)],
    ids=["boolean-integer", "integer-float", "string-number"],
)
@pytest.mark.parametrize("digest", [None, "a" * 64], ids=["null-digest", "equal-digest"])
async def test_b1_typed_json_content_conflicts(
    rollout_database, monkeypatch, table_name, first, second, digest
):
    import asyncio

    engine = rollout_database
    await _rollout_prior(engine)
    await _seed_rehearsal_runs_only(engine)
    await _migrate(engine)
    table = Base.metadata.tables[table_name]
    async with engine.begin() as conn:
        for id, value in [("a", first), ("b", second)]:
            row = {"id": id, "agent_run_id": "00000000000000000000000000", "content_digest": digest}
            payload = {"flag": value}
            if table_name == "react_traces":
                row.update(role="assistant", content="same", order_index=0, meta=payload)
            elif table_name == "tool_calls":
                row.update(tool_call_id="call", tool_name="t", params=payload)
            else:
                row.update(event_type="run.started", sequence_index=0, data=payload)
            await conn.execute(table.insert(), row)
    monkeypatch.setenv("DENDRUX_DATABASE_URL", engine.url.render_as_string(hide_password=False))
    result = await asyncio.to_thread(runner.invoke, app, ["db", "dedupe"])
    conflict = digest is None
    assert result.exit_code == (3 if conflict else 0), (result.output, result.exception)
    async with engine.connect() as conn:
        assert (await conn.execute(table.select().order_by(table.c.id))).scalars().all() == (
            ["a", "b"] if conflict else ["a"]
        )


async def test_b4_pg_blocked_upgrade_retries(rollout_database, monkeypatch):
    import asyncio
    import time

    if rollout_database.dialect.name != "postgresql":
        pytest.skip("PostgreSQL lock_timeout")
    engine = rollout_database
    await _rollout_prior(engine)
    async with engine.connect() as blocker:
        await blocker.execute(text("LOCK TABLE agent_runs IN ACCESS EXCLUSIVE MODE"))
        start = time.perf_counter()
        with pytest.raises(RuntimeError, match="5 seconds"):
            await asyncio.wait_for(_migrate(engine), 10)
        elapsed = time.perf_counter() - start
        assert 4.8 <= elapsed < 8
        print(f"blocked upgrade refusal_seconds={elapsed:.4f}")
        await blocker.rollback()
    await _migrate(engine)
    async with engine.connect() as conn:
        assert (
            await conn.execute(text("SELECT version_num FROM alembic_version"))
        ).scalar_one() == HEAD
        assert (await conn.execute(text("SHOW lock_timeout"))).scalar_one() == "0"
    # Hold the first winner after its committed barrier while a second connection tries.
    from sqlalchemy.util.concurrency import await_only

    from dendrux.db import _activate

    ready, release = asyncio.Event(), asyncio.Event()
    preflight = _activate._preflight

    def hold(conn, size):
        ready.set()
        await_only(release.wait())
        preflight(conn, size)

    monkeypatch.setattr(_activate, "_preflight", hold)
    winner = asyncio.create_task(_activate.activate_database(engine))
    try:
        await asyncio.wait_for(ready.wait(), 10)
        with pytest.raises(RuntimeError, match="Another Dendrux"):
            await _activate.activate_database(engine)
    finally:
        release.set()
        await winner


def test_b1_legacy_tool_error_content_is_not_discarded(legacy):
    engine = migrate_legacy(legacy)
    with engine.begin() as conn:
        for id, message in [("a", "private-error-a"), ("b", "private-error-b")]:
            conn.execute(
                Base.metadata.tables["tool_calls"].insert(),
                {
                    "id": id,
                    "agent_run_id": "00000000000000000000999999",
                    "tool_call_id": "call",
                    "tool_name": "t",
                    "target": "server",
                    "result": {"payload": "same"},
                    "success": False,
                    "error_message": message,
                },
            )
    result = runner.invoke(app, ["db", "dedupe"])
    assert result.exit_code == 3, result.output
    assert "private-error" not in result.output


async def test_activation_query_plans(rollout_database, record_property):
    import json

    from sqlalchemy import event

    from dendrux.db._activate import _page

    engine = rollout_database
    await _rollout_prior(engine)
    await _seed_rehearsal(engine)
    await _migrate(engine)
    async with engine.begin() as conn:
        for table_name in ("react_traces", "tool_calls"):
            table = Base.metadata.tables[table_name]
            for start in range(0, 5000, 500):
                rows = []
                for i in range(start, start + 500):
                    row = {"id": f"row-{i:05}", "agent_run_id": f"{i:026d}"}
                    if table_name == "react_traces":
                        row.update(role="assistant", content="same", order_index=0)
                    else:
                        row.update(tool_call_id=f"call-{i}", tool_name="t")
                    rows.append(row)
                await conn.execute(table.insert(), rows)
        for table_name in ("agent_runs", "react_traces", "tool_calls", "run_events"):
            await conn.execute(text(f"ANALYZE {table_name}"))
    plans = {}

    def explain(conn, cursor, statement, parameters, context, executemany):
        if statement.startswith("SELECT candidate.id"):
            prefix = (
                "EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) "
                if engine.dialect.name == "postgresql"
                else "EXPLAIN QUERY PLAN "
            )
            result = conn.exec_driver_sql(prefix + statement, parameters)
            plans[context.compiled.statement.get_final_froms()[0].original.name] = (
                result.scalar_one()[0]
                if engine.dialect.name == "postgresql"
                else [tuple(row) for row in result]
            )

    event.listen(engine.sync_engine, "before_cursor_execute", explain)
    try:
        async with engine.connect() as conn:
            for table_name, column, _ in INDEXES.values():
                end = "event-199999" if table_name == "run_events" else "row-04999"
                page = await conn.run_sync(
                    lambda c, t=table_name, k=column, e=end: _page(
                        c, Base.metadata.tables[t], k, "", e, 5000
                    )
                )
                assert len(page) == 5000
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", explain)
    record_property("dedupe_query_plans", json.dumps(plans))
    print("dedupe_query_plans", engine.dialect.name, json.dumps(plans))
    assert len(plans) == 3
    if engine.dialect.name == "postgresql":

        def nodes(node):
            yield node
            for child in node.get("Plans", []):
                yield from nodes(child)

        for table_name, plan in plans.items():
            assert any(
                node.get("Index Name") == f"{table_name}_pkey" for node in nodes(plan["Plan"])
            )
            assert not any(node["Node Type"] == "Seq Scan" for node in nodes(plan["Plan"]))


async def test_b2_pg_repairs_interrupted_invalid_index(rollout_database):
    from sqlalchemy.exc import IntegrityError

    from dendrux.db._activate import activate_database
    from dendrux.db._schema_check import schema_problems

    engine = rollout_database
    if engine.dialect.name != "postgresql":
        pytest.skip("PostgreSQL invalid concurrent indexes")
    await _rollout_prior(engine)
    await _seed_rehearsal_runs_only(engine)
    await _migrate(engine)
    async with engine.begin() as conn:
        await conn.run_sync(lambda c: add_trace(c, "a", run_id="00000000000000000000000000"))
        await conn.run_sync(lambda c: add_trace(c, "b", run_id="00000000000000000000000000"))
    async with engine.connect() as conn:
        conn = await conn.execution_options(isolation_level="AUTOCOMMIT")
        with pytest.raises(IntegrityError):
            await conn.execute(
                text(
                    "CREATE UNIQUE INDEX CONCURRENTLY uq_react_traces_run_order "
                    "ON react_traces (agent_run_id, order_index)"
                )
            )
        assert (
            await conn.execute(
                text(
                    "SELECT indisvalid FROM pg_index "
                    "WHERE indexrelid='uq_react_traces_run_order'::regclass"
                )
            )
        ).scalar_one() is False
    await activate_database(engine)
    async with engine.connect() as conn:
        assert await conn.run_sync(schema_problems) == []
        assert (await conn.execute(text("SELECT id FROM react_traces"))).scalars().all() == ["a"]


async def test_b3_mixed_index_window_and_active_replaced_names(rollout_database):

    from dendrux.db._schema_check import ensure_schema_current, schema_problems
    from dendrux.db._storage_schema import index_ddl
    from dendrux.errors import SchemaOutOfDateError

    engine = rollout_database
    await _rollout_prior(engine)
    await _migrate(engine)
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO dendrux_settings VALUES "
                "('activate_step', 'index:uq_react_traces_run_order', CURRENT_TIMESTAMP)"
            )
        )
        await conn.execute(text(index_ddl("uq_react_traces_run_order")))
        await conn.execute(text("DROP INDEX ix_react_traces_agent_run_id_order"))
    await ensure_schema_current(engine)
    async with engine.connect() as conn:
        assert await conn.run_sync(schema_problems) == []
    from dendrux.db._activate import activate_database

    await activate_database(engine)
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "CREATE INDEX ix_react_traces_agent_run_id_order "
                "ON react_traces (agent_run_id, order_index)"
            )
        )
    with pytest.raises(SchemaOutOfDateError):
        await ensure_schema_current(engine)


def test_b2_index_interruption_skips_completed_dedupe(legacy, monkeypatch):
    from dendrux.db import _activate

    engine = migrate_legacy(legacy)
    with engine.begin() as conn:
        add_trace(conn, "a")
        add_trace(conn, "b")
    original = _activate.require_index

    def stop(conn, name, table, expected):
        original(conn, name, table, expected)
        raise RuntimeError("interrupt index step")

    monkeypatch.setattr(_activate, "require_index", stop)
    assert runner.invoke(app, ["db", "activate"]).exit_code == 1
    monkeypatch.setattr(_activate, "require_index", original)

    def should_not_repeat(*args):
        pytest.fail("completed dedupe must not be repeated after an index interruption")

    monkeypatch.setattr(_activate, "_dedupe", should_not_repeat)
    result = runner.invoke(app, ["db", "activate"])
    assert result.exit_code == 0, (result.output, result.exception)


def test_additive_migration_only_uses_bounded_steps(legacy, monkeypatch):
    from dendrux.db import _migration_ddl

    statements = []
    bounded = _migration_ddl.bounded_step

    def capture(conn, statement, **kwargs):
        statements.append((statement, kwargs))
        bounded(conn, statement, **kwargs)

    monkeypatch.setattr(_migration_ddl, "bounded_step", capture)
    alembic(legacy[1], "upgrade", "head")
    assert len(statements) == sum(map(len, NEW_COLUMNS.values())) + 2
    assert all(kwargs["revision"] == HEAD for _, kwargs in statements)
    assert not any("CREATE INDEX" in sql.upper() for sql, _ in statements)


async def test_b3_create_all_active_on_both_backends(rollout_database):
    from dendrux.db._schema_check import ensure_schema_current

    engine = rollout_database
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        assert dict(
            (await conn.execute(text("SELECT key, value FROM dendrux_settings"))).all()
        ) == {"storage_format": "2", "activate_step": "done"}
    await ensure_schema_current(engine)


async def test_additive_downgrade_barrier_on_both_backends(rollout_database):
    import asyncio

    from dendrux.db._activate import activate_database

    engine = rollout_database
    await _rollout_prior(engine)
    await _migrate(engine)
    url = engine.url.render_as_string(hide_password=False)
    await asyncio.to_thread(alembic, url, "downgrade", PRIOR)
    async with engine.connect() as conn:
        assert not await conn.run_sync(lambda c: inspect(c).has_table("dendrux_settings"))
    await _migrate(engine)
    await activate_database(engine, dedupe_only=True)
    with pytest.raises(RuntimeError, match="activation"):
        await asyncio.to_thread(alembic, url, "downgrade", PRIOR)
    async with engine.connect() as conn:
        assert (
            await conn.execute(text("SELECT version_num FROM alembic_version"))
        ).scalar_one() == HEAD
        assert await conn.run_sync(lambda c: inspect(c).has_table("dendrux_settings"))


def test_duplicate_content_reads_have_a_separate_memory_bound(legacy):
    from sqlalchemy import Engine, event

    engine = migrate_legacy(legacy)
    with engine.begin() as conn:
        conn.execute(
            Base.metadata.tables["react_traces"].insert(),
            [
                {
                    "id": f"trace-{i:04}",
                    "agent_run_id": "00000000000000000000999999",
                    "role": "assistant",
                    "content": "same",
                    "order_index": 0,
                }
                for i in range(1000)
            ],
        )
    sizes = []

    def capture(conn, cursor, statement, parameters, context, executemany):
        if statement.startswith("SELECT react_traces.id, react_traces.content_digest"):
            sizes.append(len(parameters))

    event.listen(Engine, "before_cursor_execute", capture)
    try:
        result = runner.invoke(app, ["db", "dedupe"])
        assert result.exit_code == 0, result.output
    finally:
        event.remove(Engine, "before_cursor_execute", capture)
    assert sizes and max(sizes) <= 200, "large payloads must be read in small duplicate-pair groups"


async def test_b3_pg_startup_catalog_queries_remain_bounded(rollout_database):
    from sqlalchemy import event

    from dendrux.db._schema_check import ensure_schema_current

    engine = rollout_database
    if engine.dialect.name != "postgresql":
        pytest.skip("PostgreSQL catalog query budget")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    statements = []

    def capture(conn, cursor, statement, parameters, context, executemany):
        statements.append(statement)

    event.listen(engine.sync_engine, "before_cursor_execute", capture)
    try:
        await ensure_schema_current(engine)
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", capture)
    assert len(statements) <= 5


@pytest.mark.parametrize("wrong_owner", [True, False], ids=["other-table", "wrong-columns"])
async def test_activation_never_drops_a_replaced_index_with_wrong_definition(
    rollout_database, monkeypatch, wrong_owner
):
    import asyncio

    from dendrux.db._storage_schema import index_ddl

    engine = rollout_database
    await _rollout_prior(engine)
    await _migrate(engine)
    monkeypatch.setenv("DENDRUX_DATABASE_URL", engine.url.render_as_string(hide_password=False))
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO dendrux_settings VALUES "
                "('activate_step', 'drop:ix_react_traces_agent_run_id_order', CURRENT_TIMESTAMP)"
            )
        )
        for name in INDEXES:
            await conn.execute(text(index_ddl(name)))
        await conn.execute(text("DROP INDEX ix_react_traces_agent_run_id_order"))
        if wrong_owner:
            await conn.execute(text("CREATE TABLE operator_owned (value INTEGER)"))
            await conn.execute(
                text("CREATE INDEX ix_react_traces_agent_run_id_order ON operator_owned (value)")
            )
        else:
            await conn.execute(
                text(
                    "CREATE INDEX ix_react_traces_agent_run_id_order ON react_traces (order_index)"
                )
            )
    result = await asyncio.to_thread(runner.invoke, app, ["db", "activate"])
    assert result.exit_code == 1, "a name collision must refuse without dropping an unrelated index"
    async with engine.connect() as conn:
        owner = "operator_owned" if wrong_owner else "react_traces"
        indexes = await conn.run_sync(lambda c: inspect(c).get_indexes(owner))
        assert "ix_react_traces_agent_run_id_order" in {index["name"] for index in indexes}


async def test_dedupe_preserves_natural_key_blob_edges(rollout_database):
    from dendrux.db._activate import activate_database
    from dendrux.db.models import BlobRef, ReactTrace, ToolCallRecord
    from dendrux.runtime._blob_catalog import BlobServices
    from dendrux.runtime.stores._factory import store_for_engine
    from tests.unit.test_blob_catalog import make_png, store_png

    engine = rollout_database
    await _rollout_prior(engine)
    await _seed_rehearsal_runs_only(engine)
    await _migrate(engine)
    services = BlobServices.for_state_store(store_for_engine(engine))
    part = await store_png(services, "-", make_png(42))
    async with engine.begin() as conn:
        await conn.execute(
            ReactTrace.__table__.insert(),
            [
                {
                    "id": id,
                    "agent_run_id": "00000000000000000000000000",
                    "role": "user",
                    "content": "same",
                    "order_index": order,
                }
                for id, order in [("0", 0), ("1", 0), ("trace-other", 1)]
            ],
        )
        await conn.execute(
            ToolCallRecord.__table__.insert(),
            [
                {
                    "id": id,
                    "agent_run_id": "00000000000000000000000000",
                    "tool_call_id": call,
                    "tool_name": "t",
                }
                for id, call in [("a", "call"), ("other-call", "call"), ("z", "other-call")]
            ],
        )
        await conn.execute(
            BlobRef.__table__.insert(),
            [
                {
                    "id": f"edge-{i}",
                    "tenant_scope": "-",
                    "sha256": part.ref.sha256,
                    "run_id": "00000000000000000000000000",
                    "owner_kind": kind,
                    "owner_id": owner,
                    "ordinal": 0,
                }
                for i, (kind, owner) in enumerate(
                    [
                        ("trace", "0"),
                        ("trace", "1"),
                        ("tool_call", "call"),
                        ("tool_call", "other-call"),
                    ]
                )
            ],
        )
        before = (await conn.execute(BlobRef.__table__.select().order_by(BlobRef.id))).all()
    await activate_database(engine, dedupe_only=True)
    async with engine.connect() as conn:
        assert (await conn.execute(text("SELECT count(*) FROM react_traces"))).scalar_one() == 2
        assert (await conn.execute(text("SELECT count(*) FROM tool_calls"))).scalar_one() == 2
        assert (await conn.execute(BlobRef.__table__.select().order_by(BlobRef.id))).all() == before


async def test_preflight_conflict_in_later_table_deletes_nothing(rollout_database, monkeypatch):
    import asyncio

    engine = rollout_database
    await _rollout_prior(engine)
    await _seed_rehearsal_runs_only(engine)
    await _migrate(engine)
    async with engine.begin() as conn:
        for id in ("a", "b"):
            await conn.run_sync(
                lambda c, id=id: add_trace(c, id, run_id="00000000000000000000000000")
            )
        await conn.execute(
            Base.metadata.tables["run_events"].insert(),
            [
                {
                    "id": id,
                    "agent_run_id": "00000000000000000000000000",
                    "event_type": "run.started",
                    "sequence_index": 0,
                    "data": {"value": value},
                }
                for id, value in [("a", 1), ("b", 2)]
            ],
        )
        before = {
            table: (await conn.execute(text(f"SELECT count(*) FROM {table}"))).scalar_one()
            for table in Base.metadata.tables
            if table != "dendrux_settings"
        }
    monkeypatch.setenv("DENDRUX_DATABASE_URL", engine.url.render_as_string(hide_password=False))
    result = await asyncio.to_thread(runner.invoke, app, ["db", "activate", "--batch-size", "1"])
    assert result.exit_code == 3, (result.output, result.exception)
    assert "run_events" in result.output
    async with engine.connect() as conn:
        after = {
            table: (await conn.execute(text(f"SELECT count(*) FROM {table}"))).scalar_one()
            for table in before
        }
    assert after == before


async def test_resume_skips_completed_dedupe_tables(rollout_database, monkeypatch):
    from dendrux.db import _activate

    engine = rollout_database
    await _rollout_prior(engine)
    await _seed_rehearsal_runs_only(engine)
    await _migrate(engine)
    async with engine.begin() as conn:
        await conn.execute(
            Base.metadata.tables["run_events"].insert(),
            [
                {
                    "id": id,
                    "agent_run_id": "00000000000000000000000000",
                    "event_type": "run.started",
                    "sequence_index": seq,
                }
                for seq, id in enumerate(["a", "b"])
            ],
        )
    checkpoint = _activate._checkpoint

    def stop(conn, table, cursor):
        checkpoint(conn, table, cursor)
        if table == "run_events" and cursor == "a":
            raise RuntimeError("interrupt in final table")

    monkeypatch.setattr(_activate, "_checkpoint", stop)
    with pytest.raises(RuntimeError, match="interrupt in final table"):
        await _activate.activate_database(engine, batch_size=1)
    monkeypatch.setattr(_activate, "_checkpoint", checkpoint)
    pages = []
    page = _activate._page
    dedupe = _activate._dedupe
    in_dedupe = False

    def capture(conn, table, column, after, end, size):
        if in_dedupe:
            pages.append((table.name, after))
        return page(conn, table, column, after, end, size)

    def only_dedupe(conn, size):
        nonlocal in_dedupe
        in_dedupe = True
        try:
            dedupe(conn, size)
        finally:
            in_dedupe = False

    # Empty tables still reveal a rescan through their MAX(id) query.
    from sqlalchemy import event

    scanned = []

    def queries(conn, cursor, statement, parameters, context, executemany):
        if in_dedupe and statement.startswith("SELECT max("):
            scanned.append(statement)

    monkeypatch.setattr(_activate, "_page", capture)
    monkeypatch.setattr(_activate, "_dedupe", only_dedupe)
    event.listen(engine.sync_engine, "before_cursor_execute", queries)
    try:
        await _activate.activate_database(engine, batch_size=1)
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", queries)
    assert pages and pages[0] == ("run_events", "a")
    assert len(scanned) == 1 and "run_events" in scanned[0]


@pytest.mark.parametrize("index_name", list(INDEXES))
async def test_index_build_late_duplicates_restart_that_table(
    rollout_database, monkeypatch, index_name
):
    import asyncio

    from sqlalchemy.util.concurrency import await_only

    from dendrux.db import _activate

    engine = rollout_database
    await _rollout_prior(engine)
    await _seed_rehearsal_runs_only(engine)
    await _migrate(engine)
    table_name = INDEXES[index_name][0]
    table = Base.metadata.tables[table_name]
    row = {"agent_run_id": "00000000000000000000000000"}
    if table_name == "react_traces":
        row.update(role="user", content="private-sentinel", order_index=0)
    elif table_name == "tool_calls":
        row.update(tool_call_id="call", tool_name="t", params={"value": "private-sentinel"})
    else:
        row.update(event_type="run.started", sequence_index=0, data={"value": "private-sentinel"})
    async with engine.begin() as conn:
        await conn.execute(table.insert(), {**row, "id": "z"})
    original = _activate._dedupe
    inserted = False

    async def late_writer():
        async with engine.begin() as conn:
            await conn.execute(table.insert(), {**row, "id": "a"})

    def insert_after_dedupe(conn, size):
        nonlocal inserted
        original(conn, size)
        if not inserted:
            inserted = True
            if engine.dialect.name == "postgresql":
                await_only(late_writer())
            else:
                # SQLite holds EXCLUSIVE; inject on the maintenance connection.
                conn.execute(table.insert(), {**row, "id": "a"})
                conn.commit()

    monkeypatch.setattr(_activate, "_dedupe", insert_after_dedupe)
    monkeypatch.setenv("DENDRUX_DATABASE_URL", engine.url.render_as_string(hide_password=False))
    result = await asyncio.to_thread(runner.invoke, app, ["db", "activate"])
    assert result.exit_code == 3, (result.output, result.exception)
    assert "duplicates appeared after dedupe; rerun dendrux db dedupe" in result.output
    assert "private-sentinel" not in result.output
    async with engine.connect() as conn:
        settings = dict((await conn.execute(text("SELECT key, value FROM dendrux_settings"))).all())
        assert settings["activate_step"] == f"dedupe:{table_name}"
        assert f"dedupe_cursor_{table_name}" not in settings
        assert all(
            settings[f"dedupe_cursor_{name}"] == ""
            for name in ("react_traces", "tool_calls", "run_events")
            if name != table_name
        )
        assert (await conn.execute(table.select().order_by(table.c.id))).scalars().all() == [
            "a",
            "z",
        ]
    result = await asyncio.to_thread(runner.invoke, app, ["db", "activate"])
    assert result.exit_code == 0, (result.output, result.exception)
    async with engine.connect() as conn:
        assert (await conn.execute(table.select())).scalars().all() == ["a"]
        assert (
            await conn.execute(text("SELECT value FROM dendrux_settings WHERE key='activate_step'"))
        ).scalar_one() == "done"


async def test_pg_resume_after_interrupted_replaced_index_drop(rollout_database, monkeypatch):
    import time

    from sqlalchemy.util.concurrency import await_only

    from dendrux.db import _activate
    from dendrux.db._schema_check import schema_problems

    engine = rollout_database
    if engine.dialect.name != "postgresql":
        pytest.skip("PostgreSQL concurrent drop invalidation")
    await _rollout_prior(engine)
    await _migrate(engine)
    old = "ix_react_traces_agent_run_id_order"
    drop = _activate.drop_index_concurrently
    blocker = None

    async def open_blocker():
        connection = await engine.connect()
        await connection.execute(text("SELECT 1 FROM react_traces LIMIT 1"))
        return connection

    def blocked_drop(conn, name, **kwargs):
        nonlocal blocker
        if name == old and blocker is None:
            blocker = await_only(open_blocker())
        return drop(conn, name, **kwargs)

    monkeypatch.setattr(_activate, "drop_index_concurrently", blocked_drop)
    start = time.perf_counter()
    try:
        with pytest.raises(RuntimeError, match="5 seconds"):
            await _activate.activate_database(engine)
        assert 4.5 <= time.perf_counter() - start < 10
    finally:
        if blocker is not None:
            await blocker.rollback()
            await blocker.close()
    monkeypatch.setattr(_activate, "drop_index_concurrently", drop)
    async with engine.connect() as conn:
        assert (
            await conn.execute(
                text("SELECT indisvalid FROM pg_index WHERE indexrelid=CAST(:name AS regclass)"),
                {"name": old},
            )
        ).scalar_one() is False
    await _activate.activate_database(engine)
    async with engine.connect() as conn:
        assert await conn.run_sync(schema_problems) == []
        assert (
            await conn.execute(text("SELECT to_regclass(:name)"), {"name": old})
        ).scalar_one() is None
