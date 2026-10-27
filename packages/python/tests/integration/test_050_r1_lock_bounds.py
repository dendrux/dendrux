"""R1 (d7e8f9a0b1c2) bounds every lock wait and builds its indexes online.

The full upgrade starts at 0.4.1's head (``c6d7e8f9a0b1``) with an application
transaction holding ``agent_runs``. Every exclusive step must refuse within its
5 s ``lock_timeout`` with the retry message, and the index builds must not keep
writers out. Postgres only; uses a temporary schema inside ``DENDRUX_TEST_PG_URL``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from typing import Any

import pytest
from sqlalchemy import event, inspect, text
from sqlalchemy.exc import DBAPIError

from dendrux.db._schema_check import ensure_schema_current
from tests.integration._migration_schema import migration_schema
from tests.integration.test_050_t1_r1_migration import PRIOR, _alembic, _indexes, _seed

pytestmark = pytest.mark.asyncio

R1 = "d7e8f9a0b1c2"
HEAD = "e8f9a0b1c2d3"
_TIMEOUT = "Timed out acquiring the Dendrux migration lock after 5 seconds"
_R1_INDEXES = ("uq_agent_runs_tenant_idempotency_key", "ix_agent_runs_tenant_id_created_at")


def _lock_timeout_records(caplog: Any) -> list[logging.LogRecord]:
    return [r for r in caplog.records if getattr(r, "code", None) == "migration_lock_timeout"]


def _assert_lock_timeout_logged(caplog: Any, revision: str, step: str) -> None:
    """Exactly one structured line, with stable fields and no SQL text."""
    records = _lock_timeout_records(caplog)
    assert len(records) == 1, [r.getMessage() for r in records]
    record = records[0]
    assert record.levelno == logging.WARNING
    assert (record.revision, record.step, record.timeout_seconds) == (revision, step, 5)
    text_seen = (
        record.getMessage()
        + " "
        + " ".join(str(v) for k, v in vars(record).items() if k in {"revision", "step", "code"})
    )
    for sql in ("ALTER", "INDEX ", "DROP ", "CREATE", "agent_runs ("):
        assert sql not in text_seen, text_seen


@pytest.fixture
async def pg(pg_test_url: str) -> Any:
    async with migration_schema(pg_test_url, "r1_lock_bounds") as engine:
        yield pg_test_url, engine


async def _version(engine: Any) -> str:
    async with engine.connect() as conn:
        return (await conn.execute(text("SELECT version_num FROM alembic_version"))).scalar_one()


async def _columns(engine: Any) -> set[str]:
    async with engine.connect() as conn:
        return await conn.run_sync(
            lambda c: {v["name"] for v in inspect(c).get_columns("agent_runs")}
        )


async def _invalid_indexes(engine: Any) -> int:
    async with engine.connect() as conn:
        return (
            await conn.execute(
                text(
                    "SELECT count(*) FROM pg_index WHERE indrelid = 'agent_runs'::regclass "
                    "AND NOT (indisvalid AND indisready)"
                )
            )
        ).scalar_one()


async def test_upgrade_from_041_refuses_behind_an_open_transaction(pg: Any, caplog: Any) -> None:
    """A long application transaction must not make R1 wait without bound."""
    caplog.set_level(logging.WARNING)
    url, engine = pg
    await asyncio.to_thread(_alembic, url, "upgrade", PRIOR)
    await _seed(engine)
    before = await _indexes(engine)
    async with engine.connect() as blocker:
        transaction = await blocker.begin()
        # ACCESS SHARE: writers still run, but any exclusive DDL must wait.
        await blocker.execute(text("SELECT id FROM agent_runs LIMIT 1"))
        task = asyncio.create_task(asyncio.to_thread(_alembic, url, "upgrade", "head"))
        started = time.perf_counter()
        try:
            done, _ = await asyncio.wait({task}, timeout=15)
            if not done:
                pytest.fail("R1 waited more than 15 s for an application transaction")
            elapsed = time.perf_counter() - started
            with pytest.raises(RuntimeError, match=_TIMEOUT) as error:
                task.result()
            assert error.value.__suppress_context__, "no raw driver error chained"
            assert 4.5 < elapsed < 10, elapsed
            _assert_lock_timeout_logged(caplog, R1, "add_claim_epoch")
        finally:
            await transaction.rollback()
            await asyncio.gather(task, return_exceptions=True)
    # Refused before changing anything.
    assert await _version(engine) == PRIOR
    assert "claim_epoch" not in await _columns(engine)
    assert await _indexes(engine) == before

    # Once the blocker has finished, the same command completes.
    await asyncio.to_thread(_alembic, url, "upgrade", "head")
    assert await _version(engine) == HEAD
    assert await _invalid_indexes(engine) == 0
    await ensure_schema_current(engine)


async def test_writers_proceed_while_r1_builds_its_indexes(pg: Any) -> None:
    """The column add commits before the builds; the builds are concurrent and bounded."""
    url, engine = pg
    await asyncio.to_thread(_alembic, url, "upgrade", PRIOR)
    await _seed(engine)
    loop = asyncio.get_running_loop()
    reached = asyncio.Event()
    release = threading.Event()
    thread_id: int | None = None
    seen: list[tuple[str, str]] = []

    def before(conn, cursor, statement, parameters, context, executemany):  # noqa: ANN001
        if threading.get_ident() != thread_id:
            return
        if (
            "uq_agent_runs_tenant_idempotency_key" in statement
            and statement.lstrip().upper().startswith("CREATE UNIQUE INDEX")
        ):
            seen.append((statement, conn.exec_driver_sql("SHOW lock_timeout").scalar_one()))
            loop.call_soon_threadsafe(reached.set)
            assert release.wait(20)

    def migrate() -> None:
        nonlocal thread_id
        thread_id = threading.get_ident()
        _alembic(url, "upgrade", "head")

    event.listen(engine.sync_engine.__class__, "before_cursor_execute", before)
    task = asyncio.create_task(asyncio.to_thread(migrate))
    try:
        await asyncio.wait_for(reached.wait(), 20)
        async with engine.connect() as writer:
            writer = await writer.execution_options(isolation_level="AUTOCOMMIT")
            await writer.execute(text("SET lock_timeout = '2s'"))
            try:
                await writer.execute(
                    text("UPDATE agent_runs SET iteration_count = 2 WHERE id = 'r1'")
                )
            except DBAPIError:
                pytest.fail("a writer was locked out of agent_runs while R1 built its index")
        statement, timeout = seen[0]
        assert "CONCURRENTLY" in statement.upper()
        assert timeout == "5s"
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        event.remove(engine.sync_engine.__class__, "before_cursor_execute", before)
    task.result()
    assert await _version(engine) == HEAD


# Each exclusive R1 step, the action that reaches it, and the blocker that makes it wait.
# ACCESS SHARE (a read in an open transaction) blocks ALTER TABLE; concurrent index
# builds and drops wait for any open writer transaction.
_READ = "SELECT id FROM agent_runs LIMIT 1"
_WRITE = "UPDATE agent_runs SET iteration_count = 3 WHERE id = 'r4'"
_STEPS = {
    "add_column": "add_claim_epoch",
    "tenant_key_index": "build_index.uq_agent_runs_tenant_idempotency_key",
    "tenant_created_index": "build_index.ix_agent_runs_tenant_id_created_at",
    "drop_global_index": "drop_index.ix_agent_runs_idempotency_key",
    "drop_global_constraint": "drop_constraint.agent_runs_idempotency_key_key",
    "restore_global_index": "build_index.ix_agent_runs_idempotency_key",
    "drop_tenant_index": "drop_index.uq_agent_runs_tenant_idempotency_key",
    "drop_column": "drop_claim_epoch",
}
_PHASES = {
    "add_column": ("upgrade", "add column if not exists claim_epoch", _READ),
    "tenant_key_index": (
        "upgrade",
        "create unique index concurrently if not exists uq_agent_runs_tenant_idempotency_key",
        _WRITE,
    ),
    "tenant_created_index": (
        "upgrade",
        "create index concurrently if not exists ix_agent_runs_tenant_id_created_at",
        _WRITE,
    ),
    "drop_global_index": (
        "upgrade",
        "drop index concurrently if exists ix_agent_runs_idempotency_key",
        _WRITE,
    ),
    "drop_global_constraint": (
        "upgrade",
        "drop constraint if exists agent_runs_idempotency_key_key",
        _READ,
    ),
    "restore_global_index": (
        "downgrade",
        "create unique index concurrently if not exists ix_agent_runs_idempotency_key",
        _WRITE,
    ),
    "drop_tenant_index": (
        "downgrade",
        "drop index concurrently if exists uq_agent_runs_tenant_idempotency_key",
        _WRITE,
    ),
    "drop_column": ("downgrade", "drop column if exists claim_epoch", _READ),
}


@pytest.mark.parametrize("phase", list(_PHASES))
async def test_every_exclusive_step_is_bounded_and_retry_completes(
    pg: Any, phase: str, record_property: Any, caplog: Any
) -> None:
    caplog.set_level(logging.WARNING)
    url, engine = pg
    action, marker, blocker_sql = _PHASES[phase]
    await asyncio.to_thread(_alembic, url, "upgrade", PRIOR)
    await _seed(engine)
    if phase == "drop_global_constraint":
        # The 0.4.1 shape a ``create_all`` database has, stamped and then migrated.
        async with engine.begin() as conn:
            await conn.execute(text("DROP INDEX ix_agent_runs_idempotency_key"))
            await conn.execute(
                text(
                    "ALTER TABLE agent_runs ADD CONSTRAINT agent_runs_idempotency_key_key "
                    "UNIQUE (idempotency_key)"
                )
            )
    if action == "downgrade":
        await asyncio.to_thread(_alembic, url, "upgrade", "head")
    target = "head" if action == "upgrade" else PRIOR

    loop = asyncio.get_running_loop()
    reached = asyncio.Event()
    release = threading.Event()
    thread_id: int | None = None
    observed: list[str] = []
    initial: list[str] = []
    restored: list[str] = []

    def before(conn, cursor, statement, parameters, context, executemany):  # noqa: ANN001
        if threading.get_ident() != thread_id:
            return
        if "pg_try_advisory_lock(" in statement:
            initial.append(conn.exec_driver_sql("SHOW lock_timeout").scalar_one())
        if "pg_advisory_unlock(" in statement:
            restored.append(conn.exec_driver_sql("SHOW lock_timeout").scalar_one())
        if marker in " ".join(statement.lower().split()) and not observed:
            observed.append(conn.exec_driver_sql("SHOW lock_timeout").scalar_one())
            loop.call_soon_threadsafe(reached.set)
            assert release.wait(20), "the test releases the step once its blocker is in place"

    def migrate() -> None:
        nonlocal thread_id
        thread_id = threading.get_ident()
        _alembic(url, action, target)

    async with engine.connect() as blocker:
        transaction = await blocker.begin()
        event.listen(engine.sync_engine.__class__, "before_cursor_execute", before)
        task = asyncio.create_task(asyncio.to_thread(migrate))
        try:
            waiter = asyncio.ensure_future(reached.wait())
            await asyncio.wait({task, waiter}, timeout=20, return_when=asyncio.FIRST_COMPLETED)
            waiter.cancel()
            assert reached.is_set(), f"the migration never ran the {phase} step as bounded DDL"
            await blocker.execute(text(blocker_sql))
            started = time.perf_counter()
            release.set()
            done, _ = await asyncio.wait({task}, timeout=15)
            if not done:
                pytest.fail(f"the {phase} step waited more than 15 s for an open transaction")
            elapsed = time.perf_counter() - started
            with pytest.raises(RuntimeError, match=_TIMEOUT) as error:
                task.result()
            assert error.value.__suppress_context__
            assert elapsed < 10, elapsed
            assert observed == ["5s"], observed
            assert restored == initial, "the migration connection's lock_timeout is restored"
            _assert_lock_timeout_logged(caplog, R1, _STEPS[phase])
            record_property(f"r1_lock_timeout_{phase}", json.dumps({"seconds": elapsed}))
        finally:
            release.set()
            await transaction.rollback()
            await asyncio.gather(task, return_exceptions=True)
            event.remove(engine.sync_engine.__class__, "before_cursor_execute", before)

    async with engine.connect() as conn:
        assert (await conn.execute(text("SELECT count(*) FROM agent_runs"))).scalar_one() == 4

    # Rerunning the same command repairs whatever the refused step left behind.
    await asyncio.to_thread(_alembic, url, action, target)
    indexes = await _indexes(engine)
    assert await _invalid_indexes(engine) == 0
    if action == "upgrade":
        assert await _version(engine) == HEAD
        assert "claim_epoch" in await _columns(engine)
        assert set(_R1_INDEXES) <= set(indexes)
        assert "ix_agent_runs_idempotency_key" not in indexes
        assert "agent_runs_idempotency_key_key" not in indexes
        await ensure_schema_current(engine)
    else:
        assert await _version(engine) == PRIOR
        assert "claim_epoch" not in await _columns(engine)
        assert not set(_R1_INDEXES) & set(indexes)
        assert (
            "UNIQUE INDEX ix_agent_runs_idempotency_key" in indexes["ix_agent_runs_idempotency_key"]
        )
        await asyncio.to_thread(_alembic, url, "upgrade", "head")
        assert await _version(engine) == HEAD


async def test_upgrade_repairs_an_invalid_interrupted_tenant_index(pg: Any) -> None:
    """An invalid index left by an interrupted concurrent build is rebuilt on rerun."""
    url, engine = pg
    await asyncio.to_thread(_alembic, url, "upgrade", PRIOR)
    await _seed(engine)
    async with engine.connect() as conn:
        conn = await conn.execution_options(isolation_level="AUTOCOMMIT")
        # Duplicate under the tenant-scoped key, so the build fails and stays invalid.
        await conn.execute(text("DROP INDEX ix_agent_runs_idempotency_key"))
        await conn.execute(text("UPDATE agent_runs SET tenant_id = 'A', idempotency_key = NULL"))
        await conn.execute(
            text("UPDATE agent_runs SET idempotency_key = 'same' WHERE id IN ('r1', 'r2')")
        )
        with pytest.raises(DBAPIError):
            await conn.execute(
                text(
                    "CREATE UNIQUE INDEX CONCURRENTLY uq_agent_runs_tenant_idempotency_key "
                    "ON agent_runs (coalesce(tenant_id, ''), idempotency_key) "
                    "WHERE idempotency_key IS NOT NULL"
                )
            )
        await conn.execute(text("UPDATE agent_runs SET idempotency_key = NULL WHERE id = 'r2'"))
        await conn.execute(
            text(
                "CREATE UNIQUE INDEX ix_agent_runs_idempotency_key ON agent_runs (idempotency_key)"
            )
        )
    assert await _invalid_indexes(engine) == 1
    await asyncio.to_thread(_alembic, url, "upgrade", "head")
    assert await _invalid_indexes(engine) == 0
    assert await _version(engine) == HEAD
    await ensure_schema_current(engine)


async def test_downgrade_race_with_a_shared_key_refuses_without_an_invalid_index(
    pg: Any,
) -> None:
    """A cross-tenant key written during the downgrade's build refuses cleanly.

    The failed concurrent unique build would otherwise leave an invalid but still
    enforced global index that rejects the 0.5 writers' legitimate inserts.
    """
    url, engine = pg
    await asyncio.to_thread(_alembic, url, "upgrade", PRIOR)
    await _seed(engine)
    await asyncio.to_thread(_alembic, url, "upgrade", R1)
    loop = asyncio.get_running_loop()
    reached = asyncio.Event()
    release = threading.Event()
    thread_id: int | None = None

    def before(conn, cursor, statement, parameters, context, executemany):  # noqa: ANN001
        if threading.get_ident() != thread_id:
            return
        if "ix_agent_runs_idempotency_key" in statement and statement.lstrip().upper().startswith(
            "CREATE UNIQUE INDEX"
        ):
            loop.call_soon_threadsafe(reached.set)
            assert release.wait(20)

    def migrate() -> None:
        nonlocal thread_id
        thread_id = threading.get_ident()
        _alembic(url, "downgrade", PRIOR)

    event.listen(engine.sync_engine.__class__, "before_cursor_execute", before)
    task = asyncio.create_task(asyncio.to_thread(migrate))
    try:
        await asyncio.wait_for(reached.wait(), 20)
        async with engine.begin() as conn:  # a 0.5 writer, after the shared-key check
            await conn.execute(
                text(
                    "INSERT INTO agent_runs (id, tenant_id, agent_name, status, "
                    "iteration_count, delegation_level, total_input_tokens, "
                    "total_output_tokens, total_cache_read_tokens, "
                    "total_cache_creation_tokens, cancel_requested, idempotency_key, "
                    "idempotency_fingerprint, created_at, updated_at) VALUES ('late', 'B', "
                    "'a', 'success', 1, 0, 0, 0, 0, 0, false, 'k1', 'fp', now(), now())"
                )
            )
        release.set()
        with pytest.raises(RuntimeError, match="more than one tenant"):
            await asyncio.wait_for(asyncio.shield(task), 15)
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        event.remove(engine.sync_engine.__class__, "before_cursor_execute", before)
    indexes = await _indexes(engine)
    assert "ix_agent_runs_idempotency_key" not in indexes
    assert set(_R1_INDEXES) <= set(indexes)
    assert await _invalid_indexes(engine) == 0
    assert await _version(engine) == R1
    async with engine.begin() as conn:
        await conn.execute(text("UPDATE agent_runs SET idempotency_key = NULL WHERE id = 'late'"))
    await asyncio.to_thread(_alembic, url, "downgrade", PRIOR)
    assert await _version(engine) == PRIOR


async def test_r1_runs_on_an_empty_database_and_reruns_are_no_ops(pg: Any) -> None:
    url, engine = pg
    await asyncio.to_thread(_alembic, url, "upgrade", PRIOR)
    await asyncio.to_thread(_alembic, url, "upgrade", R1)
    first = await _indexes(engine)
    # Rerun the revision body on an already-migrated schema.
    await asyncio.to_thread(_alembic, url, "stamp", PRIOR)
    await asyncio.to_thread(_alembic, url, "upgrade", R1)
    assert await _indexes(engine) == first
    assert await _invalid_indexes(engine) == 0
    await asyncio.to_thread(_alembic, url, "downgrade", PRIOR)
    downgraded = await _indexes(engine)
    await asyncio.to_thread(_alembic, url, "stamp", R1)
    await asyncio.to_thread(_alembic, url, "downgrade", PRIOR)
    assert await _indexes(engine) == downgraded
    assert "claim_epoch" not in await _columns(engine)


async def test_second_revision_logs_the_same_lock_timeout_line(pg: Any, caplog: Any) -> None:
    """``e8f9a0b1c2d3`` refuses with the same message and the same structured line."""
    caplog.set_level(logging.WARNING)
    url, engine = pg
    await asyncio.to_thread(_alembic, url, "upgrade", PRIOR)
    await _seed(engine)
    await asyncio.to_thread(_alembic, url, "upgrade", R1)
    async with engine.connect() as blocker:
        transaction = await blocker.begin()
        await blocker.execute(text("SELECT id FROM agent_runs LIMIT 1"))
        try:
            with pytest.raises(RuntimeError, match=_TIMEOUT):
                await asyncio.wait_for(asyncio.to_thread(_alembic, url, "upgrade", "head"), 15)
        finally:
            await transaction.rollback()
    _assert_lock_timeout_logged(caplog, HEAD, "add_status_changed_at")
    assert await _version(engine) == R1
    await asyncio.to_thread(_alembic, url, "upgrade", "head")
    assert await _version(engine) == HEAD
