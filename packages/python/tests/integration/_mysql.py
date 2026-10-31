"""Private, disposable track-B databases; no server lifecycle operations."""

import os
import re
import uuid

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool

from dendrux.db._settings import seed_settings
from dendrux.db.models import Base
from dendrux.runtime.stores._factory import store_for_engine


@pytest_asyncio.fixture(params=["mysql", "mariadb"])
async def mysql_engine(request):
    family = request.param
    raw = os.environ.get(f"DENDRUX_TEST_{family.upper()}_URL")
    if not raw:
        pytest.skip(f"{family} test URL unset")
    url = make_url(raw)
    if (
        url.drivername != "mysql+asyncmy"
        or url.host != "127.0.0.1"
        or url.port != (53306 if family == "mysql" else 53307)
        or url.query
        or not re.fullmatch(r"dendrux_p5_b_[A-Za-z0-9_]+_test", url.database or "")
    ):
        raise pytest.UsageError("track B requires its private local MySQL-family URL")
    name = f"dendrux_p5_b_{uuid.uuid4().hex[:12]}_test"
    admin = create_async_engine(
        url._replace(database=None), poolclass=NullPool, hide_parameters=True
    )
    test_engine = create_async_engine(
        url.set(database=name),
        isolation_level="READ COMMITTED",
        connect_args={"init_command": "SET time_zone='+00:00'", "charset": "utf8mb4"},
        pool_size=20,
        max_overflow=0,
        hide_parameters=True,
    )
    try:
        async with admin.connect() as conn:
            await conn.execute(text(f"CREATE DATABASE `{name}`"))
        async with test_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            await conn.run_sync(lambda sync: seed_settings(sync, active=True))
        yield test_engine
    finally:
        await test_engine.dispose()
        try:
            async with admin.connect() as conn:
                await conn.execute(text(f"DROP DATABASE IF EXISTS `{name}`"))
        finally:
            await admin.dispose()


@pytest.fixture
def store(mysql_engine):
    return store_for_engine(mysql_engine)


def process_contender(url, jobs, barrier, results, actor):
    """Run ten callers per wave in a separate interpreter, returning safe metadata only."""
    import asyncio

    from dendrux.runtime.stores.sql.mysql import MySQLStateStore
    from dendrux.types import generate_ulid

    async def run():
        owned = MySQLStateStore.from_url(url, pool_size=10, max_overflow=0)
        try:
            await owned.get_run("absent")
            warm = await asyncio.gather(*(owned._engine.connect() for _ in range(10)))
            await asyncio.gather(*(conn.close() for conn in warm))
            for wave, job in enumerate(jobs):
                barrier.wait(timeout=30)

                async def call(index, job=job):
                    run_id = job["run_id"]
                    operation = job["operation"]
                    try:
                        if operation == "claim":
                            value = await owned.claim_paused_run(
                                run_id, expected_status="waiting_client_tool", expected_pause_id="p"
                            )
                            return (
                                {"kind": "lost"}
                                if value is False
                                else {"kind": "won", "epoch": value.claim_epoch}
                            )
                        if operation == "reclaim":
                            value = await owned.reclaim_run(
                                run_id,
                                tenant_id="tenant",
                                agent_name="recovery",
                                expected_claim_epoch=job["epoch"],
                                governance={},
                                governance_policy="raise",
                                skill_pins={},
                                loop_type="react",
                            )
                            return {"kind": "won", "epoch": value.claim_epoch}
                        if operation in {"finalize", "cancel"}:
                            value = await owned.finalize_with_event(
                                run_id,
                                status="cancelled" if operation == "cancel" else "success",
                                allowed_current_statuses=["running"],
                                event_type="run.cancelled"
                                if operation == "cancel"
                                else "run.completed",
                            )
                            return {"kind": "lost" if value is None else "won"}
                        if operation == "note":
                            value = await owned.add_note(
                                run_id,
                                note_id=generate_ulid(),
                                text="bounded",
                                tenant_id="tenant",
                                max_notes=1,
                            )
                            return {"kind": "won" if value else "lost"}
                        value = await owned.save_run_event(
                            run_id,
                            event_type="append",
                            event_id=job.get("event_id") or generate_ulid(),
                            data={"index": actor * 10 + index},
                        )
                        return {"kind": "won", "sequence": value, "index": actor * 10 + index}
                    except Exception as exc:
                        return {"kind": type(exc).__name__, "code": getattr(exc, "code", None)}

                values = await asyncio.gather(*(call(i) for i in range(10)))
                results.put((wave, actor, values))
        finally:
            await owned._engine.dispose()

    try:
        asyncio.run(run())
    except Exception as exc:
        results.put(("worker_error", actor, type(exc).__name__))
