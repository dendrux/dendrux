"""Shared explicit-URL pools, entry-point options and recovery on real PG."""

import asyncio
import importlib
import json
import uuid

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from dendrux import Agent
from dendrux.db import session
from dendrux.llm.mock import MockLLM
from dendrux.store import RunStore
from dendrux.types import LLMResponse
from tests._helpers.pg_wait import wait_pg_count


def agent(url, options=None):
    return Agent(
        name="pools",
        prompt="p",
        provider=MockLLM([LLMResponse(text="ok")]),
        database_url=url,
        database_options=options,
        strict_tenancy=False,
    )


@pytest.mark.parametrize("entry", ["agent", "store", "sweep", "env"])
@pytest.mark.parametrize("override", [False, True])
async def test_pool_defaults_and_options_on_every_entry(engine, monkeypatch, entry, override):
    if engine.dialect.name != "postgresql":
        pytest.skip("Postgres queue-pool settings")
    url = engine.url.render_as_string(hide_password=False)
    options = {"pool_pre_ping": False, "pool_recycle": 47} if override else {}
    expected = (not override, 47 if override else 1800)

    def check(target):
        assert (target.pool._pre_ping, target.pool._recycle) == expected

    if entry == "agent":
        a = agent(url, options)
        try:
            await a._resolve_state_store()
            check(a._private_engine)
        finally:
            await a.close()
    elif entry == "store":
        facade = RunStore.from_database_url(url, database_options=options)
        try:
            check(facade._owned_engine)
        finally:
            await facade.close()
    elif entry == "env":
        await session.reset_engine()
        try:
            check(await session.get_engine(url, database_options=options))
        finally:
            await session.reset_engine()
    else:
        mod = importlib.import_module("dendrux.runtime.sweep")
        original = mod._create_temp_store
        checked = []

        async def capture(*args, **kwargs):
            pair = await original(*args, **kwargs)
            check(pair[1])
            checked.append(True)
            return pair

        monkeypatch.setattr(mod, "_create_temp_store", capture)
        import datetime as dt

        await mod.sweep(
            database_url=url, stale_running=dt.timedelta(days=1), database_options=options
        )
        assert checked


async def test_thousand_agents_one_pool_and_bounded_pg_connections(
    engine, monkeypatch, record_property
):
    import sqlalchemy.ext.asyncio as sa_async

    url = engine.url.render_as_string(hide_password=False)
    options = (
        {
            "pool_size": 3,
            "max_overflow": 0,
            "connect_args": {"server_settings": {"application_name": "dendrux_t7b2_1000"}},
        }
        if engine.dialect.name == "postgresql"
        else {}
    )
    create = sa_async.create_async_engine
    created = []
    disposed = []
    dispose = AsyncEngine.dispose

    def counted(*args, **kwargs):
        assert not created, "1000 Agents must reuse one pool rather than create a second"
        candidate = create(*args, **kwargs)
        created.append(candidate)
        return candidate

    async def counted_dispose(self, *args, **kwargs):
        disposed.append(id(self))
        return await dispose(self, *args, **kwargs)

    monkeypatch.setattr(sa_async, "create_async_engine", counted)
    monkeypatch.setattr(AsyncEngine, "dispose", counted_dispose)
    agents = [agent(url, options) for _ in range(1000)]
    try:
        outcomes = await asyncio.gather(
            *(a._resolve_state_store() for a in agents), return_exceptions=True
        )
        errors = [x for x in outcomes if isinstance(x, BaseException)]
        assert not errors, str(errors[0]) if errors else ""
        assert len(created) == 1
        assert all(a._private_engine is created[0] for a in agents)
        if engine.dialect.name == "postgresql":

            async def read(a):
                async with a._private_engine.connect() as conn:
                    return (await conn.execute(text("SELECT pg_backend_pid()"))).scalar_one()

            pids = await asyncio.gather(*(read(a) for a in agents))
            async with engine.connect() as conn:
                count = (
                    await conn.execute(
                        text(
                            "SELECT count(*) FROM pg_stat_activity "
                            "WHERE datname=current_database() "
                            "AND application_name='dendrux_t7b2_1000'"
                        )
                    )
                ).scalar_one()
            assert len(set(pids)) <= count <= 3
            record_property(
                "shared_pool_1000",
                json.dumps(
                    {"agents": 1000, "pools": len(created), "connections": count, "pool_size": 3}
                ),
            )
    finally:
        await asyncio.gather(*(a.close() for a in agents))
    assert disposed.count(id(created[0])) == 1


async def test_normalized_urls_and_parallel_close_are_reference_counted(tmp_path, monkeypatch):
    from sqlalchemy import text

    url = f"sqlite+aiosqlite:///{tmp_path}/shared.db"
    a, b = agent(url), agent(f"sqlite+aiosqlite:///{tmp_path}/./shared.db")
    await asyncio.gather(a._resolve_state_store(), b._resolve_state_store())
    shared = a._private_engine
    try:
        assert shared is b._private_engine, "normalized URL aliases must reuse the engine"
        await asyncio.gather(a.close(), a.close())
        async with b._private_engine.connect() as conn:
            assert (await conn.execute(text("SELECT 1"))).scalar_one() == 1
        await asyncio.gather(b.close(), b.close())
        c = agent(url)
        try:
            await c._resolve_state_store()
            assert c._private_engine is not shared
        finally:
            await c.close()
    finally:
        await asyncio.gather(a.close(), b.close())


async def test_run_recovers_after_its_idle_backends_are_terminated(engine, record_property):
    if engine.dialect.name != "postgresql":
        pytest.skip("Postgres connection termination")
    from tests._helpers.scripted_provider import ScriptedLLM

    tag = "dendrux_t7b2_recovery"
    llm = ScriptedLLM([LLMResponse(text="recovered")], gate_at=0)
    a = Agent(
        name="recover",
        prompt="p",
        provider=llm,
        database_url=engine.url.render_as_string(hide_password=False),
        database_options={
            "pool_size": 2,
            "max_overflow": 0,
            "connect_args": {"server_settings": {"application_name": tag}},
        },
        strict_tenancy=False,
    )
    task = asyncio.create_task(a.run("go", tenant_id=None))
    try:
        await asyncio.wait_for(llm.entered.wait(), 20)
        async with engine.connect() as conn:
            pids = (
                (
                    await conn.execute(
                        text(
                            "SELECT pid FROM pg_stat_activity WHERE datname=current_database() "
                            "AND application_name=:tag"
                        ),
                        {"tag": tag},
                    )
                )
                .scalars()
                .all()
            )
            assert pids
            for pid in pids:
                assert (
                    await conn.execute(text("SELECT pg_terminate_backend(:pid)"), {"pid": pid})
                ).scalar_one()
        llm.gate.set()
        result = await asyncio.wait_for(task, 30)
        assert result.status.value == "success" and result.answer == "recovered"
        record_property("terminated_backends_recovered", len(pids))
    finally:
        llm.gate.set()
        await asyncio.gather(task, return_exceptions=True)
        await a.close()


async def test_cancelled_initializer_does_not_cancel_another_holder(tmp_path, monkeypatch):
    from dendrux.db import _shared_pool, _sqlite_sync

    entered, finish, second = asyncio.Event(), asyncio.Event(), asyncio.Event()
    initialize = _sqlite_sync.create_sqlite_schema
    acquire = _shared_pool.acquire
    calls = 0

    async def gated(engine):
        entered.set()
        await finish.wait()
        await initialize(engine)

    async def observed(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            second.set()
        return await acquire(*args, **kwargs)

    monkeypatch.setattr(_sqlite_sync, "create_sqlite_schema", gated)
    monkeypatch.setattr(_shared_pool, "acquire", observed)
    url = f"sqlite+aiosqlite:///{tmp_path}/cancel.db"
    a, b = agent(url), agent(url)
    first = asyncio.create_task(a._resolve_state_store())
    await asyncio.wait_for(entered.wait(), 20)
    other = asyncio.create_task(b._resolve_state_store())
    try:
        await asyncio.wait_for(second.wait(), 20)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        finish.set()
        assert await asyncio.wait_for(other, 20) is not None
        assert b._private_engine in _shared_pool._by_engine
    finally:
        finish.set()
        await asyncio.gather(first, other, return_exceptions=True)
        await asyncio.gather(a.close(), b.close())
    assert not any(key[0].database == str(tmp_path / "cancel.db") for key in _shared_pool._entries)


async def test_process_exit_retires_pool_and_closes_pg_connections(engine):
    if engine.dialect.name != "postgresql":
        pytest.skip("Postgres process-exit cleanup")
    import sys

    tag = "t7_exit_" + uuid.uuid4().hex
    code = r"""
import asyncio, sys
from dendrux import Agent
from dendrux.llm.mock import MockLLM
async def main():
    a = Agent(prompt="p", provider=MockLLM([]), database_url=sys.argv[1],
              database_options=dict(connect_args=dict(server_settings=dict(application_name=sys.argv[2]))))
    await a._resolve_state_store()
    print("ready", flush=True)
    await asyncio.to_thread(sys.stdin.readline)
asyncio.run(main())
"""
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        code,
        engine.url.render_as_string(hide_password=False),
        tag,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        assert await asyncio.wait_for(proc.stdout.readline(), 20) == b"ready\n"

        async def count():
            async with engine.connect() as conn:
                return int(
                    (
                        await conn.execute(
                            text(
                                "SELECT count(*) FROM pg_stat_activity WHERE application_name=:tag "
                                "AND datname=current_database()"
                            ),
                            {"tag": tag},
                        )
                    ).scalar_one()
                )

        await wait_pg_count(count, 1)
        proc.stdin.write(b"exit\n")
        await proc.stdin.drain()
        out, err = await asyncio.wait_for(proc.communicate(), 20)
        assert proc.returncode == 0, err.decode()
        assert b"exception was never retrieved" not in err.lower()
        await wait_pg_count(count, 0)
    finally:
        if proc.returncode is None:
            proc.terminate()
            await proc.wait()


async def test_pooled_runtime_has_no_session_level_sql(engine):
    if engine.dialect.name != "postgresql":
        pytest.skip("Postgres runtime SQL audit")
    import re

    from sqlalchemy import event

    from tests._helpers.scripted_provider import ScriptedLLM

    llm = ScriptedLLM([LLMResponse(text="done")], gate_at=0)
    a = Agent(
        name="sql-audit",
        prompt="p",
        provider=llm,
        database_url=engine.url.render_as_string(hide_password=False),
        database_options={
            "connect_args": {"statement_cache_size": 0, "prepared_statement_cache_size": 0}
        },
        strict_tenancy=False,
    )
    statements = []

    def capture(conn, cursor, statement, parameters, context, executemany):
        statements.append(statement)

    event.listen(engine.sync_engine.__class__, "before_cursor_execute", capture)
    task = asyncio.create_task(a.run("go", tenant_id=None))
    try:
        await asyncio.wait_for(llm.entered.wait(), 20)
        async with engine.connect() as conn:
            run_id = (
                await conn.execute(text("SELECT id FROM agent_runs WHERE agent_name='sql-audit'"))
            ).scalar_one()
        await a._lazy_store.add_note(run_id, note_id="N", text="audit", max_notes=100)
        llm.gate.set()
        assert (await asyncio.wait_for(task, 30)).status.value == "success"
    finally:
        llm.gate.set()
        await asyncio.gather(task, return_exceptions=True)
        await a.close()
        event.remove(engine.sync_engine.__class__, "before_cursor_execute", capture)
    assert any("pg_advisory_xact_lock" in s for s in statements)
    forbidden = re.compile(
        r"^\s*(?:SET\s|LISTEN\b|UNLISTEN\b)|\bpg_advisory_(?:lock|unlock)\s*\(", re.I
    )
    assert not [s for s in statements if forbidden.search(s)], (
        "runtime connections must not carry session state"
    )


async def test_parallel_creation_and_close_retains_a_live_holder(tmp_path, monkeypatch):
    from dendrux.db import _shared_pool

    url = f"sqlite+aiosqlite:///{tmp_path}/interleaved.db"
    holder = agent(url)
    await holder._resolve_state_store()
    shared = holder._private_engine
    disposed = []
    original = AsyncEngine.dispose

    async def observed(self, *args, **kwargs):
        disposed.append(self)
        await original(self, *args, **kwargs)

    monkeypatch.setattr(AsyncEngine, "dispose", observed)

    async def churn():
        for _ in range(10):
            a = agent(url)
            try:
                await a._resolve_state_store()
                assert a._private_engine is shared
                await asyncio.sleep(0)  # deliberately interleave release/acquire tasks
            finally:
                await a.close()

    try:
        await asyncio.gather(*(churn() for _ in range(25)))
        assert disposed == []
        assert _shared_pool._entries[_shared_pool._by_engine[shared]].refs == 1
    finally:
        await holder.close()
    assert disposed == [shared]
    assert shared not in _shared_pool._by_engine


@pytest.mark.parametrize("entry", ["env", "agent"])
@pytest.mark.parametrize("raw", [False, True])
@pytest.mark.parametrize("disposed", [False, True])
async def test_wrong_loop_is_refused_before_driver_ping(engine, monkeypatch, entry, raw, disposed):
    if engine.dialect.name != "postgresql":
        pytest.skip("asyncpg loop-bound pooled connections")
    from sqlalchemy.util.concurrency import greenlet_spawn

    url = engine.url.render_as_string(hide_password=False)
    a = None
    await session.reset_engine()
    try:
        if entry == "env":
            target = await session.get_engine(url)
        else:
            a = agent(url)
            await a._resolve_state_store()
            target = a._private_engine
        async with target.connect() as conn:
            assert (await conn.execute(text("SELECT 1"))).scalar_one() == 1
        if disposed:
            await target.dispose()  # the guard must survive replacement of the pool
        pinged = []
        original = target.sync_engine.dialect.do_ping

        def ping(connection):
            pinged.append(asyncio.get_running_loop())
            return original(connection)

        monkeypatch.setattr(target.sync_engine.dialect, "do_ping", ping)

        def other_loop():
            async def use():
                if raw:
                    connection = await target.raw_connection()
                    await greenlet_spawn(connection.close)
                else:
                    async with target.connect() as conn:
                        await conn.execute(text("SELECT 1"))

            asyncio.run(use())

        with pytest.raises(RuntimeError, match="different event loop"):
            await asyncio.to_thread(other_loop)
        assert not pinged, "wrong-loop rejection must precede driver pre-ping"
        async with target.connect() as conn:
            assert (await conn.execute(text("SELECT 1"))).scalar_one() == 1
    finally:
        if a is not None:
            await a.close()
        await session.reset_engine()
