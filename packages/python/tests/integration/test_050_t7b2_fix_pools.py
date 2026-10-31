"""Closed-loop retirement and guards at SQLAlchemy's public connection APIs."""

import asyncio
import gc
import logging
import threading
import time
import uuid
import warnings

import pytest
from sqlalchemy import text

from dendrux import Agent
from dendrux.llm.mock import MockLLM
from dendrux.types import LLMResponse


async def test_repeated_asyncio_jobs_with_unclosed_agents_are_bounded_after_gc(
    engine, tmp_path, caplog, record_property
):
    pg = engine.dialect.name == "postgresql"
    url = (
        engine.url.render_as_string(hide_password=False)
        if pg
        else f"sqlite+aiosqlite:///{tmp_path}/jobs.db"
    )
    tag = "t7_closed_" + uuid.uuid4().hex
    options = (
        dict(
            pool_size=2,
            max_overflow=0,
            connect_args=dict(server_settings=dict(application_name=tag)),
        )
        if pg
        else {}
    )
    errors, agents, counts, engine_ids, remaining_threads = [], [], [], [], []
    baseline_threads = set(threading.enumerate())
    caplog.set_level(logging.ERROR)

    def jobs():
        async def job():
            a = Agent(
                name="loop-job",
                prompt="p",
                provider=MockLLM([LLMResponse(text="ok")]),
                database_url=url,
                database_options=options,
                strict_tenancy=False,
            )
            agents.append(a)  # intentionally retain the unclosed holder after its loop closes
            result = await a.run("go", tenant_id=None)
            assert result.answer == "ok"
            target = a._private_engine
            engine_ids.append(id(target))
            if pg:
                async with target.connect() as conn:
                    counts.append(
                        (
                            await conn.execute(
                                text(
                                    "SELECT count(*) FROM pg_stat_activity "
                                    "WHERE datname=current_database() "
                                    "AND application_name=:tag"
                                ),
                                dict(tag=tag),
                            )
                        ).scalar_one()
                    )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ResourceWarning)
            automatic_gc = gc.isenabled()
            gc.disable()
            try:
                try:
                    for _ in range(12):
                        try:
                            asyncio.run(job())
                        except Exception as exc:
                            errors.append(type(exc).__name__ + ": " + str(exc))
                        # Collect outside a temporary loop: aiosqlite's GC stop
                        # must not enqueue a result onto a loop about to close.
                        gc.collect()
                finally:
                    for a in agents:
                        asyncio.run(a.close())
                    gc.collect()
                    deadline = time.monotonic() + 5
                    for thread in threading.enumerate():
                        if (
                            thread not in baseline_threads
                            and thread is not threading.current_thread()
                        ):
                            thread.join(timeout=max(0, deadline - time.monotonic()))
                            if thread.is_alive():
                                remaining_threads.append(thread.name)
            finally:
                if automatic_gc:
                    gc.enable()

    await asyncio.to_thread(jobs)
    assert not errors, "a closed owner loop must not poison subsequent jobs: " + str(errors)
    assert len(engine_ids) == len(set(engine_ids)) == 12
    if pg:
        assert max(counts) <= 2, "orphaned PG pools must be released after GC"
        async with engine.connect() as conn:
            assert (
                await conn.execute(
                    text(
                        "SELECT count(*) FROM pg_stat_activity "
                        "WHERE datname=current_database() "
                        "AND application_name=:tag"
                    ),
                    dict(tag=tag),
                )
            ).scalar_one() == 0
        record_property("closed_loop_peak_connections", max(counts))
    else:
        assert not remaining_threads, "orphaned SQLite workers must stop after GC"
    assert not [r for r in caplog.records if "Event loop is closed" in r.getMessage() or r.exc_info]


async def test_normal_connect_guard_does_not_rely_on_connection_init_using_raw_connection(
    engine, monkeypatch
):
    if engine.dialect.name != "postgresql":
        pytest.skip("asyncpg loop ownership")
    a = Agent(
        name="public-guard",
        prompt="p",
        database_url=engine.url.render_as_string(hide_password=False),
        strict_tenancy=False,
    )
    try:
        await a._resolve_state_store()
        target = a._private_engine

        def acquisition_must_not_happen():
            raise AssertionError("foreign-loop normal connect reached driver acquisition")

        with monkeypatch.context() as patch:
            patch.setattr(target.sync_engine, "raw_connection", acquisition_must_not_happen)

            def foreign():
                async def connect():
                    async with target.connect():
                        pass

                asyncio.run(connect())

            with pytest.raises(RuntimeError, match="different event loop"):
                await asyncio.to_thread(foreign)
    finally:
        await a.close()


async def test_loop_shutdown_drains_cancelled_run_before_retiring_pool(engine, tmp_path, caplog):
    from dendrux.runtime.stores._factory import store_for_engine
    from tests._helpers.scripted_provider import ScriptedLLM

    pg = engine.dialect.name == "postgresql"
    url = (
        engine.url.render_as_string(hide_password=False)
        if pg
        else f"sqlite+aiosqlite:///{tmp_path}/cancel.db"
    )
    holder = []
    task_holder = []

    def job():
        async def start():
            llm = ScriptedLLM([LLMResponse(text="late")], gate_at=0)
            a = Agent(
                name="shutdown", prompt="p", provider=llm, database_url=url, strict_tenancy=False
            )
            holder.append(a)
            task = asyncio.create_task(a.run("go", tenant_id=None))
            task_holder.append(task)
            await asyncio.wait_for(llm.entered.wait(), 20)

        asyncio.run(start())  # cancellation finishes its durable write before the loop closes

    caplog.set_level(logging.ERROR)
    await asyncio.to_thread(job)
    assert task_holder[0].done()
    a = holder[0]
    # Inspect via the fixture for PG; the explicit file engine for SQLite.
    if pg:
        state = store_for_engine(engine)
        records = await state.list_runs(agent_name="shutdown")
    else:
        from sqlalchemy.ext.asyncio import create_async_engine

        check = create_async_engine(url)
        try:
            records = await store_for_engine(check).list_runs(agent_name="shutdown")
        finally:
            await check.dispose()
    assert len(records) == 1 and records[0].status == "cancelled"
    assert not [r for r in caplog.records if r.exc_info]
    await a.close()
