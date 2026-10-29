"""Failure-path durability, idempotency and Postgres metadata concurrency."""

import asyncio

import pytest
from sqlalchemy import func, select, update
from sqlalchemy.exc import OperationalError

from dendrux.db.models import AgentRun, TokenUsage
from dendrux.errors import ProviderTimeoutError
from dendrux.runtime.persistence import PersistenceRecorder
from dendrux.runtime.runner import _persist_task_cancellation
from dendrux.types import LLMResponse, UsageStats
from tests._helpers.scripted_provider import tool_call
from tests.integration.test_050_t7_integrity import agent, consume, result_for
from tests.integration.test_050_t7b_wiring import FailingProvider

PRIVATE = "vendor-private-request-echo"


@pytest.mark.parametrize("entry", ["run", "stream", "retry", "resume", "resume_stream", "submit"])
@pytest.mark.parametrize("fault", ["finalize", "llm.failed"])
async def test_secondary_persistence_failure_keeps_typed_provider_error_and_safe_logs(
    store, monkeypatch, caplog, entry, fault
):
    llm = FailingProvider(
        [tool_call("client_read", pid="a")]
        if entry in {"resume", "resume_stream", "submit"}
        else []
    )
    a = agent(store, llm)
    pause = await a.run("go") if llm._responses else None
    original = (
        await agent(store, [LLMResponse(text="prior")]).run("go") if entry == "retry" else None
    )
    saved_finalize, saved_event = store.finalize_run, store.save_run_event

    async def finalize(*args, **kwargs):
        if fault == "finalize" and kwargs.get("status") == "error":
            raise RuntimeError("secondary database failure")
        return await saved_finalize(*args, **kwargs)

    async def event(*args, **kwargs):
        if fault == "llm.failed" and kwargs.get("event_type") == "llm.failed":
            raise RuntimeError("secondary evidence failure")
        return await saved_event(*args, **kwargs)

    error = ProviderTimeoutError(
        message=PRIVATE, provider="fixture", timeout_reason="call_deadline"
    )
    error.__cause__ = RuntimeError(PRIVATE)

    async def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(llm, "complete", fail)
    monkeypatch.setattr(store, "finalize_run", finalize)
    monkeypatch.setattr(store, "save_run_event", event)
    if entry in {"stream", "resume_stream"}:
        result, _ = await consume(
            a.stream("go")
            if entry == "stream"
            else a.resume_stream(pause.run_id, tool_results=[result_for(pause)])
        )
        assert result.error_info is not None and result.error_info.category == "timeout", (
            "secondary evidence must not replace ProviderError"
        )
    else:
        action = (
            a.run("go")
            if entry == "run"
            else a.retry(original.run_id)
            if entry == "retry"
            else a.resume(pause.run_id, tool_results=[result_for(pause)])
            if entry == "resume"
            else a.submit_tool_results(pause.run_id, [result_for(pause)])
        )
        with pytest.raises(ProviderTimeoutError) as caught:
            await action
        assert caught.value is error
    assert PRIVATE not in caplog.text, (
        "secondary persistence logs must not format vendor exception chains"
    )


async def test_ambiguous_usage_commit_has_one_row(store, monkeypatch):
    await store.create_run("usage", agent_name="a", input_data={})
    original = store.save_usage
    attempts = 0

    async def ambiguous(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        await original(*args, **kwargs)
        if attempts == 1:
            raise OperationalError("fixture", {}, Exception("lost commit acknowledgement"))

    monkeypatch.setattr(store, "save_usage", ambiguous)
    await PersistenceRecorder(store, "usage").on_llm_call_completed(
        "usage", LLMResponse(text="ok", usage=UsageStats(input_tokens=3, cost_usd=0.4)), 1
    )
    async with store._session_factory() as session:
        count, cost = (
            await session.execute(
                select(func.count(), func.sum(TokenUsage.cost_usd)).where(
                    TokenUsage.agent_run_id == "usage"
                )
            )
        ).one()
    assert attempts == 2
    assert count == 1 and float(cost) == 0.4, (
        "an ambiguous committed usage retry must not double bill"
    )


async def test_cancel_result_discloses_failed_persistence(store, monkeypatch):
    await store.create_run("00000000000000000000000103", agent_name="a", input_data={})

    async def fail(*args, **kwargs):
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(store, "finalize_run", fail)
    result = await _persist_task_cancellation(store, "00000000000000000000000103", interrupted=True)
    assert (await store.get_run("00000000000000000000000103")).status == "running"
    assert result.meta.get("dendrux.cancel_persisted") is False, (
        "CANCELLED must disclose a failed durable transition"
    )


async def test_stale_epoch_error_info_cannot_write_metadata(store):
    await store.create_run("stale-error", agent_name="a", input_data={}, meta={"keep": True})
    async with store._session_factory() as session:
        await session.execute(
            update(AgentRun).where(AgentRun.id == "stale-error").values(claim_epoch=18)
        )
        await session.commit()
    assert not await store.finalize_run(
        "stale-error",
        status="error",
        error="stale",
        error_info={"category": "timeout"},
        expected_current_status="running",
        claim_epoch=17,
    )
    row = await store.get_run("stale-error")
    assert row.status == "running" and row.meta == {"keep": True} and row.error is None


async def test_postgres_concurrent_meta_patches_preserve_error_info(store, engine, monkeypatch):
    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy.sql.selectable import Select

    if engine.dialect.name != "postgresql":
        pytest.skip("independent Postgres row locks")
    await store.create_run("meta-race", agent_name="a", input_data={})
    original = AsyncSession.execute
    entered, release = asyncio.Event(), asyncio.Event()
    held = False

    async def execute(session, statement, *args, **kwargs):
        nonlocal held
        result = await original(session, statement, *args, **kwargs)
        if isinstance(statement, Select) and "agent_runs" in str(statement) and not held:
            held = True
            entered.set()
            await release.wait()
        return result

    monkeypatch.setattr(AsyncSession, "execute", execute)
    patch = asyncio.create_task(store.update_run_meta("meta-race", {"dendrux.patch": 1}))
    await asyncio.wait_for(entered.wait(), 5)
    finish = asyncio.create_task(
        store.finalize_run(
            "meta-race",
            status="error",
            error="safe",
            error_info={"category": "timeout"},
            expected_current_status="running",
        )
    )
    # Let the finalizer compete while the patch transaction retains its read.
    await asyncio.sleep(0.05)
    release.set()
    await asyncio.wait_for(asyncio.gather(patch, finish), 5)
    row = await store.get_run("meta-race")
    assert row.meta == {"dendrux.patch": 1, "dendrux.error_info": {"category": "timeout"}}, (
        "metadata read/modify/write must serialize with finalization"
    )


async def test_sweep_index_migration_roundtrip(engine, tmp_path):
    from contextlib import asynccontextmanager

    from sqlalchemy import inspect, text
    from sqlalchemy.ext.asyncio import create_async_engine

    from tests.integration._migration_schema import (
        create_pre_activation_schema,
        migration_schema,
        test_database_url,
    )
    from tests.integration.test_migration_run_control import _alembic

    names = {
        "ix_agent_runs_sweep_running",
        "ix_agent_runs_sweep_waiting",
        "ix_run_events_started",
    }

    @asynccontextmanager
    async def target():
        if engine.dialect.name == "postgresql":
            async with migration_schema(test_database_url(), "t7_sweep_migration") as migrated:
                yield migrated
        else:
            migrated = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/migration.db")
            try:
                yield migrated
            finally:
                await migrated.dispose()

    async with target() as migrated:
        async with migrated.begin() as conn:
            await conn.run_sync(create_pre_activation_schema)
            for name in names:
                await conn.execute(text(f'DROP INDEX IF EXISTS "{name}"'))
            await conn.execute(
                AgentRun.__table__.insert().values(id="kept", agent_name="a", status="running")
            )
        url = migrated.url.render_as_string(hide_password=False)
        await asyncio.to_thread(_alembic, url, "stamp", "d7e8f9a0b1c2")
        await asyncio.to_thread(_alembic, url, "upgrade", "head")

        async def indexes():
            async with migrated.connect() as conn:
                if conn.dialect.name == "sqlite":
                    return set(
                        (
                            await conn.execute(
                                text("SELECT name FROM sqlite_master WHERE type = 'index'")
                            )
                        ).scalars()
                    )
                return await conn.run_sync(
                    lambda c: {
                        i["name"]
                        for table in ("agent_runs", "run_events")
                        for i in inspect(c).get_indexes(table)
                    }
                )

        assert names <= await indexes(), "migration must install all bounded sweep access paths"
        await asyncio.to_thread(_alembic, url, "downgrade", "d7e8f9a0b1c2")
        assert not names & await indexes()
        await asyncio.to_thread(_alembic, url, "upgrade", "head")
        assert names <= await indexes()
        async with migrated.connect() as conn:
            assert (await conn.execute(select(AgentRun.id))).scalars().all() == ["kept"]


@pytest.mark.parametrize("entry", ["run", "resume", "submit"])
async def test_public_cancel_reports_failed_durable_write(store, monkeypatch, entry):
    from dendrux.types import RunStatus
    from tests._helpers.scripted_provider import ScriptedLLM

    llm = ScriptedLLM(
        ([tool_call("client_read", pid="a")] if entry != "run" else [])
        + [LLMResponse(text="unused")],
        gate_at=0 if entry == "run" else 1,
    )
    a = agent(store, llm)
    pause = await a.run("go") if entry != "run" else None
    action = (
        a.run("go")
        if entry == "run"
        else a.resume(pause.run_id, tool_results=[result_for(pause)])
        if entry == "resume"
        else a.submit_tool_results(pause.run_id, [result_for(pause)])
    )
    task = asyncio.create_task(action)
    await asyncio.wait_for(llm.entered.wait(), 5)
    rid = pause.run_id if pause else next(iter(a._task_manager._owned))
    original = store.finalize_run

    async def fail(*args, **kwargs):
        if kwargs.get("status") == "cancelled":
            raise RuntimeError("database unavailable")
        return await original(*args, **kwargs)

    monkeypatch.setattr(store, "finalize_run", fail)
    await a.cancel_run(rid)
    result = await task
    assert (
        result.status == RunStatus.CANCELLED
        and result.meta.get("dendrux.cancel_persisted") is False
    )
    assert (await store.get_run(rid)).status == "running"
    await original(rid, status="cancelled", expected_current_status="running")


@pytest.mark.parametrize("resumed", [False, True])
async def test_stream_cancel_fallback_discloses_unknown_persistence(store, monkeypatch, resumed):
    from tests._helpers.scripted_provider import ScriptedLLM

    llm = ScriptedLLM([tool_call("client_read", pid="a"), LLMResponse(text="unused")])
    a = agent(store, llm)
    pause = await a.run("go") if resumed else None
    stream = (
        a.resume_stream(pause.run_id, tool_results=[result_for(pause)])
        if resumed
        else a.stream("go")
    )
    iterator = stream.__aiter__()
    await anext(iterator)

    async def fail(*args, **kwargs):
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(store, "finalize_run", fail)
    monkeypatch.setattr(store, "get_run", fail)
    await stream.aclose()
    await iterator.aclose()
    assert stream.result.meta.get("dendrux.cancel_persisted") is False, (
        "fallback cancellation must disclose unknown durability"
    )


async def test_concurrent_usage_identity_records_exactly_once(store):
    await store.create_run("usage-race", agent_name="a", input_data={})
    await asyncio.gather(
        *(
            store.save_usage(
                "usage-race",
                iteration_index=1,
                usage=UsageStats(input_tokens=3, cost_usd=0.4),
                usage_id="same-logical-call",
            )
            for _ in range(8)
        )
    )
    async with store._session_factory() as session:
        count, tokens, cost = (
            await session.execute(
                select(
                    func.count(), func.sum(TokenUsage.input_tokens), func.sum(TokenUsage.cost_usd)
                ).where(TokenUsage.agent_run_id == "usage-race")
            )
        ).one()
    assert count == 1 and tokens == 3 and float(cost) == 0.4
