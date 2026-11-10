"""Reclaim admission precedes history limits and preserves ownership."""

import asyncio
import multiprocessing
from types import SimpleNamespace

import pytest
from sqlalchemy.ext.asyncio import create_async_engine

from dendrux.errors import RunNotFoundError, RunNotReclaimableError
from dendrux.llm.mock import MockLLM
from dendrux.runtime.stores._factory import store_for_engine
from dendrux.types import Limits, LLMResponse, RunStatus
from tests._helpers.recovery_checkpoint import checkpoint
from tests.integration.test_061_secrets_continuation import SECRET
from tests.integration.test_061_secrets_reclaim import _agent, _legacy_checkpoint, _mark


@pytest.mark.parametrize("legacy", [False, True])
async def test_blocked_reclaim_precedes_history_limits(store, engine, monkeypatch, caplog, legacy):
    provider = MockLLM([LLMResponse(text="unreachable")])
    agent = _agent(store, [], provider, limits=Limits(max_rebuild_rows=1))
    item = await (
        _legacy_checkpoint(agent, store, engine) if legacy else checkpoint(agent, store, engine)
    )
    if not legacy:
        await _mark(store, item.run_id)
    reads = []
    original = store.conversation_traces

    async def history(*args, **kwargs):
        reads.append(True)
        return await original(*args, **kwargs)

    monkeypatch.setattr(store, "conversation_traces", history)
    try:
        result = await agent.reclaim(item.run_id, tenant_id="tenant", item=item)
        assert result.status == RunStatus.ERROR
        assert result.meta["failure_reason"] == "guardrail_blocked"
        assert reads == [] and provider._call_count == 0
        assert SECRET not in repr(result) and SECRET not in caplog.text
        assert (await store.get_run(item.run_id)).failure_reason == "guardrail_blocked"
    finally:
        await agent.close()


async def test_clean_reclaim_still_enforces_history_limit(store, engine):
    agent = _agent(store, [], limits=Limits(max_rebuild_rows=1))
    item = await checkpoint(agent, store, engine)
    try:
        with pytest.raises(RunNotReclaimableError) as caught:
            await agent.reclaim(item.run_id, tenant_id="tenant", item=item)
        assert caught.value.reason == "history_too_large"
    finally:
        await agent.close()


async def test_foreign_reclaim_is_refused_before_blocked_admission(store, engine, monkeypatch):
    agent = _agent(store, [], limits=Limits(max_rebuild_rows=1))
    item = await checkpoint(agent, store, engine)
    await _mark(store, item.run_id)
    before = await store.get_run(item.run_id)

    async def forbidden(*args, **kwargs):
        raise AssertionError("foreign scope reached admission")

    monkeypatch.setattr(store, "_blocked_run_ids", forbidden)
    try:
        with pytest.raises(RunNotFoundError):
            await agent.reclaim(item.run_id, tenant_id="foreign", item=item)
        after = await store.get_run(item.run_id)
        assert (after.status, after.claim_epoch) == (before.status, before.claim_epoch)
    finally:
        await agent.close()


def _reclaim_worker(url, run_id, epoch, ready, gate, answers):
    async def work():
        engine = create_async_engine(url)
        store = store_for_engine(engine)
        provider = MockLLM([LLMResponse(text="unreachable")])
        effects = []
        agent = _agent(store, effects, provider, limits=Limits(max_rebuild_rows=1))
        ready.put(True)
        await asyncio.to_thread(gate.wait, 20)
        try:
            try:
                result = await agent.reclaim(
                    run_id,
                    tenant_id="tenant",
                    item=SimpleNamespace(run_id=run_id, claim_epoch=epoch),
                )
                answer = {
                    "status": result.status.value,
                    "reason": result.meta.get("failure_reason"),
                }
            except RunNotReclaimableError as exc:
                answer = {"refused": exc.reason}
            answers.put({**answer, "calls": provider._call_count, "effects": effects})
        finally:
            await agent.close()
            await engine.dispose()

    asyncio.run(work())


async def test_two_processes_reclaim_before_history_rebuild(store, engine):
    if engine.dialect.name != "postgresql":
        pytest.skip("Postgres process race")
    seed = _agent(store, [])
    item = await checkpoint(seed, store, engine)
    await seed.close()
    await _mark(store, item.run_id)
    context = multiprocessing.get_context("spawn")
    ready, answers, gate = context.Queue(), context.Queue(), context.Event()
    processes = [
        context.Process(
            target=_reclaim_worker,
            args=(
                engine.url.render_as_string(hide_password=False),
                item.run_id,
                item.claim_epoch,
                ready,
                gate,
                answers,
            ),
        )
        for _ in range(2)
    ]
    for process in processes:
        process.start()
    try:
        for _ in processes:
            assert await asyncio.to_thread(ready.get, True, 20)
        gate.set()
        outcomes = [await asyncio.to_thread(answers.get, True, 30) for _ in processes]
    finally:
        gate.set()
        for process in processes:
            await asyncio.to_thread(process.join, 30)
            if process.is_alive():
                process.terminate()
                process.join()
            assert process.exitcode == 0
        ready.close()
        answers.close()
    assert (
        sum(o.get("status") == "error" and o.get("reason") == "guardrail_blocked" for o in outcomes)
        == 1
    )
    assert sum("refused" in o for o in outcomes) == 1
    assert all(o["calls"] == 0 and o["effects"] == [] for o in outcomes)
    assert len(await store.get_run_events(item.run_id, event_types=["run.error"])) == 1
