"""Independent submitters share a durable projected claim and terminal outcome."""

import asyncio
import json
import multiprocessing
import os

import pytest
from sqlalchemy import text

from dendrux import Agent
from dendrux.guardrails import SecretDetection
from dendrux.llm.mock import MockLLM
from dendrux.runtime.runner import resume_claimed
from dendrux.types import LLMResponse, RunStatus, ToolResult
from tests.integration.test_061_secrets_continuation import (
    SECRET,
    client_read,
    paused_agent,
    rows_are_clean,
)


def _submit_worker(url, run_id, call_id, pause_id, marker, locked, racing, release, lost):
    async def main():
        from sqlalchemy.ext.asyncio import create_async_engine

        from dendrux.errors import RunAlreadyClaimedError
        from dendrux.runtime.stores._factory import store_for_engine

        engine = create_async_engine(
            url,
            connect_args={
                "server_settings": {
                    "application_name": "dendrux-e2h-marker" if marker else "dendrux-e2h-racer"
                }
            },
        )
        store = store_for_engine(engine)
        provider = MockLLM([LLMResponse(text="must not dispatch")])
        agent = Agent(
            name="continuation",
            prompt="Answer briefly.",
            provider=provider,
            state_store=store,
            tools=[client_read],
            guardrails=[SecretDetection()],
        )
        if marker:
            original = store._write_tx

            async def commit(session, **kwargs):
                locked.set()
                assert await asyncio.to_thread(release.wait, 20)
                await original(session, **kwargs)
                os._exit(0)

            store._write_tx = commit
        else:
            assert await asyncio.to_thread(locked.wait, 20)
            racing.set()
        try:
            await agent.submit_tool_results(
                run_id,
                [
                    ToolResult(
                        name="client_read",
                        call_id=call_id,
                        payload=json.dumps(SECRET if marker else "clean"),
                    )
                ],
                tenant_id="t",
                expected_pause_id=pause_id,
            )
            raise AssertionError("submitter unexpectedly continued")
        except RunAlreadyClaimedError:
            assert not marker
            assert provider._call_count == 0
            lost.set()
        finally:
            await agent.close()
            await engine.dispose()

    asyncio.run(main())


@pytest.mark.parametrize("fmt", [1, 2])
async def test_two_process_projected_submit_crash_and_restart(store, engine, fmt):
    if engine.dialect.name != "postgresql":
        pytest.skip("Postgres process race")
    agent, _, run_id, raw, _ = await paused_agent(store, engine, fmt, guarded=True)
    context = multiprocessing.get_context("spawn")
    locked, racing, release, lost = [context.Event() for _ in range(4)]
    processes = [
        context.Process(
            target=_submit_worker,
            args=(
                engine.url.render_as_string(hide_password=False),
                run_id,
                raw["pending_tool_calls"][0]["id"],
                raw["pause_id"],
                marker,
                locked,
                racing,
                release,
                lost,
            ),
        )
        for marker in (True, False)
    ]
    try:
        for process in processes:
            process.start()
        assert await asyncio.to_thread(racing.wait, 20)
        async with asyncio.timeout(20):
            while True:
                async with engine.connect() as connection:
                    waiting = await connection.scalar(
                        text(
                            "SELECT count(*) FROM pg_stat_activity "
                            "WHERE datname = current_database() "
                            "AND application_name = 'dendrux-e2h-racer' "
                            "AND wait_event_type = 'Lock'"
                        )
                    )
                if waiting:
                    break
                await asyncio.sleep(0.05)
        release.set()
        for process in processes:
            await asyncio.to_thread(process.join, 25)
            assert process.exitcode == 0
        assert lost.is_set()
        record = await store.get_run(run_id)
        assert record.status == "running" and record.meta["dendrux.blocked"] is True
        await rows_are_clean(engine)
        provider = MockLLM([LLMResponse(text="must not dispatch")])
        restarted = Agent(
            name="continuation",
            prompt="Answer briefly.",
            provider=provider,
            state_store=store,
            tools=[client_read],
            on_governance_change="warn",
        )
        try:
            result = await resume_claimed(
                run_id,
                state_store=store,
                agent=restarted,
                provider=provider,
                pre_claim_status="waiting_client_tool",
                claim_epoch=record.claim_epoch,
            )
            assert result.status == RunStatus.ERROR and provider._call_count == 0
            assert (await store.get_run(run_id)).failure_reason == "guardrail_blocked"
        finally:
            await restarted.close()
    finally:
        release.set()
        for process in processes:
            if process.is_alive():
                process.terminate()
                await asyncio.to_thread(process.join, 5)
        await agent.close()
