"""Independent run owners stopped at the real provider-start hook."""

import asyncio
import logging
import os
from datetime import timedelta
from unittest.mock import patch

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from dendrux import Agent, tool
from dendrux.runtime.context import get_delegation_context
from dendrux.runtime.persistence import PersistenceRecorder
from dendrux.runtime.stores._factory import store_for_engine
from dendrux.types import LLMResponse
from tests._helpers.scripted_provider import ScriptedLLM


def fence_agent(store, provider):
    @tool()
    async def step() -> str:
        ctx = get_delegation_context()
        await agent.request_pause(ctx.run_id)
        return "stepped"

    agent = Agent(
        name="provider-fence",
        prompt="test",
        tools=[step],
        provider=provider,
        state_store=store,
        strict_tenancy=False,
        on_governance_change="raise",
        lease_interval=timedelta(minutes=5),
    )
    return agent


def provider_fence_worker(url, run_id, pause_id, pipe):
    async def main():
        engine = create_async_engine(url, pool_size=1, max_overflow=0)
        store = store_for_engine(engine)
        provider = ScriptedLLM([LLMResponse(text="winner")])
        agent = fence_agent(store, provider)
        codes = []

        class Codes(logging.Handler):
            def emit(self, record):
                if getattr(record, "run_id", None) == run_id:
                    codes.append(getattr(record, "code", None))

        handler = Codes()
        logger = logging.getLogger("dendrux")
        logger.addHandler(handler)
        hook = PersistenceRecorder.on_llm_call_started

        async def at_provider_start(recorder, *args, **kwargs):
            # Renew while still the real owner. Renewal must not substitute for
            # the ownership read after all processes leave this rendezvous.
            async with recorder.lease_scope():
                pass
            async with engine.connect() as conn:
                backend_pid = await conn.scalar(text("SELECT pg_backend_pid()"))
            pipe.send(
                {
                    "phase": "ready",
                    "pid": os.getpid(),
                    "backend_pid": backend_pid,
                    "epoch": recorder._store.claim_epoch,
                }
            )
            assert pipe.recv() == "release"
            await hook(recorder, *args, **kwargs)

        try:
            with patch.object(PersistenceRecorder, "on_llm_call_started", at_provider_start):
                result = await agent.submit_resume(
                    run_id, tenant_id="tenant", expected_pause_id=pause_id
                )
            pipe.send(
                {
                    "phase": "finished",
                    "calls": provider.calls,
                    "status": result.status.value,
                    "fenced": "run_fenced" in codes,
                }
            )
        finally:
            logger.removeHandler(handler)
            await agent.close()
            await engine.dispose()

    try:
        asyncio.run(main())
    except BaseException:
        import traceback

        pipe.send({"error": traceback.format_exc()})
        raise
    finally:
        pipe.close()
