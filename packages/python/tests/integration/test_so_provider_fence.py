"""Provider dispatch stops at a fresh durable ownership boundary."""

import pytest

from dendrux import Agent, SingleCall, tool
from dendrux.loops.react import ReActLoop
from dendrux.runtime.persistence import PersistenceRecorder
from dendrux.types import LLMResponse, ProviderCapabilities, RunStatus, StreamEvent, StreamEventType
from tests._fixtures.storage import pin_storage_format
from tests._helpers.h0_races import Gate, running, snapshot
from tests._helpers.scripted_provider import ScriptedLLM, tool_call


class NativeStreamProvider(ScriptedLLM):
    capabilities = ProviderCapabilities(
        supports_native_tools=True, supports_tool_call_ids=True, supports_streaming=True
    )

    async def complete_stream(self, messages, tools=None, **kwargs):
        response = await self.complete(messages, tools=tools, **kwargs)
        yield StreamEvent(type=StreamEventType.DONE, raw=response)


@pytest.mark.parametrize("storage_format", [1, 2])
@pytest.mark.parametrize("loop", [ReActLoop, SingleCall])
@pytest.mark.parametrize("stream", [False, True])
async def test_dispatch_checks_unchanged_checkpoint(
    store, engine, monkeypatch, caplog, loop, stream, storage_format
):
    await pin_storage_format(engine, storage_format)
    provider = NativeStreamProvider([LLMResponse(text="done")])
    agent = Agent(
        prompt="test", provider=provider, state_store=store, loop=loop(), strict_tenancy=False
    )
    gate = Gate()
    gate.install(monkeypatch, PersistenceRecorder, "on_llm_call_started")

    async def invoke():
        if not stream:
            return await agent.run("go")
        async with agent.stream("go") as events:
            async for _ in events:
                pass
        return events.result

    try:
        async with running(invoke(), gate) as task:
            await gate.entered.wait()
            run_id = gate.args[1]
            old = await store.get_run_control(run_id)
            assert await store.try_pause_run(
                run_id,
                status="paused",
                pause_data={"pause_id": "next"},
                claim_epoch=old.claim_epoch,
            )
            new = await store.claim_paused_run(run_id, expected_status="paused")
            assert new.claim_epoch == old.claim_epoch + 1
            before = await snapshot(engine)
            gate.release.set()
            result = await task
            assert provider.calls == 0
            assert result.status == RunStatus.RUNNING
            assert await snapshot(engine) == before
            assert any(
                getattr(record, "code", None) == "run_fenced"
                and getattr(record, "run_id", None) == run_id
                for record in caplog.records
            )
    finally:
        await agent.close()


@pytest.mark.parametrize("storage_format", [1, 2])
@pytest.mark.parametrize("stream", [False, True])
async def test_each_model_step_checks_ownership(store, engine, monkeypatch, stream, storage_format):
    await pin_storage_format(engine, storage_format)

    @tool()
    async def step() -> str:
        """Finish the first step before losing ownership."""
        return "ok"

    provider = NativeStreamProvider([tool_call("step", pid="one"), LLMResponse(text="done")])
    agent = Agent(
        prompt="test", provider=provider, tools=[step], state_store=store, strict_tenancy=False
    )
    gate = Gate()
    gate.install(
        monkeypatch,
        PersistenceRecorder,
        "on_llm_call_started",
        when=lambda recorder, run_id, iteration, **kwargs: iteration == 2,
    )

    async def invoke():
        if not stream:
            return await agent.run("go")
        async with agent.stream("go") as events:
            async for _ in events:
                pass
        return events.result

    try:
        async with running(invoke(), gate) as task:
            await gate.entered.wait()
            run_id = gate.args[1]
            old = await store.get_run_control(run_id)
            assert await store.try_pause_run(
                run_id,
                status="paused",
                pause_data={"pause_id": "next"},
                claim_epoch=old.claim_epoch,
            )
            new = await store.claim_paused_run(run_id, expected_status="paused")
            assert new.claim_epoch == old.claim_epoch + 1
            before = await snapshot(engine)
            gate.release.set()
            result = await task
            assert provider.calls == 1
            assert result.status == RunStatus.RUNNING
            assert await snapshot(engine) == before
    finally:
        await agent.close()
