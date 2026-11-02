"""The dispatch ownership read propagates refusal and leaves legacy runs alone."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from dendrux import Agent, SingleCall
from dendrux.errors import PersistenceError, RunFencedError
from dendrux.runtime.persistence import PersistenceRecorder
from dendrux.runtime.state import ClaimResult, RunControl, StoreCapabilities, claimed_store
from dendrux.types import LLMResponse, generate_ulid
from tests._helpers.scripted_provider import ScriptedLLM

RUN_ID = generate_ulid()


def recorder(control, *, storage_format=2):
    store = SimpleNamespace(
        capabilities=StoreCapabilities(checkpointing=True),
        get_run_control=AsyncMock(return_value=control),
        _requires_persisted_claim=True,
    )
    bound = claimed_store(store, RUN_ID, ClaimResult(3, storage_format, None, None))
    return PersistenceRecorder(bound, RUN_ID), store.get_run_control


@pytest.mark.parametrize(
    "control",
    [
        RunControl(claim_epoch=4, status="running"),
        RunControl(claim_epoch=3, status="paused"),
        RunControl(),
    ],
    ids=["new-owner", "no-longer-running", "missing-row"],
)
async def test_dispatch_read_refuses_lost_ownership(control, caplog):
    persistence, read = recorder(control)
    with pytest.raises(RunFencedError) as caught:
        await persistence.on_llm_call_started(RUN_ID, 1)
    assert caught.value.run_id == RUN_ID
    assert caught.value.claim_epoch == 3
    read.assert_awaited_once_with(RUN_ID)
    assert any(record.code == "run_fenced" and record.run_id == RUN_ID for record in caplog.records)


@pytest.mark.parametrize("error", [PersistenceError(retryable=False), asyncio.CancelledError()])
async def test_dispatch_read_failure_propagates(error):
    persistence, read = recorder(RunControl(claim_epoch=3, status="running"))
    read.side_effect = error
    with pytest.raises(type(error)) as caught:
        await persistence.on_llm_call_started(RUN_ID, 1)
    assert caught.value is error
    read.assert_awaited_once()


async def test_legacy_dispatch_does_not_read_ownership():
    persistence, read = recorder(RunControl(), storage_format=1)
    await persistence.on_llm_call_started(RUN_ID, 1)
    read.assert_not_awaited()


@pytest.mark.parametrize("stream", [False, True])
async def test_run_without_store_does_not_use_persistence(monkeypatch, stream):
    started = AsyncMock(side_effect=AssertionError("in-memory run used persistence"))
    monkeypatch.setattr(PersistenceRecorder, "on_llm_call_started", started)
    provider = ScriptedLLM([LLMResponse(text="done")])
    async with Agent(prompt="test", provider=provider, loop=SingleCall()) as agent:
        if stream:
            async with agent.stream("go") as events:
                async for _ in events:
                    pass
            result = events.result
        else:
            result = await agent.run("go")
    assert result.answer == "done"
    assert provider.calls == 1
    started.assert_not_awaited()
