"""Submission validation runs before claims and media writes."""

from unittest.mock import AsyncMock

import pytest

from dendrux.errors import InvalidToolResultError
from dendrux.types import ToolResult
from tests.integration.test_tenant_run_control import _pause


@pytest.mark.parametrize("stream", [False, True])
async def test_stale_token_is_never_echoed(store, stream):
    paused = await _pause("input", store, "A")
    token = "TOKEN_SENTINEL_" * 1000
    try:
        if stream:
            events = [
                e
                async for e in paused.agent.resume_stream(
                    paused.run_id, user_input="go", expected_pause_id=token, tenant_id="A"
                )
            ]
            assert token not in str(events[-1].run_result.error)
        else:
            with pytest.raises(ValueError) as caught:
                await paused.agent.resume(
                    paused.run_id, user_input="go", expected_pause_id=token, tenant_id="A"
                )
            assert token not in str(caught.value)
        assert (await store.get_run(paused.run_id)).status == "waiting_human_input"
    finally:
        await paused.agent.close()


async def test_preclaim_name_validation_precedes_transaction(store, monkeypatch):
    paused = await _pause("client", store, "A")
    raw = await store.get_pause_state(paused.run_id)
    call = raw["pending_tool_calls"][0]
    claim = AsyncMock(side_effect=AssertionError("claim opened before validating name"))
    monkeypatch.setattr(store, "submit_and_claim", claim)
    try:
        from dendrux.runtime.submit import validate_tool_results

        with pytest.raises(InvalidToolResultError):
            await validate_tool_results(
                store, paused.run_id, [ToolResult(name="wrong", call_id=call["id"], payload='"ok"')]
            )
        claim.assert_not_awaited()
    finally:
        await paused.agent.close()


@pytest.mark.parametrize("stream", [False, True])
async def test_invalid_results_precede_media_ingestion(store, monkeypatch, stream):
    from dendrux.runtime import runner

    paused = await _pause("client", store, "A")
    raw = await store.get_pause_state(paused.run_id)
    call = raw["pending_tool_calls"][0]
    ingest = AsyncMock(side_effect=AssertionError("media stored before validating results"))
    monkeypatch.setattr(runner, "_ingest_resume_images", ingest)
    result = ToolResult(name="wrong", call_id=call["id"], payload='"ok"', images=(object(),))
    try:
        with pytest.raises(InvalidToolResultError):
            if stream:
                async for _ in paused.agent.resume_stream(
                    paused.run_id, tool_results=[result], tenant_id="A"
                ):
                    pass
            else:
                await paused.agent.resume(paused.run_id, tool_results=[result], tenant_id="A")
        ingest.assert_not_awaited()
        assert (await store.get_run(paused.run_id)).status == "waiting_client_tool"
    finally:
        await paused.agent.close()
