"""Streaming client validation is observable without terminating durable work."""

from dendrux.types import RunEventType, RunStatus, ToolResult
from tests.integration.test_tenant_run_control import _pause, _results


async def test_resume_validation_error_retains_waiting_state_and_allows_resubmission(store):
    paused = await _pause("client", store, "A")
    raw = await store.get_pause_state(paused.run_id)
    call = raw["pending_tool_calls"][0]
    invalid = ToolResult(call["name"], call["id"], "PRIVATE_VALIDATION\x00")
    try:
        before = await store.get_run_control(paused.run_id)
        async with paused.agent.resume_stream(
            paused.run_id,
            tool_results=[invalid],
            tenant_id="A",
            expected_pause_id=raw["pause_id"],
        ) as stream:
            events = [event async for event in stream]
        assert events[-1].type == RunEventType.RUN_ERROR
        assert events[-1].run_result.meta["error_type"] == "InvalidToolResultError"
        assert "PRIVATE_VALIDATION" not in events[-1].error
        current = await store.get_run_control(paused.run_id)
        assert current.status == "waiting_client_tool"
        assert current.claim_epoch == before.claim_epoch
        assert (await store.get_pause_state(paused.run_id))["pause_id"] == raw["pause_id"]
        result = await paused.agent.resume(
            paused.run_id,
            tool_results=await _results(store, paused.run_id),
            tenant_id="A",
            expected_pause_id=raw["pause_id"],
        )
        assert result.status == RunStatus.SUCCESS
    finally:
        await paused.agent.close()
