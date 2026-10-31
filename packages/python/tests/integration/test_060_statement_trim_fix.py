"""Intent-read boundaries and custom-store checkpoint fallback."""

from unittest.mock import AsyncMock

import pytest

from dendrux import Agent, tool
from dendrux.llm.mock import MockLLM
from dendrux.runtime._reclaim import needs_resolution
from dendrux.runtime.runner import _load_pause_state
from dendrux.runtime.state import ClaimResult, EventWrite, claimed_store
from dendrux.types import LLMResponse, PauseState, RunStatus, ToolCall, UsageStats, generate_ulid


@pytest.mark.parametrize("targets", ["all_client", "mixed", "missing_target", "absent_call"])
async def test_intent_read_requires_every_pending_target_to_be_client(store, targets):
    run_id = generate_ulid()
    await store.create_run(run_id, "intent-boundary")
    client = ToolCall("client_read", id="client")
    server = ToolCall("server_work", id="server")
    pending = [
        {"call_id": client.id, "target": "client"},
        {"call_id": server.id, "target": "client" if targets == "all_client" else "server"},
    ]
    if targets == "missing_target":
        pending[1].pop("target")
    elif targets == "absent_call":
        pending.pop()
    if targets != "all_client":
        await store.start_tool_call(
            run_id,
            claim_epoch=0,
            row_id=generate_ulid(),
            tool_call_id=server.id,
            provider_tool_call_id=None,
            tool_name=server.name,
            target="server",
            params=None,
            iteration_index=1,
            idempotent=False,
            event=EventWrite(
                generate_ulid(),
                "tool.started",
                1,
                {"call_id": server.id, "tool_name": server.name, "idempotent": False},
                server.id,
                None,
            ),
        )
    pause = PauseState(
        agent_name="intent-boundary",
        pending_tool_calls=[client, server],
        history=[],
        steps=[],
        iteration=1,
        trace_order_offset=4,
        usage=UsageStats(),
        schema_version=2,
    )
    get_calls = AsyncMock(wraps=store.get_tool_calls)

    class ReadStore:
        get_tool_calls = get_calls

    assert await needs_resolution(ReadStore(), run_id, pause, checkpoint={"pending": pending}) == (
        targets != "all_client"
    )
    if targets == "all_client":
        get_calls.assert_not_awaited()
    else:
        get_calls.assert_awaited_once_with(run_id, iteration_index=1, limit=1000)


@pytest.mark.parametrize("omit_checkpoint", [False, True])
async def test_custom_claim_uses_checkpoint_or_falls_back_to_durable_read(store, omit_checkpoint):
    @tool(target="client")
    def client_read() -> str:
        """Read on the client."""
        return "unused"

    agent = Agent(
        name="checkpoint-fallback",
        prompt="p",
        state_store=store,
        tools=[client_read],
        provider=MockLLM([LLMResponse(tool_calls=[ToolCall("client_read", id="client")])]),
    )

    class CustomClaimStore:
        get_loop_state = AsyncMock(wraps=store.get_loop_state)

        def __getattr__(self, name):
            return getattr(store, name)

        async def claim_paused_run(self, run_id, *, expected_status):
            token = await store.claim_paused_run(run_id, expected_status=expected_status)
            assert isinstance(token, ClaimResult)
            return ClaimResult(
                token.claim_epoch,
                token.storage_format,
                None if omit_checkpoint else token.loop_state,
                token.submission,
            )

    try:
        result = await agent.run("go", tenant_id=None)
        assert result.status == RunStatus.WAITING_CLIENT_TOOL
        raw = await store.get_pause_state(result.run_id)
        checkpoint = await store.get_loop_state(result.run_id)
        custom = CustomClaimStore()
        token = await custom.claim_paused_run(
            result.run_id, expected_status=RunStatus.WAITING_CLIENT_TOOL.value
        )
        assert token.storage_format == 2
        owned = claimed_store(custom, result.run_id, token)
        pause = await _load_pause_state(owned, result.run_id, raw, agent, agent.provider)
        assert pause.iteration == checkpoint["iteration"]
        assert pause.trace_order_offset == checkpoint["trace_order_offset"]
        assert pause.provider_kwargs == checkpoint["provider_kwargs"]
        assert [m.content for m in pause.history if m.role.value == "user"] == ["go"]
        assert [call.id for call in pause.pending_tool_calls] == ["client"]
        if omit_checkpoint:
            custom.get_loop_state.assert_awaited_once_with(result.run_id)
        else:
            custom.get_loop_state.assert_not_awaited()
    finally:
        await agent.close()
