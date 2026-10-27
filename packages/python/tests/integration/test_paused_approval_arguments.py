"""A requested pause during approval execution keeps pending server arguments private."""

import json

from dendrux import Agent, tool
from dendrux.guardrails import PII
from dendrux.runtime.context import get_delegation_context
from dendrux.store import RunStore
from dendrux.types import LLMResponse
from tests._helpers.scripted_provider import ScriptedLLM, tool_calls


async def test_paused_approval_batch_hides_remaining_arguments(store):
    holder = {}

    @tool()
    async def send(to: str) -> str:
        await holder["agent"].request_pause(get_delegation_context().run_id, tenant_id="A")
        return "sent"

    @tool()
    async def send_later(to: str) -> str:
        return "sent"

    secret = "person@example.com"
    agent = Agent(
        prompt="p",
        state_store=store,
        tools=[send, send_later],
        require_approval=["send", "send_later"],
        guardrails=[PII(restore="all")],
        provider=ScriptedLLM(
            [
                tool_calls(("send", {"to": secret}, "c1"), ("send_later", {"to": secret}, "c2")),
                LLMResponse(text="done"),
            ]
        ),
    )
    holder["agent"] = agent
    reader = RunStore(store)
    try:
        paused = await agent.run("send to " + secret, tenant_id="A")
        result = await agent.submit_approval(
            paused.run_id, approved=True, expected_pause_id=paused.pause_id, tenant_id="A"
        )
        assert result.status.value == "paused"
        pending = await reader.get_pending(paused.run_id, tenant_id="A")
        assert pending.tool_calls
        assert secret not in json.dumps(pending.to_dict())
        fetched = await agent.get_result(paused.run_id, tenant_id="A")
        assert secret not in json.dumps(fetched.to_dict())
        events = await reader.get_events(paused.run_id, tenant_id="A")
        active = [e for e in events if e.event_type == "run.paused"][-1]
        view = await reader.get_event_view(paused.run_id, active, tenant_id="A")
        assert secret not in json.dumps(view.data)
        async with reader.follow(
            paused.run_id,
            tenant_id="A",
            after_sequence_index=active.sequence_index - 1,
            until_terminal=False,
            max_duration_s=2,
        ) as feed:
            async for item in feed:
                assert secret not in json.dumps(item.data)
                break
    finally:
        await agent.close()
