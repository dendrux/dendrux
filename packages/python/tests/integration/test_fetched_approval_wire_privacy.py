"""Fetched approval reasons use the live result's placeholder policy."""

import json

from dendrux import Agent, tool
from dendrux.guardrails import PII
from dendrux.types import LLMResponse, ToolDecision
from tests._helpers.scripted_provider import ScriptedLLM, tool_call


@tool()
async def send_mail(to: str) -> str:
    return "sent"


async def test_fetched_hook_approval_reason_is_redacted_without_public_mapping(store):
    private = "alice.secret@example.com"

    async def before_tool(ctx):
        return ToolDecision.require_approval(f"approve mail to {ctx.params['to']}")

    provider = ScriptedLLM(
        [
            tool_call("send_mail", {"to": "<<EMAIL_ADDRESS_1>>"}, pid="mail"),
            LLMResponse(text="done"),
        ]
    )
    async with Agent(
        prompt="p",
        tools=[send_mail],
        provider=provider,
        before_tool=before_tool,
        guardrails=[PII()],
        state_store=store,
        strict_tenancy=False,
    ) as agent:
        live = await agent.run(f"mail {private}", tenant_id="A")
        fetched = await agent.get_result(live.run_id, tenant_id="A")
        assert private not in json.dumps(live.to_dict())
        assert private not in json.dumps(fetched.to_dict())
        assert fetched.to_dict()["pending"] == live.to_dict()["pending"]
        assert "pii_mapping" not in fetched.meta
        assert "__dendrux_origins__" not in json.dumps(fetched.to_dict())
