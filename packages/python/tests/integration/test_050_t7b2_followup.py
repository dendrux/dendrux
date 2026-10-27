"""Reconnected results retain the live return policy without changing stored text."""

import pytest

from dendrux import Agent
from dendrux.guardrails import PII
from dendrux.llm.mock import MockLLM
from dendrux.types import LLMResponse


@pytest.mark.parametrize("restore", ["all", "allowlisted"])
async def test_fetched_answer_matches_live_with_pii(store, restore):
    agent = Agent(
        name="reconnect-answer",
        prompt="p",
        strict_tenancy=False,
        state_store=store,
        guardrails=[PII(restore=restore)],
        provider=MockLLM([LLMResponse(text="Sent to <<EMAIL_ADDRESS_1>>.")]),
    )
    try:
        live = await agent.run("Email alice@example.com", tenant_id="A")
        fetched = await agent.get_result(live.run_id, tenant_id="A")
        assert live.answer == "Sent to alice@example.com."
        assert fetched.answer == live.answer, "reconnecting must preserve the live answer policy"
        assert fetched.to_dict()["answer"] == live.to_dict()["answer"]
        assert (await store.get_run(live.run_id)).answer == "Sent to <<EMAIL_ADDRESS_1>>."
    finally:
        await agent.close()
