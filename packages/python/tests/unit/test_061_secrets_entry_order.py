"""Context scanning preserves the pre-entry-projection workload."""

import pytest

from dendrux import Agent
from dendrux.context_blocks import ContextBlock
from dendrux.llm.mock import MockLLM
from dendrux.loops.single import SingleCall
from dendrux.types import LLMResponse, RunStatus


@pytest.mark.parametrize("single", [False, True])
async def test_allowed_context_matches_original_scan_work(single):
    scans = []

    class Count:
        action = "block"

        async def scan(self, text):
            scans.append(len(text))
            return []

    agent = Agent(
        prompt="p",
        provider=MockLLM([LLMResponse(text="done")]),
        state_store=None,
        guardrails=[Count()],
        loop=SingleCall() if single else None,
    )
    try:
        result = await agent.run("x" * 50000, context=[ContextBlock("y" * 20000)], tenant_id="t")
        assert result.status == RunStatus.SUCCESS
        print({"single": single, "scans": scans, "characters": sum(scans)})
        assert scans == [1, 70034, 4]  # Captured from 08032496.
    finally:
        await agent.close()
