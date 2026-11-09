"""Allowed context preserves the established wire order and scanning workload."""

from pathlib import Path

import pytest

from dendrux import Agent
from dendrux.chat import ChatMessage
from dendrux.context_blocks import ContextBlock, fold_context
from dendrux.guardrails import PII
from dendrux.llm.mock import MockLLM
from dendrux.loops.single import SingleCall
from dendrux.media import Image, InMemoryBlobStore
from dendrux.types import LLMResponse, Role, RunStatus
from tests._fixtures.storage import pin_storage_format


@pytest.mark.parametrize("multimodal", [False, True])
@pytest.mark.parametrize("fmt", [1, 2])
@pytest.mark.parametrize("single", [False, True])
async def test_allowed_stable_context_keeps_original_placeholder_order(
    store, engine, fmt, single, multimodal
):
    await pin_storage_format(engine, fmt)
    provider = MockLLM([LLMResponse(text="okay") for _ in range(3)])
    agent = Agent(
        prompt="p",
        provider=provider,
        state_store=store,
        blob_store=InMemoryBlobStore(),
        guardrails=[PII(action="redact")],
        loop=SingleCall() if single else None,
    )
    png = (Path(__file__).resolve().parents[1] / "fixtures/images/zebra42.png").read_bytes()
    history = []
    context = [ContextBlock("account owner tom@example.com", placement="stable")]
    expected = fold_context(
        [],
        [ContextBlock("account owner <<EMAIL_ADDRESS_2>>", placement="stable")],
        "write to <<EMAIL_ADDRESS_1>>" + ("\n\ncc <<EMAIL_ADDRESS_3>>" if multimodal else ""),
    )[0].content
    try:
        for index in range(3):
            text = f"write to person{index}@example.com"
            content = [text, Image.from_bytes(png), "cc other@example.com"] if multimodal else text
            result = await agent.run(
                content, history=history or None, context=context, tenant_id="t"
            )
            assert result.status == RunStatus.SUCCESS
            first = next(m for m in provider.wire_messages[-1] if m.role == Role.USER)
            assert first.content == expected
            history.extend([ChatMessage.user(content), ChatMessage.assistant("okay")])
    finally:
        await agent.close()
