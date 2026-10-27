"""All caller-supplied model text is checked before a run can be persisted."""

import pytest

from dendrux import Agent
from dendrux.chat import ChatMessage
from dendrux.context_blocks import ContextBlock
from dendrux.llm.mock import MockLLM
from dendrux.runtime import runner
from dendrux.types import Limits, LLMResponse


@pytest.mark.parametrize("field", ["history", "context", "combined"])
@pytest.mark.parametrize("payload", ["é" * 51, "x\x00y", "x\ud800y"])
@pytest.mark.parametrize("entry", ["run", "stream", "runner_run", "runner_stream", "retry"])
async def test_all_input_text_is_validated_before_rows(store, field, payload, entry):
    kwargs = (
        {"history": [ChatMessage.user(payload), ChatMessage.assistant("ok")]}
        if field == "history"
        else {"context": [ContextBlock(payload)]}
        if field == "context"
        else {
            "history": [ChatMessage.user(payload), ChatMessage.assistant("ok")],
            "context": [ContextBlock("x" * 50)],
        }
    )
    provider = MockLLM([LLMResponse(text="done")])
    agent = Agent(
        prompt="p", provider=provider, state_store=store, limits=Limits(max_input_bytes=100)
    )
    try:
        with pytest.raises(ValueError):
            if entry == "run":
                await agent.run("hi", **kwargs)
            elif entry == "stream":
                async for _ in agent.stream("hi", **kwargs):
                    pass
            elif entry == "runner_run":
                await runner.run(
                    agent, user_input="hi", provider=provider, state_store=store, **kwargs
                )
            elif entry == "runner_stream":
                async for _ in runner.run_stream(
                    agent, user_input="hi", provider=provider, state_store=store, **kwargs
                ):
                    pass
            else:
                await agent.retry("missing", **kwargs)
        assert await store.count_runs() == 0
        assert provider.call_history == []
    finally:
        await agent.close()


async def test_input_limit_counts_all_fields_together(store):
    agent = Agent(
        prompt="p",
        provider=MockLLM([LLMResponse(text="done")]),
        state_store=store,
        limits=Limits(max_input_bytes=100),
    )
    try:
        with pytest.raises(ValueError):
            await agent.run(
                "x" * 40,
                history=[ChatMessage.user("x" * 30), ChatMessage.assistant("x" * 10)],
                context=[ContextBlock("x" * 30)],
            )
        assert await store.count_runs() == 0
    finally:
        await agent.close()
