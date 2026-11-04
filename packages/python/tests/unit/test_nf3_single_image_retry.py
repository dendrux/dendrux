"""A newly blocking image verdict stops SingleCall's second provider attempt."""

import pytest

from dendrux import Agent, SingleCall
from dendrux.guardrails import Finding
from dendrux.loops.base import BaseNotifier, BaseRecorder
from dendrux.loops.react import ReActLoop
from dendrux.strategies.native import NativeToolCalling
from dendrux.types import ImageViewState, LLMResponse, RunStatus
from tests.unit.test_loop_images import (
    ImageLLM,
    make_png,
    make_runtime,
    rejected,
    store_image,
    tool_msg,
    user_msg,
)


@pytest.mark.parametrize("loop_type", [ReActLoop, SingleCall])
async def test_image_guardrail_blocks_retry_before_provider_dispatch(loop_type):
    class Guard:
        guardrail_id = "test.retry"
        guardrail_version = "1"
        action = "block"

        async def scan_image(self, image):
            return [Finding("blocked", 0, 0, 1.0, "")] if self.guardrail_version == "2" else []

    guard = Guard()
    runtime = make_runtime()
    old = await store_image(runtime, make_png(seed=200))
    new = await store_image(runtime, make_png(seed=201))

    class Provider(ImageLLM):
        async def complete(self, messages, tools=None, **kwargs):
            guard.guardrail_version = "2"
            return await super().complete(messages, tools=tools, **kwargs)

    provider = Provider([rejected(old.image_id, located=True), LLMResponse(text="must not run")])
    recorded, notified, started = [], [], []

    class Recorder(BaseRecorder):
        async def on_llm_call_started(self, run_id, iteration, **kwargs):
            started.append(iteration)

        async def on_governance_event(self, run_id, event_type, iteration, data, **kwargs):
            recorded.append((event_type, data))

    class Notifier(BaseNotifier):
        async def on_governance_event(self, run_id, event_type, iteration, data, **kwargs):
            notified.append((event_type, data))

    loop = loop_type()
    agent = Agent(prompt="p", provider=provider, loop=loop, guardrails=[guard])
    result = await loop.run(
        agent=agent,
        provider=provider,
        strategy=NativeToolCalling(),
        user_input="go",
        initial_history=[tool_msg("shot", "old", old), user_msg("go", new)],
        image_runtime=runtime,
        image_view_state=ImageViewState(sent_through=1),
        recorder=Recorder(),
        notifier=Notifier(),
    )
    assert len(provider.calls) == 1
    assert started == [1]
    assert result.status == RunStatus.ERROR
    assert result.meta["failure_reason"] == "guardrail_blocked"
    assert result.iteration_count == 1
    assert result.answer is None
    assert result.usage.total_tokens == 0
    assert recorded == notified
    blocked = [data for kind, data in recorded if kind == "guardrail.blocked"]
    assert len(blocked) == 1
    assert blocked[0]["direction"] == "incoming_image"
