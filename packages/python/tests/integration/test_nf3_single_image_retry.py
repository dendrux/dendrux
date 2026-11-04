"""SingleCall retry stops on a new image guardrail block."""

import pytest

from dendrux import Agent, SingleCall, tool
from dendrux.guardrails import Finding
from dendrux.media import Image, ToolOutput
from dendrux.types import LLMResponse, Role, RunStatus
from tests._fixtures.storage import pin_storage_format
from tests._helpers.scripted_provider import tool_call
from tests.unit.test_loop_images import ImageLLM, make_png, rejected


@pytest.mark.parametrize("storage_format", [1, 2])
async def test_single_image_retry_block_on_public_entries(store, engine, storage_format):
    await pin_storage_format(engine, storage_format)

    @tool()
    async def shots() -> ToolOutput:
        """Return two distinct tool images."""
        return ToolOutput("shots", images=[Image.from_bytes(make_png(seed=n)) for n in (200, 201)])

    source = await Agent(
        prompt="p",
        provider=ImageLLM([tool_call("shots", pid="old"), LLMResponse(text="seed")]),
        tools=[shots],
        state_store=store,
        strict_tenancy=False,
    ).run("go")

    class Guard:
        guardrail_id = "test.retry"
        guardrail_version = "1"
        action = "block"

        async def scan_image(self, image):
            return [Finding("blocked", 0, 0, 1.0, "")] if self.guardrail_version == "2" else []

    guard = Guard()

    class Provider(ImageLLM):
        async def complete(self, messages, tools=None, **kwargs):
            if not self.calls:
                guard.guardrail_version = "2"
                prior = next(m for m in messages if m.role == Role.TOOL and m.images)
                self.script.insert(0, rejected(prior.images[0].image_id, located=True))
            return await super().complete(messages, tools=tools, **kwargs)

    provider = Provider([LLMResponse(text="must not run")])
    agent = Agent(
        prompt="p",
        provider=provider,
        loop=SingleCall(),
        guardrails=[guard],
        state_store=store,
        strict_tenancy=False,
    )
    result = await agent.retry(source.run_id)
    assert result.status == RunStatus.ERROR
    assert result.meta["failure_reason"] == "guardrail_blocked"
    assert len(provider.calls) == 1
    events = await store.get_run_events(result.run_id)
    assert sum(e.event_type == "guardrail.blocked" for e in events) == 1
