"""OpenRouter native-tool conformance smoke test (live, opt-in).

Heterogeneous upstreams can't be proven statically — this is the per-model
verification: offer a trivial ``echo`` tool, ask the model to call it, and
assert a parseable ToolCall comes back. Add a slug to MODELS when onboarding
a model whose tool support you want verified.

Skipped when OPENROUTER_API_KEY is not set — opt-in for CI and local dev.
"""

from __future__ import annotations

import os

import pytest

from dendrux.types import Message, Role, ToolDef

requires_openrouter_key = pytest.mark.skipif(
    not os.environ.get("OPENROUTER_API_KEY"),
    reason="OPENROUTER_API_KEY not set",
)

MODELS = [
    "deepseek/deepseek-chat",
]

ECHO_TOOL = ToolDef(
    name="echo",
    description="Echo the given text back verbatim.",
    parameters={
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    },
)


@requires_openrouter_key
@pytest.mark.parametrize("model", MODELS)
async def test_native_tool_call_round_trip(model: str) -> None:
    """The model returns a parseable native ToolCall for a trivial tool."""
    from dendrux.llm.openrouter import OpenRouterProvider

    async with OpenRouterProvider(model=model, max_tokens=200) as provider:
        response = await provider.complete(
            [
                Message(
                    role=Role.SYSTEM,
                    content="You must call the echo tool. Do not answer in text.",
                ),
                Message(role=Role.USER, content="Echo the text 'hello'."),
            ],
            [ECHO_TOOL],
            tool_choice="required",
        )

    assert response.tool_calls, f"{model} returned no tool call — check upstream routing"
    call = response.tool_calls[0]
    assert call.name == "echo"
    assert isinstance(call.params, dict)
    assert "hello" in str(call.params.get("text", "")).lower()
