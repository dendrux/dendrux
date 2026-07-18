"""OpenRouter — open-source and premium models through one API.

OpenRouter is OpenAI-Chat-Completions wire-compatible, so `OpenRouterProvider`
is a thin preset over the same transport as `OpenAIProvider` — streaming,
native tool calling, usage capture, and the evidence layer all work unchanged.
On top it adds:

  - `OPENROUTER_API_KEY` resolution + attribution headers
  - `provider.require_parameters` routing (only upstreams that honor `tools`)
  - A call-time native-tools guard: passing tools to a model that can't do
    native function calling raises an explicit error instead of silently
    returning text that ignores your tools
  - `list_models()` — the catalog as typed snapshots you filter with plain
    Python (free vs paid, tools vs not, multimodal vs text-only, price,
    context length)

This example verifies each of those against the live API and doubles as the
reference for building model pickers / cost-tier logic on top of Dendrux.

Run with:
    OPENROUTER_API_KEY=sk-or-... python examples/29_openrouter.py
    (or put the key in the repo-root .env)
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from dotenv import load_dotenv

from dendrux import Agent, tool
from dendrux.llm.openrouter import OpenRouterProvider

load_dotenv(Path(__file__).resolve().parents[3] / ".env")

MODEL = "deepseek/deepseek-chat"


@tool()
async def add(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b


async def explore_catalog() -> str:
    """list_models(): typed snapshots, filtered with plain Python."""
    print("== list_models(): catalog exploration ==")
    async with OpenRouterProvider(model=MODEL) as provider:
        models = await provider.list_models()

    print(f"Catalog size: {len(models)} models")

    free_tool_models = [m for m in models if m.is_free and m.supports_tools]
    print(f"Free + native tools: {len(free_tool_models)}")
    for m in free_tool_models[:3]:
        print(f"  - {m.id} (ctx {m.context_length})")

    multimodal = [m for m in models if m.is_multimodal]
    text_only = [m for m in models if not m.is_multimodal]
    print(f"Multimodal: {len(multimodal)}, text-only: {len(text_only)}")

    cheap_big_ctx = sorted(
        (
            m
            for m in models
            if m.supports_tools
            and not m.is_free
            and (m.context_length or 0) >= 128_000
            and m.prompt_price is not None
        ),
        key=lambda m: m.prompt_price or 0,
    )
    print("Cheapest paid tool-capable models with >=128k context:")
    for m in cheap_big_ctx[:3]:
        print(f"  - {m.id}  (${(m.prompt_price or 0) * 1e6:.3f}/1M input tokens)")

    # A model without native tools — used below to demo the guard.
    no_tools = next(m for m in models if not m.supports_tools)
    print(f"Example model WITHOUT native tools: {no_tools.id}")
    return no_tools.id


async def run_with_tools() -> None:
    """Native tool calling through OpenRouter — same code path as OpenAI/Anthropic."""
    print(f"\n== agent.run() with tools on {MODEL} ==")
    agent = Agent(
        provider=OpenRouterProvider(model=MODEL),
        prompt="You are a calculator. Use the add tool to add numbers, then state the result.",
        tools=[add],
    )
    async with agent:
        result = await agent.run("What is 15 + 27?")
        print(f"Answer: {result.answer}")
        print(f"Steps: {result.iteration_count}, Tokens: {result.usage.total_tokens}")


async def run_via_recipe_string() -> None:
    """The `openrouter:model` recipe string — provider built for you."""
    print("\n== recipe string: Agent(provider='openrouter:<model>') ==")
    agent = Agent(
        provider=f"openrouter:{MODEL}",
        prompt="Answer in one short sentence.",
    )
    async with agent:
        result = await agent.run("What is the capital of France?")
        print(f"Answer: {result.answer}")


async def demo_native_tools_guard(no_tools_model: str) -> None:
    """Tools + incapable model → explicit error, not silent text."""
    print(f"\n== native-tools guard on {no_tools_model} ==")
    agent = Agent(
        provider=OpenRouterProvider(model=no_tools_model),
        prompt="You are a calculator.",
        tools=[add],
    )
    async with agent:
        try:
            await agent.run("What is 1 + 1?")
        except ValueError as exc:
            # The guard raises before any request is sent — the run fails
            # loudly instead of "succeeding" with an answer that silently
            # ignored the tools. Without tools, the same model runs fine.
            print(f"Guard raised as designed:\n  {exc}")


async def main() -> None:
    no_tools_model = await explore_catalog()
    await run_with_tools()
    await run_via_recipe_string()
    await demo_native_tools_guard(no_tools_model)


if __name__ == "__main__":
    asyncio.run(main())
