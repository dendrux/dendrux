"""OpenAI-compatible backend — a local OSS model via Ollama, no API key, no cost.

This is the pattern for running *any* OpenAI-compatible server (Ollama, vLLM,
SGLang, Groq, Together, LM Studio): point the same `OpenAIProvider` at its
`base_url`. Ollama exposes an OpenAI-compatible Chat Completions endpoint at
`http://localhost:11434/v1`, so the provider that talks to api.openai.com talks
to a local model with only a `base_url` change. Because the base_url is not the
official OpenAI endpoint, the provider sends `max_tokens` (not
`max_completion_tokens`) — the param compatible backends expect.

Tool calling on OSS models:
    Native tool calling is the same code path here as for OpenAI/Anthropic —
    dendrux always asks for and reads the structured `tool_calls` field; it
    never scrapes tool calls out of text. The *transport* is native everywhere.
    What varies is the *model*: strong models reliably emit a structured call,
    but small models (e.g. llama3.2:3b) sometimes answer with the call written
    as plain text instead — in which case there's no `tool_calls` to run and
    dendrux returns that text as the answer. Mitigations: pick a tool-capable
    model (qwen2.5:7b / llama3.1:8b are far more reliable than a 3B) and give a
    direct, tool-forcing prompt. This is a model property, not a dendrux or
    Ollama limitation.

Prereqs:
    ollama pull llama3.2:3b      # a small chat+tools model (default below)
    # Ollama serves on http://localhost:11434 by default

Run with:
    python examples/27_openai_compatible_ollama.py

Override the defaults if needed:
    OLLAMA_MODEL=qwen2.5:7b OLLAMA_BASE_URL=http://localhost:11434/v1 \
        python examples/27_openai_compatible_ollama.py
"""

from __future__ import annotations

import asyncio
import os

from dendrux import Agent, tool
from dendrux.llm.openai import OpenAIProvider
from dendrux.types import RunEventType

MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")


@tool()
async def add(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b


def build_agent() -> Agent:
    # api_key is required by the OpenAI client but unused by Ollama — any
    # non-empty placeholder works.
    return Agent(
        provider=OpenAIProvider(model=MODEL, base_url=BASE_URL, api_key="ollama"),
        prompt="You are a calculator. Use the add tool to add numbers, then state the result.",
        tools=[add],
    )


async def run_once() -> None:
    print(f"== agent.run() against {MODEL} @ {BASE_URL} ==")
    async with build_agent() as agent:
        result = await agent.run("What is 15 + 27?")
        print(f"Answer: {result.answer}")
        print(f"Steps: {result.iteration_count}, Tokens: {result.usage.total_tokens}")


async def run_streaming() -> None:
    print("\n== agent.stream() ==")
    async with build_agent() as agent:
        stream = agent.stream("Add 100 and 23, then tell me the total.")
        async with stream:
            async for event in stream:
                if event.type == RunEventType.TEXT_DELTA:
                    print(event.text, end="", flush=True)
                elif event.type == RunEventType.TOOL_USE_END:
                    params = event.tool_call.params if event.tool_call else {}
                    print(f"\n  >> {event.tool_call.name}({params})")
                elif event.type == RunEventType.TOOL_RESULT:
                    print(f"  << {event.tool_result.payload}")
                elif event.type == RunEventType.RUN_COMPLETED:
                    print()  # newline after the streamed answer


async def main() -> None:
    await run_once()
    await run_streaming()


if __name__ == "__main__":
    asyncio.run(main())
