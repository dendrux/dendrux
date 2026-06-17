"""Reasoning / thinking through the dendrux runtime (OpenAI).

OpenAI counterpart to 25_reasoning.py — same PUBLIC runtime API
(``agent.run()`` / ``agent.stream()``). Uses the Responses provider, the only
OpenAI path that returns reasoning *text* (a summary); the Chat provider
reports ``reasoning_tokens`` only.

No tools here: OpenAI reasoning items aren't yet round-tripped across tool
turns, so this keeps to a single reasoning turn (see 25_reasoning.py for the
tool-using, per-step variant on Anthropic).

Run with:
    OPENAI_API_KEY=sk-... python examples/26_reasoning_openai.py
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from dotenv import load_dotenv

from dendrux import Agent
from dendrux.llm.openai_responses import OpenAIResponsesProvider
from dendrux.loops import SingleCall
from dendrux.types import RunEventType

# Load .env from repo root (examples/ → python/ → packages/ → dendrux/)
load_dotenv(Path(__file__).resolve().parents[3] / ".env")

PROMPT = (
    "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the "
    "ball. How much does the ball cost? Give the number and one line of why."
)


def _agent() -> Agent:
    return Agent(
        name="ReasoningDemoOpenAI",
        provider=OpenAIResponsesProvider(
            model="gpt-5.5", thinking=True, effort="medium", show_thinking=True
        ),
        loop=SingleCall(),
        prompt="Think it through, then answer concisely.",
    )


async def main() -> None:
    # --- agent.run(): reasoning surfaces in the result ---
    async with _agent() as agent:
        result = await agent.run(PROMPT)

    print("=== agent.run() ===")
    print("answer:", (result.answer or "").strip())
    print("reasoning_tokens:", result.usage.reasoning_tokens)
    # SingleCall has no per-step list, so the single call's reasoning is on meta.
    if result.meta.get("reasoning"):
        print("thinking:", result.meta["reasoning"][:160].strip())

    # --- agent.stream(): reasoning streams live, before the answer ---
    print("\n=== agent.stream() ===")
    async with _agent() as agent:
        stream = agent.stream(PROMPT)
        async with stream:
            in_reasoning = in_answer = False
            async for ev in stream:
                if ev.type == RunEventType.REASONING_DELTA:
                    if not in_reasoning:
                        print("[thinking] ", end="", flush=True)
                        in_reasoning = True
                    print(ev.text, end="", flush=True)
                elif ev.type == RunEventType.TEXT_DELTA:
                    if not in_answer:
                        print("\n[answer] ", end="", flush=True)
                        in_answer = True
                    print(ev.text, end="", flush=True)
                elif ev.type == RunEventType.RUN_COMPLETED:
                    print(f"\n\nreasoning_tokens={ev.run_result.usage.reasoning_tokens}")


if __name__ == "__main__":
    asyncio.run(main())
