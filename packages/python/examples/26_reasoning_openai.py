"""Reasoning / thinking on OpenAI — surface the reasoning summary + token count.

The OpenAI counterpart to examples/25_reasoning.py. The same three controls
(thinking / effort / show_thinking) work across providers.

Uses the Responses API provider, because only the Responses API returns
reasoning *text* (a summary — never the raw chain-of-thought). The Chat
Completions provider (OpenAIProvider) reports ``reasoning_tokens`` only, no text.

Run with:
    OPENAI_API_KEY=sk-... python examples/26_reasoning_openai.py
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from dotenv import load_dotenv

from dendrux.llm.openai_responses import OpenAIResponsesProvider
from dendrux.types import Message, Role, StreamEventType

# Load .env from repo root (examples/ → python/ → packages/ → dendrux/)
load_dotenv(Path(__file__).resolve().parents[3] / ".env")

MODEL = "gpt-5.5"
PROMPT = (
    "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the "
    "ball. How much does the ball cost? Give the number and one line of why."
)


async def main() -> None:
    # --- Thinking ON: reasoning summary returned ---
    async with OpenAIResponsesProvider(
        model=MODEL,
        thinking=True,
        effort="medium",
        show_thinking=True,
        max_output_tokens=3000,
    ) as provider:
        resp = await provider.complete([Message(role=Role.USER, content=PROMPT)])

    print("=== thinking ON (effort=medium) ===")
    print("\n[reasoning summary]")
    print(resp.reasoning or "(none returned)")
    print("\n[answer]")
    print(resp.text or "")
    print(
        f"\n[usage] output_tokens={resp.usage.output_tokens} "
        f"reasoning_tokens={resp.usage.reasoning_tokens} "
        f"reasoning_blocks={len(resp.reasoning_blocks or [])}"
    )

    # --- Thinking OFF (default): backward-compatible, no summary surfaced ---
    async with OpenAIResponsesProvider(model=MODEL, max_output_tokens=3000) as provider:
        resp = await provider.complete([Message(role=Role.USER, content=PROMPT)])

    print("\n=== thinking OFF (default) ===")
    print(f"reasoning is None: {resp.reasoning is None}")
    print("\n[answer]")
    print(resp.text or "")

    # --- Thinking ON, STREAMING: reasoning summary streams before the answer ---
    print("\n=== thinking ON, STREAMING ===")
    async with OpenAIResponsesProvider(
        model=MODEL,
        thinking=True,
        effort="medium",
        show_thinking=True,
        max_output_tokens=3000,
    ) as provider:
        in_reasoning = in_answer = False
        async for ev in provider.complete_stream([Message(role=Role.USER, content=PROMPT)]):
            if ev.type == StreamEventType.REASONING_DELTA:
                if not in_reasoning:
                    print("[thinking] ", end="", flush=True)
                    in_reasoning = True
                print(ev.text, end="", flush=True)
            elif ev.type == StreamEventType.TEXT_DELTA:
                if not in_answer:
                    print("\n\n[answer] ", end="", flush=True)
                    in_answer = True
                print(ev.text, end="", flush=True)
            elif ev.type == StreamEventType.DONE:
                u = ev.raw.usage
                print(
                    f"\n\n[usage] output_tokens={u.output_tokens} "
                    f"reasoning_tokens={u.reasoning_tokens}"
                )


if __name__ == "__main__":
    asyncio.run(main())
