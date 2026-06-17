"""Reasoning / thinking — surface a model's reasoning summary + token count.

Phase 1 wires reasoning through the provider's ``complete()`` path. This
example calls the Anthropic provider directly to show, for a thinking-enabled
call:
  - the answer text
  - the summarized thinking the model surfaced (``response.reasoning``)
  - the reasoning token count (billed within output tokens)
  - how many reasoning blocks were captured for multi-turn replay

It contrasts that with the default (thinking off), which surfaces nothing —
proving the feature is opt-in and backward-compatible.

(Surfacing reasoning through ``agent.run()`` lands with the runner roll-up +
``RunResult.reasoning`` step; this example exercises the provider path that is
wired today.)

Run with:
    ANTHROPIC_API_KEY=sk-... python examples/25_reasoning.py
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from dotenv import load_dotenv

from dendrux.llm.anthropic import AnthropicProvider
from dendrux.types import Message, Role, StreamEventType

# Load .env from repo root (examples/ → python/ → packages/ → dendrux/)
load_dotenv(Path(__file__).resolve().parents[3] / ".env")

PROMPT = (
    "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the "
    "ball. How much does the ball cost? Give the number and one line of why."
)


async def main() -> None:
    # --- Thinking ON: adaptive, high effort, summary visible ---
    async with AnthropicProvider(
        model="claude-sonnet-4-6",
        thinking=True,
        effort="high",
        show_thinking=True,
        max_tokens=2000,
    ) as provider:
        resp = await provider.complete([Message(role=Role.USER, content=PROMPT)])

    print("=== thinking ON (adaptive, effort=high) ===")
    print("\n[reasoning summary]")
    print(resp.reasoning or "(none returned)")
    print("\n[answer]")
    print(resp.text or "")
    print(
        f"\n[usage] output_tokens={resp.usage.output_tokens} "
        f"reasoning_tokens={resp.usage.reasoning_tokens} "
        f"reasoning_blocks={len(resp.reasoning_blocks or [])}"
    )

    # --- Thinking OFF (default): backward-compatible, nothing surfaced ---
    async with AnthropicProvider(model="claude-sonnet-4-6", max_tokens=2000) as provider:
        resp = await provider.complete([Message(role=Role.USER, content=PROMPT)])

    print("\n=== thinking OFF (default) ===")
    print(f"reasoning is None: {resp.reasoning is None}")
    print(f"reasoning_blocks is None: {resp.reasoning_blocks is None}")
    print("\n[answer]")
    print(resp.text or "")

    # --- Thinking ON, STREAMING: watch reasoning stream before the answer ---
    print("\n=== thinking ON, STREAMING ===")
    async with AnthropicProvider(
        model="claude-sonnet-4-6", thinking=True, effort="high", max_tokens=2000
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
