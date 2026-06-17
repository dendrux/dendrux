"""Reasoning / thinking through the dendrux runtime (Anthropic).

Thinking is turned on at the provider; everything else uses the PUBLIC runtime
API — ``agent.run()`` and ``agent.stream()``. The runtime surfaces reasoning
as a first-class feature:

  - per step:   ``RunResult.steps[i].meta["reasoning"]`` (the model's thinking)
  - rolled up:  ``RunResult.usage.reasoning_tokens``
  - streamed:   ``RunEventType.REASONING_DELTA`` events (live, before the answer)
  - persisted:  reasoning text + token counts land in the DB (see RunStore)

Run with:
    ANTHROPIC_API_KEY=sk-... python examples/25_reasoning.py
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from dotenv import load_dotenv

from dendrux import Agent, tool
from dendrux.llm.anthropic import AnthropicProvider
from dendrux.store import RunStore
from dendrux.types import Budget, RunEventType

# Load .env from repo root (examples/ → python/ → packages/ → dendrux/)
load_dotenv(Path(__file__).resolve().parents[3] / ".env")

# Persist to a local SQLite DB so we can read the reasoning back out after.
_DB_PATH = Path.home() / ".dendrux" / "reasoning_example.db"
_DB_URL = f"sqlite+aiosqlite:///{_DB_PATH}"


@tool()
async def get_population(city: str) -> str:
    """Return the population of a city."""
    return {"paris": "2.1 million", "tokyo": "14 million"}.get(city.lower(), "unknown")


def _agent() -> Agent:
    return Agent(
        name="ReasoningDemo",
        # Thinking is the only reasoning-specific config — it lives on the provider.
        provider=AnthropicProvider(model="claude-sonnet-4-6", thinking=True, effort="high"),
        prompt="Use get_population for each city before answering. Think it through.",
        tools=[get_population],
        budget=Budget(max_tokens=100_000),
        database_url=_DB_URL,
    )


async def main() -> None:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    # --- agent.run(): tool-using ReAct + thinking, persisted to SQLite ---
    async with _agent() as agent:
        result = await agent.run("Which has more people, Paris or Tokyo? Use the tool for both.")

    print("=== agent.run() (ReAct + tools + thinking + persistence) ===")
    print("answer:", (result.answer or "").strip())
    print(f"iterations={result.iteration_count}  reasoning_tokens={result.usage.reasoning_tokens}")
    for i, step in enumerate(result.steps):
        reasoning = step.meta.get("reasoning")
        if reasoning:
            print(f"  step {i} thinking: {reasoning[:100].strip()}...")

    # --- read the persisted reasoning back via the public RunStore ---
    async with RunStore.from_database_url(_DB_URL) as store:
        run = await store.get_run(result.run_id)
        calls = await store.get_llm_calls(result.run_id)
    print("\n=== persisted (RunStore) ===")
    print("run total_reasoning_tokens:", run.total_reasoning_tokens if run else None)
    print("per-call reasoning_tokens:", [c.reasoning_tokens for c in calls])
    print("per-call reasoning text present:", [bool(c.reasoning) for c in calls])

    # --- agent.stream(): reasoning streams live, before the answer ---
    print("\n=== agent.stream() ===")
    async with _agent() as agent:
        stream = agent.stream("What is Tokyo's population? Use the tool.")
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
