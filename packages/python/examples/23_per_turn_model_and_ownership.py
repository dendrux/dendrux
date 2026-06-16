"""Example 23: Per-turn model selection + provider recipe form + pooling.

Shows three things you want when building chatbots / web endpoints:

1. Per-turn model override — pass ``model=`` to ``run()`` to switch model
   for a single turn without rebuilding the agent. The model actually used
   is recorded per LLM call (correct billing/observability).

2. Provider recipe form — ``provider="anthropic:claude-haiku-4-5"`` builds
   the provider for you (mirrors ``database_url``), reading the API key from
   the environment. ``async with`` closes it. Pass a provider instance when
   you need full configuration.

3. Pooling — reuse one provider instance across many agents. ``close()``
   would close the shared provider, so do NOT call it per request; close
   the pooled provider once at shutdown.

See docs/recipes/web-endpoint.mdx for the full architecture.

Run:
    ANTHROPIC_API_KEY=sk-... python examples/23_per_turn_model_and_ownership.py
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from dotenv import load_dotenv

from dendrux import Agent
from dendrux.llm.anthropic import AnthropicProvider
from dendrux.store import RunStore

load_dotenv(Path(__file__).resolve().parents[3] / ".env")


async def per_turn_model(db_url: str) -> None:
    """One agent, two turns, two different models — recorded correctly."""
    print("\n=== 1. Per-turn model override (recipe-form provider) ===")
    # Recipe-string provider: the agent builds it, async with closes it.
    async with Agent(
        provider="anthropic:claude-haiku-4-5",  # default model
        prompt="Reply in one short sentence.",
        database_url=db_url,
        name="chat",
    ) as agent:
        default = await agent.run("Say hello.")
        overridden = await agent.run("Say hello.", model="claude-sonnet-4-6")

    async with RunStore.from_database_url(db_url) as store:
        for label, result in [("default", default), ("override", overridden)]:
            calls = await store.get_llm_calls(result.run_id)
            print(f"  {label:9} -> recorded model: {calls[0].model}")


async def pooled_provider() -> None:
    """One provider instance reused across many agents, closed once.

    close() would close the shared provider, so we do NOT call it per
    request — the lightweight agent is garbage-collected and we close the
    pooled provider once at shutdown. (In a real app you'd also share one
    engine via ``state_store=`` or ``DENDRUX_DATABASE_URL`` rather than a
    per-agent ``database_url``; these agents run ephemerally to stay focused.)
    """
    print("\n=== 2. Pooled provider (don't close() per request) ===")
    provider = AnthropicProvider(model="claude-haiku-4-5")  # built once, reused
    try:
        for i in (1, 2):
            agent = Agent(provider=provider, prompt="Reply in one word.")
            result = await agent.run("Ping?")  # no agent.close() -> provider stays open
            print(f"  request {i}: {result.answer!r}  (provider reused)")
    finally:
        await provider.close()  # close the pooled provider once
    print("  pooled provider closed at shutdown.")


async def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        db_url = f"sqlite+aiosqlite:///{tmp}/ex23.db"
        await per_turn_model(db_url)
        await pooled_provider()
    print("\nDone — recipe-form provider closed by `async with`;")
    print("pooled instance closed once at shutdown.")


if __name__ == "__main__":
    asyncio.run(main())
