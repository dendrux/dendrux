"""Context blocks demo — per-run context + cross-run cache hits.

Shows the "context out of prompt" pattern: project instructions live in
``context=`` as a ``stable`` block (not in the agent's ``prompt=``), so they
fold into the FIRST user message and sit in the cached prefix. Re-supply the
same stable block every turn and Anthropic reads it from cache on turn 2+.

    Turn 1 → cache WRITE (cache_creation_input_tokens > 0)
    Turn 2 → cache READ  (cache_read_input_tokens > 0)   ← the win

Dynamic context (retrieved docs for this turn) folds into the CURRENT user
message instead — volatile tail, re-sent each turn, never cached across turns.

The stable block is deliberately long so the cached prefix crosses Anthropic's
per-model minimum (2,048 tokens for Sonnet 4.6). Below it, caching is silently
skipped. OpenAI auto-caches prefixes >= 1,024 tokens but the hit is
eventually-consistent, so its cache_read may lag by a call or two.

Run with (keys in .env at repo root):
    python examples/28_context_blocks.py
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

from dotenv import load_dotenv

from dendrux import Agent, ContextBlock
from dendrux.chat import ChatMessage
from dendrux.llm.anthropic import AnthropicProvider
from dendrux.llm.openai import OpenAIProvider
from dendrux.loops.single import SingleCall

if TYPE_CHECKING:
    from dendrux.llm.base import LLMProvider
    from dendrux.types import RunResult

load_dotenv(Path(__file__).resolve().parents[3] / ".env")


# A project's runtime instructions — the kind of thing you'd normally be
# tempted to shove into the agent prompt. It belongs in context= (stable):
# it's project state, not agent identity. Repeated sections keep it above the
# cache minimum without needing a real 3k-token document on hand.
_SECTION = """
## {n}. Engineering policy section {n}

Testing: every change ships with tests written before the implementation
(types first, then tests, then code, then refactor). New code carries at
least 80% line coverage. The full check suite must pass before any commit;
no exceptions for "trivial" changes, because trivial changes are exactly the
ones that regress silently. Prefer table-driven tests for pure functions and
end-to-end tests for anything crossing a process or network boundary.

Code review: all changes land through a branch and a pull request; never push
to the main branch directly. Reviews check correctness first, then clarity,
then consistency with the surrounding code — match the file's existing naming,
comment density, and idioms rather than importing your personal style. A review
is not a rubber stamp: at least one reviewer must be able to explain what the
change does and why it is safe.

Architecture: each module depends only on the layers below it. Side effects
live at the edges; the core is pure and deterministic so it can be tested
without mocks. Persisted state is written once, atomically, and never rewritten
in place. Public APIs are typed and documented; private helpers are neither
unless the logic is non-obvious.
"""

PROJECT_INSTRUCTIONS = "# Project Orbit — Engineering Handbook\n" + "\n".join(
    _SECTION.format(n=i) for i in range(1, 13)
)


def _fmt(label: str, r: RunResult) -> str:
    u = r.usage
    write = u.cache_creation_input_tokens
    read = u.cache_read_input_tokens
    return (
        f"  {label:<8} input={u.input_tokens:>6}  "
        f"cache_write={('-' if write is None else write):>6}  "
        f"cache_read={('-' if read is None else read):>6}"
    )


async def demo(provider: LLMProvider, label: str) -> None:
    print(f"\n=== {label} ===")
    agent = Agent(
        provider=provider,
        prompt="You are a concise assistant for Project Orbit.",
        tools=[],
        loop=SingleCall(),
    )
    stable = [
        ContextBlock(
            PROJECT_INSTRUCTIONS,
            kind="project_instructions",
            placement="stable",
            source="project:orbit",
        )
    ]

    q1 = "In one sentence, what is the testing policy?"
    r1 = await agent.run(q1, context=stable)
    print(_fmt("turn 1", r1))

    # Turn 2 — same stable block, prior turn passed as history. The stable head
    # (project instructions + first turn) is byte-identical → cache read.
    history = [ChatMessage.user(q1), ChatMessage.assistant(r1.answer)]
    r2 = await agent.run(
        "In one sentence, what is the code review policy?",
        history=history,
        context=stable,
    )
    print(_fmt("turn 2", r2))

    read = r2.usage.cache_read_input_tokens
    if read:
        print(f"  ✅ turn 2 read {read} tokens from cache (stable context hit)")
    else:
        print("  ⚠️  no cache read on turn 2 (prefix below minimum, or cache cold)")


async def main() -> None:
    await demo(AnthropicProvider(model="claude-sonnet-4-6"), "Anthropic (explicit cache_control)")
    await demo(OpenAIProvider(model="gpt-4o-mini"), "OpenAI (auto prefix cache)")


if __name__ == "__main__":
    asyncio.run(main())
