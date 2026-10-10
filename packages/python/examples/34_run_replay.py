"""Run replay demo — memory across runs with ``runs=``.

A tool-using chat where each turn is a fresh ``agent.run()`` (an Agent built
per user message, as a web endpoint would). The app keeps only the ordered
run ids of the conversation; Dendrux replays each earlier run's tool calls,
results and reasoning from its own traces, so turn 2 does not re-explore the
schema turn 1 already learned.

Per turn it prints every LLM call's cache read / write and ``result.replay``.
Expect turn 2+ to read the whole earlier conversation from cache. Then a
final turn with ``replay_last=1`` shows the sliding window: older runs
collapse to question + answer and replayed reasoning is dropped with a
warning.

Run with (ANTHROPIC_API_KEY in .env at repo root):
    python examples/34_run_replay.py
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from dotenv import load_dotenv

from dendrux import Agent, tool
from dendrux.llm.anthropic import AnthropicProvider
from dendrux.store import RunStore

load_dotenv(Path(__file__).resolve().parents[3] / ".env")

TABLES = {
    "orders": ["id", "customer_id", "plan_id", "amount_usd", "status", "created_at"],
    "plans": ["id", "name", "monthly_price_usd", "tier"],
    "refunds": ["id", "order_id", "amount_usd", "refunded_at"],
}


@tool()
async def list_tables() -> list[str]:
    """List the tables in the analytics database."""
    return sorted(TABLES)


@tool()
async def describe_table(table: str) -> list[str]:
    """Return the column names of a table.

    Args:
        table: Table name from list_tables.
    """
    return TABLES.get(table, [])


@tool()
async def run_sql(query: str) -> list[dict]:
    """Run a read-only SQL query and return rows.

    Args:
        query: A SELECT statement.
    """
    if "group" in query.lower():
        return [{"plan": "pro", "sum": 30000}, {"plan": "basic", "sum": 12000}]
    return [{"sum": 42000}]


# Frozen for the whole conversation, including the line about tools from
# earlier turns (adding it later would itself be a prompt edit).
PROMPT = "\n".join(
    [
        "You are an analytics assistant. Answer with the tools provided.",
        "Explore the schema before writing SQL you have not written before in this "
        "conversation; if earlier turns already explored it, reuse what you learned.",
        "Tools used in earlier turns may no longer be available; only call tools "
        "offered in this request.",
        "Keep answers to one sentence.",
        *(
            f"Policy {i:02d}: exclude test accounts, filter paid orders, and round to "
            "whole dollars before reporting any metric."
            for i in range(1, 61)
        ),
    ]
)

QUESTIONS = [
    "What was total revenue last month?",
    "Break that down by plan.",
    "Which plan should we focus on?",
]


def build_agent(db_url: str) -> Agent:
    return Agent(
        name="AnalyticsChat",
        provider=AnthropicProvider(model="claude-sonnet-4-6", thinking=True, cache_ttl="1h"),
        prompt=PROMPT,
        tools=[list_tables, describe_table, run_sql],
        database_url=db_url,
    )


async def turn(db_url: str, question: str, run_ids: list[str], replay_last: int | str) -> str:
    agent = build_agent(db_url)
    try:
        result = await agent.run(question, runs=run_ids, replay_last=replay_last)
        assert agent._private_engine is not None
        calls = await RunStore.from_engine(agent._private_engine).get_llm_calls(result.run_id)
    finally:
        await agent.close()

    print(f"\n> {question}   (runs={len(run_ids)}, replay_last={replay_last!r})")
    for c in calls:
        print(
            f"  call {c.iteration}: cache_read={c.cache_read_input_tokens} "
            f"cache_write={c.cache_creation_input_tokens} fresh={c.input_tokens}"
        )
    print(f"  answer: {result.answer}")
    print(f"  replay: {result.replay}")
    return result.run_id


async def main() -> None:
    db_url = f"sqlite+aiosqlite:///{tempfile.mkdtemp()}/replay.db"
    run_ids: list[str] = []
    for question in QUESTIONS:
        run_ids.append(await turn(db_url, question, list(run_ids), "all"))
    await turn(db_url, "Summarise the conversation.", run_ids, 1)


if __name__ == "__main__":
    asyncio.run(main())
