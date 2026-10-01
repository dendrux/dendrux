"""Cost, cancellation, stream idempotency, and the answer contract — in one run.

Four behaviours that production callers hit together, verified live:

  1. `Agent(pricing=PriceTable({...}))` fills `cost_usd` for providers that
     report tokens only (Anthropic here). `cost_source` says who priced it.
  2. `RunResult.answer` is the final turn only. Earlier assistant text from
     a tool-using run lives in the traces, not in `answer`.
  3. Cancelling the `run()` coroutine (`asyncio.wait_for` timeout) finalizes
     the persisted row as `cancelled` and emits `run.cancelled`, instead of
     leaving it `running` forever.
  4. `agent.stream(..., idempotency_key=)` returns one cached terminal event
     on a repeat, and `RUN_ERROR` on a conflicting input, without touching
     the original run.

Run with:
    ANTHROPIC_API_KEY=sk-ant-... python examples/33_cost_cancel_idempotency.py
    (or put the key in the repo-root .env)
"""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

from dotenv import load_dotenv

from dendrux import Agent, ModelPricing, PriceTable, tool
from dendrux.store import RunStore
from dendrux.types import RunEventType

load_dotenv(Path(__file__).resolve().parents[3] / ".env")

PROVIDER = "anthropic:claude-haiku-4-5"  # recipe string: the agent owns the client
DB_URL = f"sqlite+aiosqlite:///{Path.home() / '.dendrux' / 'example_33.db'}"

# USD per million tokens, copied from the vendor's pricing page. Dendrux ships
# no price list: you own the rates, it owns the arithmetic.
PRICING = PriceTable(
    {
        "claude-haiku-4-5*": ModelPricing(input=1.0, output=5.0, cache_read=0.10, cache_write=1.25),
        "claude-sonnet-4-6": ModelPricing(
            input=3.0, output=15.0, cache_read=0.30, cache_write=3.75
        ),
    }
)


@tool()
async def lookup_city(name: str) -> str:
    """Return a fact about a city."""
    return f"{name} sits on a river and is known for its old town."


def section(title: str) -> None:
    print(f"\n== {title} ==")


async def priced_run_and_final_answer(store: RunStore) -> None:
    """Parts 1 and 2: table-priced cost and the final-turn answer contract."""
    section("1. Pricing: cost_usd from your PriceTable")
    async with Agent(
        provider=PROVIDER,
        prompt="Use lookup_city before answering. Keep the final answer to one sentence.",
        tools=[lookup_city],
        database_url=DB_URL,
        pricing=PRICING,
    ) as agent:
        result = await agent.run(
            "Tell me about Prague. Say 'Checking...' first, then call the tool, then answer."
        )
    usage = result.usage
    print(f"status       : {result.status.value}")
    print(f"tokens       : {usage.input_tokens} in / {usage.output_tokens} out")
    print(f"cost_usd     : {usage.cost_usd:.6f}  (cost_source={usage.cost_source!r})")

    calls = await store.get_llm_calls(result.run_id)
    for call in calls:
        print(f"  call {call.iteration}: {call.model}  ${call.cost_usd:.6f}")

    section("2. Answer contract: final turn only")
    print(f"result.answer: {result.answer!r}")
    traces = await store.get_traces(result.run_id)
    assistant_text = [t.content for t in traces if t.role == "assistant" and t.content]
    print(f"assistant turns recorded in traces: {len(assistant_text)}")
    for i, text in enumerate(assistant_text, 1):
        print(f"  turn {i}: {text[:70]!r}")

    section("1b. Same run without a table: cost is unknown, never 0")
    async with Agent(provider=PROVIDER, prompt="One short sentence.", database_url=DB_URL) as plain:
        r = await plain.run("What is 2 + 2?")
    print(f"cost_usd={r.usage.cost_usd!r}  cost_source={r.usage.cost_source!r}")


async def timeout_cancels_run(store: RunStore) -> None:
    """Part 3: a timed-out run() coroutine finalizes the row as cancelled."""
    section("3. asyncio.wait_for timeout on run()")
    started: list[str] = []

    class CaptureRunId:
        async def on_run_started(self, run_id: str, **_: object) -> None:
            started.append(run_id)

        def __getattr__(self, _name: str):  # every other hook is a no-op
            async def noop(*_: object, **__: object) -> None:
                return None

            return noop

    async with Agent(
        provider=PROVIDER,
        prompt="Write a long essay, at least 800 words.",
        database_url=DB_URL,
        pricing=PRICING,
    ) as agent:
        try:
            await asyncio.wait_for(
                agent.run("An essay about the ocean.", notifier=CaptureRunId()), timeout=1.5
            )
        except TimeoutError:
            print("wait_for raised TimeoutError, as expected")

    run_id = started[0]
    record = await store.get_run(run_id)
    events = [e for e in await store.get_events(run_id)]
    cancelled = [e for e in events if e.event_type == "run.cancelled"]
    print(f"row status   : {record.status}")
    print(f"events       : {[e.event_type for e in events]}")
    print(f"cancel reason: {cancelled[0].data.get('reason') if cancelled else None}")


async def stream_idempotency(store: RunStore) -> None:
    """Part 4: idempotency_key on stream() — cache hit and conflict."""
    section("4. stream() with idempotency_key")
    async with Agent(
        provider=PROVIDER,
        prompt="Answer in one short sentence.",
        database_url=DB_URL,
        pricing=PRICING,
    ) as agent:
        key = f"demo-{uuid.uuid4().hex[:8]}"

        first = agent.stream("Name one primary colour.", idempotency_key=key)
        async with first:
            text = "".join(
                [e.text async for e in first if e.type == RunEventType.TEXT_DELTA and e.text]
            )
        print(f"first run    : {first.run_id}  answer={text!r}")
        events_before = len(await store.get_events(first.run_id))

        replay = agent.stream("Name one primary colour.", idempotency_key=key)
        provisional_id = replay.run_id
        async with replay:
            replay_events = [e async for e in replay]
        print(
            f"replay       : {len(replay_events)} event ({replay_events[0].type.value}), "
            f"run_id resolved {provisional_id != replay.run_id and replay.run_id == first.run_id}"
        )
        print(
            f"replay answer: {replay.result.answer!r}  (no provider call, no new events: "
            f"{len(await store.get_events(first.run_id)) == events_before})"
        )

        conflict = agent.stream("Name one secondary colour.", idempotency_key=key)
        async with conflict:
            conflict_events = [e async for e in conflict]
        print(
            f"conflict     : {conflict_events[-1].type.value}  "
            f"error_type={conflict.result.meta.get('error_type')!r}"
        )
        print(f"original run : {(await store.get_run(first.run_id)).status}")


async def main() -> None:
    Path.home().joinpath(".dendrux").mkdir(exist_ok=True)
    async with RunStore.from_database_url(DB_URL) as store:
        await priced_run_and_final_answer(store)
        await timeout_cancels_run(store)
        await stream_idempotency(store)
    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
