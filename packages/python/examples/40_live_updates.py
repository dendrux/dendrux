"""Live updates: wake a follower the moment a run changes.

A mock provider drives one run that pauses for a client tool. A reader follows
the run with `RunStore.follow` while a writer submits the tool result. No API
key and no network are needed; the database is a temporary SQLite file.

Two modes:

    python examples/40_live_updates.py            # default: no live updates, the follower polls
    python examples/40_live_updates.py loopback   # an in-process bus delivers hints

The default needs nothing. The "loopback" bus is a small `LiveUpdates`
subclass that passes hints through a queue inside this process. In a real
deployment the bus is shared by every process:

    PostgresStateStore(engine, live_updates=True)            # Postgres LISTEN/NOTIFY
    MySQLStateStore(engine, live_updates=RedisUpdates(url))  # Redis Pub/Sub

Whichever you use, the reader is `RunStore(state)` over the same kind of store
the writer uses. `RunStore.from_engine` and `RunStore.from_database_url` never
listen. A hint only wakes the reader earlier; the database is always the truth,
so the events are the same in both modes.
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING

from dendrux import Agent, tool
from dendrux.db.session import get_engine, reset_engine
from dendrux.llm.mock import MockLLM
from dendrux.store import FollowEnd, LiveUpdates, RunStore, SQLiteStateStore
from dendrux.types import LLMResponse, RunStatus, ToolCall, ToolResult

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence


@tool(target="client")
async def read_selection() -> str:
    """Read the cells the user has selected in the spreadsheet."""
    return ""


class LoopbackUpdates(LiveUpdates):
    """Deliver hints inside one process. A real bus does the same over a network."""

    def __init__(self, **limits) -> None:
        super().__init__(**limits)
        self._queue: asyncio.Queue[str] = asyncio.Queue()

    async def send(self, messages: Sequence[str]) -> None:
        for message in messages:
            self._queue.put_nowait(message)

    async def receive(self) -> AsyncIterator[str | None]:
        yield None  # the subscription is live
        while True:
            yield await self._queue.get()


async def follow_to_end(reads: RunStore, run_id: str, seen: list[str]) -> str:
    async with reads.follow(run_id, tenant_id=None) as follow:
        async for item in follow:
            if isinstance(item, FollowEnd):
                return item.reason
            event_type = getattr(item, "event_type", None)
            if event_type:
                seen.append(event_type)
    return "closed"


async def main(mode: str) -> int:
    if mode not in ("poll", "loopback"):
        print("Usage: python examples/40_live_updates.py [loopback]")
        return 2

    bus = LoopbackUpdates() if mode == "loopback" else None
    with tempfile.TemporaryDirectory() as tmp:
        engine = await get_engine(f"sqlite+aiosqlite:///{Path(tmp) / 'live.db'}")
        state = SQLiteStateStore(engine, live_updates=bus)
        reads = RunStore(state)  # the form that listens, when the store has live updates
        provider = MockLLM(
            [
                LLMResponse(tool_calls=[ToolCall("read_selection", {})]),
                LLMResponse(text="The selection sums to 42."),
            ]
        )
        try:
            async with Agent(
                name="sheet-helper",
                prompt="Answer questions about the user's selection.",
                provider=provider,
                tools=[read_selection],
                state_store=state,
                strict_tenancy=False,  # single-tenant example
            ) as agent:
                paused = await agent.run("What is the selection total?")
                assert paused.status == RunStatus.WAITING_CLIENT_TOOL
                call = paused.pending.tool_calls[0]
                print(f"mode: {mode}; paused run {paused.run_id}, waiting for {call.name!r}")

                seen: list[str] = []
                follower = asyncio.create_task(follow_to_end(reads, paused.run_id, seen))
                await asyncio.sleep(1.0)  # the follower is now idle and waiting

                started = time.monotonic()
                finished = await agent.submit_tool_results(
                    paused.run_id,
                    [ToolResult(call.name, call.id, '"6 + 36"')],
                    tenant_id=None,
                    expected_pause_id=paused.pause_id,
                )
                reason = await asyncio.wait_for(follower, timeout=30)
                delay = time.monotonic() - started

            print(f"finished: {finished.status.value}: {finished.answer}")
            print(f"follow ended: {reason}; saw {len(seen)} events, last {seen[-1]}")
            print(f"seconds from submit to the end of the follow: {delay:.2f}")
            if bus is not None:
                stats = bus.stats()
                print(f"hints sent: {stats.sent}; received: {stats.received}")
            else:
                print("live updates are off: the follower polls the database")
        finally:
            await reads.close()  # stops the reader; never closes the store
            await state.close()  # closes the live-update connections; the engine stays yours
            await reset_engine()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "poll")))
