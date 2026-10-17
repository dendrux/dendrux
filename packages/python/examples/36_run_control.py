"""Run control: per-call approval, a before_tool policy hook, and durable pause.

Three short runs on one agent, driven by a mock provider (no API key needed):

1. Two gated `refund` calls in one turn. A human approves one and denies the
   other with `submit_approval(decisions=...)`.
2. A `before_tool` hook that gates a large email for approval, answers a
   destructive delete with a canned result (`substitute`) and lets a lookup
   run. The lookup was never gated, so it executes on resume without a
   decision.
3. `request_pause` while a tool is running. The run parks as `paused`
   with its full history and `submit_resume` continues it.

Run with:
    python examples/36_run_control.py
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from typing import Any

from dendrux import Agent, ApprovalDecision, ToolCallContext, ToolDecision, tool
from dendrux.llm.mock import MockLLM
from dendrux.store import RunStore, StoredEvent
from dendrux.types import LLMResponse, RunStatus, ToolCall

REFUNDED: list[int] = []
EMAILS: list[int] = []
DELETED: list[int] = []

crunch_entered = asyncio.Event()
crunch_release = asyncio.Event()
seen_run: dict[str, str] = {}


@tool()
async def refund(order_id: int) -> str:
    """Refund an order."""
    REFUNDED.append(order_id)
    return f"Refunded order {order_id}"


@tool()
async def send_email(to: list[str], body: str) -> str:
    """Send an email to the given recipients."""
    EMAILS.append(len(to))
    return f"Sent to {len(to)} recipients"


@tool()
async def delete_rows(count: int) -> str:
    """Delete rows from the orders table."""
    DELETED.append(count)
    return f"Deleted {count} rows"


@tool()
async def lookup(customer: str) -> str:
    """Look up a customer record."""
    return f"Record for {customer}"


@tool()
async def crunch(job: str) -> str:
    """A slow job. Waits until the example releases it."""
    await crunch_release.wait()
    return f"Finished {job}"


def policy(call: ToolCallContext) -> ToolDecision | None:
    """Evaluated before every tool call. Returning None allows the call."""
    if call.tool_name == "crunch":
        seen_run["run_id"] = call.run_id
        crunch_entered.set()
    if call.tool_name == "send_email" and len(call.params.get("to", [])) > 10:
        return ToolDecision.require_approval("More than 10 recipients")
    if call.tool_name == "delete_rows" and call.params.get("count", 0) > 100:
        return ToolDecision.substitute({"status": "held_for_review"})
    return None


def call(name: str, params: dict[str, Any], pid: str) -> ToolCall:
    return ToolCall(name=name, params=params, provider_tool_call_id=pid)


SCRIPT = [
    # Run 1: two gated refunds in one turn, then the closing answer.
    LLMResponse(
        tool_calls=[call("refund", {"order_id": 1}, "p1"), call("refund", {"order_id": 2}, "p2")]
    ),
    LLMResponse(text="Order 1 is refunded. Order 2 was declined."),
    # Run 2: one hook-gated call, one substituted call, one allowed call.
    LLMResponse(
        tool_calls=[
            call(
                "send_email",
                {"to": [f"user{i}@example.com" for i in range(12)], "body": "Hi"},
                "p3",
            ),
            call("delete_rows", {"count": 500}, "p4"),
            call("lookup", {"customer": "acme"}, "p5"),
        ]
    ),
    LLMResponse(text="Email sent, delete held for review, record found."),
    # Run 3: a slow tool, paused mid-run, then the closing answer after resume.
    LLMResponse(tool_calls=[call("crunch", {"job": "nightly"}, "p6")]),
    LLMResponse(text="The nightly job finished."),
]


def data(event: StoredEvent) -> dict[str, Any]:
    """An event's payload (never None for the events read here)."""
    return event.data or {}


async def pending_calls(store: RunStore, run_id: str) -> list[dict[str, Any]]:
    """The calls a run is waiting on, read from its `run.paused` event."""
    events = await store.get_events(run_id, event_types=["run.paused"])
    calls: list[dict[str, Any]] = data(events[-1])["pending_tool_calls"]
    return calls


async def per_call_approval(agent: Agent, store: RunStore) -> None:
    print("--- 1. per-call approval ---")
    paused = await agent.run("Refund orders 1 and 2.")
    assert paused.status is RunStatus.WAITING_APPROVAL
    first, second = await pending_calls(store, paused.run_id)

    done = await agent.submit_approval(
        paused.run_id,
        decisions={
            first["id"]: ApprovalDecision.approve(),
            second["id"]: ApprovalDecision.deny("Order 2 is outside the refund window"),
        },
    )
    decided = (await store.get_events(paused.run_id, event_types=["approval.decided"]))[0]
    print("status:", done.status.value)
    print("refunded:", REFUNDED)
    print("decision:", data(decided)["decision"])
    for item in data(decided)["decisions"]:
        print(f"  {item['tool_name']} -> {item['decision']} ({item['reason']})")


async def policy_hook(agent: Agent, store: RunStore) -> None:
    print("\n--- 2. before_tool policy ---")
    paused = await agent.run("Email everyone, clean up old orders and look up acme.")
    assert paused.status is RunStatus.WAITING_APPROVAL
    gates = await store.get_events(paused.run_id, event_types=["approval.requested"])
    substituted = await store.get_events(paused.run_id, event_types=["policy.substituted"])
    print("approval requested for:", [data(g)["tool_name"] for g in gates])
    print("reason:", data(gates[0])["reason"], "| source:", data(gates[0])["source"])
    print("substituted:", [data(s)["tool_name"] for s in substituted])

    # Only the hook-gated call needs a decision; the held lookup runs on its own.
    done = await agent.submit_approval(
        paused.run_id, decisions={data(gates[0])["call_id"]: ApprovalDecision.approve()}
    )
    print("status:", done.status.value)
    print("emails sent:", EMAILS, "| rows deleted:", DELETED)


async def pause_and_resume(agent: Agent, store: RunStore) -> None:
    print("\n--- 3. request_pause and submit_resume ---")
    run_task = asyncio.create_task(agent.run("Run the nightly job."))
    await crunch_entered.wait()
    run_id = seen_run["run_id"]

    outcome = await agent.request_pause(run_id, reason="deploy window")
    print("pause requested:", outcome.status_before.value, "->", outcome.affected)
    crunch_release.set()  # the tool is never preempted; it finishes first

    paused = await run_task
    print("run() returned:", paused.status.value)
    event = (await store.get_events(run_id, event_types=["run.paused"]))[-1]
    print("run.paused:", data(event)["reason"], "-", data(event)["message"])

    done = await agent.submit_resume(run_id)
    print("after resume:", done.status.value, "|", done.answer)


async def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        url = f"sqlite+aiosqlite:///{Path(tmp) / 'run_control.db'}"
        async with Agent(
            name="Ops",
            provider=MockLLM(SCRIPT),
            prompt="You are an operations assistant.",
            tools=[refund, send_email, delete_rows, lookup, crunch],
            require_approval=["refund"],
            before_tool=policy,
            database_url=url,
        ) as agent:
            store = RunStore.from_database_url(url)
            async with store:
                await per_call_approval(agent, store)
                await policy_hook(agent, store)
                await pause_and_resume(agent, store)


if __name__ == "__main__":
    asyncio.run(main())
