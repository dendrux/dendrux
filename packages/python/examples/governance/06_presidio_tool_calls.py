"""Governance: Presidio-backed PII + tool call validation.

Demonstrates the end-to-end redaction round-trip with Presidio's NLP engine:

  - ``PII(engine="presidio")`` catches PERSON, LOCATION, DATE_TIME,
    EMAIL_ADDRESS, PHONE_NUMBER — things the default regex engine cannot
    match on its own.
  - A server tool prints what it actually received at runtime (the real,
    deanonymized values — tools live inside the developer's trust boundary).
  - After the run, we read everything through the public ``RunStore`` to
    assert three invariants:
      1. DB persisted the raw user input (ground truth).
      2. The tool invocation row stores the real parameters it ran with.
      3. ``pii_mapping`` is the audit key linking raw DB rows to the
         placeholder view the LLM provider API actually saw.

Requires:
    pip install dendrux[anthropic,presidio]
    python -m spacy download en_core_web_lg

Run with:
    ANTHROPIC_API_KEY=sk-... python examples/governance/06_presidio_tool_calls.py
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from dotenv import load_dotenv

from dendrux import Agent, tool
from dendrux.guardrails import PII
from dendrux.llm.anthropic import AnthropicProvider
from dendrux.notifiers.console import ConsoleNotifier
from dendrux.store import RunStore
from dendrux.types import RunStatus

load_dotenv(Path(__file__).resolve().parents[4] / ".env")

DB_PATH = Path(__file__).resolve().parent / "presidio_demo.db"
DB_URL = f"sqlite+aiosqlite:///{DB_PATH}"

tool_invocations: list[dict[str, str]] = []


@tool()
async def process_refund(
    customer_name: str,
    customer_email: str,
    order_date: str,
    shipping_city: str,
) -> str:
    """Issue a refund for a customer order.

    Tool lives inside the developer's trust boundary — it must receive
    real, deanonymized values so the refund actually reaches the right
    inbox.
    """
    tool_invocations.append(
        {
            "customer_name": customer_name,
            "customer_email": customer_email,
            "order_date": order_date,
            "shipping_city": shipping_city,
        }
    )
    return (
        f"Refund issued for {customer_name} ({customer_email}), "
        f"order placed {order_date}, shipping to {shipping_city}."
    )


async def main() -> None:
    if DB_PATH.exists():
        DB_PATH.unlink()

    user_input = (
        "Refund Alice Johnson's order from March 14 2026. "
        "Her email is alice.johnson@example.com, "
        "she's in San Francisco, phone 415-555-0143."
    )

    async with Agent(
        provider=AnthropicProvider(model="claude-sonnet-4-6"),
        prompt=(
            "You are a customer support agent. When asked to issue a refund, "
            "call process_refund with the customer's name, email, order date, "
            "and shipping city extracted from the user's message."
        ),
        tools=[process_refund],
        guardrails=[PII(engine="presidio")],
        database_url=DB_URL,
    ) as agent:
        notifier = ConsoleNotifier()
        result = await agent.run(user_input, notifier=notifier)
        notifier.print_summary(result)
        assert result.status == RunStatus.SUCCESS, result.error

    # Re-open the public read facade. Everything below uses only public API —
    # no private imports, no underscore accessors.
    async with RunStore.from_database_url(DB_URL) as store:
        traces = await store.get_traces(result.run_id)
        tool_calls = await store.get_tool_invocations(result.run_id)
        mapping = await store.get_pii_mapping(result.run_id)

    print("\n" + "=" * 60)
    print("VALIDATION")
    print("=" * 60)

    # 1. Tool received real values at runtime (captured in-process).
    assert len(tool_invocations) == 1, "expected exactly one refund call"
    call = tool_invocations[0]
    assert "<<" not in call["customer_email"], (
        "Tool received a placeholder — deanonymize never fired"
    )
    print("\n[1] Tool function received REAL values at runtime:")
    for k, v in call.items():
        print(f"    {k:17} {v!r}")

    # 2. DB stored the raw user input (no redaction leaked into persistence).
    user_trace = next(t for t in traces if t.role == "user")
    assert "alice.johnson@example.com" in user_trace.content, (
        "DB did not store raw input — redaction leaked into persistence"
    )
    print("\n[2] DB stored RAW user input (ground truth — via RunStore):")
    print(f"    {user_trace.content}")

    # 3. DB tool_invocation row has the real params the tool ran with.
    assert len(tool_calls) == 1
    persisted_params = tool_calls[0].params or {}
    assert persisted_params.get("customer_email") == "alice.johnson@example.com"
    print("\n[3] DB persisted tool params RAW:")
    for k, v in persisted_params.items():
        print(f"    {k:17} {v!r}")

    # 4. pii_mapping — the audit key that links raw DB rows to the LLM view.
    expected = {"EMAIL_ADDRESS", "PERSON", "LOCATION", "PHONE_NUMBER"}
    found = {p.strip("<>").rsplit("_", 1)[0] for p in mapping}
    missing = expected - found
    assert not missing, f"Presidio should have detected {missing}"
    print("\n[4] pii_mapping (audit key — links raw DB to LLM-eye view):")
    for placeholder, real in mapping.items():
        print(f"    {placeholder:28}  ->  {real!r}")

    print("\n" + "=" * 60)
    print("INVARIANT VERIFIED")
    print("=" * 60)
    print(
        "  LLM provider API saw:  placeholders (mapping keys)\n"
        "  DB persisted:          raw values (ground truth)\n"
        "  Tool executed with:    real deanonymized values\n"
        "  Audit key:             pii_mapping links the two views"
    )


if __name__ == "__main__":
    asyncio.run(main())
