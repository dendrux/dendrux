"""Governance: Dashboard — see all governance events in the timeline.

Runs an agent with all four governance layers enabled and persistence.
After the run, launches the dashboard so you can see deny, approval,
budget, and guardrail events in the timeline.

Run with:
    ANTHROPIC_API_KEY=sk-... python examples/governance/05_dashboard.py

Then open http://localhost:8001 and click on the run.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from dotenv import load_dotenv

from dendrux import Agent, tool
from dendrux.guardrails import PII
from dendrux.llm.anthropic import AnthropicProvider
from dendrux.notifiers.console import ConsoleNotifier
from dendrux.types import Budget, RunStatus

load_dotenv(Path(__file__).resolve().parents[4] / ".env")

DB_PATH = Path(__file__).parent / "dashboard_demo.db"
DB_URL = f"sqlite+aiosqlite:///{DB_PATH}"


@tool()
async def search(query: str) -> str:
    """Search for customer information."""
    return f"Found: customer record for '{query}', email: jane@example.com"


@tool()
async def refund(order_id: int) -> str:
    """Refund an order. Requires manager approval."""
    return f"Refunded ${order_id * 10:.2f} for order {order_id}"


@tool()
async def delete_account(user_id: int) -> str:
    """Permanently delete a user account."""
    return f"Deleted user {user_id}"


async def main() -> None:
    async with Agent(
        provider=AnthropicProvider(model="claude-sonnet-4-6"),
        prompt=(
            "You are a customer support agent. "
            "Always use the appropriate tool for each action. "
            "When asked to delete an account, call delete_account. "
            "When asked for a refund, call refund. "
            "When asked to search, call search."
        ),
        tools=[search, refund, delete_account],
        deny=["delete_account"],
        require_approval=["refund"],
        budget=Budget(max_tokens=3000),
        guardrails=[PII()],
        database_url=DB_URL,
        name="GovernanceAgent",
    ) as agent:
        notifier = ConsoleNotifier()

        # Run 1 — triggers deny + approval + budget + guardrail events
        print("=" * 60)
        print("Starting run with all governance layers...")
        print("=" * 60 + "\n")

        result = await agent.run(
            "Search for user 42, delete their account, and refund order 456",
            notifier=notifier,
        )

        if result.status == RunStatus.WAITING_APPROVAL:
            pause = result.meta["pause_state"]
            print(f"\nRun paused — {result.status.value}")
            for tc in pause.pending_tool_calls:
                print(f"  Pending: {tc.name}({tc.params})")

            # Auto-approve for the demo
            print("\nAuto-approving for demo...\n")
            result = await agent.resume(result.run_id, notifier=notifier)

        notifier.print_summary(result)
        print(f"\nAnswer: {result.answer}")
        print(f"\nRun ID: {result.run_id}")
        print(f"Database: {DB_PATH}")

    # View in dashboard
    print("\n" + "=" * 60)
    print("To see governance events in the dashboard timeline:")
    print(f"  dendrux dashboard --db {DB_PATH}")
    print("Then open http://localhost:8001 and click on the run.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
