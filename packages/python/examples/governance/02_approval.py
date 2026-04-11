"""Governance: Approval (HITL) — human sign-off before tool execution.

The agent has three tools. refund requires human approval before
executing. When the LLM calls refund, the run pauses and waits
for the operator to approve or reject.

Also demonstrates deny + approval working together:
delete_account is denied outright, refund requires approval,
search executes freely.

Requires persistence (SQLite) for pause/resume.

Run with:
    ANTHROPIC_API_KEY=sk-... python examples/governance/02_approval.py
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from dotenv import load_dotenv

from dendrux import Agent, tool
from dendrux.llm.anthropic import AnthropicProvider
from dendrux.notifiers.console import ConsoleNotifier
from dendrux.types import RunStatus, ToolResult

load_dotenv(Path(__file__).resolve().parents[4] / ".env")

DB_PATH = Path(__file__).parent / "approval_demo.db"
DB_URL = f"sqlite+aiosqlite:///{DB_PATH}"


@tool()
async def search(query: str) -> str:
    """Search for customer information."""
    return f"Found: customer record for '{query}'"


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
            "When asked for a refund, call the refund tool. "
            "When asked to delete an account, call delete_account."
        ),
        tools=[search, refund, delete_account],
        deny=["delete_account"],
        require_approval=["refund"],
        database_url=DB_URL,
    ) as agent:
        notifier = ConsoleNotifier()

        # Start the run — will pause when LLM calls refund
        print("Starting run...\n")
        result = await agent.run(
            "Refund order 456 and delete user 42's account",
            notifier=notifier,
        )

        if result.status != RunStatus.WAITING_APPROVAL:
            notifier.print_summary(result)
            print(f"\nAnswer: {result.answer}")
            return

        # Show what's pending
        pause = result.meta["pause_state"]
        print(f"\n{'=' * 50}")
        print(f"Run paused — {result.status.value}")
        print(f"Run ID: {result.run_id}")
        print(f"{'=' * 50}")
        for tc in pause.pending_tool_calls:
            print(f"  Pending: {tc.name}({tc.params})")
        print()

        # Ask for approval
        choice = input("Approve? [y/n]: ").strip().lower()

        if choice == "y":
            # Approve — framework executes the pending tools
            print("\nApproving...\n")
            result = await agent.resume(result.run_id, notifier=notifier)
        else:
            # Reject — provide rejection reason as tool error
            reason = input("Rejection reason: ").strip() or "Rejected by operator"
            print(f"\nRejecting: {reason}\n")
            import json

            result = await agent.resume(
                result.run_id,
                tool_results=[
                    ToolResult(
                        name=tc.name,
                        call_id=tc.id,
                        payload=json.dumps({"rejected": True, "reason": reason}),
                        success=False,
                        error=reason,
                    )
                    for tc in pause.pending_tool_calls
                ],
                notifier=notifier,
            )

        notifier.print_summary(result)
        print(f"\nAnswer: {result.answer}")

    # Clean up demo DB
    DB_PATH.unlink(missing_ok=True)


if __name__ == "__main__":
    asyncio.run(main())
