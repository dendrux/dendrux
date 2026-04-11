"""Governance: Tool Deny — block dangerous tools deterministically.

The agent has three tools but delete_account is denied by policy.
When the LLM tries to use it, the framework returns a synthesized
error and the model adapts — the tool never executes.

Shows both batch (agent.run) and streaming (agent.stream) paths.

Run with:
    ANTHROPIC_API_KEY=sk-... python examples/governance/01_tool_deny.py
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from dotenv import load_dotenv

from dendrux import Agent, tool
from dendrux.llm.anthropic import AnthropicProvider
from dendrux.notifiers.console import ConsoleNotifier
from dendrux.types import RunEventType

load_dotenv(Path(__file__).resolve().parents[4] / ".env")


@tool()
async def search(query: str) -> str:
    """Search for customer information."""
    return f"Found: customer record for '{query}'"


@tool()
async def refund(order_id: int) -> str:
    """Refund an order."""
    return f"Refunded order {order_id}"


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
            "When asked for a refund, call refund."
        ),
        tools=[search, refund, delete_account],
        deny=["delete_account"],
    ) as agent:
        # --- Batch mode ---
        print("=" * 60)
        print("BATCH MODE")
        print("=" * 60)

        notifier = ConsoleNotifier()
        result = await agent.run(
            "Delete user 42's account and refund order 456",
            notifier=notifier,
        )
        notifier.print_summary(result)
        print(f"\nAnswer: {result.answer}")

        # --- Streaming mode ---
        print("\n" + "=" * 60)
        print("STREAMING MODE")
        print("=" * 60 + "\n")

        async with agent.stream("Delete user 99's account and search for their records") as stream:
            async for event in stream:
                if event.type == RunEventType.TEXT_DELTA:
                    print(event.text, end="", flush=True)
                elif event.type == RunEventType.TOOL_RESULT:
                    tr = event.tool_result
                    if tr.success:
                        print(f"\n  [tool] {tr.name}: {tr.payload}")
                    else:
                        print(f"\n  [denied] {tr.name}: {tr.error}")
                elif event.type == RunEventType.RUN_COMPLETED:
                    print(f"\n\nStatus: {event.run_result.status.value}")

        print()


if __name__ == "__main__":
    asyncio.run(main())
