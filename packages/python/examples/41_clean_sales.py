"""Approve sales cleanup. Supply ANTHROPIC_API_KEY in the process environment.

Run: python examples/41_clean_sales.py (requires db and anthropic extras).
SALES_DEMO_DIR defaults to sales-demo; DENDRUX_DATABASE_URL can select prepared Postgres.
"""

import asyncio
import os
from pathlib import Path

from _sales_sheet import SalesSheet

from dendrux import Agent, Budget
from dendrux.guardrails import SecretDetection
from dendrux.llm.anthropic import AnthropicProvider
from dendrux.notifiers.console import ConsoleNotifier
from dendrux.types import RunStatus


async def main() -> None:
    """Run a bounded cleanup conversation with explicit tenant and pause tokens."""
    root = Path(os.environ.get("SALES_DEMO_DIR", "sales-demo")).resolve()
    sheet = SalesSheet(root)
    sheet.initialize()
    db_url = os.environ.get("DENDRUX_DATABASE_URL", f"sqlite+aiosqlite:///{root / 'runs.db'}")
    async with Agent(
        name="SalesCleaner",
        database_url=db_url,
        provider=AnthropicProvider(model="claude-haiku-4-5", call_timeout=120),
        prompt="Read the sheet, normalize regions, then post a review notice. "
        "Stop if the batch is declined.",
        tools=[sheet.read_sheet, sheet.write_region, sheet.post_review_notice],
        require_approval=["write_region"],
        guardrails=[SecretDetection()],
        max_iterations=8,
        budget=Budget(max_tokens=20000, on_exceed="cancel"),
    ) as agent:
        notifier = ConsoleNotifier()
        result = await agent.run(
            "Clean the sales regions and post a review notice.",
            tenant_id="sales-demo",
            notifier=notifier,
        )
        for _ in range(4):
            if result.status != RunStatus.WAITING_APPROVAL:
                break
            if result.pending is None or result.pause_id is None:
                raise RuntimeError("Reload pending work before deciding")
            print([(c.name, c.params) for c in result.pending.tool_calls])
            result = await agent.submit_approval(
                result.run_id,
                approved=input("Approve batch? [y/N] ").strip().lower() == "y",
                tenant_id="sales-demo",
                expected_pause_id=result.pause_id,
                notifier=notifier,
            )
        print(result.run_id, result.status.value, result.answer or result.error or "Still paused")


if __name__ == "__main__":
    asyncio.run(main())
