"""Tools and agents for example 17.

One agent, two providers (Anthropic / OpenAI). Same tools, same
governance, same DB — only the LLM differs. Governance is configured
in ``_base_kwargs`` so both agents share it identically.
"""

import os

from dendrux import Agent, tool
from dendrux.guardrails import PII, Pattern, SecretDetection
from dendrux.llm.anthropic import AnthropicProvider
from dendrux.llm.openai import OpenAIProvider
from dendrux.types import Budget

# ---------------------------------------------------------------------------
# Tools — one per pause type, plus one deny-target.
# ---------------------------------------------------------------------------


@tool()
async def lookup_order(order_id: str) -> dict[str, str]:
    """Look up an order by ID. Server-side, no pause."""
    # Fake catalog for the demo; real apps hit their DB.
    catalog = {
        "4421": {"customer_email": "jane@acme.com", "status": "shipped", "total": "$210.00"},
        "5102": {"customer_email": "bob@example.com", "status": "pending", "total": "$48.50"},
    }
    return catalog.get(order_id, {"error": f"order {order_id} not found"})


@tool(target="client")
async def contact_customer(email: str, message: str) -> str:
    """Ask the user's client (browser) to deliver a message. Pauses the run."""
    return ""  # filled in by the browser via submit_tool_results


@tool()
async def issue_refund(order_id: str, amount_usd: float) -> str:
    """Refund an order. Requires human approval before it runs."""
    return f"Refunded order {order_id} for ${amount_usd:.2f}."


@tool()
async def delete_account(user_id: str) -> str:
    """Delete an account. Denied — should never run."""
    raise RuntimeError("unreachable: delete_account is on the deny list")


TOOLS = [lookup_order, contact_customer, issue_refund, delete_account]


# ---------------------------------------------------------------------------
# Governance (shared across providers)
# ---------------------------------------------------------------------------


def _governance_kwargs() -> dict:
    """Build the governance kwargs both agents share.

    Three guardrails, three distinct actions:
      1. PII            — redact emails/phones/etc., deanonymized at tools.
      2. SecretDetection — block the run if credentials show up.
      3. Custom pattern  — warn (audit only) when internal ticket IDs appear.
    """
    return {
        "deny": ["delete_account"],
        "require_approval": ["issue_refund"],
        "budget": Budget(max_tokens=20_000),
        "guardrails": [
            PII(action="redact"),
            SecretDetection(action="block"),
            PII(
                action="warn",
                include_defaults=False,
                extra_patterns=[Pattern("INTERNAL_TICKET", r"TKT-\d{6,}")],
            ),
        ],
    }


PROMPT = (
    "You are a support agent for Duskwell Supply Co. You can look up orders,"
    "contact customers (client-side — the user's browser runs this tool), "
    "issue refunds (requires human approval), and ask the user to clarify "
    "when you need more info. Prefer short answers. Confirm destructive "
    "actions before running them."
)


# ---------------------------------------------------------------------------
# Provider-specific agent builders
# ---------------------------------------------------------------------------


def build_anthropic_agent(*, database_url: str) -> Agent:
    return Agent(
        name="SupportAgent",
        provider=AnthropicProvider(
            model="claude-sonnet-4-6",
            api_key=os.environ.get("ANTHROPIC_API_KEY"),
        ),
        database_url=database_url,
        prompt=PROMPT,
        tools=TOOLS,
        **_governance_kwargs(),
    )


def build_openai_agent(*, database_url: str) -> Agent:
    return Agent(
        name="SupportAgent",
        provider=OpenAIProvider(
            model="gpt-4o-mini",
            api_key=os.environ.get("OPENAI_API_KEY"),
        ),
        database_url=database_url,
        prompt=PROMPT,
        tools=TOOLS,
        **_governance_kwargs(),
    )
