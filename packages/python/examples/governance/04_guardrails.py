"""Governance: Guardrails — PII redaction and secret detection.

Demonstrates:
  - PII(action="redact") — emails/phones replaced with <<EMAIL_ADDRESS_1>> placeholders
  - SecretDetection(action="block") — AWS keys terminate the run
  - Custom Pattern — domain-specific PII detection
  - Tools receive real values (deanonymized), LLM sees placeholders

Run with:
    ANTHROPIC_API_KEY=sk-... python examples/governance/04_guardrails.py
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from dotenv import load_dotenv

from dendrux import Agent, tool
from dendrux.guardrails import PII, Pattern, SecretDetection
from dendrux.llm.anthropic import AnthropicProvider
from dendrux.notifiers.console import ConsoleNotifier
from dendrux.types import RunStatus

load_dotenv(Path(__file__).resolve().parents[4] / ".env")


@tool()
async def send_email(to: str, subject: str, body: str) -> str:
    """Send an email to a customer."""
    return f"Email sent to {to}: [{subject}] {body}"


@tool()
async def lookup_customer(query: str) -> str:
    """Look up customer information."""
    return f"Customer '{query}': jane@example.com, phone +1-555-123-4567, EMP-789012"


async def demo_pii_redaction() -> None:
    """PII is redacted before the LLM sees it, tools get real values."""
    print("=" * 60)
    print("DEMO 1: PII Redaction")
    print("=" * 60 + "\n")

    async with Agent(
        provider=AnthropicProvider(model="claude-sonnet-4-6"),
        prompt=(
            "You are a customer support agent. "
            "When asked to contact a customer, first look them up, "
            "then send them an email using the information you found."
        ),
        tools=[send_email, lookup_customer],
        guardrails=[
            PII(
                extra_patterns=[Pattern("EMPLOYEE_ID", r"EMP-\d{6}")],
            ),
        ],
    ) as agent:
        notifier = ConsoleNotifier()
        result = await agent.run(
            "Look up customer John and send them a hello email",
            notifier=notifier,
        )
        notifier.print_summary(result)
        print(f"\nAnswer: {result.answer}\n")


async def demo_secret_block() -> None:
    """Secrets in user input block the run before the LLM sees them."""
    print("=" * 60)
    print("DEMO 2: Secret Detection (block)")
    print("=" * 60 + "\n")

    async with Agent(
        provider=AnthropicProvider(model="claude-sonnet-4-6"),
        prompt="You are a helpful assistant.",
        tools=[lookup_customer],
        guardrails=[SecretDetection()],
    ) as agent:
        notifier = ConsoleNotifier()
        result = await agent.run(
            "Store this config: aws_key=AKIAIOSFODNN7EXAMPLE",
            notifier=notifier,
        )

        print(f"\nStatus: {result.status.value}")
        print(f"Error: {result.error}")
        assert result.status == RunStatus.ERROR
        print("(Run blocked — secret never reached the LLM)\n")


async def demo_warn_mode() -> None:
    """Warn mode logs findings without modifying content."""
    print("=" * 60)
    print("DEMO 3: Shadow Rollout (warn mode)")
    print("=" * 60 + "\n")

    async with Agent(
        provider=AnthropicProvider(model="claude-sonnet-4-6"),
        prompt="You are a helpful assistant. Answer briefly.",
        tools=[],
        guardrails=[PII(action="warn")],
    ) as agent:
        notifier = ConsoleNotifier()
        result = await agent.run(
            "Summarize: Contact jane@example.com or call +1-555-123-4567",
            notifier=notifier,
        )
        notifier.print_summary(result)
        print(f"\nAnswer: {result.answer}")
        print("(PII detected and logged, but content was NOT modified)\n")


async def main() -> None:
    await demo_pii_redaction()
    await demo_secret_block()
    await demo_warn_mode()


if __name__ == "__main__":
    asyncio.run(main())
