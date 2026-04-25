"""19 — CLI chatbot with Postgres + governance (PII, PromptInjection, Budget)."""

from __future__ import annotations

import asyncio
from pathlib import Path

from dotenv import load_dotenv

from dendrux import Agent
from dendrux.chat import ChatMessage
from dendrux.guardrails import PII, Pattern, PromptInjection
from dendrux.llm.anthropic import AnthropicProvider
from dendrux.notifiers.console import ConsoleNotifier
from dendrux.types import Budget

load_dotenv(Path(__file__).resolve().parents[3] / ".env")

DB_URL = "postgresql+asyncpg://postgres:root@localhost:5432/dendrux"

INJECTION_PATTERNS = [
    Pattern(
        "INSTRUCTION_OVERRIDE",
        r"(?i)\bignore\s+(?:all\s+|the\s+)?(?:previous|prior|above)\s+"
        r"(?:instructions?|system\s+prompt|rules)\b",
    ),
    Pattern(
        "SYSTEM_PROMPT_LEAK",
        r"(?i)\b(reveal|show|print|leak|repeat)\s+(your|the)\s+system\s+prompt\b",
    ),
]


async def main() -> None:
    async with Agent(
        name="ChatBotPG",
        provider=AnthropicProvider(model="claude-sonnet-4-6"),
        database_url=DB_URL,
        prompt="You are a friendly assistant. Reply briefly.",
        tools=[],
        guardrails=[
            PII(engine="presidio"),
            PromptInjection(action="block", patterns=INJECTION_PATTERNS),
        ],
        budget=Budget(max_tokens=5_000),
    ) as agent:
        history: list[ChatMessage] = []
        notifier = ConsoleNotifier()
        print("Chatbot ready (Postgres + governance). Ctrl+D or 'quit' to exit.\n")
        while True:
            try:
                user = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not user or user.lower() in {"quit", "exit"}:
                break
            result = await agent.run(user_input=user, history=history, notifier=notifier)
            if result.answer:
                print(f"bot> {result.answer}\n")
                history.append(ChatMessage.user(user))
                history.append(ChatMessage.assistant(result.answer))
            else:
                print(f"bot> [{result.status.value}] {result.error or 'no answer'}\n")


if __name__ == "__main__":
    asyncio.run(main())
