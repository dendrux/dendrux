"""18 — CLI chatbot with SQLite persistence."""

from __future__ import annotations

import asyncio
from pathlib import Path

from dotenv import load_dotenv

from dendrux import Agent
from dendrux.chat import ChatMessage
from dendrux.llm.anthropic import AnthropicProvider

load_dotenv(Path(__file__).resolve().parents[3] / ".env")


async def main() -> None:
    db_path = Path.home() / ".dendrux" / "chatbot.db"
    async with Agent(
        name="ChatBot",
        provider=AnthropicProvider(model="claude-sonnet-4-6"),
        database_url=f"sqlite+aiosqlite:///{db_path}",
        prompt="You are a friendly assistant. Reply briefly.",
        tools=[],
    ) as agent:
        history: list[ChatMessage] = []
        print("Chatbot ready. Ctrl+D or 'quit' to exit.\n")
        while True:
            try:
                user = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not user or user.lower() in {"quit", "exit"}:
                break
            result = await agent.run(user_input=user, history=history)
            print(f"bot> {result.answer}\n")
            history.append(ChatMessage.user(user))
            history.append(ChatMessage.assistant(result.answer))


if __name__ == "__main__":
    asyncio.run(main())
