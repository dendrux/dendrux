"""Hello World — minimal Dendrite agent with a tool.

Run with:
    ANTHROPIC_API_KEY=sk-... python examples/01_hello_world.py
"""

from dendrite import Agent, tool


@tool()
async def add(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b


agent = Agent(
    prompt="You are a helpful calculator. Use the add tool when asked to add numbers.",
    tools=[add],
)

if __name__ == "__main__":
    import asyncio
    import os
    from pathlib import Path

    from dotenv import load_dotenv

    from dendrite.llm.anthropic import AnthropicProvider

    # Load .env from repo root (three levels up: examples/ → python/ → packages/ → dendrite/)
    load_dotenv(Path(__file__).resolve().parents[3] / ".env")

    async def main() -> None:
        provider = AnthropicProvider(
            model="claude-sonnet-4-6",
            api_key=os.environ.get("ANTHROPIC_API_KEY"),
        )
        async with Agent(
            provider=provider,
            prompt=agent.prompt,
            tools=agent.tools,
        ) as a:
            result = await a.run("What is 15 + 27?")
            print(f"Answer: {result.answer}")
            print(f"Steps: {result.iteration_count}, Tokens: {result.usage.total_tokens}")

    asyncio.run(main())
