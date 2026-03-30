"""Scrape Agent — deep content reader.

Takes a URL, scrapes it via Firecrawl, and returns a focused
summary of the content.
"""

from __future__ import annotations

from dendrite import Agent
from dendrite.llm.anthropic import AnthropicProvider

from tools.firecrawl_tools import firecrawl_scrape

SCRAPE_PROMPT = """\
You are a content extraction specialist. Your job is to read a web page and return
a clear, structured summary of its key information.

Instructions:
- Scrape the given URL using the firecrawl_scrape tool
- Extract the most important facts, data points, and arguments
- Organize the summary with clear sections
- Preserve specific numbers, dates, and quotes when relevant
- Note the source URL for attribution
- If the page is inaccessible or empty, say so honestly
"""


async def run_scrape(url: str) -> str:
    """Run the scrape agent on a URL and return its answer."""
    async with Agent(
        name="ScrapeAgent",
        provider=AnthropicProvider(model="claude-sonnet-4-6"),
        prompt=SCRAPE_PROMPT,
        tools=[firecrawl_scrape],
    ) as agent:
        result = await agent.run(f"Read and summarize this page: {url}")
        return result.answer or "Scrape agent returned no answer."
