"""Search Agent — web research specialist.

Takes a query, searches the web via Firecrawl, and returns
a structured summary of findings.
"""

from __future__ import annotations

from dendrux import Agent
from dendrux.llm.anthropic import AnthropicProvider

from tools.firecrawl_tools import firecrawl_search

SEARCH_PROMPT = """\
You are a web research specialist. Your job is to search for information on a given query
and return a clear, factual summary of what you find.

Instructions:
- Search for the query using the firecrawl_search tool
- You may search multiple times with different phrasings to get broader coverage
- Synthesize the results into a structured summary
- Include source URLs for key claims
- Focus on facts, not opinions
- If results are thin, say so honestly — don't fabricate
"""


async def run_search(query: str) -> str:
    """Run the search agent on a query and return its answer."""
    async with Agent(
        name="SearchAgent",
        provider=AnthropicProvider(model="claude-sonnet-4-6"),
        prompt=SEARCH_PROMPT,
        tools=[firecrawl_search],
    ) as agent:
        result = await agent.run(query)
        return result.answer or "Search agent returned no answer."
