"""Research Agent — multi-agent composition with Dendrux.

Demonstrates the agent-as-tool pattern: an orchestrator agent delegates
to specialist sub-agents (search, scrape) that each have their own
tools, prompts, and reasoning loops.

Architecture:
    Orchestrator Agent
      +-- tool: research_topic(query)  -> runs SearchAgent internally
      |     +-- tool: firecrawl_search(query)  -> Firecrawl API
      +-- tool: deep_read(url)         -> runs ScrapeAgent internally
      |     +-- tool: firecrawl_scrape(url)    -> Firecrawl API
      +-- tool: save_report(content)   -> writes .md to disk

Requirements:
    pip install firecrawl-py python-dotenv

Usage:
    cd examples/04_research_agent
    ANTHROPIC_API_KEY=sk-... FIRECRAWL_API_KEY=fc-... python main.py "your topic"
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv

from dendrux import Agent, tool
from dendrux.llm.anthropic import AnthropicProvider

from agents.search_agent import run_search
from agents.scrape_agent import run_scrape

load_dotenv()

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


# --- Orchestrator tools: each delegates to a sub-agent ---


@tool(max_calls_per_run=3, timeout_seconds=120)
async def research_topic(query: str) -> str:
    """Search the web for a query and get summarized findings.

    You have a maximum of 3 search calls — plan your queries wisely.
    """
    print(f"  [search] {query}")
    return await run_search(query)


@tool(max_calls_per_run=2, timeout_seconds=120)
async def deep_read(url: str) -> str:
    """Read and summarize a specific web page in depth.

    You have a maximum of 2 deep reads — use them on the most valuable sources.
    """
    print(f"  [deep_read] {url}")
    return await run_scrape(url)


@tool()
async def save_report(filename: str, content: str) -> str:
    """Save the final research report as a markdown file.

    Args:
        filename: Name for the file (without extension, e.g. 'quantum_computing')
        content: The full markdown content of the report
    """
    filepath = OUTPUT_DIR / f"{filename}.md"
    filepath.write_text(content, encoding="utf-8")
    return f"Report saved to {filepath}"


ORCHESTRATOR_PROMPT = """\
You are a research orchestrator. Your job is to produce a comprehensive, well-structured
research report on a given topic.

Budget:
- You have exactly 3 search calls (research_topic) and 2 deep reads (deep_read)
- Plan your queries carefully to maximize coverage within these limits

Your workflow:
1. Break the topic into 2-3 focused research queries (not more than 3 — that's your budget)
2. Use research_topic for each query
3. If a search result mentions a high-value source, use deep_read to get full content (max 2)
4. Synthesize all findings into a structured markdown report
5. Save the report using save_report

Report format:
- Title (# heading)
- Executive Summary (2-3 paragraphs)
- Key sections with findings (## headings)
- Each claim should cite its source URL
- Conclusion
- Sources list at the end

Quality standards:
- Be factual — only include information found in your research
- Cite sources for every major claim
- If information is conflicting, present both sides
- If coverage is thin on a subtopic, say so honestly
- Aim for depth over breadth
"""


async def main() -> None:
    topic = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else None
    if not topic:
        print("Usage: python main.py \"your research topic\"")
        sys.exit(1)

    print(f"Researching: {topic}\n")

    db_path = Path(__file__).parent / "research.db"

    async with Agent(
        name="ResearchOrchestrator",
        provider=AnthropicProvider(model="claude-sonnet-4-6"),
        database_url=f"sqlite+aiosqlite:///{db_path}",
        prompt=ORCHESTRATOR_PROMPT,
        tools=[research_topic, deep_read, save_report],
        max_iterations=25,
    ) as agent:
        result = await agent.run(f"Research this topic thoroughly: {topic}")

        print(f"\n{'='*60}")
        print(f"Status: {result.status.value}")
        print(f"Iterations: {result.iteration_count}")
        print(f"Tokens: {result.usage.total_tokens}")
        print(f"Run ID: {result.run_id}")
        print(f"{'='*60}")
        if result.answer:
            print(f"\n{result.answer}")


if __name__ == "__main__":
    asyncio.run(main())
