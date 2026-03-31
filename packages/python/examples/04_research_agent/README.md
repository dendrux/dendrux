# Example 04: Research Agent (Multi-Agent Composition)

Demonstrates the **agent-as-tool** pattern — an orchestrator agent delegates to specialist sub-agents, each with their own tools, prompts, and reasoning loops.

## Architecture

```
Orchestrator (ResearchOrchestrator)
  |
  +-- research_topic(query)     --> SearchAgent --> firecrawl_search
  +-- deep_read(url)            --> ScrapeAgent --> firecrawl_scrape
  +-- save_report(filename)     --> writes .md to output/
```

Each sub-agent is a full Dendrux `Agent` — it gets its own system prompt, makes its own LLM calls, and reasons independently. The orchestrator sees them as regular tools.

## Setup

```bash
cd packages/python
pip install -e ".[anthropic,db]"
pip install firecrawl-py python-dotenv
```

Create a `.env` file in `examples/04_research_agent/`:

```
ANTHROPIC_API_KEY=sk-ant-...
FIRECRAWL_API_KEY=fc-...
```

Get a Firecrawl API key at https://firecrawl.dev (free tier available).

## Run

```bash
cd examples/04_research_agent
python main.py "quantum computing breakthroughs 2025"
```

The orchestrator will:
1. Break the topic into focused queries
2. Delegate each query to a SearchAgent (max 3 searches)
3. Optionally deep-read promising URLs via ScrapeAgent (max 2 reads)
4. Synthesize findings into a markdown report
5. Save to `output/<topic>.md`

## Token Budget

Sub-agent calls are expensive (each one does its own LLM reasoning loop), so call counts are capped using Dendrux's built-in `max_calls_per_run`:

```python
@tool(max_calls_per_run=3, timeout_seconds=120)
async def research_topic(query: str) -> str:
    ...
```

- **3** search calls (`research_topic`)
- **2** deep reads (`deep_read`)

When a tool hits its limit, Dendrux returns a graceful message to the LLM — no crash, the agent adapts and moves to synthesis.

## Inspect

With persistence enabled, inspect any run after it completes:

```bash
dendrux runs
dendrux traces <run_id> --tools
```

## Files

```
04_research_agent/
├── main.py                  # Orchestrator agent + entry point
├── agents/
│   ├── search_agent.py      # SearchAgent: query -> Firecrawl search -> summary
│   └── scrape_agent.py      # ScrapeAgent: URL -> Firecrawl scrape -> summary
├── tools/
│   └── firecrawl_tools.py   # Raw Firecrawl SDK wrappers
└── output/                  # Generated reports
```
