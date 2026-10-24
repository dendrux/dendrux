# Example 04: Research Agent (Multi-Agent Composition)

Demonstrates the **agent-as-tool** pattern — an orchestrator agent delegates to specialist sub-agents, each with their own tools, prompts, and reasoning loops.

## Architecture

```
Orchestrator (ResearchOrchestrator)
  |
  +-- research_topic(query)     --> SearchAgent --> Firecrawl search
  +-- deep_read(url)            --> ScrapeAgent --> Firecrawl scrape
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

Get a Firecrawl API key at [firecrawl.dev](https://firecrawl.dev) (free tier available).

## Run

```bash
cd examples/04_research_agent
python main.py "quantum computing breakthroughs 2025"
```

The orchestrator will:
1. Break the topic into focused queries
2. Delegate each to a SearchAgent (max 3 searches)
3. Optionally deep-read promising URLs via ScrapeAgent (max 2 reads)
4. Synthesize findings into a markdown report
5. Save to `output/<topic>.md`

## Streaming Mode

Run the same flow with real-time event streaming:

```bash
python stream_test.py "quantum computing 2025"
```

Shows token-by-token text, tool start/end events, and a completion summary.

## What It Shows

- **Agent-as-tool composition** — sub-agents run inside `@tool()` functions
- **`max_calls_per_run`** — runtime-enforced tool call limits (no manual counters)
- **`timeout_seconds`** — explicit timeouts for long-running sub-agent tools
- **`ConsoleNotifier`** — rich terminal output showing iterations, tool calls, and summary
- **Persistence** — all agents share a local `research.db` for run inspection
- **Streaming** — `stream_test.py` demonstrates `agent.stream()` with event handling
- **Multi-provider** — swap `AnthropicProvider` → `OpenAIProvider` in one line

## Inspect Runs

With persistence enabled, inspect any run after it completes:

```bash
dendrux runs
dendrux traces <run_id> --tools
```

## Files

```
04_research_agent/
├── main.py                  # Orchestrator agent + entry point
├── stream_test.py           # Streaming variant with event handling
├── agents/
│   ├── search_agent.py      # SearchAgent: query -> Firecrawl search -> summary
│   └── scrape_agent.py      # ScrapeAgent: URL -> Firecrawl scrape -> summary
├── tools/
│   └── firecrawl_tools.py   # Raw Firecrawl SDK v4 wrappers
└── output/                  # Generated reports
```
