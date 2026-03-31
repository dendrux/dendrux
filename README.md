# 🌿 Dendrite

The Python framework for building agents that use tools, persist state, and pause for human input.

> 🧪 **Research project** — built with [Claude Code](https://claude.ai/code) and [Codex](https://openai.com/codex). This is an actively developed research repo, not a production-hardened library. APIs may change. Use it to learn, experiment, and build — but evaluate carefully before depending on it in production.

## 📦 Install

```bash
pip install -e ".[all]"          # everything (Anthropic + OpenAI + DB + bridge)
pip install -e ".[anthropic,db]" # just Anthropic + SQLite
pip install -e ".[openai,db]"    # just OpenAI + SQLite
```

## 🚀 Quick Start

```python
import asyncio
from dendrite import Agent, tool
from dendrite.llm.anthropic import AnthropicProvider

@tool()
async def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

async def main():
    async with Agent(
        provider=AnthropicProvider(model="claude-sonnet-4-6"),
        prompt="You are a calculator.",
        tools=[add],
    ) as agent:
        result = await agent.run("What is 15 + 27?")
        print(result.answer)

asyncio.run(main())
```

```bash
ANTHROPIC_API_KEY=sk-... python my_agent.py
```

That's it. Tools are plain Python functions. The agent calls them, reasons about results, and returns an answer.

---

## 🤖 Provider Choices

Dendrite works with multiple LLM providers. Swap one import — everything else stays the same.

### Anthropic (Claude)

```bash
pip install -e ".[anthropic,db]"
```

```python
from dendrite.llm.anthropic import AnthropicProvider

provider = AnthropicProvider(model="claude-sonnet-4-6")
# Uses ANTHROPIC_API_KEY from environment
```

### OpenAI (Chat Completions)

Works with OpenAI and any compatible API — vLLM, SGLang, Groq, Together, Ollama.

```bash
pip install -e ".[openai,db]"
```

```python
from dendrite.llm.openai import OpenAIProvider

# OpenAI
provider = OpenAIProvider(model="gpt-4o")

# vLLM / SGLang (local)
provider = OpenAIProvider(model="llama-3-70b", base_url="http://localhost:8000/v1")

# Groq
provider = OpenAIProvider(model="llama-3.3-70b", base_url="https://api.groq.com/openai/v1")
```

### OpenAI (Responses API)

Use this when you need OpenAI's built-in tools (web search) alongside your own tools.

```python
from dendrite.llm.openai_responses import OpenAIResponsesProvider

provider = OpenAIResponsesProvider(
    model="gpt-4o",
    builtin_tools=["web_search_preview"],
)
```

> **Note:** Reasoning models (o-series, GPT-5) with multi-turn tool calling have a [known limitation](packages/python/src/dendrite/llm/openai_responses.py) — reasoning items are not preserved between turns. Use OpenAIProvider (Chat Completions) for reasoning model tool loops.

---

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in your project root:

```bash
# Required — pick one (or both)
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...

# Optional — database location (default: ~/.dendrite/dendrite.db)
DENDRITE_DATABASE_URL=sqlite+aiosqlite:///path/to/dendrite.db
```

Load it in your code:

```python
from dotenv import load_dotenv
load_dotenv()
```

### Provider Settings

All providers accept these at construction. Per-call kwargs override defaults.

```python
provider = AnthropicProvider(
    model="claude-sonnet-4-6",
    max_tokens=8_000,        # default 16K
    temperature=0.7,          # default: model decides
    timeout=300,              # HTTP timeout in seconds (default 120)
    max_retries=3,            # auto-retry on transient errors (default 3)
)

provider = OpenAIProvider(
    model="gpt-4o",
    max_tokens=8_000,
    temperature=0.7,
    reasoning_effort="medium",  # for o-series / GPT-5
    timeout=300,
    base_url="http://...",      # for vLLM/SGLang/Groq (optional)
)
```

You can also override per call:

```python
result = await agent.run("be creative", temperature=1.0, max_tokens=2000)
```

### Database

Dendrite persists runs, traces, tool calls, and token usage when you provide a `database_url`.

**SQLite (zero config):**

```python
from pathlib import Path

agent = Agent(
    provider=provider,
    database_url=f"sqlite+aiosqlite:///{Path.home() / '.dendrite' / 'dendrite.db'}",
    prompt="...",
    tools=[...],
)
```

Tables are auto-created on first use. No migrations needed.

**Postgres:**

```bash
export DENDRITE_DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/dendrite
dendrite db migrate
```

**Environment variable:** Set `DENDRITE_DATABASE_URL` once and skip passing `database_url` to every agent. The CLI and dashboard also use this env var automatically.

**No database:** Omit `database_url` — the agent runs fine, just no persistence.

---

## ✨ Features

### 📟 Terminal Output (ConsoleObserver)

See what your agent is doing in real time:

```python
from dendrite.observers.console import ConsoleObserver

result = await agent.run("do the thing", observer=ConsoleObserver())
```

Shows iterations, tool calls with params, success/failure, token counts, and a summary panel. Opt-in — your agent code doesn't change.

### 🔒 Tool Constraints

Control how often tools can be called and how long they can run:

```python
@tool(max_calls_per_run=3, timeout_seconds=120)
async def search(query: str) -> str:
    """Search the web. Limited to 3 calls per run."""
    ...
```

When a tool hits its limit, Dendrite returns a graceful message to the LLM — no crash, the agent adapts. If a tool times out, you get a warning with a hint to increase the timeout.

### ⚡ Parallel Tool Execution

When the LLM returns multiple tool calls in one response, Dendrite executes them concurrently:

```python
@tool(parallel=True)    # default — runs concurrently with other parallel tools
async def fast_lookup(id: str) -> str: ...

@tool(parallel=False)   # barrier — runs alone, in order
async def write_to_db(data: str) -> str: ...
```

### ⏸️ Client Tool Pause/Resume

Define tools that run on the client (browser, mobile, Excel). The agent pauses and waits:

```python
@tool(target="client")
async def read_excel_range(sheet: str, range: str) -> str:
    """Read cells from the user's spreadsheet."""
    return ""
```

Mount the bridge for HTTP-based interaction:

```python
from dendrite import bridge

app.mount("/dendrite", bridge(agent, allow_insecure_dev_mode=True))
```

The bridge handles SSE streaming, tool result submission, polling, and cancellation. See [`examples/03_client_tools/`](packages/python/examples/03_client_tools/) for a working demo.

### 🔔 Custom Observers

Build your own observer in 10 lines — Slack, Telegram, webhooks, anything:

```python
class SlackObserver:
    async def on_message_appended(self, message, iteration):
        pass

    async def on_llm_call_completed(self, response, iteration, **kwargs):
        pass

    async def on_tool_completed(self, tool_call, tool_result, iteration):
        status = "ok" if tool_result.success else "failed"
        await post_to_slack(f"Tool {tool_call.name} {status}")
```

Compose multiple observers:

```python
from dendrite.observers.composite import CompositeObserver

result = await agent.run(
    "do the thing",
    observer=CompositeObserver([ConsoleObserver(), SlackObserver()]),
)
```

### 🔐 Persistence + Redaction

Scrub sensitive data before it's stored:

```python
agent = Agent(
    provider=provider,
    database_url="sqlite+aiosqlite:///...",
    redact=lambda text: text.replace("sk-ant-", "[REDACTED]"),
    ...
)
```

### 🧩 Multi-Agent Composition

Use agents as tools inside other agents:

```python
@tool(max_calls_per_run=3, timeout_seconds=120)
async def research(query: str) -> str:
    """Delegate to a specialist research agent."""
    async with Agent(
        provider=OpenAIProvider(model="gpt-4o"),
        prompt="You are a research specialist...",
        tools=[web_search],
    ) as sub_agent:
        result = await sub_agent.run(query)
        return result.answer
```

See [`examples/04_research_agent/`](packages/python/examples/04_research_agent/) for a full multi-agent example.

---

## 📂 Examples

| Example | What it shows |
|---------|--------------|
| [`01_hello_world.py`](packages/python/examples/01_hello_world.py) | Minimal agent with one tool |
| [`02_persistent_agent.py`](packages/python/examples/02_persistent_agent.py) | SQLite persistence + CLI inspection |
| [`03_client_tools/`](packages/python/examples/03_client_tools/) | Client tool pause/resume with bridge |
| [`04_research_agent/`](packages/python/examples/04_research_agent/) | Multi-agent composition with Firecrawl |

---

## 🖥️ CLI and Dashboard

### 📟 CLI

```bash
dendrite runs                    # List all runs
dendrite runs --status success   # Filter by status
dendrite traces <run_id>         # Full conversation trace
dendrite traces <run_id> --tools # With tool call details
dendrite db migrate              # Run Alembic migrations (Postgres)
dendrite db status               # Check migration status
```

### 📊 Dashboard

```bash
dendrite dashboard               # Launch at http://localhost:8001
dendrite dashboard --db ./my.db  # Point at a specific database
dendrite dashboard --port 9000   # Custom port
```

The dashboard shows runs, timelines, tool calls, token usage, and LLM payloads. All data is read-only.

---

## 🐛 Troubleshooting

<details>
<summary><strong>LLM request timed out</strong></summary>

The default HTTP timeout is 120s. For large outputs (reports, long reasoning), increase it:

```python
provider = AnthropicProvider(model="claude-sonnet-4-6", timeout=300)
```

</details>

<details>
<summary><strong>Tool timed out (default)</strong></summary>

Default tool timeout is 120s. For slow tools (sub-agents, API calls), set it explicitly:

```python
@tool(timeout_seconds=300)
async def slow_tool(query: str) -> str: ...
```

The warning message tells you exactly which tool and suggests the fix.

</details>

<details>
<summary><strong>No runs found / wrong database</strong></summary>

The CLI reads from `DENDRITE_DATABASE_URL` or `~/.dendrite/dendrite.db` by default. If your agent writes to a different path:

```bash
dendrite dashboard --db ./path/to/your.db
```

</details>

<details>
<summary><strong>Traces not saved</strong></summary>

Persistence requires `database_url` on the Agent. Without it, the agent runs but nothing is stored:

```python
agent = Agent(provider=provider, database_url="sqlite+aiosqlite:///...", ...)
```

</details>

<details>
<summary><strong>Agent context manager</strong></summary>

Always use `async with Agent(...) as agent:` — this ensures the HTTP client and database connections are closed cleanly. Without it, connections may leak.

</details>

---

## 📊 Current Status (v0.1.0a1)

| Feature | Status |
|---------|--------|
| Agent loop + ReAct reasoning | Shipped |
| Tool calling (sync + async, parallel, timeouts, max_calls) | Shipped |
| Anthropic Claude provider | Shipped |
| OpenAI Chat Completions provider | Shipped |
| OpenAI Responses API provider | Shipped (reasoning model tool loops limited) |
| Compatible APIs (vLLM, SGLang, Groq, Ollama) | Shipped |
| SQLite + Postgres persistence | Shipped |
| CLI + Dashboard | Shipped |
| Observer system (console, composite, custom) | Shipped |
| Client tool pause/resume | Shipped |
| Bridge transport (SSE) | Experimental |
| Token-level streaming | Planned |
| Gemini provider | Planned |
| TypeScript client SDK | Planned |

---

## 🧑‍💻 Development

```bash
cd packages/python
pip install -e ".[dev,db,anthropic,openai,bridge]"
make ci    # lint + typecheck + tests
```

| Command | What it does |
|---------|-------------|
| `make ci` | Lint + typecheck + tests |
| `make test` | Tests only |
| `make format` | Auto-fix formatting |
| `make lint` | Check without fixing |
| `make typecheck` | mypy strict mode |

<details>
<summary>Install extras</summary>

| Extra | What it adds |
|-------|-------------|
| `anthropic` | Anthropic Claude SDK |
| `openai` | OpenAI SDK |
| `db` | SQLAlchemy, aiosqlite, Alembic |
| `bridge` | FastAPI, uvicorn |
| `postgres` | asyncpg |
| `dev` | pytest, ruff, mypy |

</details>

---

## 🤖 How This Was Built

Dendrite is built through AI pair programming using [Claude Code](https://claude.ai/code) and [Codex](https://openai.com/codex). Every commit is co-authored, every design decision discussed before implementation. The architecture, tests, and documentation were developed collaboratively between a human developer and AI assistants.

This is a research project exploring what's possible when AI tools build AI infrastructure. The code is real, tested, and functional — but it reflects the pace and priorities of research, not the hardening of a production release.

---

## 📄 License

[Apache 2.0](LICENSE)
