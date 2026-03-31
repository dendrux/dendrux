# 🌿 Dendrite

> *The runtime for agents that act in the real world.*

Dendrite is a Python framework for building agents with tool calling, persistence, and full observability. Its core differentiator is the **client tool bridge**: an agent can pause when it needs a client-side tool like Excel, a browser, or a mobile app, wait for the client to execute it, then resume reasoning.

---

## ✨ What You Can Do Today

- 🛠️ Define agents with plain Python tools
- 🤖 Run agents locally with Anthropic Claude
- 💾 Persist runs, traces, tool calls, and token usage to SQLite or Postgres
- 🔍 Inspect runs and traces from the CLI
- 🌐 Mount a bridge for SSE streaming and client tool interaction
- ⏸️ Pause for client-side tools, submit results, and resume the run

---

## 📦 Install

```bash
git clone https://github.com/anmolgautam/dendrite.git
cd dendrite/packages/python
pip install -e ".[anthropic,db,bridge]"
```

For development:

```bash
pip install -e ".[dev,db,anthropic,bridge]"
```

<details>
<summary>📋 Install extras</summary>

| Extra | What it adds |
|-------|-------------|
| `anthropic` | Anthropic Claude SDK |
| `db` | SQLAlchemy, aiosqlite, Alembic |
| `bridge` | FastAPI, uvicorn (for client tool bridge) |
| `dev` | pytest, ruff, mypy, python-dotenv |
| `postgres` | asyncpg (for Postgres instead of SQLite) |

</details>

---

## 🚀 After Installing

> All commands below assume you are in `packages/python/`. The `dendrite` CLI, examples, and `make` targets all run from this directory.

### 1. Verify the install

```bash
dendrite --version
```

You should see `0.1.0a1`. You're good to go! 🎉

### 2. Set your API key

```bash
cp ../../.env.example ../../.env
```

Edit `.env` and add your Anthropic API key:

```
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

### 3. Try the examples

Run them in order — each one builds on the previous:

```bash
# 🧮 Example 1: Minimal agent with one tool
ANTHROPIC_API_KEY=sk-... python examples/01_hello_world.py

# 💾 Example 2: Agent with persistence — traces saved to SQLite
ANTHROPIC_API_KEY=sk-... python examples/02_persistent_agent.py

# 🔍 Inspect what happened
dendrite runs
dendrite traces <run_id> --tools

# ⏸️ Example 3: Client tool bridge — agent pauses for your input
ANTHROPIC_API_KEY=sk-... python examples/03_client_tools/server.py
# Open http://localhost:8000

# 🔬 Example 4: Multi-agent research — orchestrator delegates to sub-agents
cd examples/04_research_agent
ANTHROPIC_API_KEY=sk-... FIRECRAWL_API_KEY=fc-... python main.py "quantum computing"
# Report saved to output/
```

For the client tool demo (example 3), use a prompt like:

> *"Look up the AAPL stock price, then read cell A1 from my spreadsheet."*

The agent will call the server tool immediately, pause for the client tool, wait for your input in the browser, then resume and finish.

For the research agent (example 4), each sub-agent is a full Dendrite Agent with its own tools and reasoning loop. See [`examples/04_research_agent/`](packages/python/examples/04_research_agent/) for details.

### 4. Build your own agent

Create a Python file with three things — tools, an agent, and a run call:

```python
import asyncio
from dendrite import Agent, tool
from dendrite.llm.anthropic import AnthropicProvider

# 1. Define tools — plain async functions
@tool()
async def my_tool(input: str) -> str:
    """Describe what this tool does — the LLM reads this."""
    return f"Result for {input}"

# 2. Create agent with provider and run it
async def main():
    async with Agent(
        provider=AnthropicProvider(model="claude-sonnet-4-6"),
        prompt="You are a helpful assistant. Use my_tool when needed.",
        tools=[my_tool],
    ) as agent:
        result = await agent.run("Do the thing")
        print(result.answer)

asyncio.run(main())
```

Run it with:

```bash
ANTHROPIC_API_KEY=sk-... python my_agent.py
```

See `examples/01_hello_world.py` as a minimal template.

---

## 🧩 Three Ways To Use Dendrite

### 1. 🧮 Local agent run

```python
from dendrite import Agent, tool
from dendrite.llm.anthropic import AnthropicProvider

@tool()
async def add(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b

async with Agent(
    provider=AnthropicProvider(model="claude-sonnet-4-6"),
    prompt="You are a helpful calculator.",
    tools=[add],
) as agent:
    result = await agent.run("What is 15 + 27?")
    print(result.answer)
```

### 2. 💾 Persistent runs and inspection

```python
async with Agent(
    provider=AnthropicProvider(model="claude-sonnet-4-6"),
    database_url=f"sqlite+aiosqlite:///{Path.home() / '.dendrite' / 'dendrite.db'}",
    prompt="You are a helpful calculator.",
    tools=[add],
) as agent:
    result = await agent.run("What is 15 + 27?")
    # Every step is now persisted to the database
```

Once persistence is enabled, inspect runs and traces anytime:

```bash
dendrite runs
dendrite runs --status success
dendrite traces <run_id>
dendrite traces <run_id> --tools
```

### 3. ⏸️ Client tool pause/resume with bridge

Define a client tool with `target="client"` — the agent pauses instead of executing it:

```python
@tool(target="client")
async def read_excel_range(sheet: str, range: str) -> str:
    """Read cells from the user's spreadsheet.

    Runs on the client — agent pauses until the result is submitted.
    """
    return ""
```

Mount the bridge for HTTP-based pause/resume *(experimental)*:

```python
from dendrite import Agent, bridge

agent = Agent(
    provider=AnthropicProvider(model="claude-sonnet-4-6"),
    database_url=f"sqlite+aiosqlite:///{Path.home() / '.dendrite' / 'dendrite.db'}",
    prompt="You are a spreadsheet analyst.",
    tools=[lookup_price, read_excel_range],
)

# Developer owns run creation
@app.post("/chat")
async def chat(request):
    result = await agent.run(request.input)
    return {"run_id": result.run_id, "status": result.status.value}

# Bridge handles paused-run interaction
app.mount("/dendrite", bridge(agent, allow_insecure_dev_mode=True))
```

**The HTTP flow:**

```
Browser/Client              Your Server + Bridge           Agent
      |                           |                          |
  1.  |-- POST /chat ---------->  |-- agent.run() -------->  |
      |                           |                          |-- LLM call
      |                           |                          |-- server tool
      |                           |                          |-- client tool -> PAUSE
      | <-- {run_id, status} ---- |                          |
      |                           |                          |
  2.  |-- GET /dendrite/runs/{id}/events (SSE) ----------->  |
      | <-- snapshot (waiting_client_tool) ---------------- |
      |                           |                          |
  3.  |-- POST /dendrite/runs/{id}/tool-results ---------->  |
      | <-- 200 (accepted) ------ |-- resume_claimed() --->  |-- LLM call
  4.  | <-- run.completed (SSE)   | <-- RunResult --------  |
```

See the working demo: [`examples/03_client_tools/`](packages/python/examples/03_client_tools/)

---

## 🔧 How Tool Calling Works

Dendrite uses **native tool calling** — no prompt engineering, no XML parsing, no regex hacks.

When you decorate a function with `@tool()`, Dendrite extracts the function signature and docstring into a JSON schema. This schema is passed directly to the LLM provider's native tool API (Anthropic's `tool_use` blocks, OpenAI's `function_calling`). The provider returns structured tool call objects, not text that needs parsing.

```
@tool() decorator          NativeToolCalling strategy         Anthropic API
      │                              │                              │
      ├─ extract signature ──>       │                              │
      ├─ generate JSON schema ──>    ├─ pass schemas to provider ─> ├─ tools parameter
      │                              │                              │
      │                              │  <── tool_use blocks ──────  ├─ Claude responds
      │                              ├─ normalize to ToolCall ──>   │
      │                              ├─ format result as TOOL msg ─>│
```

**What this means:**
- Tool definitions go to the API as structured schemas, not injected into the system prompt
- Tool calls come back as typed objects, not parsed from free text
- Tool results are correlated by `call_id` — no ambiguity about which call a result belongs to
- The strategy layer is swappable — same agent can use different strategies without changing tool code

`NativeToolCalling` is the default and works automatically. For advanced use, you can build a custom strategy (e.g. prompt-based for providers without native tool calling) by subclassing `Strategy`:

```python
from dendrite.strategies.base import Strategy

class MyCustomStrategy(Strategy):
    def build_messages(self, *, system_prompt, history, tool_defs):
        # Control what messages and tool schemas go to the LLM
        ...

    def parse_response(self, response):
        # Interpret the LLM response into an AgentStep
        ...

    def format_tool_result(self, result):
        # Format tool results for the next LLM turn
        ...
```

The strategy never calls the LLM directly — it prepares inputs and interprets outputs. The loop handles execution.

---

## 💾 Persistence and Database

Dendrite supports SQLite and Postgres.

### SQLite (default — zero config)

SQLite is the default. It auto-creates tables on first use. No migrations needed.

If `DENDRITE_DATABASE_URL` is not set, Dendrite uses:

```
sqlite+aiosqlite:///./dendrite.db
```

### Postgres

For Postgres, set `DENDRITE_DATABASE_URL` and run Alembic migrations before using persistence:

```bash
export DENDRITE_DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/dendrite
dendrite db migrate
dendrite db status
```

> **Note:** `dendrite db migrate` runs `alembic upgrade head`. The CLI resolves Alembic config automatically and works from any directory.

### ⚠️ Schema changes after pulling new code

SQLite auto-create only works on a **fresh** database. If you pull code that adds new columns to an existing `dendrite.db`:

```bash
# Option 1: Delete and recreate (local dev)
rm dendrite.db

# Option 2: Run migrations (preserves data)
dendrite db migrate
```

Postgres always requires `dendrite db migrate` after schema changes.

---

## 📟 CLI Cheatsheet

```bash
# List runs
dendrite runs
dendrite runs --limit 50
dendrite runs --status success
dendrite runs --tenant tenant_123

# Inspect traces
dendrite traces <run_id>
dendrite traces <run_id> --tools

# Database
dendrite db migrate
dendrite db status
```

---

## 🐛 Common Debugging

<details>
<summary><strong>"No runs found"</strong></summary>

You are probably looking at the wrong database. Check:

```bash
echo $DENDRITE_DATABASE_URL
```

If unset, Dendrite uses local SQLite at `./dendrite.db`. Make sure you are running CLI commands from the same directory where the agent ran.

</details>

<details>
<summary><strong>"I ran an agent but no traces were saved"</strong></summary>

Persistence is enabled by passing `database_url` to the Agent:

```python
agent = Agent(
    provider=provider,
    database_url="sqlite+aiosqlite:///path/to/dendrite.db",
    prompt="...",
    tools=[...],
)
result = await agent.run("...")
```

Without `database_url` (or `state_store`), the agent still runs but nothing is stored.

</details>

<details>
<summary><strong>"Postgres tables are missing"</strong></summary>

You likely skipped migrations:

```bash
dendrite db migrate
dendrite db status
```

SQLite auto-creates tables. Postgres does not.

</details>

<details>
<summary><strong>"SQLite errors after pulling new code"</strong></summary>

If a new column was added and you have an existing `dendrite.db`:

```bash
# Delete and recreate (simplest for local dev)
rm dendrite.db

# Or run migrations to update in place
dendrite db migrate
```

</details>

<details>
<summary><strong>"The client tool demo does not pause"</strong></summary>

Your prompt probably did not trigger the client tool. Use an explicit prompt like:

> *"Look up the AAPL stock price, then read cell A1 from my spreadsheet."*

The pause only happens when the model chooses a tool with `target="client"`.

</details>

<details>
<summary><strong>"My hosted run is stuck in waiting_client_tool"</strong></summary>

That is expected until the client submits tool results. Resume it by posting to `POST /runs/{run_id}/tool-results`:

```json
{
  "tool_results": [
    {
      "tool_call_id": "...",
      "tool_name": "read_excel_range",
      "result": "Revenue: $394B"
    }
  ]
}
```

</details>

<details>
<summary><strong>"I get 401 on run endpoints"</strong></summary>

HMAC auth is enabled and the request is missing a valid bearer token for that specific run.

For local demos, use `allow_insecure_dev_mode=True` when creating the bridge.

For real deployments, pass `secret=` to `bridge()` and send the token in the header:

```
Authorization: Bearer drn_...
```

</details>

<details>
<summary><strong>"How do I inspect what actually happened?"</strong></summary>

```bash
dendrite runs                       # List all runs with status
dendrite traces <run_id>            # Full conversation trace
dendrite traces <run_id> --tools    # Trace + tool call details
```

`dendrite traces --tools` is the fastest way to see both the conversation and tool execution records.

</details>

---

## 🗂️ Project Structure

```
packages/python/
├── src/dendrite/
│   ├── agent.py            # Agent — definition + runtime facade
│   ├── tool.py             # @tool decorator + schema generation
│   ├── types.py            # Core types (Message, ToolCall, RunResult)
│   ├── auth.py             # Run-scoped HMAC token utilities
│   ├── llm/                # LLM providers (Anthropic, Mock)
│   ├── loops/              # Execution loops (ReAct)
│   ├── strategies/         # Tool calling strategies
│   ├── runtime/            # Runner, observer, state store
│   ├── db/                 # SQLAlchemy models, Alembic migrations
│   ├── bridge/             # Mountable transport (SSE, pause/resume)
│   └── cli/                # CLI commands (runs, traces, db, dashboard)
├── examples/
│   ├── 01_hello_world.py
│   ├── 02_persistent_agent.py
│   ├── 03_client_tools/
│   └── 04_research_agent/
└── tests/                  # 449 tests, 92% coverage
```

---

## 📊 Current Status (v0.1.0a1)

| Feature | Status |
|---------|--------|
| Agent loop + ReAct reasoning | ✅ Shipped |
| Tool calling (sync + async, timeouts) | ✅ Shipped |
| Anthropic Claude provider | ✅ Shipped |
| OpenAI Chat Completions provider | ✅ Shipped |
| OpenAI Responses API provider | ⚠️ Shipped (reasoning model tool loops limited) |
| Compatible API support (vLLM, SGLang, Groq, Ollama) | ✅ Shipped (via OpenAIProvider base_url) |
| SQLite + Postgres persistence | ✅ Shipped |
| CLI (traces, runs, db, dashboard) | ✅ Shipped |
| Token usage tracking + redaction | ✅ Shipped |
| Pause/resume for client tools | ✅ Shipped |
| Bridge transport (SSE + persist-first handoff) | ⚠️ Experimental |
| Run-scoped HMAC auth | ✅ Shipped |
| Agent API (provider, database_url, run/resume) | ✅ Shipped |
| Parallel tool execution + max_calls_per_run | ✅ Shipped |
| Token-level streaming (agent.stream()) | 🔜 Planned |
| TypeScript client SDK | 🔜 Planned |
| Gemini provider | 🔜 Planned |
| Tool sandbox / isolation | 🔜 Planned |

---

## ⚙️ Requirements

- Python 3.11+
- An Anthropic API key (for running agents with Claude)

Tests run without an API key — they use `MockLLM` for deterministic testing.

---

## 🧑‍💻 Development

```bash
cd packages/python
pip install -e ".[dev,db,anthropic,bridge]"
make ci
```

| Command | What it does |
|---------|-------------|
| `make ci` | Lint + typecheck + tests — **run before every commit** |
| `make test` | Tests only |
| `make format` | Auto-fix formatting and lint |
| `make lint` | Check without fixing |
| `make typecheck` | mypy strict mode |

---

## 🤖 Built with Claude Code

Dendrite is built through AI pair programming using [Claude Code](https://claude.ai/code). Every commit is co-authored, every design decision discussed before implementation. We use a 5-layer doc architecture to keep the LLM aligned across conversations — vision docs describe the destination, the status map describes reality.

---

## 📄 License

[Apache 2.0](LICENSE)
