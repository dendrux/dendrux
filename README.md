# 🌿 Dendrite

> *The runtime for agents that act in the real world.*

Dendrite is a Python framework for building agents with tool calling, persistence, and full observability. Its core differentiator is the **client tool bridge**: an agent can pause when it needs a client-side tool like Excel, a browser, or a mobile app, wait for the client to execute it, then resume reasoning.

---

## ✨ What You Can Do Today

- 🛠️ Define agents with plain Python tools
- 🤖 Run agents locally with Anthropic Claude
- 💾 Persist runs, traces, tool calls, and token usage to SQLite or Postgres
- 🔍 Inspect runs and traces from the CLI
- 🌐 Host agents behind a FastAPI server with SSE streaming
- ⏸️ Pause for client-side tools, submit results, and resume the run

---

## 📦 Install

```bash
git clone https://github.com/anmolgautam/dendrite.git
cd dendrite/packages/python
pip install -e ".[anthropic,db,server]"
```

For development:

```bash
pip install -e ".[dev,db,anthropic,server]"
```

<details>
<summary>📋 Install extras</summary>

| Extra | What it adds |
|-------|-------------|
| `anthropic` | Anthropic Claude SDK |
| `db` | SQLAlchemy, aiosqlite, Alembic |
| `server` | FastAPI, uvicorn |
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
dendrite run examples/01_hello_world.py -i "What is 15 + 27?"

# 💾 Example 2: Agent with persistence — traces saved to SQLite
python examples/02_persistent_agent.py

# 🔍 Inspect what happened
dendrite runs
dendrite traces <run_id> --tools

# ⏸️ Example 3: Client tool bridge — agent pauses for your input
python examples/03_client_tools/server.py
# Open http://localhost:8000
```

For the client tool demo, use a prompt like:

> *"Look up the AAPL stock price, then read cell A1 from my spreadsheet."*

The agent will call the server tool immediately, pause for the client tool, wait for your input in the browser, then resume and finish.

### 4. Build your own agent

Create a Python file with three things — tools, an agent, and a run call:

```python
from dendrite import Agent, tool, run
from dendrite.llm.anthropic import AnthropicProvider

# 1. Define tools — plain async functions
@tool()
async def my_tool(input: str) -> str:
    """Describe what this tool does — the LLM reads this."""
    return f"Result for {input}"

# 2. Define an agent
agent = Agent(
    name="MyAgent",
    model="claude-sonnet-4-6",
    prompt="You are a helpful assistant. Use my_tool when needed.",
    tools=[my_tool],
)

# 3. Run it
import asyncio

async def main():
    provider = AnthropicProvider(
        api_key="sk-ant-...",
        model=agent.model,
    )
    result = await run(agent, provider=provider, user_input="Do the thing")
    print(result.answer)

asyncio.run(main())
```

Run it with:

```bash
python my_agent.py
```

Or use the CLI (auto-detects the agent in the file):

```bash
dendrite run my_agent.py -i "Do the thing"
```

See `examples/01_hello_world.py` as a minimal template.

---

## 🧩 Three Ways To Use Dendrite

### 1. 🧮 Local agent run

```python
from dendrite import Agent, tool, run
from dendrite.llm.anthropic import AnthropicProvider

@tool()
async def add(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b

agent = Agent(
    name="Calculator",
    model="claude-sonnet-4-6",
    prompt="You are a helpful calculator.",
    tools=[add],
)

provider = AnthropicProvider(
    api_key="sk-ant-...",
    model=agent.model,
)

result = await run(
    agent,
    provider=provider,
    user_input="What is 15 + 27?",
)

print(result.answer)
```

### 2. 💾 Persistent runs and inspection

```python
from dendrite.db.session import get_engine
from dendrite.runtime.state import SQLAlchemyStateStore

engine = await get_engine()  # Auto-creates ./dendrite.db
store = SQLAlchemyStateStore(engine)

result = await run(
    agent,
    provider=provider,
    user_input="What is 15 + 27?",
    state_store=store,  # Every step is now persisted
)
```

Once persistence is enabled, inspect runs and traces anytime:

```bash
dendrite runs
dendrite runs --status success
dendrite traces <run_id>
dendrite traces <run_id> --tools
```

### 3. ⏸️ Hosted agents with client tool pause/resume

Define a client tool with `target="client"` — the agent pauses instead of executing it:

```python
@tool(target="client")
async def read_excel_range(sheet: str, range: str) -> str:
    """Read cells from the user's spreadsheet.

    Runs on the client — agent pauses until the result is submitted.
    """
    return ""
```

Host agents over HTTP using FastAPI:

```python
from dendrite.server import create_app, AgentRegistry, HostedAgentConfig

registry = AgentRegistry()
registry.register(HostedAgentConfig(
    agent=agent,
    provider_factory=lambda: AnthropicProvider(
        api_key="sk-ant-...",
        model=agent.model,
    ),
))

dendrite_app = create_app(
    state_store=store,
    registry=registry,
    hmac_secret="your-secret",  # or allow_insecure_dev_mode=True
)

# Mount on your FastAPI app
app.mount("/dendrite", dendrite_app)
```

**The HTTP flow:**

```
Browser/Client              Dendrite Server              Agent Loop
      │                           │                          │
  1.  ├── POST /runs ──────────>  │ ── run() ─────────────>  │
      │                           │                          ├─ LLM call
  2.  ├── GET /events (SSE) ───>  │                          ├─ server tool ✅
      │  <── run.step ──────────  │                          ├─ client tool → PAUSE ⏸️
      │  <── run.paused ────────  │                          │
      │                           │                          │  (waiting...)
  3.  ├── POST /tool-results ──>  │ ── resume() ──────────>  │
      │                           │                          ├─ LLM call
  4.  │  <── run.completed ─────  │ <── RunResult ─────────  │  ✅
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

`NativeToolCalling` is the default. You can pass it explicitly or swap in a different strategy:

```python
from dendrite.strategies.native import NativeToolCalling

# Default — you don't need to pass this, but you can
result = await run(
    agent,
    provider=provider,
    user_input="What is 15 + 27?",
    strategy=NativeToolCalling(),  # explicit
)
```

To build a custom strategy (e.g. prompt-based for providers without native tool calling), subclass `Strategy`:

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

# Use it
result = await run(agent, provider=provider, user_input="...", strategy=MyCustomStrategy())
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
# Run an agent
dendrite run examples/01_hello_world.py -i "What is 15 + 27?"

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

Persistence is only enabled when you pass a `state_store`:

```python
result = await run(
    agent,
    provider=provider,
    user_input="...",
    state_store=SQLAlchemyStateStore(engine),  # Required for persistence
)
```

Without `state_store`, the agent still runs but nothing is stored.

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

For local demos, use `allow_insecure_dev_mode=True` when creating the app.

For real deployments, send the token returned by `POST /runs` in the header:

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
│   ├── agent.py            # Agent definition
│   ├── tool.py             # @tool decorator + schema generation
│   ├── types.py            # Core types (Message, ToolCall, RunResult)
│   ├── llm/                # LLM providers (Anthropic, Mock)
│   ├── loops/              # Execution loops (ReAct)
│   ├── strategies/         # Tool calling strategies
│   ├── runtime/            # Runner, observer, state store
│   ├── db/                 # SQLAlchemy models, Alembic migrations
│   ├── server/             # FastAPI app, SSE, HMAC auth
│   └── cli/                # CLI commands
├── examples/
│   ├── 01_hello_world.py
│   ├── 02_persistent_agent.py
│   └── 03_client_tools/
└── tests/                  # 351 tests, 89% coverage
```

---

## 📊 Current Status (v0.1.0a1)

| Feature | Status |
|---------|--------|
| Agent loop + ReAct reasoning | ✅ Shipped |
| Tool calling (sync + async, timeouts) | ✅ Shipped |
| Anthropic Claude provider | ✅ Shipped |
| SQLite + Postgres persistence | ✅ Shipped |
| CLI (run, traces, runs, db) | ✅ Shipped |
| Token usage tracking + redaction | ✅ Shipped |
| Pause/resume for client tools | ✅ Shipped |
| FastAPI hosting + SSE transport | ✅ Shipped |
| Run-scoped HMAC auth | ✅ Shipped |
| Worker pool / crash recovery | 🔜 Planned |
| TypeScript client SDK | 🔜 Planned |
| OpenAI + multi-provider support | 🔜 Planned |
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
pip install -e ".[dev,db,anthropic,server]"
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
