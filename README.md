# 🌿 Dendrux

The async Python runtime for agents that survive failure, persist everything, and bridge to the real world.

[![CI](https://github.com/dendrux/dendrux/actions/workflows/ci.yml/badge.svg)](https://github.com/dendrux/dendrux/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)

> `v0.1.0a5` - core API stabilizing, actively developed.

---

## Why Dendrux?

Agent frameworks assume tools run on the server and finish instantly. Real agents crash mid-run, need human approval, interact with browsers and spreadsheets, and must explain what they did after the fact.

Dendrux is the runtime that handles this. Tools are plain Python functions. State survives restarts. Crashed runs get swept. Failed runs retry with prior context reconstructed from persisted traces (the DB stores raw values — guardrails redact at the LLM boundary only, so retries replay the original conversation). And every LLM call, tool execution, pause, and failure is persisted as evidence.

## Quick Start

```python
import asyncio
from dendrux import Agent, tool
from dendrux.llm.anthropic import AnthropicProvider

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

Three concepts: `Agent`, `@tool()`, provider. Everything else is opt-in.

Production agent in 10 lines. Add PII redaction, tool deny, human approval, and budget tracking in 5 more:

```python
agent = Agent(
    provider=provider,
    tools=[search, refund, delete_account],
    prompt="You are a customer support agent.",
    deny=["delete_account"],
    require_approval=["refund"],
    budget=Budget(max_tokens=30_000),
    guardrails=[PII(), SecretDetection(action="block")],
)
```

---

## Install

```bash
pip install "dendrux[all]"            # everything (Anthropic + OpenAI + DB + http)
pip install "dendrux[anthropic,db]"   # just Anthropic + SQLite
pip install "dendrux[openai,db]"      # just OpenAI + SQLite
```

<details>
<summary>Development install (from source)</summary>

```bash
git clone https://github.com/dendrux/dendrux.git
cd dendrux/packages/python
pip install -e ".[dev,all]"
```

</details>

---

## Six Pillars

Dendrux is built around six design commitments:

| # | Pillar | What it means |
|---|--------|---------------|
| 1 | **Survive failure** | Durable writes, sweep, retry, idempotency. Runs never lie about state. |
| 2 | **Control execution** | Tool constraints, timeouts, parallel/sequential policy, delegation depth guards. |
| 3 | **Govern behavior** | Tool deny/approval, advisory budgets, PII redaction, secret detection. Four layers of runtime governance. |
| 4 | **Explain everything** | Every LLM call, tool execution, pause, and lifecycle event is persisted as evidence. |
| 5 | **Coordinate agents** | Parent-child delegation with automatic linking, depth guards, and lifecycle coupling. |
| 6 | **Pause for the real world** | Client-side tool pause/resume for spreadsheets, browsers, and desktops with domain-aware constraints. |

---

## Features

### 🔄 Survive Failure

Runs crash. Processes die. Dendrux handles it.

**Durable persistence**: every trace, tool call, and lifecycle event is written through a durability layer with retry and exponential backoff. Transient DB failures (connection drops, lock timeouts, SQLite busy) are retried automatically. Logical errors propagate immediately. LLM interactions and token usage are persisted best-effort.

**Sweep stale runs**: call `sweep()` at app startup to detect runs that were RUNNING when the process died or WAITING when no one came back. They're marked ERROR with structured failure reasons.

```python
from dendrux import sweep
from datetime import timedelta

results = await sweep(
    database_url="sqlite+aiosqlite:///runs.db",
    stale_running=timedelta(minutes=20),
    abandoned_waiting=timedelta(hours=2),
)
```

**Retry terminal runs**: a failed, cancelled, or timed-out run can be retried with prior context from persisted traces. The DB stores raw values (guardrails redact at the LLM boundary only), so retry replays the full original conversation. Same agent or a different one, different model, different tools.

```python
result = await agent.retry("01JR...")
```

**Idempotency**: pass a key to prevent duplicate runs. Same key + same input returns the existing run. Same key + different input raises `IdempotencyConflictError`.

```python
result = await agent.run(
    "Analyze Q3 report",
    idempotency_key="q3-analysis-2026",
)
```

### 🛡️ Control Execution

**Tool constraints**: control how often tools run and how long they take. When a limit is hit, the agent gets a graceful message and adapts. No crash.

```python
@tool(max_calls_per_run=3, timeout_seconds=120)
async def search(query: str) -> str:
    """Search the web. Limited to 3 calls per run."""
    ...
```

**Parallel execution**: when the LLM returns multiple tool calls, Dendrux executes them concurrently by default. Mark tools `parallel=False` for sequential execution.

```python
@tool(parallel=True)     # runs concurrently (default)
async def fast_lookup(id: str) -> str: ...

@tool(parallel=False)    # runs alone, in order
async def write_to_db(data: str) -> str: ...
```

**Delegation depth guards**: nested agent calls are tracked automatically. Runaway recursion is caught and stopped.

### 🔐 Govern Behavior

Four layers of runtime governance, all opt-in via kwargs on `Agent`:

```python
from dendrux import Agent, Budget
from dendrux.guardrails import PII, SecretDetection, Pattern

agent = Agent(
    provider=provider,
    tools=[search, refund, delete_account],
    prompt="You are a customer support agent.",
    deny=["delete_account"],                     # block tools deterministically
    require_approval=["refund"],                 # pause for human sign-off
    budget=Budget(max_tokens=30_000),            # advisory spend tracking
    guardrails=[                                 # content scanning at LLM boundary
        PII(extra_patterns=[Pattern("EMPLOYEE_ID", r"EMP-\d{6}")]),
        SecretDetection(action="block"),
    ],
)
```

**Tool deny**: `deny=["tool_name"]` blocks tools deterministically. The model gets a synthesized error and adapts. The tool never executes.

**Approval (HITL)**: `require_approval=["tool_name"]` pauses the run for human sign-off. Approve with `agent.submit_approval(run_id, approved=True)`, reject with `agent.submit_approval(run_id, approved=False, rejection_reason="...")`.

**Advisory budget**: `budget=Budget(max_tokens=N)` fires governance events at configurable thresholds (50%, 75%, 90%) and when usage exceeds the cap. Advisory only - the run continues. Developers observe and act.

**Guardrails**: `guardrails=[PII(), SecretDetection()]` scans content crossing the LLM boundary. Three actions:
- `redact` - PII replaced with `<<EMAIL_ADDRESS_1>>` placeholders. Tools receive real values (deanonymized). Run-scoped mapping persisted for audit.
- `block` - run terminates immediately. The LLM never sees the content.
- `warn` - findings logged, content unchanged. Shadow rollout before promoting to redact.

Custom patterns via `Pattern("NAME", r"regex")`. Extensible `Guardrail` protocol for custom scanners (async `scan()` supports LLM-as-judge). Two detection engines: `PII()` uses a zero-dependency regex scanner by default, `PII(engine="presidio")` opts in to Microsoft Presidio's NLP-backed recognizers (~18 entities, install `dendrux[presidio]`).

**Pipeline per iteration:**

```
                        ┌─────────────────┐
  User Input ──────────▶│ Incoming Guard  │ scan all messages, redact / block / warn
                        └────────┬────────┘
                                 ▼
                        ┌─────────────────┐
                        │    LLM Call     │ model sees <<EMAIL_ADDRESS_1>> placeholders
                        └────────┬────────┘
                                 ▼
                        ┌─────────────────┐
                        │  Output Guard   │ scan response text + tool call params
                        └────────┬────────┘
                                 ▼
  ┌───────┐   ┌──────────────┐   ┌──────────┐   ┌─────────┐   ┌────────┐
  │ Deny  │──▶│ Deanonymize  │──▶│ Approval │──▶│ Execute │──▶│ Budget │
  └───────┘   └──────────────┘   └──────────┘   └─────────┘   └────────┘
  block by    placeholders →     HITL           server /       advisory
  policy      real values        sign-off       client tools   tracking

  ─────────────────────────────────────────────────────────────────────────
  Events: policy.denied │ approval.* │ budget.* │ guardrail.*
  Recorder (fail-closed)  +  Notifier (best-effort)
```

All governance events flow through both the fail-closed recorder (DB audit trail) and best-effort notifier (console, SSE, custom).

### 🔍 Explain Everything

Two event seams, separated by design:

- **Recorder** (internal): authoritative persistence. Fail-closed with durability retry. If a trace can't be written, the run stops.
- **Notifier** (external): best-effort UI notifications. Exceptions swallowed. Never kills a run.

```python
from dendrux.notifiers.console import ConsoleNotifier

result = await agent.run("do the thing", notifier=ConsoleNotifier())
```

Every run persists: traces with full message content, tool calls with parameters and results, LLM interactions with request/response payloads, token usage, timing, delegation links, and lifecycle events.

**PII policy**: the DB is the authoritative execution record and stores raw values. Guardrails redact at the LLM boundary only — what the provider API sees — and the placeholder→real bijection is persisted on `agent_runs.pii_mapping` as the audit key, so dashboards can replay either view from the same rows. See [PII redaction](docs/architecture/pii-redaction.mdx).

### 🌳 Coordinate Agents

Use agents as tools inside other agents. Parent-child relationships are tracked automatically via `contextvars`. Zero developer code.

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

The state store links parent and child runs. The dashboard shows the full delegation tree. Depth guards prevent runaway recursion.

### ⏸️ Pause for the Real World

**Client-side tools**: define tools that run on the client (browser, mobile, Excel). The agent pauses and waits.

```python
@tool(target="client")
async def read_excel_range(sheet: str, range: str) -> str:
    """Read cells from the user's spreadsheet."""
    return ""

# Dendrux ships reads + SSE. You wire the writes around agent methods.
from dendrux.http import make_read_router
from dendrux.store import RunStore
from dendrux.types import ToolResult
from pydantic import BaseModel

class ResultItem(BaseModel):
    tool_call_id: str
    tool_name: str
    payload: str  # JSON-encoded string

class ResultsBody(BaseModel):
    results: list[ResultItem]

store = RunStore.from_database_url(db_url)
app.include_router(make_read_router(store=store, authorize=require_user), prefix="/api")

@app.post("/runs/{run_id}/tool-results")
async def submit(run_id: str, body: ResultsBody, user=Depends(require_user)):
    # Convert the app's DTO into Dendrux's ToolResult — the method expects
    # list[ToolResult], not arbitrary Pydantic models.
    results = [
        ToolResult(name=r.tool_name, call_id=r.tool_call_id, payload=r.payload)
        for r in body.results
    ]
    return await agent.submit_tool_results(run_id, results)
```

Pause/resume orchestration, CAS-safe claim, and SSE transport are Dendrux's job. Auth, schema, error shape, and framework choice are yours.

### 🔀 Streaming

Stream events as they happen: token-by-token text, tool calls, lifecycle events:

```python
# Full event stream
async for event in agent.stream("analyze revenue"):
    print(event.type, event.text or "")

# Just the text
async for chunk in agent.stream("analyze revenue").text():
    print(chunk, end="")

# With cleanup
async with agent.stream("analyze revenue") as stream:
    async for event in stream:
        ...
```

Works with all providers. If the consumer breaks early, the run is cancelled cleanly.

---

## Architecture

<p align="center">
  <img src="docs/architecture.svg" alt="Dendrux architecture layers" width="680">
</p>

The loop never touches provider-specific APIs. The strategy never calls the LLM. Each layer has one job.

**Persistence flows through two separated paths:**
- **Evidence** (right): Recorder → StateStore. Fail-closed, durable retry.
- **Display** (left): Notifier → Console / custom. Best-effort, exceptions swallowed.

---

## Providers

Swap one import. Everything else stays the same.

**Anthropic (Claude)**
```python
from dendrux.llm.anthropic import AnthropicProvider
provider = AnthropicProvider(model="claude-sonnet-4-6")
```

**OpenAI (Chat Completions)**, also works with vLLM, SGLang, Groq, Together, Ollama
```python
from dendrux.llm.openai import OpenAIProvider
provider = OpenAIProvider(model="gpt-4o")
provider = OpenAIProvider(model="llama-3-70b", base_url="http://localhost:8000/v1")
```

**OpenAI (Responses API)**, for built-in tools like web search alongside your own
```python
from dendrux.llm.openai_responses import OpenAIResponsesProvider
provider = OpenAIResponsesProvider(model="gpt-4o", builtin_tools=["web_search_preview"])
```

All providers accept `max_tokens`, `temperature`, `timeout`, and `max_retries` at construction or per-call.

> **Note:** Reasoning models (o-series, GPT-5) with multi-turn tool calling have a [known limitation](packages/python/src/dendrux/llm/openai_responses.py). Reasoning items are not preserved between turns. Use OpenAIProvider (Chat Completions) for reasoning model tool loops.

---

## Database

SQLite works with zero config. Tables auto-create on first use.

```python
agent = Agent(
    provider=provider,
    database_url="sqlite+aiosqlite:///runs.db",
    prompt="...",
    tools=[...],
)
```

Postgres supported via Alembic migrations: `dendrux db migrate`

Set `DENDRUX_DATABASE_URL` once to skip passing `database_url` to every agent.

Omit `database_url` entirely for ephemeral runs. No persistence, no overhead.

---

## CLI and Dashboard

```bash
dendrux runs                    # list all runs
dendrux runs --status error     # filter by status
dendrux traces <run_id>         # full conversation trace
dendrux traces <run_id> --tools # with tool call details
dendrux db migrate              # run Alembic migrations (Postgres)
dendrux dashboard               # launch web dashboard at :8001
```

The dashboard shows runs, timelines, tool calls, token usage, delegation trees, and LLM payloads. All data is read-only. Traces and tool calls are rendered raw; when a PII guardrail is active, `agent_runs.pii_mapping` carries the bijection the LLM saw and can be applied at render time to produce an LLM-eye view.

---

## Examples

| Example | What it shows |
|---------|---------------|
| [`01_hello_world.py`](packages/python/examples/01_hello_world.py) | Minimal agent with one tool |
| [`02_persistent_agent.py`](packages/python/examples/02_persistent_agent.py) | SQLite persistence + CLI inspection |
| [`03_client_tools/`](packages/python/examples/03_client_tools/) | Client-side tool pause/resume |
| [`04_research_agent/`](packages/python/examples/04_research_agent/) | Multi-agent delegation with Firecrawl |
| [`05_streaming_text.py`](packages/python/examples/05_streaming_text.py) | Token-by-token streaming |
| [`06_streaming_tools.py`](packages/python/examples/06_streaming_tools.py) | Tool calls in streaming mode |
| [`07_streaming_openai.py`](packages/python/examples/07_streaming_openai.py) | OpenAI Chat Completions streaming |
| [`08_streaming_openai_responses.py`](packages/python/examples/08_streaming_openai_responses.py) | OpenAI Responses API streaming |
| [`09_client_tools_streaming/`](packages/python/examples/09_client_tools_streaming/) | Client-side tools over streaming NDJSON |
| [`10_single_call.py`](packages/python/examples/10_single_call.py) | One-shot LLM call, no loop |
| [`11_structured_output.py`](packages/python/examples/11_structured_output.py) | Typed Pydantic models from LLM |
| [`12_structured_output_stream.py`](packages/python/examples/12_structured_output_stream.py) | Structured output in streaming mode |
| [`13_mcp_filesystem.py`](packages/python/examples/13_mcp_filesystem.py) | Agent with an MCP tool source |
| [`14_skills/`](packages/python/examples/14_skills/) | Filesystem-loaded skill packs |
| [`15_cache_efficiency.py`](packages/python/examples/15_cache_efficiency.py) | Anthropic prompt-cache hit measurement |
| [`16_cache_efficiency_openai.py`](packages/python/examples/16_cache_efficiency_openai.py) | OpenAI prompt-cache hit measurement |
| [`17_full_stack_app/`](packages/python/examples/17_full_stack_app/) | **Canonical full-stack recipe:** read router + dashboard + all 4 governance layers + 4 agent methods in one FastAPI app |
| **Governance** | |
| [`governance/01_tool_deny.py`](packages/python/examples/governance/01_tool_deny.py) | Block tools deterministically (batch + streaming) |
| [`governance/02_approval.py`](packages/python/examples/governance/02_approval.py) | Human-in-the-loop approval with approve/reject CLI |
| [`governance/03_budget.py`](packages/python/examples/governance/03_budget.py) | Advisory token budget with threshold warnings |
| [`governance/04_guardrails.py`](packages/python/examples/governance/04_guardrails.py) | PII redaction, secret detection, warn mode |
| [`governance/05_dashboard.py`](packages/python/examples/governance/05_dashboard.py) | Dashboard pointed at a seeded run database |

```bash
cd packages/python/examples
ANTHROPIC_API_KEY=sk-... python 01_hello_world.py
```

---

## The Numbers

| Metric | Value |
|--------|-------|
| Source code | 17,600+ lines |
| Test code | 25,200+ lines |
| Test functions | 1,225 |
| Test-to-source ratio | 1.43 : 1 |
| Min coverage enforced | 80% |
| Python versions tested | 3.11, 3.12, 3.13 |
| CI checks | lint, format, types, tests |
| Alembic migrations | 10 |
| Examples | 17 runnable scripts + 4 HTTP demos |
| Development started | March 11, 2026 |

---

## Status

Dendrux is in active development (`v0.1.0a5`). The core API is stabilizing. `Agent`, `tool`, `run`, `stream`, `retry`, `resume`, `submit_tool_results`, `submit_input`, `submit_approval`, `cancel_run`, `sweep`, `RunStore`, and `make_read_router` are the public surface and unlikely to break. Internal modules may still change.

| Layer | Status |
|-------|--------|
| Agent loop + ReAct reasoning | Stable |
| Tool calling (parallel, timeouts, constraints) | Stable |
| Providers (Anthropic, OpenAI Chat, OpenAI Responses) | Stable |
| Compatible APIs (vLLM, SGLang, Groq, Ollama) | Stable |
| Persistence (SQLite + Postgres) | Stable |
| Durability (retry, sweep, idempotency) | Stabilizing |
| Retry terminal runs | Stabilizing |
| Streaming (text + events) | Stabilizing |
| Delegation context (parent-child linking) | Stabilizing |
| Governance (deny, approval, budget, guardrails) | Stabilizing |
| Recorder/Notifier split | Stable |
| CLI + Dashboard | Stabilizing |
| Client-side tool pause/resume | Stabilizing |
| Public HTTP surface (`make_read_router`, `RunStore`) | Stabilizing |

Built with [Claude Code](https://claude.ai/code) and [Codex](https://openai.com/codex). 1,225 tests, strict types, Apache 2.0.

## What Dendrux Is Not

Dendrux is not a hosted platform, a task queue, or a deployment tool. You own your server, your auth, your infra. Dendrux gives you `agent.run()` and gets out of the way.

## License

[Apache 2.0](LICENSE)
