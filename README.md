<p align="center">
  <img src="https://raw.githubusercontent.com/dendrux/dendrux/main/assets/logo.svg" alt="Dendrux" width="96" height="96" />
</p>

<h1 align="center">Dendrux</h1>

<p align="center">
  The async Python runtime for agents that survive failure, persist everything, and bridge to the real world.
</p>

<p align="center">
  <a href="https://www.dendrux.dev">Website</a>
  ·
  <a href="https://www.dendrux.dev/docs">Docs</a>
  ·
  <a href="https://github.com/dendrux/dendrux">GitHub</a>
</p>

<p align="center">
  <a href="https://github.com/dendrux/dendrux/actions/workflows/ci.yml"><img src="https://github.com/dendrux/dendrux/actions/workflows/ci.yml/badge.svg" alt="CI" /></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.11%2B-blue.svg" alt="Python 3.11+" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-green.svg" alt="License: Apache 2.0" /></a>
</p>

> `v0.5.0` - core API stabilizing, actively developed.

<!-- VERIFY-0.5.0: the version banner and the 'What's new in 0.5.0' block are written ahead of the release PR; confirm the version number and the release date when the four version sites are bumped -->

---

## What's new in 0.5.0

0.5.0 ("Trust") makes Dendrux safe by default for multi-tenant, production use. Read the [upgrade guide](docs/upgrading-to-0.5.0.mdx) before you deploy: run `dendrux db migrate`, then roll your workers (rolling deploys are supported).

- **Tenant-scoped reads.** Pass `tenant_id=` on every RunStore read and Agent write. A foreign run looks like a missing one. Your routes authenticate the actor and authorize user-level run ownership. See [Multi-tenancy](docs/architecture/multi-tenancy.mdx).
- **Payloads hidden by default.** `RunStore.follow` and `get_event_view` hide historical payloads by default. Active client calls remain actionable and may contain personal data; authorize the actor before returning them. Raw RunStore reads require your own response projection.
- **Build your own read routes.** `RunStore.get_pending` (what a run is waiting on) and `RunStore.follow` (a hardened live feed that closes when the run does, shares one poller per run and is bounded) are library calls for routes you own. See [Build your own read routes](docs/recipes/build-your-own-read-routes.mdx).
- **No more silent provider failures.** Every failed request is a typed `ProviderError` (`.category`, `.retryable`, `.safe_message`); empty streams, refusals and failed Responses calls end the run `error` with `RunResult.error_info` instead of an empty `success`. Calls have deadlines (`call_timeout`, `stream_idle_timeout`). See [Handling provider errors](docs/recipes/handling-provider-errors.mdx).
- **Dollar budgets must be priceable.** A `max_cost_usd` budget for a model with no price raises `BudgetConfigurationError` instead of never firing. `Budget(count_cache_tokens=True)` opts into counting cache tokens.
- **Tools are validated and coerced.** Model-typed parameters arrive as model instances, `Field(...)` constraints are enforced, and what a person approves is exactly what runs. Sync tools run on a bounded thread pool, and results, inputs, metadata and notes have size caps (`Agent(limits=Limits(...))`).
- **PII restore you control.** `PII(restore="allowlisted")` with `@tool(receives_pii=...)` returns real values only to the tools you name. The default is unchanged. See [PII restore](docs/architecture/pii-restore.mdx).
- **Run integrity.** Claims are fenced, and a resume under changed governance emits `governance.changed` (or raises with `on_governance_change="raise"`).
- **Migration.** Run `dendrux db migrate` (revision `d7e8f9a0b1c2`). Several defaults flip in 0.6; 0.5 prints a `FutureWarning` for each. See the [CHANGELOG](packages/python/CHANGELOG.md).

[//]: # "VERIFY-0.5.0: Confirm expected_pause_id support before documenting stale-submit protection."

## What's new in 0.4.1

- **Truncated answers are flagged**: `result.truncated` and `result.stop_reason` say when the final answer was cut off at the token limit, on all four providers. The run still succeeds, so check the flag.
- **Anthropic strict mode**: `@tool(strict=True)` and structured output are sent strict on Anthropic when the schema allows it, with an automatic non-strict fallback and an `AnthropicProvider(strict=False)` opt-out.
- **`tenant_id=` on run-control writes**: `submit_tool_results`, `submit_input`, `submit_approval`, `submit_resume`, `resume` and `resume_stream` reject runs outside the tenant before writing anything.
- No migration. See the [CHANGELOG](packages/python/CHANGELOG.md).

## What's new in 0.4.0

- **Images**: `agent.run(["What is on this receipt?", Image.from_path("receipt.png")])`, tools that return `ToolOutput(result, images=[...])`, client tools that attach `ToolResult.images`, and MCP image results all reach the model as images. Images are stored once as tenant-scoped blobs (database by default, or your own `BlobStore` such as S3); runs, pauses and events hold references only. Anthropic, OpenAI Chat and Responses, and OpenRouter are supported; non-vision models get a typed error or a placeholder. See the [Images guide](docs/recipes/images.mdx).
- **Migration**: run `dendrux db migrate` (new revision `c6d7e8f9a0b1`), even if you do not use images. See the [CHANGELOG](packages/python/CHANGELOG.md).

## What's new in 0.3.0

- **Per-call approvals**: `agent.submit_approval(run_id, decisions={call_id: ApprovalDecision.approve() | .deny(reason)})` decides each gated call on its own. Client-target calls are never executed server-side on approval (a security fix).
- **`before_tool` policy hook**: one function sees each parsed tool call (local, MCP, skill) that passes static `deny=` and skill preflight, before approval, and returns allow, deny, require approval or a substituted result. Fails closed.
- **Run control**: `agent.request_pause(run_id)` parks a running run as `paused` and `agent.submit_resume(run_id)` continues it. `cancel_run(run_id, cascade=True)` stops delegated children too, and `cancel_run` now interrupts an in-flight `run()` in the same process.
- **Inbox notes**: `agent.submit_note(run_id, "...")` tells a run something while it works; the model sees it at the next iteration.
- **Dollar budgets**: `Budget(max_cost_usd=..., on_exceed="pause" | "cancel", include_children=True)` enforces token and dollar limits across a run and its children.
- **Subtree reads**: `RunStore.get_subtree(run_id)` returns a delegation tree with live rolled-up usage, plus `get_notes`, `event_types=` and `before=` filters.
- **`MCPConnection.call_tool`** runs a server tool directly from your app (outside Agent governance).
- **Gateway fixes** for OpenAI-compatible endpoints, including a typed `ContextLengthExceededError`.
- **Migration**: run `dendrux db migrate` (new revision `b5c6d7e8f9a0`). See the [CHANGELOG](packages/python/CHANGELOG.md).

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
pip install "dendrux[anthropic]"      # Anthropic + core runtime
pip install "dendrux[openai]"         # OpenAI + core runtime
pip install "dendrux[mcp]"            # managed MCP client runtime
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
| 3 | **Govern behavior** | Tool deny/approval/policy hook, token and dollar budgets, PII redaction, secret detection. Four layers of runtime governance. |
| 4 | **Explain everything** | Every LLM call, tool execution, pause, and lifecycle event is persisted as evidence. |
| 5 | **Coordinate agents** | Parent-child delegation with automatic linking, depth guards, and lifecycle coupling. |
| 6 | **Pause for the real world** | Client-side tool pause/resume for spreadsheets, browsers, and desktops with domain-aware constraints. |

---

## Features

### 🔄 Survive Failure

Runs crash. Processes die. Dendrux handles it.

**Durable persistence**: every trace, tool call, and lifecycle event is written through a durability layer with retry and exponential backoff. Transient DB failures (connection drops, lock timeouts, SQLite busy) are retried automatically. Logical errors propagate immediately. LLM interactions and token usage are persisted best-effort.

**Sweep stale runs**: call `sweep()` at app startup (or on a schedule you own) to detect runs that were RUNNING when the process died or WAITING when no one came back. They're marked ERROR with structured failure reasons. Claims are fenced with an epoch, so a worker whose run was taken over stops instead of overwriting the new owner.

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

**Idempotency**: pass a key to prevent duplicate runs. Same key + same input returns the existing run. Same key + different input raises `IdempotencyConflictError`. Keys are unique per tenant.

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
    budget=Budget(max_tokens=30_000),            # spend limit (warn / pause / cancel)
    guardrails=[                                 # content scanning at LLM boundary
        PII(extra_patterns=[Pattern("EMPLOYEE_ID", r"EMP-\d{6}")]),
        SecretDetection(action="block"),
    ],
)
```

**Tool deny**: `deny=["tool_name"]` blocks tools deterministically. The model gets a synthesized error and adapts. The tool never executes.

**Approval (HITL)**: `require_approval=["tool_name"]` pauses the run for human sign-off. Approve with `agent.submit_approval(run_id, approved=True)`, reject with `agent.submit_approval(run_id, approved=False, rejection_reason="...")`, or decide each call with `decisions={call_id: ApprovalDecision.approve()}`.

**Tool policy**: `before_tool=hook` decides each call from its arguments: `ToolDecision.deny(...)`, `.require_approval(...)`, `.substitute(result)` or `None` to allow.

**Budget**: `budget=Budget(max_tokens=N, max_cost_usd=D)` fires governance events at configurable thresholds (50%, 75%, 90%) and when usage reaches a limit, counting delegated child runs. By default (`on_exceed="warn"`) it is advisory and the run continues; `on_exceed="pause"` parks the run for a person to raise the limit and resume, and `"cancel"` ends it. Pause and cancel take effect at the next iteration boundary.

**Guardrails**: `guardrails=[PII(), SecretDetection()]` scans content crossing the LLM boundary. Three actions:
- `redact` - PII replaced with `<<EMAIL_ADDRESS_1>>` placeholders. Tools receive real values (deanonymized); with `PII(restore="allowlisted")` only tools marked `@tool(receives_pii=...)` do. Run-scoped mapping persisted for audit.
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
  block by    placeholders →     HITL           server /       spend
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

**PII policy**: the DB is the authoritative execution record and stores raw values. Guardrails redact at the LLM boundary only — what the provider API sees — and the placeholder→real bijection is persisted on `agent_runs.pii_mapping` as the audit key, so inspection tools can replay either view from the same rows. See [PII redaction](docs/architecture/pii-redaction.mdx).

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

The state store links parent and child runs. Read the delegation tree with `RunStore.get_subtree(..., tenant_id=...)` and render it in your app. Depth guards prevent runaway recursion.

### 📚 Skills

Package instructions, reference files, and scripts as a skill. The model sees a catalogue, pulls a body only when it needs it, and reads files or runs scripts on demand.

```python
from dendrux import Agent, Skill, SkillExecutionResult, SkillExecutor

triage = Skill(
    name="ticket-triage",
    description="Classify a support ticket and pick a priority.",
    body="Read references/policy.md, then classify the ticket.",
    files={"references/policy.md": "P1: outage. P2: broken feature. P3: question."},
)

agent = Agent(provider=provider, prompt="You are a support agent.", skills=[triage])
result = await agent.run("The export button does nothing.", activate_skills=["ticket-triage"])
```

Skills build from code, `SKILL.md` text, or a directory. Scripts under `scripts/` run only through a `SkillExecutor` you write (a subprocess, a container, a hosted sandbox); the model's tool calls obey `deny` and `require_approval` like any other tool. A run records the digest of every skill it started with and refuses to resume against changed content. See [Skills](docs/architecture/skills.mdx).

### ⏸️ Pause for the Real World

**Client-side tools**: define tools that run on the client (browser, mobile, Excel). The agent pauses and waits.

```python
@tool(target="client")
async def read_excel_range(sheet: str, range: str) -> str:
    """Read cells from the user's spreadsheet."""
    return ""

# Dendrux gives you the reads as library calls. You own the routes.
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

@app.get("/runs/{run_id}/pending")
async def pending(run_id: str, user=Depends(require_user)):
    # What is the run waiting on? The tenant is required; a foreign run is a 404.
    pause = await store.get_pending(run_id, tenant_id=user.org_id)   # or UNSCOPED
    return pause.to_dict() if pause is not None else None

@app.post("/runs/{run_id}/tool-results")
async def submit(run_id: str, body: ResultsBody, user=Depends(require_user)):
    # Convert the app's DTO into Dendrux's ToolResult — the method expects
    # list[ToolResult], not arbitrary Pydantic models.
    results = [
        ToolResult(name=r.tool_name, call_id=r.tool_call_id, payload=r.payload)
        for r in body.results
    ]
    return await agent.submit_tool_results(run_id, results, tenant_id=user.org_id)
```

Pause/resume orchestration and the CAS-safe claim are Dendrux's job. Auth, schema, error shape, and framework choice are yours; for the client's side, build `GET /runs/{id}/pending` over `RunStore.get_pending`, and add live updates with `RunStore.follow` if you want them (see [Client-side tools](docs/recipes/client-tools.mdx)). Never pipe `agent.stream()` into a browser SSE response: closing the tab cancels the run.

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

All providers accept `max_tokens`, `temperature`, `max_retries`, and the deadlines `call_timeout` and `stream_idle_timeout` (default 600 s each) at construction or per-call.

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

Postgres supported via Alembic migrations: `dendrux db migrate` (check with `dendrux db check`).

In a server, create one state store at startup and pass `state_store=` to every agent; do not pass `database_url=` per request. Falling back to the `DENDRUX_DATABASE_URL` environment variable is deprecated in 0.5.0 (the CLI still reads it).

Omit `database_url` entirely for ephemeral runs. No persistence, no overhead.

---

## CLI and inspection

```bash
dendrux runs --tenant acme --limit 50
dendrux traces RUN_ID --limit 100
dendrux db check
```

Use RunStore reads plus your own tracing and operator UI. See
[Observability](docs/recipes/observability.mdx) for bounded inspection and
[HTTP integration](docs/reference/http-api.mdx) for app-owned routes.

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
| [`14_skills/`](packages/python/examples/14_skills/) | Skills with files and a script executor: model-driven and explicit activation, deny policy |
| [`15_cache_efficiency.py`](packages/python/examples/15_cache_efficiency.py) | Anthropic prompt-cache hit measurement |
| [`16_cache_efficiency_openai.py`](packages/python/examples/16_cache_efficiency_openai.py) | OpenAI prompt-cache hit measurement |
| [`17_full_stack_app/`](packages/python/examples/17_full_stack_app/) | **Canonical full-stack recipe:** developer-owned routes, header auth, RunStore pending reads and governance |
| [`30_mcp_agent.py`](packages/python/examples/30_mcp_agent.py) | Local tools + an SDK v2 MCP source with shared `MCPHost` lifecycle |
| [`36_run_control.py`](packages/python/examples/36_run_control.py) | Per-call approvals, a `before_tool` policy hook and `request_pause` / `submit_resume`, with a mock provider (no API key) |
| [`37_images.py`](packages/python/examples/37_images.py) | A user image plus a tool that returns an image, stored and inspected (mock provider by default, no API key) |
| [`38_images_client_tool.py`](packages/python/examples/38_images_client_tool.py) | Client-tool screenshot: pause, submit the image from a second Agent, resume (mock provider by default) |
| **Governance** | |
| [`governance/01_tool_deny.py`](packages/python/examples/governance/01_tool_deny.py) | Block tools deterministically (batch + streaming) |
| [`governance/02_approval.py`](packages/python/examples/governance/02_approval.py) | Human-in-the-loop approval with approve/reject CLI |
| [`governance/03_budget.py`](packages/python/examples/governance/03_budget.py) | Token budget with threshold warnings |
| [`governance/04_guardrails.py`](packages/python/examples/governance/04_guardrails.py) | PII redaction, secret detection, warn mode |

```bash
cd packages/python/examples
ANTHROPIC_API_KEY=sk-... python 01_hello_world.py
```

---

## The Numbers

| Metric | Value |
|--------|-------|
| Source code | 55,000+ lines |
| Test code | 90,000+ lines |
| Test cases | 6,100+ (about 5,400 run on SQLite alone; the rest need Postgres) |
| Test-to-source ratio | 1.6 : 1 |
| Min coverage enforced | 80% |
| Python versions tested | 3.11, 3.12, 3.13 |
| CI checks | lint, format, types, tests |
| Alembic migrations | 16 |
| Examples | 34 runnable scripts + 6 example apps |
| Development started | March 11, 2026 |

---

## Status

Dendrux is in active development (`v0.5.0`). The core API is stabilizing. `Agent`, `tool`, `run`, `stream`, `retry`, `resume`, `submit_tool_results`, `submit_input`, `submit_approval`, `cancel_run`, `sweep`, `RunStore` are the public surface and unlikely to break. Internal modules may still change.

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
| CLI | Stabilizing |
| Client-side tool pause/resume | Stabilizing |
| Public read API (`RunStore`) | Stabilizing |

Built with [Claude Code](https://claude.ai/code) and [Codex](https://openai.com/codex). 6,100+ tests, strict types, Apache 2.0.

<!-- VERIFY-0.5.0: refresh the counts in The Numbers (lines of code, test cases, migrations, examples) from the release candidate -->

## What Dendrux Is Not

Dendrux is not a hosted platform, a task queue, or a deployment tool. You own your server, your auth, your infra. Dendrux gives you `agent.run()` and gets out of the way. It provides developer-called recovery primitives (sweep, heartbeat, fencing) but never runs background workers or schedules them.

## License

[Apache 2.0](LICENSE)
