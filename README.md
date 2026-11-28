<p align="center">
  <img src="https://raw.githubusercontent.com/dendrux/dendrux/main/assets/logo.svg" alt="Dendrux" width="96" height="96" />
</p>

<h1 align="center">Dendrux</h1>

<p align="center">
  A Python library for agent runs that pause for people, approvals and client-side tools, and resume in any process by id.
</p>

<p align="center">
  <a href="https://www.dendrux.dev">Website</a>
  ·
  <a href="https://www.dendrux.dev/docs">Docs</a>
  ·
  <a href="https://www.dendrux.dev/docs/changelog">Changelog</a>
  ·
  <a href="https://pypi.org/project/dendrux/">PyPI</a>
</p>

<p align="center">
  <a href="https://github.com/dendrux/dendrux/actions/workflows/ci.yml"><img src="https://github.com/dendrux/dendrux/actions/workflows/ci.yml/badge.svg" alt="CI" /></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.11%2B-blue.svg" alt="Python 3.11+" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-green.svg" alt="License: Apache 2.0" /></a>
</p>

> `v0.8.0` - actively developed. Until 1.0 a minor release can change behaviour; each one says so in its upgrade guide.

---

## Why Dendrux

Real agents have to stop and wait. A refund needs a manager's approval. A question needs a person's answer. A tool has to run in the user's browser, spreadsheet or desktop app, not on your server.

Dendrux makes that the normal case. A run **pauses**, its full state is saved to your database, and it **resumes later in any process, by run id**: the process that started it can be gone. You answer the pause through ordinary methods on `Agent` (`submit_approval`, `submit_input`, `submit_tool_results`) from your own routes.

What makes that safe to rely on:

- **Durable state.** The run's state is committed to your database as it goes, so a fresh process can pick it up. Every pause has a `pause_id`, so a stale browser tab cannot answer a newer pause.
- **Evidence.** Every LLM call, tool call, pause and decision is recorded, and you read it back through `RunStore`.
- **Recovery you call.** `sweep()` finds runs whose worker stopped, and `agent.reclaim()` continues them from their last committed step. You decide when and where these run; Dendrux starts no background workers.

Not sure it fits? [Is Dendrux for you?](docs/is-dendrux-for-you.mdx) is a two-minute check.

## Install

```bash
pip install "dendrux[anthropic]"     # Anthropic provider + core runtime (SQLite included)
pip install "dendrux[openai]"        # OpenAI provider (also OpenRouter and OpenAI-compatible endpoints)
pip install "dendrux[postgres]"      # Postgres driver
pip install "dendrux[all]"           # Anthropic, OpenAI, Postgres, MCP, Presidio, OpenTelemetry
```

MySQL and MariaDB (`dendrux[mysql]`) and Redis live updates (`dendrux[redis]`) are separate extras, not part of `all`. Python 3.11 or later.

## Quick start

An agent that pauses before a refund, then resumes after a person approves it:

```python
import asyncio
from dendrux import Agent, tool
from dendrux.llm.anthropic import AnthropicProvider


@tool()
async def refund(order_id: int) -> str:
    """Issue a refund for an order. A person must approve it first."""
    return f"Refunded order {order_id}"


def build_agent() -> Agent:
    return Agent(
        provider=AnthropicProvider(model="claude-sonnet-4-6"),
        prompt="You are a support agent. Use the refund tool when asked.",
        tools=[refund],
        require_approval=["refund"],
        database_url="sqlite+aiosqlite:///./runs.db",
        strict_tenancy=False,  # single-tenant example; see Multi-tenancy
    )


async def main():
    # The run pauses before the refund and is saved to runs.db.
    async with build_agent() as agent:
        paused = await agent.run("Please refund order 42.")
    print(paused.status.value, paused.run_id)

    # Later, from any process: approve the pause the person was shown.
    async with build_agent() as agent:
        done = await agent.submit_approval(
            paused.run_id, approved=True, expected_pause_id=paused.pause_id
        )
    print(done.status.value, done.answer)


asyncio.run(main())
```

```bash
ANTHROPIC_API_KEY=sk-... python quickstart.py
```

The two halves can run in different processes or on different machines; they share only the database. The [Quickstart](docs/quickstart.mdx) splits them into two scripts.

## What you get

**Pause and resume**

- **Approvals.** `require_approval=[...]` pauses before a tool runs; approve or deny each call with `submit_approval(..., expected_pause_id=...)`. [Approval](docs/architecture/approval.mdx)
- **Client-side tools.** `@tool(target="client")` pauses the run until your browser, spreadsheet or app posts the result back. [Client-side tools](docs/recipes/client-tools.mdx)
- **Human input, notes and pauses on request.** `submit_input`, `submit_note`, `request_pause` and `submit_resume`. [Pause and resume](docs/architecture/pause-resume.mdx)

**Subagents**

- **Managed subagents.** `Agent(subagents=[...])` lets an agent hand tasks to other agents. A child's pause surfaces on the root run, and one root budget covers the whole tree.
- **Parallel fan-out.** Several subagent calls in one model turn run at once, up to `Limits.max_parallel_subagents` (default 4). `RunResult.pendings` lists every pause in the tree, and runs waiting on children report `waiting_for="subagents"`. [Managed subagents](docs/guides/subagents.mdx)

**Run control**

- **Cancel.** `agent.cancel_run(...)`, or `dendrux.cancel_run(run_id, tenant_id=..., state_store=...)` from a route or script without building an Agent or loading a provider. Subagents are cancelled too. [Cancelling a run](docs/recipes/cancel-run.mdx)
- **Run ids up front.** `agent.run(..., on_started=callback)` hands you the run id as soon as the run is saved, before the model starts. [Agent reference](docs/reference/agent.mdx)
- **Retry, replay and idempotency.** `agent.retry(run_id)` starts a new run seeded with a finished run's history, and `idempotency_key=` stops duplicate runs. [Run replay](docs/recipes/run-replay.mdx)

**Recovery**

- **`sweep()` and `reclaim()`.** A sweep marks runs whose worker stopped; `agent.reclaim()` continues them from their last committed step, through the root for subagent trees. Claims are fenced, so an old worker that wakes up stops instead of overwriting. You call both from your own jobs. [Recovery](docs/guides/recovery.mdx)

**Governance**

- **Tool access.** `deny=[...]` and a `before_tool=` policy hook that can deny, require approval or substitute a result. [Tool policy](docs/architecture/tool-policy.mdx)
- **Budgets.** Token and dollar limits across a run and its children: warn, pause or cancel. [Budget](docs/architecture/budget.mdx)
- **Guardrails.** PII redaction at the model boundary, secret detection and prompt-injection patterns. [Guardrails](docs/architecture/guardrails.mdx)
- **Tenancy.** Strict tenancy is on by default: every call and read names a `tenant_id`, and a foreign run looks like a missing one. [Multi-tenancy](docs/architecture/multi-tenancy.mdx)

**Reads and live updates**

- **`RunStore`.** Library reads for routes you own: `get_pending`, `get_pendings`, `follow` (a bounded live feed), `get_event_view`, `get_subtree` and `list_runs`. [RunStore](docs/reference/run-store.mdx), [Build your own read routes](docs/recipes/build-your-own-read-routes.mdx)
- **Optional live updates.** Postgres or Redis hints wake `follow` early. Off by default; the database stays the source of truth. [Scaling: live updates](docs/guides/scaling.mdx#live-updates)
- **Observability.** A fail-closed recorder, a best-effort notifier and OpenTelemetry. [Observability](docs/recipes/observability.mdx)

**Models and content**

- **Streaming** with `agent.stream(...)`, **structured output** with `output_type=`, **images** in input and tool results, **skills**, **MCP** tool sources, context blocks, reasoning and prompt caching. [Images](docs/recipes/images.mdx), [Skills](docs/architecture/skills.mdx), [MCP](docs/architecture/mcp.mdx)

**Testing**

- **`dendrux.testing.FakeModel`.** A scripted model for tests: answer by agent name and turn, then assert on what each agent was asked. [Testing with FakeModel](docs/recipes/testing-with-fake-model.mdx)

## Providers and databases

| Providers | Import |
|---|---|
| Anthropic | `dendrux.llm.anthropic.AnthropicProvider` |
| OpenAI Chat Completions, and OpenAI-compatible endpoints via `base_url=` | `dendrux.llm.openai.OpenAIProvider` |
| OpenAI Responses API | `dendrux.llm.openai_responses.OpenAIResponsesProvider` |
| OpenRouter | `dendrux.llm.openrouter.OpenRouterProvider` |

Every provider failure is a typed `ProviderError`, and calls have deadlines. See [Models](docs/architecture/models.mdx) and [Handling provider errors](docs/recipes/handling-provider-errors.mdx).

| Database | Notes |
|---|---|
| SQLite | Zero config, tables created on first use. Good for local work and single-node apps. |
| PostgreSQL | `dendrux[postgres]`. Run `dendrux db migrate` once as a deploy job. |
| MySQL 8.4 and MariaDB 10.11 | `dendrux[mysql]` and `MySQLStateStore`. [MySQL and MariaDB](docs/guides/mysql.mdx) |
| Your own SQL database | Subclass `SQLStateStore` and certify it with the store test kit. [Build your own store](docs/guides/build-your-own-store.mdx) |

In a server, build one store at startup and pass `state_store=` to every Agent. See [Scaling workers](docs/guides/scaling.mdx).

## Examples

Runnable scripts live in [`packages/python/examples/`](packages/python/examples/). A few to start with:

| Example | What it shows |
|---|---|
| [`01_hello_world.py`](packages/python/examples/01_hello_world.py) | The smallest agent with one tool |
| [`03_client_tools/`](packages/python/examples/03_client_tools/) | A browser-side tool: pause, post the result, resume |
| [`17_full_stack_app/`](packages/python/examples/17_full_stack_app/) | A full app: your own routes, auth, pending reads and governance |
| [`36_run_control.py`](packages/python/examples/36_run_control.py) | Per-call approvals, a `before_tool` hook, pause and resume (no API key) |
| [`37_images.py`](packages/python/examples/37_images.py) | Images in input and tool results (no API key by default) |
| [`39_mysql_store.py`](packages/python/examples/39_mysql_store.py) | `MySQLStateStore` on MySQL or MariaDB |
| [`40_live_updates.py`](packages/python/examples/40_live_updates.py) | `RunStore.follow` with optional live updates |
| [`governance/`](packages/python/examples/governance/) | Tool deny, approval, budgets and guardrails |

```bash
cd packages/python/examples
ANTHROPIC_API_KEY=sk-... python 01_hello_world.py
```

## Documentation

Everything is at [dendrux.dev/docs](https://www.dendrux.dev/docs): the [Quickstart](docs/quickstart.mdx), the [architecture pages](docs/architecture/runs.mdx), task recipes and the [API reference](docs/reference/agent.mdx). The source of the docs is in [`docs/`](docs/).

## What Dendrux is not

Dendrux is a library, not a hosted platform, a task queue or a deployment tool. You own your server, routes, auth, workers and scaling. Dendrux gives you `agent.run()`, the pause and resume methods, persistence and read primitives. Its recovery tools (`sweep`, `reclaim`, heartbeats, fencing) are functions you call; it never runs background workers or schedules anything.

## Changelog

What changed in each release is on the website at [dendrux.dev/docs/changelog](https://www.dendrux.dev/docs/changelog) ([source](docs/changelog.mdx)). The full list is in [`packages/python/CHANGELOG.md`](packages/python/CHANGELOG.md), and each release from [0.5.0](docs/upgrading-to-0.5.0.mdx) to [0.8.0](docs/upgrading-to-0.8.0.mdx) has an upgrade guide.

## License

[Apache 2.0](LICENSE)
