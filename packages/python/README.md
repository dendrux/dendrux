# dendrite

> Python SDK for Dendrite — the runtime for agents that act in the real world.

**Version:** 0.1.0a1

## Install

```bash
cd packages/python
pip install -e ".[anthropic,db,server]"
```

| Extra | What it adds |
|-------|-------------|
| `anthropic` | Anthropic Claude SDK |
| `db` | SQLAlchemy, aiosqlite, Alembic |
| `server` | FastAPI, uvicorn |
| `dev` | pytest, ruff, mypy, python-dotenv |
| `postgres` | asyncpg |

## Minimal Example

```python
from dendrite import Agent, tool
from dendrite.llm.anthropic import AnthropicProvider

@tool()
async def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

async with Agent(
    provider=AnthropicProvider(model="claude-sonnet-4-6"),
    prompt="You are a calculator.",
    tools=[add],
) as agent:
    result = await agent.run("What is 15 + 27?")
    print(result.answer)
```

## API Quick Reference

### Core

| Import | What it does |
|--------|-------------|
| `from dendrite import Agent` | Define an agent (provider, prompt, tools, limits) |
| `from dendrite import tool` | `@tool()` decorator — turns a function into an agent tool |
| `from dendrite import bridge` | `bridge(agent)` — mountable FastAPI app for pause/resume transport |
| `from dendrite import run` | `await run(agent, provider=..., user_input=...)` — low-level runner |

### Providers

| Import | What it does |
|--------|-------------|
| `from dendrite.llm.anthropic import AnthropicProvider` | Claude API provider |
| `from dendrite.llm.mock import MockLLM` | Deterministic mock for testing |

### `agent.run()` Parameters

```python
async with Agent(
    provider=provider,                  # Required: LLM provider
    prompt="...",                       # Required: system prompt
    tools=[add],                        # Optional: tool functions
    database_url=f"sqlite+aiosqlite:///{Path.home() / '.dendrite' / 'dendrite.db'}",
    redact=my_scrubber,                 # Optional: scrub persisted strings
) as agent:
    result = await agent.run(
        "What is 15 + 27?",
        tenant_id="org-123",            # Optional: multi-tenant isolation
        metadata={"thread": "t1"},      # Optional: your linking data
    )
```

### `RunResult`

```python
result.answer          # str | None — the agent's final answer
result.status          # RunStatus — SUCCESS, ERROR, MAX_ITERATIONS, WAITING_CLIENT_TOOL
result.steps           # list[AgentStep] — full reasoning chain
result.iteration_count # int — how many loop iterations ran
result.usage           # UsageStats — input_tokens, output_tokens, total_tokens, cost_usd
result.run_id          # str — unique run identifier (ULID)
```

## Alpha Limitations

- **No tool sandbox** — tools run in-process with full host privileges. Only run tools you trust.
- **Opt-in trace redaction** — pass `redact=` to `run()` to scrub persisted content. Not enabled by default.
- **Anthropic-only** — other LLM providers planned for a future sprint.

## Full Documentation

See the [main README](../../README.md) for:

- Quick start guide with all three examples
- Hosted agent setup with client tool pause/resume
- Database and migration guide (SQLite + Postgres)
- CLI cheatsheet
- Common debugging
