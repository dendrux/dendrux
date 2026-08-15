# dendrux

> Python SDK for Dendrux — the framework for building agents with tools, persistence, and observability.

**Version:** 0.2.0a14

## Install

```bash
pip install "dendrux[all]"              # Everything
pip install "dendrux[anthropic]"        # Anthropic + core runtime
pip install "dendrux[openai]"           # OpenAI + core runtime
pip install "dendrux[mcp]"              # Managed MCP client runtime
```

## Quick Example

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

## Managed MCP client

`MCPRuntime` is the application-scoped owner of process-local MCP connections.
Bind cheaply for each request, hand an explicit tool view to an ephemeral
`Agent`, and close the runtime once during application shutdown:

```python
from dendrux import Agent
from dendrux.mcp import MCPRuntime, MCPSource

runtime = MCPRuntime(max_connections=100, max_in_flight_calls=100)

connection = runtime.bind(
    tenant_key=current_user.id,
    connection_key=github_connection.id,
    source=MCPSource.http("github", github_connection.server_url),
    credentials=credential_provider,
)

agent = Agent(
    provider=provider,
    prompt="Help with this repository.",
    tool_sources=[connection.tools(allowed_tools=["search_code", "create_issue"])],
)

try:
    result = await agent.run(user_message)
finally:
    await agent.close()  # Releases this Agent's lease.

# Once, during application shutdown:
await runtime.close()
```

Dendrux is the MCP client. The vendor, your platform, or a subprocess you
configure hosts the MCP server. See the
[MCP architecture guide](https://www.dendrux.dev/docs/architecture/mcp) and
[multi-user example](https://github.com/dendrux/dendrux/blob/main/packages/python/examples/31_mcp_runtime_multi_user.py).
To test a real hosted server, run the
[read-only GitHub example](https://github.com/dendrux/dendrux/blob/main/packages/python/examples/32_mcp_github_remote.py).

## Providers

| Provider | Import | Use case |
|----------|--------|----------|
| Anthropic | `from dendrux.llm.anthropic import AnthropicProvider` | Claude models |
| OpenAI | `from dendrux.llm.openai import OpenAIProvider` | GPT models + vLLM, SGLang, Groq, Ollama |
| OpenAI Responses | `from dendrux.llm.openai_responses import OpenAIResponsesProvider` | GPT + built-in tools (web search) |
| Mock | `from dendrux.llm.mock import MockLLM` | Deterministic testing |

## API Quick Reference

```python
from pathlib import Path
from dendrux.observers.console import ConsoleObserver

async with Agent(
    provider=provider,                  # Required: LLM provider
    prompt="...",                        # Required: system prompt
    tools=[add],                         # Optional: tool functions
    database_url=f"sqlite+aiosqlite:///{Path.home() / '.dendrux' / 'dendrux.db'}",
    redact=my_scrubber,                  # Optional: scrub persisted strings
) as agent:
    result = await agent.run(
        "What is 15 + 27?",
        observer=ConsoleObserver(),      # Optional: terminal output
        tenant_id="org-123",             # Optional: multi-tenant isolation
        metadata={"thread": "t1"},       # Optional: your linking data
    )
```

### RunResult

```python
result.answer          # str | None — the agent's final answer
result.status          # RunStatus — SUCCESS, ERROR, MAX_ITERATIONS, WAITING_CLIENT_TOOL, CANCELLED
result.steps           # list[AgentStep] — full reasoning chain
result.iteration_count # int — how many loop iterations ran
result.usage           # UsageStats — input_tokens, output_tokens, total_tokens
result.run_id          # str — unique run identifier (ULID)
result.error           # str | None — error message if status is ERROR
```

### Tool Options

```python
@tool()                                  # Basic server tool
@tool(target="client")                   # Client-side — agent pauses
@tool(max_calls_per_run=3)               # Limit calls per run
@tool(timeout_seconds=120)               # Custom timeout (default 120s)
@tool(parallel=False)                    # Run alone, not concurrently
```

## Full Documentation

See the [full documentation on GitHub](https://github.com/dendrux/dendrux) for provider setup, configuration, database guide, CLI, dashboard, observer system, and examples.
