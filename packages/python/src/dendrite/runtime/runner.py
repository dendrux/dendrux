"""Agent runner — the entry point for executing agents.

Takes an Agent definition and runs it through the loop with an explicit
provider and strategy. This is the top-level API developers interact with.

Sprint 1: caller provides the LLM provider instance, defaults to
NativeToolCalling strategy and ReActLoop. Future sprints add provider
registry (model string → provider resolution), strategy selection from
agent config, and more loop types.

Sprint 2 adds optional state_store for persistence. When provided:
  - Runner owns the run_id (generates it, passes to loop)
  - PersistenceObserver records traces, tool calls, and usage
  - finalize_run() is called in try/finally to guarantee persistence
  - Observer failures are logged, never kill the run
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from dendrite.loops.react import ReActLoop
from dendrite.strategies.native import NativeToolCalling
from dendrite.types import RunStatus, UsageStats, generate_ulid

if TYPE_CHECKING:
    from dendrite.agent import Agent
    from dendrite.llm.base import LLMProvider
    from dendrite.loops.base import Loop
    from dendrite.runtime.state import StateStore
    from dendrite.strategies.base import Strategy
    from dendrite.types import RunResult

logger = logging.getLogger(__name__)


async def run(
    agent: Agent,
    *,
    provider: LLMProvider,
    user_input: str,
    strategy: Strategy | None = None,
    loop: Loop | None = None,
    state_store: StateStore | None = None,
    tenant_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    **kwargs: Any,
) -> RunResult:
    """Run an agent to completion.

    This is the primary API for executing a Dendrite agent. It wires
    together the agent definition, LLM provider, strategy, and loop,
    then executes the loop until completion.

    Args:
        agent: Agent definition (model, tools, prompt, limits).
        provider: LLM provider to use for this run.
        user_input: The user's input to process.
        strategy: Communication strategy. Defaults to NativeToolCalling.
        loop: Execution loop. Defaults to ReActLoop.
        state_store: Optional persistence backend. If provided, the run
            is persisted to the database with full traces.
        tenant_id: Optional tenant ID for multi-tenant isolation.
        metadata: Optional developer linking data (thread_id, user_id, etc.).
            Stored in agent_runs.meta — Dendrite stores it, never reads it.
        **kwargs: Reserved for future use.

    Returns:
        RunResult with status, answer, steps, and usage stats.

    Usage:
        from dendrite import Agent, tool, run
        from dendrite.llm import AnthropicProvider

        @tool()
        async def add(a: int, b: int) -> int:
            return a + b

        agent = Agent(
            model="claude-sonnet-4-6",
            tools=[add],
            prompt="You are a calculator.",
        )
        provider = AnthropicProvider(api_key="sk-...", model="claude-sonnet-4-6")
        result = await run(agent, provider=provider, user_input="What is 15 + 27?")
        print(result.answer)
    """
    resolved_strategy = strategy or NativeToolCalling()
    resolved_loop = loop or ReActLoop()

    # Runner owns run_id — single source of truth
    run_id = generate_ulid()
    observer = None

    if state_store is not None:
        # Create the run record before the loop starts
        await state_store.create_run(
            run_id,
            agent.name,
            input_data={"input": user_input},
            model=agent.model,
            strategy=type(resolved_strategy).__name__,
            tenant_id=tenant_id,
            meta=metadata,
        )

        # Create persistence observer
        from dendrite.runtime.observer import PersistenceObserver
        from dendrite.tool import get_tool_def

        target_lookup = {}
        for fn in agent.tools:
            td = get_tool_def(fn)
            target_lookup[td.name] = td.target
        observer = PersistenceObserver(
            state_store,
            run_id,
            model=agent.model,
            provider_name=type(provider).__name__,
            target_lookup=target_lookup,
        )

    try:
        result = await resolved_loop.run(
            agent=agent,
            provider=provider,
            strategy=resolved_strategy,
            user_input=user_input,
            run_id=run_id,
            observer=observer,
        )

        # Finalize with success or max_iterations
        if state_store is not None:
            await state_store.finalize_run(
                run_id,
                status=result.status.value,
                answer=result.answer,
                iteration_count=result.iteration_count,
                total_usage=result.usage,
            )

        return result

    except Exception as exc:
        # Persist ERROR status before re-raising
        if state_store is not None:
            try:
                await state_store.finalize_run(
                    run_id,
                    status=RunStatus.ERROR.value,
                    error=str(exc),
                    total_usage=UsageStats(),
                )
            except Exception:
                logger.warning("Failed to persist ERROR status for run %s", run_id, exc_info=True)
        raise
