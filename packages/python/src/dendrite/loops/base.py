"""Loop protocol — how the agent iterates.

A loop orchestrates the cycle of calling the LLM, interpreting the response,
executing tools, and feeding results back. It uses a Strategy for LLM
communication and a Provider for actual LLM calls.

The loop never touches provider-specific APIs or prompt formatting — that's
the strategy's job. The loop is pure orchestration.

Observation: loops accept an optional LoopObserver that receives notifications
at each history mutation and LLM call. The observer is how persistence,
logging, metrics, and streaming plug in without teaching the loop about
databases or transports.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from dendrite.agent import Agent
    from dendrite.llm.base import LLMProvider
    from dendrite.strategies.base import Strategy
    from dendrite.types import LLMResponse, Message, RunResult, ToolCall, ToolResult


@runtime_checkable
class LoopObserver(Protocol):
    """Observer for loop events — the seam for persistence and observability.

    The loop fires these callbacks at the exact points where history mutates
    and provider.complete() returns. Implementations decide what to do:
    persist to DB, log, emit metrics, stream to SSE, etc.

    Failure policy: observers should not raise. If they do, the loop logs
    a warning and continues execution. Observability failures must not
    kill agent runs.
    """

    async def on_message_appended(self, message: Message, iteration: int) -> None:
        """Called when a message is appended to the conversation history.

        Fires for: initial user message (iteration=0), assistant responses,
        and tool result messages. SYSTEM prompt is not in history — it's
        rebuilt by the strategy each iteration.

        Maps to: react_traces table.
        """
        ...

    async def on_llm_call_completed(self, response: LLMResponse, iteration: int) -> None:
        """Called after provider.complete() returns.

        Carries token usage in response.usage. Separate from
        on_message_appended because usage data doesn't belong in traces.

        Maps to: token_usage table.
        """
        ...

    async def on_tool_completed(
        self, tool_call: ToolCall, tool_result: ToolResult, iteration: int
    ) -> None:
        """Called after _execute_tool() returns.

        Consumes tool_result.duration_ms — no re-timing. Records both
        tool_call.id (Dendrite ULID) and tool_call.provider_tool_call_id.

        Maps to: tool_calls table.
        """
        ...


class Loop(ABC):
    """Base class for agent execution loops.

    Subclasses implement the iteration pattern:
        ReActLoop      — think → act → observe → repeat
        SingleShot     — one LLM call, no tools (planned)
        PlanAndExecute — plan upfront, then execute steps (planned)
    """

    @abstractmethod
    async def run(
        self,
        *,
        agent: Agent,
        provider: LLMProvider,
        strategy: Strategy,
        user_input: str,
        run_id: str | None = None,
        observer: LoopObserver | None = None,
    ) -> RunResult:
        """Execute the agent loop until completion.

        Args:
            agent: Agent definition (tools, prompt, limits).
            provider: LLM provider to call.
            strategy: Strategy for message building and response parsing.
            user_input: The user's input to process.
            run_id: Optional runner-provided ID. If None, loop generates one.
            observer: Optional observer for persistence/logging hooks.

        Returns:
            RunResult with status, answer, steps, and usage.
        """
