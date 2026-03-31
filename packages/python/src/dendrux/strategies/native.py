"""NativeToolCalling strategy.

Uses the LLM provider's native tool_use API (Anthropic tool_use blocks,
OpenAI function_calling). Zero prompt engineering for tool format — the
provider handles tool definition serialization and response parsing.

The strategy's job is minimal:
  - Pass tool definitions through to the provider
  - Read tool_calls from LLMResponse and wrap in AgentStep
  - Format tool results as Role.TOOL messages with call_id correlation
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from dendrux.strategies.base import Strategy
from dendrux.types import (
    AgentStep,
    Finish,
    Message,
    Role,
)

if TYPE_CHECKING:
    from dendrux.types import LLMResponse, ToolDef, ToolResult


class NativeToolCalling(Strategy):
    """Strategy that uses the provider's native tool calling API.

    The recommended strategy for providers that support it (Anthropic, OpenAI).
    Developer writes zero parsing code — the provider returns structured
    tool_use blocks, this strategy wraps them into AgentSteps.

    Sprint 1 compromise: AgentStep.action models a single action, so when the
    LLM returns multiple tool calls, the first becomes step.action and the full
    ordered list is stored in step.meta["all_tool_calls"]. The loop uses this
    to execute all calls before the next LLM turn. A future sprint may replace
    this with a typed multi-action representation on AgentStep.

    Usage:
        strategy = NativeToolCalling()
        messages, tools = strategy.build_messages(
            system_prompt="You are helpful",
            history=[user_msg],
            tool_defs=agent.get_tool_defs(),
        )
        response = await provider.complete(messages, tools=tools)
        step = strategy.parse_response(response)
    """

    def build_messages(
        self,
        *,
        system_prompt: str,
        history: list[Message],
        tool_defs: list[ToolDef],
    ) -> tuple[list[Message], list[ToolDef] | None]:
        """Prepend system prompt to history, pass tools through to provider.

        NativeToolCalling doesn't modify the prompt — tools go directly
        to the provider's API, not into the system prompt text.
        """
        messages = [Message(role=Role.SYSTEM, content=system_prompt), *history]
        tools = tool_defs if tool_defs else None
        return messages, tools

    def parse_response(self, response: LLMResponse) -> AgentStep:
        """Parse LLM response into an AgentStep.

        If the response has tool_calls → return the first ToolCall as action.
        If text only → treat as Finish (the agent is done reasoning).
        """
        if response.tool_calls:
            # Native tool calling: provider returned structured tool_use blocks.
            # AgentStep.action models a single action, so we store the first
            # tool call there. The full ordered list goes in meta["all_tool_calls"]
            # for the loop to execute all of them before the next LLM turn.
            tc = response.tool_calls[0]
            return AgentStep(
                reasoning=response.text,
                action=tc,
                raw_response=response.text,
                meta={"all_tool_calls": response.tool_calls},
            )

        # No tool calls — the LLM is done. Text is the final answer.
        return AgentStep(
            reasoning=None,
            action=Finish(answer=response.text or ""),
            raw_response=response.text,
        )

    def format_tool_result(self, result: ToolResult) -> Message:
        """Format tool result as a TOOL message with call_id correlation.

        The provider adapter will convert this to its native format
        (e.g., Anthropic tool_result block in a user message).
        """
        return Message(
            role=Role.TOOL,
            content=result.payload,
            name=result.name,
            call_id=result.call_id,
            meta={"is_error": True} if not result.success else {},
        )
