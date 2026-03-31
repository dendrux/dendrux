"""Strategy protocol — how the agent communicates with the LLM.

A strategy controls the translation between Dendrux's agent loop and the
LLM provider. It decides:
  - What messages to send (build_messages)
  - How to interpret the response (parse_response)
  - How to format tool results for the next turn (format_tool_result)
  - Whether to pass tools to the provider or embed them in the prompt

The loop calls these methods each iteration. The strategy never calls the
LLM directly — it prepares inputs and interprets outputs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dendrux.types import AgentStep, LLMResponse, Message, ToolDef, ToolResult


class Strategy(ABC):
    """Base class for LLM communication strategies.

    Subclasses implement the translation between the agent loop's
    universal types and the specific format the LLM expects/produces.

    Built-in strategies:
        NativeToolCalling — uses provider's native tool_use API
        PromptBasedReAct  — injects tools into the prompt as text (planned)
    """

    @abstractmethod
    def build_messages(
        self,
        *,
        system_prompt: str,
        history: list[Message],
        tool_defs: list[ToolDef],
    ) -> tuple[list[Message], list[ToolDef] | None]:
        """Prepare messages and tools for the LLM call.

        Args:
            system_prompt: The agent's system prompt.
            history: Conversation so far (user input + prior turns).
            tool_defs: Available tool definitions.

        Returns:
            (messages, tools_for_provider):
            - messages: The full message list to send to the provider.
            - tools_for_provider: ToolDefs to pass to provider.complete(),
              or None if tools are embedded in the prompt text.
        """

    @abstractmethod
    def parse_response(self, response: LLMResponse) -> AgentStep:
        """Parse an LLM response into an AgentStep.

        The strategy interprets the response based on how it structured the
        request. NativeToolCalling reads tool_calls; PromptBasedReAct parses text.

        Args:
            response: Normalized response from the LLM provider.

        Returns:
            AgentStep with reasoning + action (ToolCall, Finish, or Clarification).
        """

    @abstractmethod
    def format_tool_result(self, result: ToolResult) -> Message:
        """Format a tool execution result as a message for the next LLM turn.

        NativeToolCalling creates a Role.TOOL message with call_id correlation.
        PromptBasedReAct creates a Role.USER message with observation text.

        Args:
            result: The result of executing a tool.

        Returns:
            Message to append to conversation history.
        """
