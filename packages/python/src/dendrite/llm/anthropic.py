"""Anthropic Messages API provider.

Thin adapter between Dendrite's universal types and the Anthropic Messages API.
Handles bidirectional conversion:
  - Outbound: Message/ToolDef → Anthropic MessageParam/ToolParam
  - Inbound:  Anthropic response → LLMResponse/ToolCall

No business logic, no agent loop awareness. Just shape translation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import anthropic
import httpx

from dendrite.llm.base import LLMProvider
from dendrite.types import (
    LLMResponse,
    ProviderCapabilities,
    Role,
    ToolCall,
    UsageStats,
)

if TYPE_CHECKING:
    from dendrite.types import Message, ToolDef

# Anthropic-specific kwargs that complete() will forward to the API.
# Anything not in this set is silently ignored to prevent leaking
# unsupported params into the API call.
_SUPPORTED_KWARGS = frozenset(
    {"temperature", "top_p", "top_k", "stop_sequences", "model", "max_tokens"}
)


class AnthropicProvider(LLMProvider):
    """Anthropic Messages API provider.

    Converts Dendrite's universal Message/ToolDef types to Anthropic's native
    format and normalizes responses back. Sprint 1: batch only (complete()),
    streaming uses the base class fallback.

    Capabilities reflect what this adapter *implements*, not what the
    Anthropic API can theoretically do. Streaming, thinking, and multimodal
    will be enabled when their corresponding code paths are added.

    Usage:
        provider = AnthropicProvider(api_key="sk-...", model="claude-sonnet-4-6")
        response = await provider.complete(messages, tools=tools)
    """

    capabilities = ProviderCapabilities(
        supports_native_tools=True,
        supports_tool_call_ids=True,
        supports_streaming=False,  # No complete_stream() override yet
        supports_streaming_tool_deltas=False,
        supports_thinking=False,  # Not implemented in Sprint 1
        supports_multimodal=False,  # Not implemented in Sprint 1
        supports_system_prompt=True,
        supports_parallel_tool_calls=True,
        max_context_tokens=200_000,
    )

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        timeout: float = 120.0,
        max_retries: int = 3,
    ) -> None:
        self._client = anthropic.AsyncAnthropic(
            api_key=api_key,
            timeout=httpx.Timeout(timeout, connect=10.0),
            max_retries=max_retries,
        )
        self._model = model

    def __repr__(self) -> str:
        return f"AnthropicProvider(model={self._model!r})"

    async def close(self) -> None:
        """Close the underlying HTTP client and release connections."""
        await self._client.close()

    async def __aenter__(self) -> AnthropicProvider:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolDef] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Send messages to Claude and return a normalized response.

        kwargs override constructor defaults (e.g. model, max_tokens).
        Only Anthropic-supported kwargs are forwarded; unknown keys are ignored.
        """
        system_prompt, api_messages = self._convert_messages(messages)
        api_tools = self._convert_tools(tools) if tools else anthropic.NOT_GIVEN

        api_kwargs: dict[str, Any] = {
            "model": kwargs.pop("model", self._model),
            "max_tokens": kwargs.pop("max_tokens", 16_000),
            "messages": api_messages,
            "system": system_prompt if system_prompt else anthropic.NOT_GIVEN,
            "tools": api_tools,
        }

        # Forward only supported kwargs — ignore the rest
        for key in _SUPPORTED_KWARGS - {"model", "max_tokens"}:
            if key in kwargs:
                api_kwargs[key] = kwargs.pop(key)

        # Capture provider request payload before the call (exclude non-serializable NOT_GIVEN)
        captured_request = {k: v for k, v in api_kwargs.items() if v is not anthropic.NOT_GIVEN}

        response = await self._client.messages.create(**api_kwargs)
        llm_response = self._normalize_response(response)

        # Attach adapter-boundary payloads for evidence layer
        llm_response.provider_request = captured_request
        llm_response.provider_response = response.model_dump()

        return llm_response

    # ------------------------------------------------------------------
    # Outbound conversions: Dendrite → Anthropic
    # ------------------------------------------------------------------

    def _convert_messages(self, messages: list[Message]) -> tuple[str, list[dict[str, Any]]]:
        """Convert Dendrite messages to Anthropic format.

        Returns (system_prompt, api_messages) where:
        - SYSTEM messages are extracted and joined into a single string
        - ASSISTANT messages with tool_calls become content block arrays
        - TOOL messages with native-tool correlation (call_id → provider_tool_call_id)
          become tool_result blocks in user messages
        - TOOL messages without native correlation become plain user messages
          (strategy is responsible for formatting these as text observations)
        - USER/ASSISTANT (text-only) pass through normally
        """
        system_parts: list[str] = []
        api_messages: list[dict[str, Any]] = []

        # Build provider ID index: Dendrite call_id → ToolCall.
        # Used to resolve provider_tool_call_id for tool_result blocks.
        call_index: dict[str, ToolCall] = {}
        for msg in messages:
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    if tc.id in call_index:
                        err = (
                            f"Duplicate Dendrite call_id '{tc.id}' in conversation "
                            f"history. Tool calls must have unique IDs."
                        )
                        raise ValueError(err)
                    call_index[tc.id] = tc

        for msg in messages:
            if msg.role == Role.SYSTEM:
                system_parts.append(msg.content)

            elif msg.role == Role.USER:
                api_messages.append({"role": "user", "content": msg.content})

            elif msg.role == Role.ASSISTANT:
                if msg.tool_calls:
                    # Assistant with tool calls → content blocks
                    content: list[dict[str, Any]] = []
                    if msg.content:
                        content.append({"type": "text", "text": msg.content})
                    for tc in msg.tool_calls:
                        content.append(
                            {
                                "type": "tool_use",
                                "id": tc.provider_tool_call_id or tc.id,
                                "name": tc.name,
                                "input": tc.params,
                            }
                        )
                    api_messages.append({"role": "assistant", "content": content})
                else:
                    api_messages.append({"role": "assistant", "content": msg.content})

            elif msg.role == Role.TOOL:
                if msg.call_id is None:
                    raise ValueError(
                        "TOOL message missing call_id — this violates Message.__post_init__"
                    )
                original_call = call_index.get(msg.call_id)

                if original_call is None:
                    raise ValueError(
                        f"TOOL message references call_id '{msg.call_id}' "
                        f"but no matching ToolCall found in conversation history."
                    )

                if original_call.provider_tool_call_id:
                    # Native tool correlation path — emit tool_result block
                    self._append_tool_result(
                        api_messages,
                        tool_use_id=original_call.provider_tool_call_id,
                        content=msg.content,
                        is_error=bool(msg.meta.get("is_error")),
                    )
                else:
                    # No native correlation (e.g. PromptBasedReAct via
                    # NativeToolCalling — shouldn't happen, but be safe).
                    # Emit as plain user message so the strategy's text
                    # formatting is preserved.
                    api_messages.append({"role": "user", "content": msg.content})

        system_prompt = "\n\n".join(system_parts)
        return system_prompt, api_messages

    def _append_tool_result(
        self,
        api_messages: list[dict[str, Any]],
        *,
        tool_use_id: str,
        content: str,
        is_error: bool,
    ) -> None:
        """Append a tool_result block, merging with the previous user message if consecutive."""
        tool_result_block: dict[str, Any] = {
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": content,
        }
        if is_error:
            tool_result_block["is_error"] = True

        # Merge consecutive tool results into a single user message
        if (
            api_messages
            and api_messages[-1]["role"] == "user"
            and isinstance(api_messages[-1]["content"], list)
            and api_messages[-1]["content"]
            and api_messages[-1]["content"][0].get("type") == "tool_result"
        ):
            api_messages[-1]["content"].append(tool_result_block)
        else:
            api_messages.append({"role": "user", "content": [tool_result_block]})

    def _convert_tools(self, tools: list[ToolDef]) -> list[dict[str, Any]]:
        """Convert Dendrite ToolDefs to Anthropic tool format."""
        return [
            {
                "name": td.name,
                "description": td.description,
                "input_schema": td.parameters,
            }
            for td in tools
        ]

    # ------------------------------------------------------------------
    # Inbound conversions: Anthropic → Dendrite
    # ------------------------------------------------------------------

    def _normalize_response(self, response: anthropic.types.Message) -> LLMResponse:
        """Convert Anthropic response to Dendrite LLMResponse.

        Text blocks are concatenated with no separator — each block is
        already self-contained. This avoids injecting content that doesn't
        exist in the provider output.
        """
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        name=block.name,
                        params=dict(block.input) if block.input else {},
                        provider_tool_call_id=block.id,
                    )
                )

        usage = UsageStats(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            total_tokens=response.usage.input_tokens + response.usage.output_tokens,
        )

        return LLMResponse(
            text="".join(text_parts) if text_parts else None,
            tool_calls=tool_calls if tool_calls else None,
            raw=response,
            usage=usage,
        )
