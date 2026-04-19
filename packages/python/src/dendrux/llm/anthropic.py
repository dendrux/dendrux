"""Anthropic Messages API provider.

Thin adapter between Dendrux's universal types and the Anthropic Messages API.
Handles bidirectional conversion:
  - Outbound: Message/ToolDef → Anthropic MessageParam/ToolParam
  - Inbound:  Anthropic response → LLMResponse/ToolCall

No business logic, no agent loop awareness. Just shape translation.
"""

from __future__ import annotations

import copy
import json
from typing import TYPE_CHECKING, Any, Literal

import anthropic
import httpx

from dendrux.llm._helpers import (
    build_call_index,
    connection_error,
    parse_tool_json_lossy,
    resolve_tool_message_call,
    timeout_error,
)
from dendrux.llm._retry_telemetry import (
    begin_call_attempt_tracking,
    call_attempt_tracking,
    end_call_attempt_tracking,
    make_telemetry_http_client,
)
from dendrux.llm.base import LLMProvider
from dendrux.types import (
    LLMResponse,
    ProviderCapabilities,
    Role,
    StreamEvent,
    StreamEventType,
    ToolCall,
    UsageStats,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from dendrux.types import Message, ToolDef

# Anthropic-specific kwargs that complete() will forward to the API.
# Anything not in this set is silently ignored to prevent leaking
# unsupported params into the API call.
_SUPPORTED_KWARGS = frozenset(
    {"temperature", "top_p", "top_k", "stop_sequences", "model", "max_tokens"}
)


class AnthropicProvider(LLMProvider):
    """Anthropic Messages API provider.

    Converts Dendrux's universal Message/ToolDef types to Anthropic's native
    format and normalizes responses back. Supports both batch (complete())
    and streaming (complete_stream()) with token-by-token text deltas and
    tool call events.

    Capabilities reflect what this adapter *implements*, not what the
    Anthropic API can theoretically do. Thinking and multimodal will be
    enabled when their corresponding code paths are added.

    Usage:
        provider = AnthropicProvider(api_key="sk-...", model="claude-sonnet-4-6")
        response = await provider.complete(messages, tools=tools)
    """

    capabilities = ProviderCapabilities(
        supports_native_tools=True,
        supports_tool_call_ids=True,
        supports_streaming=True,
        supports_streaming_tool_deltas=True,
        supports_thinking=False,  # Not implemented yet
        supports_multimodal=False,  # Not implemented yet
        supports_system_prompt=True,
        supports_parallel_tool_calls=True,
        supports_structured_output=True,
        max_context_tokens=200_000,
    )

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        max_tokens: int = 16_000,
        temperature: float | None = None,
        timeout: float = 120.0,
        max_retries: int = 3,
        cache_ttl: Literal["5m", "1h"] | None = None,
    ) -> None:
        """Create an Anthropic provider.

        Args:
            model: Model identifier (e.g. "claude-sonnet-4-6").
            api_key: API key. Defaults to ANTHROPIC_API_KEY env var.
            max_tokens: Maximum output tokens per call. Override per-call via kwargs.
            temperature: Sampling temperature. None = model default. Override per-call.
            timeout: HTTP request timeout in seconds.
            max_retries: Number of automatic retries on transient errors.
            cache_ttl: Anthropic prompt-cache TTL. ``None`` (default) omits the
                field so Anthropic uses its 5-minute default. Set ``"1h"`` for
                long-running workflows where iterations span more than 5 minutes
                (costs 2× base input on cache creation, reads stay cheap).
        """
        self._client = anthropic.AsyncAnthropic(
            api_key=api_key,
            http_client=make_telemetry_http_client(httpx.Timeout(timeout, connect=10.0)),
            max_retries=max_retries,
        )
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._timeout = timeout
        self._cache_ttl = cache_ttl

    @property
    def model(self) -> str:
        """The model identifier this provider is configured to use."""
        return self._model

    def __repr__(self) -> str:
        return f"AnthropicProvider(model={self._model!r})"

    async def close(self) -> None:
        """Close the underlying HTTP client and release connections."""
        await self._client.close()

    async def __aenter__(self) -> AnthropicProvider:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    def _cache_control_marker(self) -> dict[str, Any]:
        """Build the cache_control marker for system + last-message breakpoints.

        Includes ``ttl`` only when the provider was configured with one;
        otherwise Anthropic applies its default 5m.
        """
        marker: dict[str, Any] = {"type": "ephemeral"}
        if self._cache_ttl is not None:
            marker["ttl"] = self._cache_ttl
        return marker

    def _apply_cache_control(
        self,
        system_prompt: str,
        api_messages: list[dict[str, Any]],
    ) -> tuple[Any, list[dict[str, Any]]]:
        """Apply cache_control to system block and last message's last block.

        The marker on the last message warms the next iteration's cache —
        the call that writes it pays full creation; the next call reads it.

        Returns ``(system, messages)`` where:
        - ``system`` is a block list with cache_control, or ``NOT_GIVEN`` when
          no system prompt was provided.
        - ``messages`` is a fresh list with the last message deep-copied and
          augmented with cache_control. The caller's input is never mutated.
        """
        marker = self._cache_control_marker()

        if system_prompt:
            system_blocks: Any = [{"type": "text", "text": system_prompt, "cache_control": marker}]
        else:
            system_blocks = anthropic.NOT_GIVEN

        if not api_messages:
            return system_blocks, api_messages

        new_messages = list(api_messages)
        last = copy.deepcopy(new_messages[-1])
        content = last["content"]
        if isinstance(content, str):
            last["content"] = [{"type": "text", "text": content, "cache_control": marker}]
        elif isinstance(content, list) and content:
            content[-1]["cache_control"] = marker
        new_messages[-1] = last
        return system_blocks, new_messages

    def _build_api_kwargs(
        self,
        messages: list[Message],
        tools: list[ToolDef] | None,
        kwargs: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Build Anthropic API kwargs from Dendrux messages and caller kwargs.

        Returns (api_kwargs, captured_request) where captured_request is the
        serializable subset for the evidence layer.
        """
        system_prompt, api_messages = self._convert_messages(messages)
        api_tools = self._convert_tools(tools) if tools else anthropic.NOT_GIVEN
        system_blocks, cached_messages = self._apply_cache_control(system_prompt, api_messages)

        api_kwargs: dict[str, Any] = {
            "model": kwargs.pop("model", self._model),
            "max_tokens": kwargs.pop("max_tokens", self._max_tokens),
            "messages": cached_messages,
            "system": system_blocks,
            "tools": api_tools,
        }

        if "temperature" in kwargs:
            api_kwargs["temperature"] = kwargs.pop("temperature")
        elif self._temperature is not None:
            api_kwargs["temperature"] = self._temperature

        for key in _SUPPORTED_KWARGS - {"model", "max_tokens", "temperature"}:
            if key in kwargs:
                api_kwargs[key] = kwargs.pop(key)

        captured_request = {k: v for k, v in api_kwargs.items() if v is not anthropic.NOT_GIVEN}
        return api_kwargs, captured_request

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolDef] | None = None,
        *,
        output_schema: dict[str, Any] | None = None,
        run_id: str | None = None,
        cache_key_prefix: str | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Send messages to Claude and return a normalized response.

        kwargs override constructor defaults (e.g. model, max_tokens).
        Only Anthropic-supported kwargs are forwarded; unknown keys are ignored.

        ``run_id`` and ``cache_key_prefix`` are accepted for protocol
        compatibility with OpenAI providers but unused — Anthropic caching
        is keyed by byte-stable prefix and ``cache_control`` markers, not
        by an explicit cache key.

        When output_schema is provided, injects a synthetic tool with the
        schema and forces tool_choice. The tool_use response is extracted
        and normalized into LLMResponse.text as a JSON string, with
        tool_calls set to None (preserving the SingleCall invariant).
        """
        api_kwargs, captured_request = self._build_api_kwargs(messages, tools, kwargs)

        # Structured output: inject synthetic tool + forced tool_choice
        is_structured = output_schema is not None
        if is_structured:
            api_kwargs["tools"] = [
                {
                    "name": "structured_output",
                    "description": "Return your response in this exact format.",
                    "input_schema": output_schema,
                }
            ]
            api_kwargs["tool_choice"] = {"type": "tool", "name": "structured_output"}
            # Update captured_request so evidence reflects the actual API call
            captured_request["tools"] = api_kwargs["tools"]
            captured_request["tool_choice"] = api_kwargs["tool_choice"]

        try:
            with call_attempt_tracking():
                response = await self._client.messages.create(**api_kwargs)
        except anthropic.APITimeoutError:
            raise timeout_error("AnthropicProvider", self._timeout) from None
        except anthropic.APIConnectionError as exc:
            raise connection_error("Anthropic API", self._model, exc) from exc

        if is_structured:
            llm_response = self._normalize_structured_response(response)
        else:
            llm_response = self._normalize_response(response)

        # Attach adapter-boundary payloads for evidence layer
        llm_response.provider_request = captured_request
        llm_response.provider_response = response.model_dump()

        return llm_response

    async def complete_stream(
        self,
        messages: list[Message],
        tools: list[ToolDef] | None = None,
        *,
        output_schema: dict[str, Any] | None = None,
        run_id: str | None = None,
        cache_key_prefix: str | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream LLM response as events, token by token.

        Text deltas are yielded immediately as they arrive.
        Tool call arguments are accumulated internally and yielded as
        complete TOOL_USE_END events when each content block finishes.

        ``run_id`` / ``cache_key_prefix`` accepted for protocol parity but
        unused (Anthropic caching is byte-based via ``cache_control``).

        At stream end, yields a DONE event carrying the full LLMResponse
        (with usage stats and provider payloads) for the loop to consume.
        """
        api_kwargs, captured_request = self._build_api_kwargs(messages, tools, kwargs)

        # Accumulators for building the final LLMResponse
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        # Per-block state for tool call assembly
        _current_tool_name: str | None = None
        _current_tool_id: str | None = None
        _current_tool_json_parts: list[str] = []

        attempt_token = begin_call_attempt_tracking()
        try:
            async with self._client.messages.stream(**api_kwargs) as stream:
                async for event in stream:
                    if event.type == "content_block_start":
                        block = event.content_block
                        if block.type == "tool_use":
                            _current_tool_name = block.name
                            _current_tool_id = block.id
                            _current_tool_json_parts = []
                            yield StreamEvent(
                                type=StreamEventType.TOOL_USE_START,
                                tool_name=block.name,
                                tool_call_id=block.id,
                            )

                    elif event.type == "content_block_delta":
                        delta = event.delta
                        if delta.type == "text_delta":
                            text_parts.append(delta.text)
                            yield StreamEvent(
                                type=StreamEventType.TEXT_DELTA,
                                text=delta.text,
                            )
                        elif delta.type == "input_json_delta":
                            _current_tool_json_parts.append(delta.partial_json)

                    elif event.type == "content_block_stop":
                        if _current_tool_name is not None:
                            raw_json = "".join(_current_tool_json_parts)
                            params = parse_tool_json_lossy(
                                raw_json,
                                provider="anthropic",
                                model=self._model,
                                tool_name=_current_tool_name,
                                call_id=_current_tool_id or "",
                            )
                            tc = ToolCall(
                                name=_current_tool_name,
                                params=params,
                                provider_tool_call_id=_current_tool_id,
                            )
                            tool_calls.append(tc)
                            yield StreamEvent(
                                type=StreamEventType.TOOL_USE_END,
                                tool_call=tc,
                                tool_name=tc.name,
                                tool_call_id=_current_tool_id,
                            )
                            _current_tool_name = None
                            _current_tool_id = None
                            _current_tool_json_parts = []

                # Get the final message for usage stats and provider response
                final_message = await stream.get_final_message()

        except anthropic.APITimeoutError:
            raise timeout_error("AnthropicProvider", self._timeout) from None
        except anthropic.APIConnectionError as exc:
            raise connection_error("Anthropic API", self._model, exc, streaming=True) from exc
        finally:
            end_call_attempt_tracking(attempt_token)

        usage = UsageStats(
            input_tokens=final_message.usage.input_tokens,
            output_tokens=final_message.usage.output_tokens,
            total_tokens=final_message.usage.input_tokens + final_message.usage.output_tokens,
            cache_read_input_tokens=getattr(final_message.usage, "cache_read_input_tokens", None),
            cache_creation_input_tokens=getattr(
                final_message.usage, "cache_creation_input_tokens", None
            ),
        )

        llm_response = LLMResponse(
            text="".join(text_parts) if text_parts else None,
            tool_calls=tool_calls if tool_calls else None,
            raw=final_message,
            usage=usage,
        )
        llm_response.provider_request = captured_request
        llm_response.provider_response = final_message.model_dump()

        yield StreamEvent(type=StreamEventType.DONE, raw=llm_response)

    # ------------------------------------------------------------------
    # Outbound conversions: Dendrux → Anthropic
    # ------------------------------------------------------------------

    def _convert_messages(self, messages: list[Message]) -> tuple[str, list[dict[str, Any]]]:
        """Convert Dendrux messages to Anthropic format.

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
        call_index = build_call_index(messages)

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
                original_call = resolve_tool_message_call(msg, call_index)

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
        """Convert Dendrux ToolDefs to Anthropic tool format."""
        return [
            {
                "name": td.name,
                "description": td.description,
                "input_schema": td.parameters,
            }
            for td in tools
        ]

    # ------------------------------------------------------------------
    # Inbound conversions: Anthropic → Dendrux
    # ------------------------------------------------------------------

    def _normalize_structured_response(self, response: anthropic.types.Message) -> LLMResponse:
        """Extract structured output from a forced tool_use response.

        The synthetic "structured_output" tool's input is the structured
        data. We serialize it to JSON and put it in LLMResponse.text,
        with tool_calls=None so SingleCall's invariant is preserved.
        """
        structured_data: dict[str, Any] | None = None
        for block in response.content:
            if block.type == "tool_use" and block.name == "structured_output":
                structured_data = dict(block.input) if block.input else {}
                break

        if structured_data is None:
            raise RuntimeError(
                "Anthropic structured output: expected a tool_use block named "
                "'structured_output' but none was found in the response."
            )

        usage = UsageStats(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            total_tokens=response.usage.input_tokens + response.usage.output_tokens,
            cache_read_input_tokens=getattr(response.usage, "cache_read_input_tokens", None),
            cache_creation_input_tokens=getattr(
                response.usage, "cache_creation_input_tokens", None
            ),
        )

        return LLMResponse(
            text=json.dumps(structured_data),
            tool_calls=None,
            raw=response,
            usage=usage,
        )

    def _normalize_response(self, response: anthropic.types.Message) -> LLMResponse:
        """Convert Anthropic response to Dendrux LLMResponse.

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
            cache_read_input_tokens=getattr(response.usage, "cache_read_input_tokens", None),
            cache_creation_input_tokens=getattr(
                response.usage, "cache_creation_input_tokens", None
            ),
        )

        return LLMResponse(
            text="".join(text_parts) if text_parts else None,
            tool_calls=tool_calls if tool_calls else None,
            raw=response,
            usage=usage,
        )
