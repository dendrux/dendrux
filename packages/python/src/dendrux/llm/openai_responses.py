"""OpenAI Responses API provider.

Thin adapter between Dendrux's universal types and the OpenAI Responses API.
Use this provider when you need OpenAI's built-in tools (web_search,
code_interpreter, file_search) alongside custom Dendrux tools.

For standard function calling without built-in tools, use OpenAIProvider
(Chat Completions) instead — it works with any OpenAI-compatible API.

No business logic, no agent loop awareness. Just shape translation.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

import httpx
import openai

from dendrux.llm._helpers import (
    build_call_index,
    connection_error,
    parse_tool_json_lossy,
    parse_tool_json_strict,
    resolve_tool_message_call,
    timeout_error,
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

logger = logging.getLogger(__name__)

# Responses-API kwargs that complete() will forward.
_SUPPORTED_KWARGS = frozenset(
    {
        "temperature",
        "top_p",
        "max_output_tokens",
        "model",
        "tool_choice",
    }
)

# Built-in tool types supported via builtin_tools parameter.
# Only web_search_preview works with a bare {"type": "..."} dict.
# code_interpreter and file_search require additional config (container,
# vector_store_ids) — pass them directly via complete() kwargs when needed.
_BUILTIN_TOOL_TYPES = frozenset({"web_search_preview"})


class OpenAIResponsesProvider(LLMProvider):
    """OpenAI Responses API provider.

    Use this when you need OpenAI's built-in tools (web search)
    alongside custom Dendrux function tools.

    For standard function calling without built-in tools, use
    OpenAIProvider (Chat Completions) instead.

    Limitation — reasoning models with tool calling:
        Reasoning models (o-series, GPT-5) may return internal reasoning
        items alongside function calls. Dendrux's Message type does not
        preserve these items between turns, so multi-turn tool calling
        with reasoning models may lose context. For reasoning models
        without tool calling, or for non-reasoning models with tools,
        this provider works correctly. This will be fixed when Dendrux
        adds first-class thinking/reasoning support.

    Usage:
        # With built-in web search + custom tools
        provider = OpenAIResponsesProvider(
            model="gpt-4o",
            builtin_tools=["web_search_preview"],
        )

        # With reasoning effort (o-series / GPT-5)
        provider = OpenAIResponsesProvider(
            model="gpt-5",
            reasoning_effort="medium",
        )
    """

    capabilities = ProviderCapabilities(
        supports_native_tools=True,
        supports_tool_call_ids=True,
        supports_streaming=True,
        supports_streaming_tool_deltas=True,
        supports_thinking=False,
        supports_multimodal=False,
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
        builtin_tools: list[str] | None = None,
        max_output_tokens: int = 16_000,
        temperature: float | None = None,
        reasoning_effort: str | None = None,
        timeout: float = 120.0,
        max_retries: int = 3,
    ) -> None:
        """Create an OpenAI Responses API provider.

        Args:
            model: Model identifier (e.g. "gpt-4o", "gpt-5").
            api_key: API key. Defaults to OPENAI_API_KEY env var.
            builtin_tools: OpenAI built-in tools to enable (e.g. "web_search_preview").
            max_output_tokens: Maximum output tokens per call. Override per-call.
            temperature: Sampling temperature. None = model default.
            reasoning_effort: Reasoning depth ("low", "medium", "high").
            timeout: HTTP request timeout in seconds.
            max_retries: Number of automatic retries on transient errors.
        """
        for t in builtin_tools or []:
            if t not in _BUILTIN_TOOL_TYPES:
                raise ValueError(
                    f"Unknown built-in tool '{t}'. "
                    f"Supported: {', '.join(sorted(_BUILTIN_TOOL_TYPES))}"
                )

        self._client = openai.AsyncOpenAI(
            api_key=api_key,
            timeout=httpx.Timeout(timeout, connect=10.0),
            max_retries=max_retries,
        )
        self._model = model
        self._builtin_tools = builtin_tools or []
        self._max_output_tokens = max_output_tokens
        self._temperature = temperature
        self._reasoning_effort = reasoning_effort
        self._timeout = timeout

    @property
    def model(self) -> str:
        """The model identifier this provider is configured to use."""
        return self._model

    def __repr__(self) -> str:
        parts = [f"model={self._model!r}"]
        if self._builtin_tools:
            parts.append(f"builtin_tools={self._builtin_tools!r}")
        return f"OpenAIResponsesProvider({', '.join(parts)})"

    async def close(self) -> None:
        """Close the underlying HTTP client and release connections."""
        await self._client.close()

    async def __aenter__(self) -> OpenAIResponsesProvider:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    def _build_api_kwargs(
        self,
        messages: list[Message],
        tools: list[ToolDef] | None,
        kwargs: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Build Responses API kwargs from Dendrux messages and caller kwargs.

        Returns (api_kwargs, captured_request) where captured_request is the
        serializable subset for the evidence layer.
        """
        instructions, input_items = self._convert_messages(messages)
        api_tools = self._build_tools(tools)

        max_output_tokens = kwargs.pop(
            "max_output_tokens",
            kwargs.pop("max_tokens", self._max_output_tokens),
        )

        api_kwargs: dict[str, Any] = {
            "model": kwargs.pop("model", self._model),
            "input": input_items,
            "max_output_tokens": max_output_tokens,
        }

        # Only pass instructions if non-empty
        if instructions:
            api_kwargs["instructions"] = instructions

        # Only pass tools if non-empty
        if api_tools:
            api_kwargs["tools"] = api_tools

        # Apply constructor defaults for optional params (per-call kwargs override)
        for attr, key in [
            (self._temperature, "temperature"),
        ]:
            if key in kwargs:
                api_kwargs[key] = kwargs.pop(key)
            elif attr is not None:
                api_kwargs[key] = attr

        # Handle reasoning_effort — Responses API uses nested reasoning object
        reasoning_effort = kwargs.pop("reasoning_effort", self._reasoning_effort)
        if reasoning_effort is not None:
            api_kwargs["reasoning"] = {"effort": reasoning_effort}

        # Forward remaining supported kwargs
        already_handled = {
            "model",
            "max_output_tokens",
            "max_tokens",
            "temperature",
        }
        for key in _SUPPORTED_KWARGS - already_handled:
            if key in kwargs:
                api_kwargs[key] = kwargs.pop(key)

        captured_request = dict(api_kwargs)
        return api_kwargs, captured_request

    def _inject_structured_output(
        self,
        api_kwargs: dict[str, Any],
        captured_request: dict[str, Any],
        output_schema: dict[str, Any],
    ) -> None:
        """Inject text.format for structured output and update evidence."""
        from dendrux.llm._schema import normalize_for_openai_strict

        strict_schema = normalize_for_openai_strict(output_schema)
        schema_name = output_schema.get("title", "structured_output")
        text_format = {
            "format": {
                "type": "json_schema",
                "name": schema_name,
                "strict": True,
                "schema": strict_schema,
            },
        }
        api_kwargs["text"] = text_format
        captured_request["text"] = text_format

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolDef] | None = None,
        *,
        output_schema: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Send messages to the model via the Responses API.

        kwargs override constructor defaults (e.g. model, max_output_tokens).
        Only supported kwargs are forwarded; unknown keys are ignored.

        When output_schema is provided, sets text.format to json_schema
        with strict mode.
        """
        api_kwargs, captured_request = self._build_api_kwargs(messages, tools, kwargs)

        if output_schema is not None:
            self._inject_structured_output(api_kwargs, captured_request, output_schema)

        try:
            response = await self._client.responses.create(**api_kwargs)
        except openai.APITimeoutError:
            raise timeout_error("OpenAIResponsesProvider", self._timeout) from None
        except openai.APIConnectionError as exc:
            raise connection_error("OpenAI Responses API", self._model, exc) from exc

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
        **kwargs: Any,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream LLM response as events via the Responses API.

        The Responses API has explicit event types for each phase of output
        (content deltas, function call starts/args/done), making the mapping
        to StreamEvent straightforward.

        At stream end, yields a DONE event carrying the full LLMResponse.

        Raises NotImplementedError if the endpoint rejects streaming.
        """
        api_kwargs, captured_request = self._build_api_kwargs(messages, tools, kwargs)

        if output_schema is not None:
            self._inject_structured_output(api_kwargs, captured_request, output_schema)
        api_kwargs["stream"] = True

        try:
            stream = await self._client.responses.create(**api_kwargs)
        except openai.BadRequestError as exc:
            raise NotImplementedError(
                f"Streaming request rejected by the Responses API endpoint. "
                f"Use agent.run() for batch completion. "
                f"Original error: {exc.message}"
            ) from exc
        except openai.APITimeoutError:
            raise timeout_error("OpenAIResponsesProvider", self._timeout) from None
        except openai.APIConnectionError as exc:
            raise connection_error(
                "OpenAI Responses API",
                self._model,
                exc,
                streaming=True,
            ) from exc

        # Accumulators for building the final LLMResponse
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        usage = UsageStats()

        # Track active function calls: item_id → (name, call_id)
        _active_calls: dict[str, tuple[str, str]] = {}
        _completed_response: Any = None

        async for event in stream:
            event_type = getattr(event, "type", None)

            # --- Text deltas ---
            if event_type == "response.output_text.delta":
                delta_text = event.delta
                if delta_text:
                    text_parts.append(delta_text)
                    yield StreamEvent(type=StreamEventType.TEXT_DELTA, text=delta_text)

            # --- Function call lifecycle (ordered: added → delta → done) ---
            elif event_type == "response.output_item.added":
                item = event.item
                if getattr(item, "type", None) == "function_call":
                    item_id = item.id
                    name = item.name
                    call_id = getattr(item, "call_id", item_id)
                    _active_calls[item_id] = (name, call_id)
                    yield StreamEvent(
                        type=StreamEventType.TOOL_USE_START,
                        tool_name=name,
                        tool_call_id=call_id,
                    )

            elif event_type == "response.function_call_arguments.delta":
                # Incremental arg fragments — done event gives the complete
                # string, so no accumulation needed here.
                pass

            elif event_type == "response.function_call_arguments.done":
                item_id = event.item_id
                raw_args = event.arguments
                name, call_id = _active_calls.pop(item_id, ("unknown", item_id))
                if name == "unknown":
                    logger.warning(
                        "Tool call completed with no matching start event — "
                        "provider=%s model=%s item_id=%s",
                        "openai_responses",
                        self._model,
                        item_id,
                    )
                params = parse_tool_json_lossy(
                    raw_args or "",
                    provider="openai_responses",
                    model=self._model,
                    tool_name=name,
                    call_id=call_id,
                )
                tc = ToolCall(
                    name=name,
                    params=params,
                    provider_tool_call_id=call_id,
                )
                tool_calls.append(tc)
                yield StreamEvent(
                    type=StreamEventType.TOOL_USE_END,
                    tool_call=tc,
                    tool_name=tc.name,
                    tool_call_id=call_id,
                )

            # --- Response completed — extract usage ---
            elif event_type == "response.completed":
                _completed_response = event.response
                if hasattr(_completed_response, "usage") and _completed_response.usage:
                    usage = UsageStats(
                        input_tokens=getattr(_completed_response.usage, "input_tokens", 0),
                        output_tokens=getattr(_completed_response.usage, "output_tokens", 0),
                        total_tokens=getattr(_completed_response.usage, "total_tokens", 0),
                    )

            # --- Response failed ---
            elif event_type == "response.failed":
                error_msg = getattr(event, "error", "Unknown streaming error")
                raise RuntimeError(f"Responses API stream failed: {error_msg}")

        # Assemble the final LLMResponse
        llm_response = LLMResponse(
            text="".join(text_parts) if text_parts else None,
            tool_calls=tool_calls if tool_calls else None,
            raw=None,
            usage=usage,
        )
        llm_response.provider_request = captured_request
        if _completed_response is not None and hasattr(_completed_response, "model_dump"):
            llm_response.provider_response = _completed_response.model_dump()

        yield StreamEvent(type=StreamEventType.DONE, raw=llm_response)

    # ------------------------------------------------------------------
    # Outbound conversions: Dendrux → OpenAI Responses API
    # ------------------------------------------------------------------

    def _convert_messages(self, messages: list[Message]) -> tuple[str, list[dict[str, Any]]]:
        """Convert Dendrux messages to Responses API input format.

        Returns (instructions, input_items) where:
          - SYSTEM messages → joined into instructions string
          - USER → {role: "user", content: ...}
          - ASSISTANT → {role: "assistant", content: ...}
          - ASSISTANT with tool_calls → function_call items
          - TOOL → function_call_output items
        """
        instructions_parts: list[str] = []
        input_items: list[dict[str, Any]] = []
        call_index = build_call_index(messages)

        for msg in messages:
            if msg.role == Role.SYSTEM:
                instructions_parts.append(msg.content)

            elif msg.role == Role.USER:
                input_items.append({"role": "user", "content": msg.content})

            elif msg.role == Role.ASSISTANT:
                if msg.tool_calls:
                    # Emit text content if present
                    if msg.content:
                        input_items.append({"role": "assistant", "content": msg.content})
                    # Emit function_call items
                    for tc in msg.tool_calls:
                        input_items.append(
                            {
                                "type": "function_call",
                                "call_id": tc.provider_tool_call_id or tc.id,
                                "name": tc.name,
                                "arguments": json.dumps(tc.params),
                            }
                        )
                else:
                    input_items.append({"role": "assistant", "content": msg.content})

            elif msg.role == Role.TOOL:
                original_call = resolve_tool_message_call(msg, call_index)
                input_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": original_call.provider_tool_call_id or original_call.id,
                        "output": msg.content,
                    }
                )

        instructions = "\n\n".join(instructions_parts)
        return instructions, input_items

    def _build_tools(self, dendrux_tools: list[ToolDef] | None) -> list[dict[str, Any]]:
        """Build the tools array: built-in tools + Dendrux function tools."""
        api_tools: list[dict[str, Any]] = []

        # Add built-in tools (web_search, code_interpreter, etc.)
        for builtin in self._builtin_tools:
            api_tools.append({"type": builtin})

        # Add Dendrux function tools (flatter format than Chat Completions)
        if dendrux_tools:
            for td in dendrux_tools:
                api_tools.append(
                    {
                        "type": "function",
                        "name": td.name,
                        "description": td.description,
                        "parameters": td.parameters,
                    }
                )

        return api_tools

    # ------------------------------------------------------------------
    # Inbound conversions: OpenAI Responses → Dendrux
    # ------------------------------------------------------------------

    def _normalize_response(self, response: Any) -> LLMResponse:
        """Convert Responses API response to Dendrux LLMResponse.

        The response.output is a list of items. We extract:
          - output_text items → text
          - function_call items → ToolCall list
          - Built-in tool outputs (web_search results, etc.) are consumed
            by the model internally — they don't appear as ToolCalls.
        """
        text = response.output_text if hasattr(response, "output_text") else None

        # Extract function calls from output items
        tool_calls: list[ToolCall] = []
        for item in response.output:
            if getattr(item, "type", None) == "function_call":
                params = parse_tool_json_strict(
                    item.arguments or "",
                    tool_name=item.name,
                    call_id=item.call_id,
                )
                tool_calls.append(
                    ToolCall(
                        name=item.name,
                        params=params,
                        provider_tool_call_id=item.call_id,
                    )
                )

        # Extract usage
        usage = UsageStats()
        if hasattr(response, "usage") and response.usage:
            usage = UsageStats(
                input_tokens=getattr(response.usage, "input_tokens", 0),
                output_tokens=getattr(response.usage, "output_tokens", 0),
                total_tokens=getattr(response.usage, "total_tokens", 0),
            )

        return LLMResponse(
            text=text if text else None,
            tool_calls=tool_calls if tool_calls else None,
            raw=response,
            usage=usage,
        )
