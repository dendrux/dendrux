"""OpenAI Chat Completions API provider.

Thin adapter between Dendrux's universal types and the OpenAI Chat Completions API.
Handles bidirectional conversion:
  - Outbound: Message/ToolDef → OpenAI ChatCompletionMessageParam/tool
  - Inbound:  OpenAI ChatCompletion → LLMResponse/ToolCall

Works with any OpenAI-compatible API (vLLM, SGLang, Groq, Together, Ollama, etc.)
by setting base_url.

No business logic, no agent loop awareness. Just shape translation.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, Literal

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

logger = logging.getLogger(__name__)


_OPENAI_DEFAULT_BASE_URL = "https://api.openai.com/v1/"


def _build_usage_with_cache(usage: Any) -> UsageStats:
    """Build UsageStats from an OpenAI Chat Completions usage object.

    Reads cached_tokens from usage.prompt_tokens_details when present
    (compatible backends may omit it). Normalizes input_tokens to mean
    fresh-only by subtracting cached from prompt_tokens, so cross-provider
    semantics match Anthropic's native behavior. Clamps to zero so a
    misbehaving vendor that reports cached > prompt does not propagate
    negative counts into the rollup or budget guardrails.
    """
    prompt_tokens = usage.prompt_tokens or 0
    completion_tokens = usage.completion_tokens or 0
    details = getattr(usage, "prompt_tokens_details", None)
    cache_read: int | None = getattr(details, "cached_tokens", None) if details else None
    fresh_input = max(0, prompt_tokens - (cache_read or 0))
    return UsageStats(
        input_tokens=fresh_input,
        output_tokens=completion_tokens,
        total_tokens=fresh_input + completion_tokens,
        cache_read_input_tokens=cache_read,
        cache_creation_input_tokens=None,
    )


# OpenAI-specific kwargs that complete() will forward to the API.
_SUPPORTED_KWARGS = frozenset(
    {
        "temperature",
        "top_p",
        "stop",
        "max_tokens",
        "max_completion_tokens",
        "model",
        "frequency_penalty",
        "presence_penalty",
        "seed",
        "tool_choice",
        "reasoning_effort",
    }
)


class _ToolCallBuffer:
    """Accumulator for one streaming tool call, keyed by chunk index.

    Handles any arrival order of name, id, and argument fragments —
    some OpenAI-compatible backends don't guarantee name-first delivery.
    """

    __slots__ = ("name", "tool_call_id", "json_parts", "start_emitted")

    def __init__(self) -> None:
        self.name: str | None = None
        self.tool_call_id: str | None = None
        self.json_parts: list[str] = []
        self.start_emitted: bool = False


class OpenAIProvider(LLMProvider):
    """OpenAI Chat Completions API provider.

    Works with OpenAI and any OpenAI-compatible API (vLLM, SGLang, Groq,
    Together, Ollama, LM Studio) via base_url.

    Usage:
        # OpenAI
        provider = OpenAIProvider(model="gpt-4o")

        # vLLM / SGLang
        provider = OpenAIProvider(
            model="meta-llama/Llama-3-70B",
            base_url="http://localhost:8000/v1",
        )

        # Groq
        provider = OpenAIProvider(
            model="llama-3.3-70b",
            base_url="https://api.groq.com/openai/v1",
            api_key="gsk-...",
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
        max_context_tokens=128_000,
    )

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        max_tokens: int = 16_000,
        temperature: float | None = None,
        reasoning_effort: str | None = None,
        timeout: float = 120.0,
        max_retries: int = 3,
        prompt_cache_retention: Literal["in-memory", "24h"] | None = None,
    ) -> None:
        """Create an OpenAI Chat Completions provider.

        Args:
            model: Model identifier (e.g. "gpt-4o", "gpt-4o-mini").
            api_key: API key. Defaults to OPENAI_API_KEY env var.
            base_url: Override for compatible APIs (vLLM, SGLang, Groq, etc.).
            max_tokens: Maximum output tokens per call. Override per-call via kwargs.
            temperature: Sampling temperature. None = model default. For GPT-4o family.
            reasoning_effort: Reasoning depth ("low", "medium", "high"). For o-series/GPT-5.
            timeout: HTTP request timeout in seconds.
            max_retries: Number of automatic retries on transient errors.
            prompt_cache_retention: OpenAI prompt-cache retention. ``None``
                (default) omits the field so OpenAI uses its in-memory default
                (5–10 min idle, up to 1 h). Set ``"24h"`` for long-running
                workflows on supported models. Hyphenated literal per the SDK.
        """
        self._client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=make_telemetry_http_client(httpx.Timeout(timeout, connect=10.0)),
            max_retries=max_retries,
        )
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._reasoning_effort = reasoning_effort
        self._timeout = timeout
        self._prompt_cache_retention = prompt_cache_retention

    @property
    def model(self) -> str:
        """The model identifier this provider is configured to use."""
        return self._model

    def __repr__(self) -> str:
        base = str(self._client.base_url)
        if base != "https://api.openai.com/v1/":
            return f"OpenAIProvider(model={self._model!r}, base_url={base!r})"
        return f"OpenAIProvider(model={self._model!r})"

    async def close(self) -> None:
        """Close the underlying HTTP client and release connections."""
        await self._client.close()

    async def __aenter__(self) -> OpenAIProvider:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    def _build_api_kwargs(
        self,
        messages: list[Message],
        tools: list[ToolDef] | None,
        kwargs: dict[str, Any],
        *,
        run_id: str | None = None,
        cache_key_prefix: str | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Build OpenAI API kwargs from Dendrux messages and caller kwargs.

        Returns (api_kwargs, captured_request) where captured_request is the
        serializable subset for the evidence layer.

        Cache routing: ``cache_key_prefix`` (typically ``"{agent_name}:{model}"``)
        takes priority; ``run_id`` is the fallback. The provider's
        ``prompt_cache_retention`` is forwarded only when set, so vendor
        defaults apply otherwise.
        """
        api_messages = self._convert_messages(messages)
        api_tools = self._convert_tools(tools) if tools else openai.NOT_GIVEN

        # Resolve max_tokens: per-call kwargs override constructor default
        max_tokens = kwargs.pop(
            "max_completion_tokens",
            kwargs.pop("max_tokens", self._max_tokens),
        )

        api_kwargs: dict[str, Any] = {
            "model": kwargs.pop("model", self._model),
            "messages": api_messages,
            "tools": api_tools,
            "max_tokens": max_tokens,
        }

        # Apply constructor defaults for optional params (per-call kwargs override)
        for attr, key in [
            (self._temperature, "temperature"),
            (self._reasoning_effort, "reasoning_effort"),
        ]:
            if key in kwargs:
                api_kwargs[key] = kwargs.pop(key)
            elif attr is not None:
                api_kwargs[key] = attr

        # Forward remaining supported kwargs — ignore the rest
        already_handled = {
            "model",
            "max_tokens",
            "max_completion_tokens",
            "temperature",
            "reasoning_effort",
        }
        for key in _SUPPORTED_KWARGS - already_handled:
            if key in kwargs:
                api_kwargs[key] = kwargs.pop(key)

        # Cache routing — derive key from prefix or run_id.
        # Skip on non-OpenAI base_urls (Groq/Together/vLLM/etc.) since some
        # compatible backends reject unknown OpenAI-specific request fields.
        if str(self._client.base_url) == _OPENAI_DEFAULT_BASE_URL:
            cache_key_source = cache_key_prefix or run_id
            if cache_key_source:
                api_kwargs["prompt_cache_key"] = f"dendrux:{cache_key_source}"
            if self._prompt_cache_retention is not None:
                api_kwargs["prompt_cache_retention"] = self._prompt_cache_retention

        captured_request = {k: v for k, v in api_kwargs.items() if v is not openai.NOT_GIVEN}
        return api_kwargs, captured_request

    def _inject_structured_output(
        self,
        api_kwargs: dict[str, Any],
        captured_request: dict[str, Any],
        output_schema: dict[str, Any],
    ) -> None:
        """Inject response_format for structured output and update evidence."""
        from dendrux.llm._schema import normalize_for_openai_strict

        strict_schema = normalize_for_openai_strict(output_schema)
        schema_name = output_schema.get("title", "structured_output")
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "strict": True,
                "schema": strict_schema,
            },
        }
        api_kwargs["response_format"] = response_format
        captured_request["response_format"] = response_format

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
        """Send messages to the model and return a normalized response.

        kwargs override constructor defaults (e.g. model, max_tokens, temperature).
        Only supported kwargs are forwarded; unknown keys are ignored.

        ``cache_key_prefix`` (typically ``"{agent_name}:{model}"``) and
        ``run_id`` together drive the ``prompt_cache_key`` sent to OpenAI —
        prefix preferred, run_id as fallback.

        When output_schema is provided, sets response_format to json_schema
        with strict mode. The response is a JSON string in message.content.
        """
        api_kwargs, captured_request = self._build_api_kwargs(
            messages, tools, kwargs, run_id=run_id, cache_key_prefix=cache_key_prefix
        )

        if output_schema is not None:
            self._inject_structured_output(api_kwargs, captured_request, output_schema)

        try:
            with call_attempt_tracking():
                response = await self._client.chat.completions.create(**api_kwargs)
        except openai.APITimeoutError:
            raise timeout_error("OpenAIProvider", self._timeout) from None
        except openai.APIConnectionError as exc:
            raise connection_error("OpenAI API", self._model, exc) from exc

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
        Tool call arguments are buffered by index and flushed as complete
        TOOL_USE_END events when the terminal finish_reason arrives.

        ``cache_key_prefix`` / ``run_id`` drive ``prompt_cache_key`` for
        OpenAI cache pool routing.

        At stream end, yields a DONE event carrying the full LLMResponse.

        Raises NotImplementedError if the endpoint rejects streaming (common
        with some OpenAI-compatible backends).
        """
        api_kwargs, captured_request = self._build_api_kwargs(
            messages, tools, kwargs, run_id=run_id, cache_key_prefix=cache_key_prefix
        )

        if output_schema is not None:
            self._inject_structured_output(api_kwargs, captured_request, output_schema)
        api_kwargs["stream"] = True
        api_kwargs["stream_options"] = {"include_usage": True}

        attempt_token = begin_call_attempt_tracking()
        try:
            stream = await self._client.chat.completions.create(**api_kwargs)
        except openai.BadRequestError as exc:
            raise NotImplementedError(
                f"Streaming request rejected by endpoint at {self._client.base_url}. "
                f"If this endpoint does not support streaming, use agent.run() instead. "
                f"Original error: {exc.message}"
            ) from exc
        except openai.APITimeoutError:
            raise timeout_error("OpenAIProvider", self._timeout) from None
        except openai.APIConnectionError as exc:
            raise connection_error("OpenAI API", self._model, exc, streaming=True) from exc
        finally:
            # Counter only spans the request-establishing call. Once the
            # stream is open the SDK no longer retries, so iteration below
            # does not need the counter.
            end_call_attempt_tracking(attempt_token)

        # Accumulators for building the final LLMResponse
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        usage = UsageStats()
        finish_reason: str | None = None

        # Tool call buffers keyed by chunk index — no ordering assumptions.
        # Name, id, and arguments may arrive in any order for a given index
        # (some OpenAI-compatible backends don't guarantee name-first).
        _tool_buffers: dict[int, _ToolCallBuffer] = {}

        async for chunk in stream:
            # Usage arrives in the final chunk
            if chunk.usage:
                usage = _build_usage_with_cache(chunk.usage)

            if not chunk.choices:
                continue

            choice = chunk.choices[0]
            delta = choice.delta

            # --- Text deltas ---
            if delta.content:
                text_parts.append(delta.content)
                yield StreamEvent(type=StreamEventType.TEXT_DELTA, text=delta.content)

            # --- Tool call deltas ---
            if delta.tool_calls:
                for delta_tc in delta.tool_calls:
                    idx = delta_tc.index

                    # Ensure buffer exists for this index
                    if idx not in _tool_buffers:
                        _tool_buffers[idx] = _ToolCallBuffer()

                    buf = _tool_buffers[idx]

                    # Capture id (first occurrence)
                    if delta_tc.id and buf.tool_call_id is None:
                        buf.tool_call_id = delta_tc.id

                    if delta_tc.function:
                        # Capture name (first occurrence)
                        if delta_tc.function.name and buf.name is None:
                            buf.name = delta_tc.function.name

                        # Accumulate argument fragments
                        if delta_tc.function.arguments:
                            buf.json_parts.append(delta_tc.function.arguments)

                    # Emit TOOL_USE_START once we know the name (deferred
                    # if args arrived before name for this index).
                    if buf.name and not buf.start_emitted:
                        buf.start_emitted = True
                        yield StreamEvent(
                            type=StreamEventType.TOOL_USE_START,
                            tool_name=buf.name,
                            tool_call_id=buf.tool_call_id,
                        )

            # --- Flush all buffered tool calls on terminal finish_reason ---
            if choice.finish_reason is not None:
                finish_reason = choice.finish_reason
                if _tool_buffers:
                    for flush_idx in sorted(_tool_buffers):
                        buf = _tool_buffers[flush_idx]
                        raw_json = "".join(buf.json_parts)
                        params = parse_tool_json_lossy(
                            raw_json,
                            provider="openai",
                            model=self._model,
                            tool_name=buf.name or "",
                            call_id=buf.tool_call_id or "",
                        )

                        # Ensure START was emitted even if name arrived late
                        if buf.name and not buf.start_emitted:
                            buf.start_emitted = True
                            yield StreamEvent(
                                type=StreamEventType.TOOL_USE_START,
                                tool_name=buf.name,
                                tool_call_id=buf.tool_call_id,
                            )

                        if buf.name is None:
                            logger.warning(
                                "Tool call completed with no name — "
                                "provider=%s model=%s call_id=%s",
                                "openai",
                                self._model,
                                buf.tool_call_id,
                            )

                        tc = ToolCall(
                            name=buf.name or "unknown",
                            params=params,
                            provider_tool_call_id=buf.tool_call_id,
                        )
                        tool_calls.append(tc)
                        yield StreamEvent(
                            type=StreamEventType.TOOL_USE_END,
                            tool_call=tc,
                            tool_name=tc.name,
                            tool_call_id=buf.tool_call_id,
                        )
                    _tool_buffers.clear()

        # Assemble the final LLMResponse
        llm_response = LLMResponse(
            text="".join(text_parts) if text_parts else None,
            tool_calls=tool_calls if tool_calls else None,
            raw=None,
            usage=usage,
        )
        llm_response.provider_request = captured_request
        llm_response.provider_response = {
            "object": "chat.completion.chunked",
            "finish_reason": finish_reason,
            "usage": {
                "prompt_tokens": usage.input_tokens,
                "completion_tokens": usage.output_tokens,
                "total_tokens": usage.total_tokens,
            },
        }

        yield StreamEvent(type=StreamEventType.DONE, raw=llm_response)

    # ------------------------------------------------------------------
    # Outbound conversions: Dendrux → OpenAI
    # ------------------------------------------------------------------

    def _convert_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        """Convert Dendrux messages to OpenAI Chat Completions format.

        Role mapping:
          - SYSTEM → "system" (or "developer" — both work, "system" is universal)
          - USER → "user"
          - ASSISTANT → "assistant", with tool_calls array if present
          - TOOL → "tool" with tool_call_id correlation
        """
        api_messages: list[dict[str, Any]] = []
        call_index = build_call_index(messages)

        for msg in messages:
            if msg.role == Role.SYSTEM:
                api_messages.append({"role": "system", "content": msg.content})

            elif msg.role == Role.USER:
                api_messages.append({"role": "user", "content": msg.content})

            elif msg.role == Role.ASSISTANT:
                if msg.tool_calls:
                    # Assistant with tool calls
                    api_msg: dict[str, Any] = {"role": "assistant"}
                    if msg.content:
                        api_msg["content"] = msg.content
                    api_msg["tool_calls"] = [
                        {
                            "id": tc.provider_tool_call_id or tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.params),
                            },
                        }
                        for tc in msg.tool_calls
                    ]
                    api_messages.append(api_msg)
                else:
                    api_messages.append({"role": "assistant", "content": msg.content})

            elif msg.role == Role.TOOL:
                original_call = resolve_tool_message_call(msg, call_index)

                api_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": original_call.provider_tool_call_id or original_call.id,
                        "content": msg.content,
                    }
                )

        return api_messages

    def _convert_tools(self, tools: list[ToolDef]) -> list[dict[str, Any]]:
        """Convert Dendrux ToolDefs to OpenAI function tool format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": td.name,
                    "description": td.description,
                    "parameters": td.parameters,
                },
            }
            for td in tools
        ]

    # ------------------------------------------------------------------
    # Inbound conversions: OpenAI → Dendrux
    # ------------------------------------------------------------------

    def _normalize_response(self, response: Any) -> LLMResponse:
        """Convert OpenAI ChatCompletion to Dendrux LLMResponse."""
        if not response.choices:
            raise RuntimeError(
                f"OpenAI returned a ChatCompletion with no choices. Model: {self._model}"
            )
        choice = response.choices[0]
        message = choice.message

        # Extract text
        text = message.content

        # Extract tool calls
        tool_calls: list[ToolCall] | None = None
        if message.tool_calls:
            tool_calls = []
            for tc in message.tool_calls:
                params = parse_tool_json_strict(
                    tc.function.arguments or "",
                    tool_name=tc.function.name,
                    call_id=tc.id,
                )
                tool_calls.append(
                    ToolCall(
                        name=tc.function.name,
                        params=params,
                        provider_tool_call_id=tc.id,
                    )
                )

        # Extract usage
        usage = UsageStats()
        if response.usage:
            usage = _build_usage_with_cache(response.usage)

        return LLMResponse(
            text=text,
            tool_calls=tool_calls,
            raw=response,
            usage=usage,
        )
