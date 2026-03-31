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
from typing import TYPE_CHECKING, Any

import httpx
import openai

from dendrux.llm.base import LLMProvider
from dendrux.types import (
    LLMResponse,
    ProviderCapabilities,
    Role,
    ToolCall,
    UsageStats,
)

if TYPE_CHECKING:
    from dendrux.types import Message, ToolDef

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
        supports_streaming=False,
        supports_streaming_tool_deltas=False,
        supports_thinking=False,
        supports_multimodal=False,
        supports_system_prompt=True,
        supports_parallel_tool_calls=True,
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
        """
        self._client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=httpx.Timeout(timeout, connect=10.0),
            max_retries=max_retries,
        )
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._reasoning_effort = reasoning_effort
        self._timeout = timeout

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

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolDef] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Send messages to the model and return a normalized response.

        kwargs override constructor defaults (e.g. model, max_tokens, temperature).
        Only supported kwargs are forwarded; unknown keys are ignored.
        """
        api_messages = self._convert_messages(messages)
        api_tools = self._convert_tools(tools) if tools else openai.NOT_GIVEN

        # Resolve max_tokens: per-call kwargs override constructor default
        max_tokens = kwargs.pop(
            "max_completion_tokens", kwargs.pop("max_tokens", self._max_tokens),
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
        already_handled = {"model", "max_tokens", "max_completion_tokens", "temperature", "reasoning_effort"}
        for key in _SUPPORTED_KWARGS - already_handled:
            if key in kwargs:
                api_kwargs[key] = kwargs.pop(key)

        # Capture provider request payload
        captured_request = {k: v for k, v in api_kwargs.items() if v is not openai.NOT_GIVEN}

        try:
            response = await self._client.chat.completions.create(**api_kwargs)
        except openai.APITimeoutError:
            raise TimeoutError(
                f"LLM request timed out after {self._timeout}s. "
                f"The model may need more time for large outputs. "
                f"Increase timeout: OpenAIProvider(model=..., timeout=300)"
            ) from None

        llm_response = self._normalize_response(response)

        # Attach adapter-boundary payloads for evidence layer
        llm_response.provider_request = captured_request
        llm_response.provider_response = response.model_dump()

        return llm_response

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

        # Build provider ID index: Dendrux call_id → ToolCall
        call_index: dict[str, ToolCall] = {}
        for msg in messages:
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    if tc.id in call_index:
                        raise ValueError(
                            f"Duplicate Dendrux call_id '{tc.id}' in conversation "
                            f"history. Tool calls must have unique IDs."
                        )
                    call_index[tc.id] = tc

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
        choice = response.choices[0]
        message = choice.message

        # Extract text
        text = message.content

        # Extract tool calls
        tool_calls: list[ToolCall] | None = None
        if message.tool_calls:
            tool_calls = []
            for tc in message.tool_calls:
                if tc.function.arguments:
                    try:
                        params = json.loads(tc.function.arguments)
                    except json.JSONDecodeError as e:
                        raise ValueError(
                            f"Tool call '{tc.function.name}' (id={tc.id}) returned "
                            f"invalid JSON arguments: {tc.function.arguments!r}"
                        ) from e
                else:
                    params = {}
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
            usage = UsageStats(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            )

        return LLMResponse(
            text=text,
            tool_calls=tool_calls,
            raw=response,
            usage=usage,
        )
