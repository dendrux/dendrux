"""Structured LLM call helper.

Reusable helper for making a single LLM call with a Pydantic output
schema, validating the response, and returning both the raw JSON and
typed model instance.

Used by SingleCall when output_type is set. Decoupled from loops and
agents — only knows about providers and Pydantic models.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import BaseModel, ValidationError

from dendrux.types import StructuredOutputValidationError

if TYPE_CHECKING:
    from dendrux.llm.base import LLMProvider
    from dendrux.types import LLMResponse, Message

T = TypeVar("T", bound=BaseModel)


async def structured_complete(
    provider: LLMProvider,
    messages: list[Message],
    output_type: type[T],
    **provider_kwargs: Any,
) -> tuple[LLMResponse, T]:
    """Make a structured LLM call and validate the response.

    1. Derives JSON Schema from the Pydantic model
    2. Calls provider.complete() with output_schema
    3. Validates the JSON response against the model
    4. Returns (LLMResponse, validated_instance)

    Raises:
        ValueError: If the provider does not support structured output.
        StructuredOutputValidationError: If the response fails Pydantic
            validation (malformed JSON, wrong types, missing fields).

    Args:
        provider: The LLM provider to call.
        messages: Messages to send (system + user).
        output_type: Pydantic BaseModel subclass to validate against.
        **provider_kwargs: Forwarded to provider.complete() (temperature, etc.).

    Returns:
        (response, output): The raw LLMResponse and the validated model instance.
    """
    if not provider.capabilities.supports_structured_output:
        raise ValueError(
            f"{type(provider).__name__} does not support structured output. "
            f"Use AnthropicProvider, OpenAIProvider, or OpenAIResponsesProvider."
        )

    schema = output_type.model_json_schema()

    response = await provider.complete(
        messages,
        tools=None,
        output_schema=schema,
        **provider_kwargs,
    )

    raw_text = response.text or ""

    try:
        output = output_type.model_validate_json(raw_text)
    except ValidationError as exc:
        raise StructuredOutputValidationError(
            raw_response=raw_text,
            output_type_name=output_type.__name__,
            validation_error=str(exc),
        ) from exc

    return response, output
