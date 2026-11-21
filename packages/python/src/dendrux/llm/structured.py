"""Structured LLM call helper.

Reusable helper for making a single LLM call with a Pydantic output
schema, validating the response, and returning both the raw JSON and
typed model instance.

Used by SingleCall when output_type is set. Decoupled from loops and
agents — only knows about providers and Pydantic models.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar, get_args

from pydantic import BaseModel, ValidationError
from pydantic_core.core_schema import ErrorType

from dendrux.llm._calls import complete_call
from dendrux.types import StructuredOutputValidationError

if TYPE_CHECKING:
    from dendrux.llm.base import LLMProvider
    from dendrux.types import LLMResponse, Message

T = TypeVar("T", bound=BaseModel)

_MAX_SUMMARISED_KINDS = 10
_KNOWN_ERROR_TYPES = frozenset(get_args(ErrorType))


def _safe_error_type(error_type: str) -> str:
    """Keep Pydantic's built-in error kinds; a custom kind may be built from model text."""
    return error_type if error_type in _KNOWN_ERROR_TYPES else "custom_error"


def _value_free_summary(exc: ValidationError) -> str:
    """Describe a validation failure by error count and kinds only.

    Pydantic's own text carries the rejected input and custom validator
    messages, and error locations can hold model-chosen keys, so neither is
    copied. Kinds are allowlisted, deduplicated in first-seen order and capped.
    """
    errors = exc.errors(include_input=False, include_url=False, include_context=False)
    kinds = list(dict.fromkeys(_safe_error_type(item["type"]) for item in errors))
    shown = ", ".join(kinds[:_MAX_SUMMARISED_KINDS])
    more = len(kinds) - _MAX_SUMMARISED_KINDS
    suffix = f", and {more} more" if more > 0 else ""
    count = len(errors)
    return f"{count} validation error{'s' if count != 1 else ''} ({shown}{suffix})"


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
            validation (malformed JSON, wrong types, missing fields). It
            carries the call's ``usage``, ``model`` and ``stop_reason``
            (usage unpriced here; the
            SingleCall loop prices and records the call before re-raising).

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

    response = await complete_call(
        provider,
        messages,
        tools=None,
        output_schema=schema,
        **provider_kwargs,
    )

    raw_text = response.text or ""

    try:
        output = output_type.model_validate_json(raw_text)
    except ValidationError as exc:
        error = StructuredOutputValidationError(
            raw_response=raw_text,
            output_type_name=output_type.__name__,
            validation_error=_value_free_summary(exc),
            usage=response.usage,
            model=response.model,
            stop_reason=response.stop_reason,
        )
        error._response = response
        # ``from None``: the Pydantic cause carries the rejected values and would
        # be formatted by any diagnostic that logs this error's chain.
        raise error from None

    return response, output
