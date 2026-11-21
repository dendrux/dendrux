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

_MAX_SUMMARISED_ERRORS = 10
_KNOWN_ERROR_TYPES = frozenset(get_args(ErrorType))


def _safe_error_type(error_type: str) -> str:
    """Keep Pydantic's built-in error kinds; a custom kind may be built from model text."""
    return error_type if error_type in _KNOWN_ERROR_TYPES else "custom_error"


def _declared_names(node: object, names: set[str]) -> set[str]:
    """Collect every property name a JSON Schema declares, at any depth."""
    if isinstance(node, dict):
        props = node.get("properties")
        if isinstance(props, dict):
            names.update(key for key in props if isinstance(key, str))
        for value in node.values():
            _declared_names(value, names)
    elif isinstance(node, list):
        for item in node:
            _declared_names(item, names)
    return names


def _value_free_summary(exc: ValidationError, schema: dict[str, Any]) -> str:
    """Describe a validation failure by error kind and location only.

    Pydantic's own text carries the rejected input and custom validator
    messages, both model output. Locations keep only names the schema
    declares (a model-chosen key shows as ``*``) and list indexes.
    """
    declared = _declared_names(schema, set())
    errors = exc.errors(include_input=False, include_url=False, include_context=False)
    parts = []
    for item in errors[:_MAX_SUMMARISED_ERRORS]:
        loc = ".".join(str(p) if isinstance(p, int) or p in declared else "*" for p in item["loc"])
        parts.append(f"{loc or '<root>'}: {_safe_error_type(item['type'])}")
    more = len(errors) - len(parts)
    suffix = f"; and {more} more" if more > 0 else ""
    count = len(errors)
    return f"{count} validation error{'s' if count != 1 else ''} ({'; '.join(parts)}{suffix})"


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
            validation_error=_value_free_summary(exc, schema),
            usage=response.usage,
            model=response.model,
            stop_reason=response.stop_reason,
        )
        error._response = response
        raise error from exc

    return response, output
