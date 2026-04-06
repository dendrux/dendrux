"""Tests for structured output (output_type on SingleCall)."""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import BaseModel

from dendrux.agent import Agent
from dendrux.llm._schema import normalize_for_openai_strict
from dendrux.llm.mock import MockLLM
from dendrux.llm.structured import structured_complete
from dendrux.loops.single import SingleCall
from dendrux.types import (
    LLMResponse,
    Message,
    ProviderCapabilities,
    Role,
    RunStatus,
    StructuredOutputValidationError,
    UsageStats,
)

# ------------------------------------------------------------------
# Test models
# ------------------------------------------------------------------


class Sentiment(BaseModel):
    label: str
    confidence: float
    reasoning: str


class NestedModel(BaseModel):
    name: str
    tags: list[str]
    metadata: dict[str, Any]


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _structured_mock(output: BaseModel) -> MockLLM:
    """Create a MockLLM that returns the model as JSON text."""
    usage = UsageStats(input_tokens=10, output_tokens=20, total_tokens=30)
    mock = MockLLM(
        [LLMResponse(text=output.model_dump_json(), usage=usage)],
    )
    mock.capabilities = ProviderCapabilities(supports_structured_output=True)
    return mock


def _make_agent(provider: MockLLM, **overrides) -> Agent:
    defaults: dict[str, Any] = {
        "provider": provider,
        "prompt": "Classify sentiment.",
        "tools": [],
        "loop": SingleCall(),
    }
    defaults.update(overrides)
    return Agent(**defaults)


# ------------------------------------------------------------------
# Agent creation validation
# ------------------------------------------------------------------


class TestAgentValidation:
    """output_type validation at Agent creation time."""

    def test_output_type_requires_single_call(self):
        """output_type without SingleCall raises ValueError."""
        with pytest.raises(ValueError, match="does not use SingleCall"):
            Agent(
                provider=MockLLM([]),
                prompt="test",
                tools=[],
                output_type=Sentiment,
            )

    def test_output_type_with_react_loop_rejected(self):
        """output_type with explicit ReActLoop raises ValueError."""
        from dendrux.loops.react import ReActLoop

        with pytest.raises(ValueError, match="does not use SingleCall"):
            Agent(
                provider=MockLLM([]),
                prompt="test",
                tools=[],
                loop=ReActLoop(),
                output_type=Sentiment,
            )

    def test_output_type_with_single_call_accepted(self):
        """output_type with SingleCall creates successfully."""
        agent = Agent(
            provider=MockLLM([]),
            prompt="test",
            tools=[],
            loop=SingleCall(),
            output_type=Sentiment,
        )
        assert agent.output_type is Sentiment

    def test_output_type_none_is_default(self):
        """No output_type means None."""
        agent = Agent(
            provider=MockLLM([]),
            prompt="test",
            tools=[],
            loop=SingleCall(),
        )
        assert agent.output_type is None


# ------------------------------------------------------------------
# Structured output via agent.run()
# ------------------------------------------------------------------


class TestStructuredRun:
    """End-to-end structured output through agent.run()."""

    @pytest.mark.asyncio
    async def test_basic_structured_output(self):
        """Agent with output_type returns validated model in result.output."""
        expected = Sentiment(label="negative", confidence=0.78, reasoning="Shipping bad")
        provider = _structured_mock(expected)

        agent = _make_agent(provider, output_type=Sentiment)
        result = await agent.run("The product is okay but shipping was terrible")

        assert result.status == RunStatus.SUCCESS
        assert result.output is not None
        assert isinstance(result.output, Sentiment)
        assert result.output.label == "negative"
        assert result.output.confidence == 0.78
        assert result.output.reasoning == "Shipping bad"
        # answer is raw JSON string
        assert result.answer is not None
        parsed = json.loads(result.answer)
        assert parsed["label"] == "negative"

    @pytest.mark.asyncio
    async def test_structured_output_per_call_override(self):
        """output_type on run() overrides agent default."""
        expected = Sentiment(label="positive", confidence=0.9, reasoning="Great")
        provider = _structured_mock(expected)

        # Agent has no output_type, but run() passes one
        agent = _make_agent(provider)
        result = await agent.run(
            "This is great!",
            output_type=Sentiment,
        )

        assert result.output is not None
        assert isinstance(result.output, Sentiment)
        assert result.output.label == "positive"

    @pytest.mark.asyncio
    async def test_structured_output_none_override(self):
        """output_type=None on run() disables agent-level output_type."""
        provider = MockLLM([LLMResponse(text="Just plain text")])
        provider.capabilities = ProviderCapabilities(supports_structured_output=True)

        agent = _make_agent(provider, output_type=Sentiment)
        result = await agent.run("Hello", output_type=None)

        assert result.output is None
        assert result.answer == "Just plain text"

    @pytest.mark.asyncio
    async def test_no_output_type_returns_none(self):
        """Without output_type, result.output is None."""
        provider = MockLLM([LLMResponse(text="Just text")])

        agent = _make_agent(provider)
        result = await agent.run("Hello")

        assert result.output is None
        assert result.answer == "Just text"


# ------------------------------------------------------------------
# Validation failure
# ------------------------------------------------------------------


class TestValidationFailure:
    """StructuredOutputValidationError on bad LLM responses."""

    @pytest.mark.asyncio
    async def test_invalid_json_raises(self):
        """Non-JSON response raises StructuredOutputValidationError."""
        provider = MockLLM([LLMResponse(text="not json at all")])
        provider.capabilities = ProviderCapabilities(supports_structured_output=True)

        agent = _make_agent(provider, output_type=Sentiment)

        with pytest.raises(StructuredOutputValidationError, match="Sentiment"):
            await agent.run("test")

    @pytest.mark.asyncio
    async def test_wrong_types_raises(self):
        """JSON with wrong types raises StructuredOutputValidationError."""
        bad_json = json.dumps({"label": "positive", "confidence": "not a float", "reasoning": 123})
        provider = MockLLM([LLMResponse(text=bad_json)])
        provider.capabilities = ProviderCapabilities(supports_structured_output=True)

        agent = _make_agent(provider, output_type=Sentiment)

        with pytest.raises(StructuredOutputValidationError):
            await agent.run("test")

    @pytest.mark.asyncio
    async def test_missing_fields_raises(self):
        """JSON missing required fields raises StructuredOutputValidationError."""
        bad_json = json.dumps({"label": "positive"})
        provider = MockLLM([LLMResponse(text=bad_json)])
        provider.capabilities = ProviderCapabilities(supports_structured_output=True)

        agent = _make_agent(provider, output_type=Sentiment)

        with pytest.raises(StructuredOutputValidationError, match="Sentiment"):
            await agent.run("test")


# ------------------------------------------------------------------
# Provider capability check
# ------------------------------------------------------------------


class TestProviderCapability:
    """structured_complete rejects unsupported providers."""

    @pytest.mark.asyncio
    async def test_unsupported_provider_raises(self):
        """Provider without supports_structured_output raises ValueError."""
        provider = MockLLM([LLMResponse(text="{}")])
        # Default capabilities: supports_structured_output=False
        messages = [Message(role=Role.USER, content="test")]

        with pytest.raises(ValueError, match="does not support structured output"):
            await structured_complete(provider, messages, Sentiment)


# ------------------------------------------------------------------
# structured_complete helper
# ------------------------------------------------------------------


class TestStructuredComplete:
    """Direct tests for the structured_complete helper."""

    @pytest.mark.asyncio
    async def test_returns_response_and_model(self):
        """Returns (LLMResponse, validated_model) tuple."""
        expected = Sentiment(label="neutral", confidence=0.5, reasoning="Mixed")
        provider = _structured_mock(expected)

        messages = [
            Message(role=Role.SYSTEM, content="Classify."),
            Message(role=Role.USER, content="It's okay"),
        ]
        response, output = await structured_complete(provider, messages, Sentiment)

        assert isinstance(output, Sentiment)
        assert output.label == "neutral"
        assert response.text is not None
        assert response.usage.input_tokens == 10

    @pytest.mark.asyncio
    async def test_passes_output_schema_to_provider(self):
        """Verifies the schema is passed to provider.complete()."""
        expected = Sentiment(label="positive", confidence=0.9, reasoning="Great")
        provider = _structured_mock(expected)

        messages = [Message(role=Role.USER, content="test")]
        await structured_complete(provider, messages, Sentiment)

        assert len(provider.call_history) == 1
        call = provider.call_history[0]
        assert call["output_schema"] is not None
        schema = call["output_schema"]
        assert schema["type"] == "object"
        assert "label" in schema["properties"]
        assert "confidence" in schema["properties"]


# ------------------------------------------------------------------
# OpenAI schema normalization
# ------------------------------------------------------------------


class TestOpenAISchemaNoermalization:
    """Tests for _schema.normalize_for_openai_strict."""

    def test_adds_additional_properties_false(self):
        """Ensures additionalProperties: false on object types."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name"],
        }
        result = normalize_for_openai_strict(schema)

        assert result["additionalProperties"] is False
        # All properties should be in required
        assert set(result["required"]) == {"name", "age"}

    def test_preserves_existing_required(self):
        """Doesn't lose existing required fields."""
        schema = {
            "type": "object",
            "properties": {"a": {"type": "string"}, "b": {"type": "string"}},
            "required": ["a"],
        }
        result = normalize_for_openai_strict(schema)
        assert set(result["required"]) == {"a", "b"}

    def test_nested_objects(self):
        """Normalizes nested object schemas."""
        schema = {
            "type": "object",
            "properties": {
                "inner": {
                    "type": "object",
                    "properties": {
                        "value": {"type": "string"},
                    },
                },
            },
        }
        result = normalize_for_openai_strict(schema)

        assert result["additionalProperties"] is False
        inner = result["properties"]["inner"]
        assert inner["additionalProperties"] is False
        assert inner["required"] == ["value"]

    def test_array_items(self):
        """Normalizes object schemas inside array items."""
        schema = {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"id": {"type": "integer"}},
                    },
                },
            },
        }
        result = normalize_for_openai_strict(schema)

        item_schema = result["properties"]["items"]["items"]
        assert item_schema["additionalProperties"] is False

    def test_preserves_non_object_types(self):
        """Doesn't modify string/number/array types."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "minLength": 1},
                "score": {"type": "number", "minimum": 0},
            },
        }
        result = normalize_for_openai_strict(schema)

        assert result["properties"]["name"]["minLength"] == 1
        assert result["properties"]["score"]["minimum"] == 0

    def test_does_not_mutate_original(self):
        """Returns a copy, doesn't modify the input."""
        schema = {
            "type": "object",
            "properties": {"x": {"type": "string"}},
        }
        result = normalize_for_openai_strict(schema)

        assert "additionalProperties" not in schema
        assert result["additionalProperties"] is False

    def test_defs_normalized(self):
        """$defs schemas are also normalized."""
        schema = {
            "type": "object",
            "properties": {"ref": {"$ref": "#/$defs/Inner"}},
            "$defs": {
                "Inner": {
                    "type": "object",
                    "properties": {"val": {"type": "string"}},
                },
            },
        }
        result = normalize_for_openai_strict(schema)

        inner_def = result["$defs"]["Inner"]
        assert inner_def["additionalProperties"] is False

    def test_pydantic_model_schema(self):
        """Works with real Pydantic model_json_schema() output."""
        schema = Sentiment.model_json_schema()
        result = normalize_for_openai_strict(schema)

        assert result["additionalProperties"] is False
        assert set(result["required"]) == {"label", "confidence", "reasoning"}


# ------------------------------------------------------------------
# Streaming guard
# ------------------------------------------------------------------


class TestStreamingGuard:
    """Structured output is rejected in streaming mode for v1."""

    def test_agent_stream_rejects_output_type(self):
        """agent.stream() raises NotImplementedError when output_type is set."""
        provider = MockLLM([])
        provider.capabilities = ProviderCapabilities(supports_structured_output=True)
        agent = _make_agent(provider, output_type=Sentiment)

        with pytest.raises(NotImplementedError, match="streaming"):
            agent.stream("test")

    @pytest.mark.asyncio
    async def test_single_call_run_stream_rejects_output_type(self):
        """SingleCall.run_stream raises NotImplementedError with output_type."""
        from dendrux.strategies.native import NativeToolCalling

        provider = MockLLM([])
        agent = _make_agent(provider)
        loop = SingleCall()

        with pytest.raises(NotImplementedError, match="streaming"):
            gen = loop.run_stream(
                agent=agent,
                provider=provider,
                strategy=NativeToolCalling(),
                user_input="test",
                output_type=Sentiment,
            )
            # Must advance the generator to trigger the error
            await gen.__anext__()


# ------------------------------------------------------------------
# Idempotency fingerprint
# ------------------------------------------------------------------


class TestIdempotencyFingerprint:
    """output_type changes the fingerprint."""

    def test_different_output_types_different_fingerprint(self):
        """Same input but different output_type produces different fingerprint."""
        from dendrux.types import compute_idempotency_fingerprint

        fp1 = compute_idempotency_fingerprint("agent", "input")
        fp2 = compute_idempotency_fingerprint("agent", "input", output_type_name="Sentiment")
        fp3 = compute_idempotency_fingerprint("agent", "input", output_type_name="Analysis")

        assert fp1 != fp2
        assert fp2 != fp3
        assert fp1 != fp3

    def test_same_output_type_same_fingerprint(self):
        """Same input and output_type produces same fingerprint."""
        from dendrux.types import compute_idempotency_fingerprint

        fp1 = compute_idempotency_fingerprint("agent", "input", output_type_name="Sentiment")
        fp2 = compute_idempotency_fingerprint("agent", "input", output_type_name="Sentiment")

        assert fp1 == fp2

    def test_none_output_type_matches_omitted(self):
        """output_type_name=None produces same fingerprint as omitted."""
        from dendrux.types import compute_idempotency_fingerprint

        fp1 = compute_idempotency_fingerprint("agent", "input")
        fp2 = compute_idempotency_fingerprint("agent", "input", output_type_name=None)

        assert fp1 == fp2
