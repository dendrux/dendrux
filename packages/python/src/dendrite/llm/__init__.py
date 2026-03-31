"""LLM provider abstraction layer."""

from dendrite.llm.base import LLMProvider
from dendrite.llm.mock import MockLLM

__all__ = ["LLMProvider", "MockLLM"]

# Optional provider imports — only available when extras are installed.
# Usage: from dendrite.llm import AnthropicProvider  (requires pip install dendrite[anthropic])


def __getattr__(name: str) -> type:  # noqa: N807
    if name == "AnthropicProvider":
        from dendrite.llm.anthropic import AnthropicProvider

        return AnthropicProvider
    if name == "OpenAIProvider":
        from dendrite.llm.openai import OpenAIProvider

        return OpenAIProvider
    if name == "OpenAIResponsesProvider":
        from dendrite.llm.openai_responses import OpenAIResponsesProvider

        return OpenAIResponsesProvider
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
