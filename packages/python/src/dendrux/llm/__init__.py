"""LLM provider abstraction layer."""

from dendrux.llm.base import LLMProvider
from dendrux.llm.mock import MockLLM

__all__ = ["LLMProvider", "MockLLM"]

# Optional provider imports — only available when extras are installed.
# Usage: from dendrux.llm import AnthropicProvider  (requires pip install dendrux[anthropic])


def __getattr__(name: str) -> type:  # noqa: N807
    if name == "AnthropicProvider":
        from dendrux.llm.anthropic import AnthropicProvider

        return AnthropicProvider
    if name == "OpenAIProvider":
        from dendrux.llm.openai import OpenAIProvider

        return OpenAIProvider
    if name == "OpenAIResponsesProvider":
        from dendrux.llm.openai_responses import OpenAIResponsesProvider

        return OpenAIResponsesProvider
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
