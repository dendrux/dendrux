"""Opt-in testing helpers. Nothing here is imported by Dendrux itself."""

from dendrux.llm.mock import MockLLM
from dendrux.testing.fake_model import (
    FakeCall,
    FakeModel,
    FakeModelScriptError,
    FakeProvider,
    call,
    calls,
    say,
    structured,
    when,
)

__all__ = [
    "FakeCall",
    "FakeModel",
    "FakeModelScriptError",
    "FakeProvider",
    "MockLLM",
    "call",
    "calls",
    "say",
    "structured",
    "when",
]
