"""Guardrails — content scanning at the LLM boundary.

Guardrails detect PII, secrets, and other sensitive content in text
crossing the LLM boundary. The framework applies actions (redact,
block, warn) based on findings.

Usage:
    from dendrux.guardrails import PII, SecretDetection, Pattern

    agent = Agent(
        provider=...,
        tools=[search],
        prompt="...",
        guardrails=[
            PII(action="redact"),
            SecretDetection(action="block"),
        ],
    )
"""

from dendrux.guardrails._engine import GuardrailEngine
from dendrux.guardrails._pii import PII
from dendrux.guardrails._prompt_injection import PromptInjection
from dendrux.guardrails._protocol import Finding, Guardrail, Pattern
from dendrux.guardrails._secrets import SecretDetection

__all__ = [
    "Finding",
    "Guardrail",
    "GuardrailEngine",
    "PII",
    "Pattern",
    "PromptInjection",
    "SecretDetection",
]
