"""PromptInjection guardrail — regex-based detection of injection attempts.

Plugs into the existing ``GuardrailEngine`` and reuses two hooks the
loops already invoke:

* ``scan_incoming`` walks every message handed to ``provider.complete()``
  on each iteration. This covers user input AND tool-result re-entry
  (tool result messages are part of the prompt on the next turn).
* ``scan_outgoing`` runs over the LLM's response text + tool-call params
  before they are persisted, so an LLM that complies with an earlier
  injection still trips the guardrail.

Coverage is non-streaming only. ``Agent.stream()`` and ``Agent.resume_stream``
reject ``guardrails=`` because the outgoing scan needs the full response;
this is a known gap tracked separately.

Unlike ``PII``, no default patterns ship: prompt-injection threats are
domain-specific (the same regex that protects a support agent breaks a
security-research agent that summarizes jailbreak papers), so devs supply
``patterns=[Pattern(name, regex), ...]`` based on their threat model.
The recipes docs ship a copy-paste menu organized by app type.

``redact`` is intentionally not supported — there is no real value to
deanonymize, and replacing the matched span with a placeholder just
leaves a confusing token mid-context. Use ``block`` (default) or ``warn``.

Note: the ``patterns=`` kwarg here intentionally differs from
``PII(extra_patterns=..., include_defaults=...)`` because PII has built-in
defaults to extend, and this guardrail does not.
"""

from __future__ import annotations

import re
from typing import Literal

from dendrux.guardrails._protocol import Finding, Pattern

Engine = Literal["regex"]


class PromptInjection:
    """Detect prompt-injection attempts via regex patterns.

    Args:
        action: ``"block"`` (default) terminates the run when a pattern
            matches. ``"warn"`` logs the finding and lets the run
            continue. ``"redact"`` is not supported.
        engine: ``"regex"`` (only option in v1). A classifier engine is
            planned for v2.
        patterns: Required list of ``Pattern`` objects. No defaults ship
            — see ``docs/recipes/prompt-injection-patterns.mdx`` for
            starter snippets organized by app type.
    """

    def __init__(
        self,
        *,
        action: Literal["block", "warn"] = "block",
        engine: Engine = "regex",
        patterns: list[Pattern] | None = None,
    ) -> None:
        if action not in ("block", "warn"):
            raise ValueError(
                f"Invalid action: {action!r}. Must be 'block' or 'warn'. "
                "PromptInjection does not support 'redact' — there is no value "
                "to deanonymize, and replacing the match leaves confusing "
                "tokens mid-context."
            )
        if engine != "regex":
            raise ValueError(
                f"Invalid engine: {engine!r}. Only 'regex' is supported in v1. "
                "A 'classifier' engine is planned for v2."
            )
        if not patterns:
            raise ValueError(
                "PromptInjection(engine='regex') requires patterns=[Pattern(...), ...]. "
                "No defaults ship — prompt-injection threats are domain-specific. "
                "See docs/recipes/prompt-injection-patterns.mdx for starter snippets."
            )

        self.action: Literal["block", "warn"] = action
        self.engine: Engine = engine
        self._patterns: list[Pattern] = list(patterns)
        self._compiled: list[tuple[str, re.Pattern[str]]] = [
            (p.name, re.compile(p.regex)) for p in self._patterns
        ]

    async def scan(self, text: str) -> list[Finding]:
        """Detect injection patterns in text. Framework applies the action."""
        findings: list[Finding] = []
        for entity_type, pattern in self._compiled:
            for match in pattern.finditer(text):
                findings.append(
                    Finding(
                        entity_type=entity_type,
                        start=match.start(),
                        end=match.end(),
                        score=1.0,
                        text=match.group(),
                    )
                )
        findings.sort(key=lambda f: f.start)
        return findings
