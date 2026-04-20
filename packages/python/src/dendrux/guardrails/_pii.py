"""PII guardrail — pluggable detection backend.

Default ``engine="regex"`` uses compiled patterns for the five canonical
entities (EMAIL_ADDRESS, PHONE_NUMBER, US_SSN, CREDIT_CARD, IP_ADDRESS).

``engine="presidio"`` uses Microsoft Presidio's ``AnalyzerEngine`` for
NLP-backed detection of ~18 entities (adds PERSON, LOCATION, DATE_TIME,
IBAN_CODE, US_PASSPORT, etc.). Requires ``pip install dendrux[presidio]``.

Entity names are canonical to Presidio's vocabulary across both engines
so dashboards/tests see one set of names regardless of backend.
"""

from __future__ import annotations

import re
from typing import Literal, Protocol

from dendrux.guardrails._protocol import Finding, Pattern

Engine = Literal["regex", "presidio"]


_DEFAULT_PATTERNS: list[Pattern] = [
    Pattern("EMAIL_ADDRESS", r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"),
    Pattern("PHONE_NUMBER", r"\+?1?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}"),
    Pattern("US_SSN", r"\b\d{3}-\d{2}-\d{4}\b"),
    Pattern(
        "CREDIT_CARD",
        r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
    ),
    Pattern("IP_ADDRESS", r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"),
]


class _Scanner(Protocol):
    async def scan(self, text: str) -> list[Finding]: ...


class _RegexScanner:
    def __init__(self, patterns: list[Pattern]) -> None:
        self._compiled: list[tuple[str, re.Pattern[str]]] = [
            (p.name, re.compile(p.regex)) for p in patterns
        ]

    async def scan(self, text: str) -> list[Finding]:
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


class _PresidioScanner:
    """Presidio-backed scanner. Uses Presidio's default recognizers for the
    built-in entities (~18 types including PERSON, LOCATION) and wraps any
    extra_patterns as PatternRecognizer instances.
    """

    def __init__(self, extra_patterns: list[Pattern]) -> None:
        try:
            from presidio_analyzer import (
                AnalyzerEngine,
                PatternRecognizer,
            )
            from presidio_analyzer import (
                Pattern as PresidioPattern,
            )
        except ImportError as e:
            raise ImportError(
                "PII(engine='presidio') requires presidio. Install dendrux[presidio]."
            ) from e

        self._analyzer = AnalyzerEngine()
        for p in extra_patterns:
            recognizer = PatternRecognizer(
                supported_entity=p.name,
                patterns=[PresidioPattern(name=p.name, regex=p.regex, score=1.0)],
            )
            self._analyzer.registry.add_recognizer(recognizer)

    async def scan(self, text: str) -> list[Finding]:
        results = self._analyzer.analyze(text=text, language="en")
        findings = [
            Finding(
                entity_type=r.entity_type,
                start=r.start,
                end=r.end,
                score=r.score,
                text=text[r.start : r.end],
            )
            for r in results
        ]
        findings.sort(key=lambda f: f.start)
        return findings


class PII:
    """Detects PII and produces Findings. Framework applies the action.

    Args:
        action: ``"redact"`` (default) replaces PII with ``<<TYPE_N>>``
            placeholders at the LLM boundary. ``"block"`` terminates the
            run. ``"warn"`` logs but does not modify.
        engine: ``"regex"`` (default) uses compiled patterns for the five
            canonical entities. ``"presidio"`` uses Microsoft Presidio's
            ``AnalyzerEngine`` — requires ``pip install dendrux[presidio]``.
        extra_patterns: Additional ``Pattern`` objects for domain-specific
            PII. Applied by both engines (Presidio wraps each as a
            ``PatternRecognizer``).
        include_defaults: Include the five built-in entities. Set ``False``
            to use only ``extra_patterns``.
    """

    def __init__(
        self,
        *,
        action: Literal["redact", "block", "warn"] = "redact",
        engine: Engine = "regex",
        extra_patterns: list[Pattern] | None = None,
        include_defaults: bool = True,
    ) -> None:
        if action not in ("redact", "block", "warn"):
            raise ValueError(f"Invalid action: {action!r}. Must be 'redact', 'block', or 'warn'.")
        if engine not in ("regex", "presidio"):
            raise ValueError(f"Invalid engine: {engine!r}. Must be 'regex' or 'presidio'.")
        self.action: Literal["redact", "block", "warn"] = action
        self.engine: Engine = engine

        if engine == "regex":
            patterns: list[Pattern] = []
            if include_defaults:
                patterns.extend(_DEFAULT_PATTERNS)
            if extra_patterns:
                patterns.extend(extra_patterns)
            self._scanner: _Scanner = _RegexScanner(patterns)
        else:
            # Presidio ships built-in recognizers for the five canonical
            # entities (and 13 more). include_defaults is a regex-only
            # knob here — Presidio's defaults are always active.
            self._scanner = _PresidioScanner(extra_patterns or [])

    async def scan(self, text: str) -> list[Finding]:
        return await self._scanner.scan(text)
