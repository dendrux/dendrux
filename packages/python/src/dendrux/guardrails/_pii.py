"""PII guardrail — pluggable detection backend.

Default ``engine="regex"`` uses compiled patterns for the five canonical
entities (EMAIL_ADDRESS, PHONE_NUMBER, US_SSN, CREDIT_CARD, IP_ADDRESS).
The built-in expressions are linear-time, validated (Luhn for cards, octets
<= 255 for IPs, E.164 or separators for phones; a bare 10-digit number only
after a phone keyword), and scans of large text run off the event loop.

``engine="presidio"`` uses Microsoft Presidio's ``AnalyzerEngine`` for
NLP-backed detection of ~18 entities (adds PERSON, LOCATION, DATE_TIME,
IBAN_CODE, US_PASSPORT, etc.). Requires ``pip install dendrux[presidio]``.

Entity names are canonical to Presidio's vocabulary across both engines
so dashboards/tests see one set of names regardless of backend.
"""

from __future__ import annotations

import asyncio
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Literal, Protocol

from dendrux._sentinel import _UnsetType
from dendrux.guardrails._detectors import (
    Detector,
    e164_digit_count,
    luhn_valid,
    phone_context,
    run_scan,
    scan_detectors,
)
from dendrux.guardrails._protocol import UNSCANNABLE, Finding, Pattern

Engine = Literal["regex", "presidio"]
RestoreMode = Literal["all", "allowlisted"]

_RESTORE_NOT_CHOSEN = _UnsetType()

# Presidio scans in flight (running plus queued) per scanner.
PRESIDIO_MAX_PENDING = 16


# Every expression is bounded so one scan is O(len(text)); see ``_detectors``.
_OCTET = r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)"

_DEFAULT_DETECTORS: list[Detector] = [
    Detector(
        "EMAIL_ADDRESS",
        re.compile(r"[a-zA-Z0-9._%+\-]{1,64}@[a-zA-Z0-9.\-]{1,255}\.[a-zA-Z]{2,63}"),
        prefilter="@",
    ),
    # E.164 (with or without separators), then separator-formatted national
    # numbers, then a bare 10-digit run only right after a phone keyword.
    # Candidates are delimited by what follows (``(?!\d)``) and by a bounded
    # group count, so a number followed by an extension or another field is
    # still found at its own boundary. Five groups cover common EU formats.
    Detector(
        "PHONE_NUMBER",
        re.compile(r"(?<![\w+])\+[1-9]\d{7,14}(?!\d)"),
        prefilter="+",
    ),
    Detector(
        "PHONE_NUMBER",
        re.compile(r"(?<![\w+])\+\d{1,3}(?:[-.\s]?\(\d{1,4}\)|[-.\s]\d{1,4}){2,5}(?!\d)"),
        validate=e164_digit_count,
        prefilter="+",
    ),
    Detector(
        "PHONE_NUMBER",
        re.compile(
            r"(?<![\d\-.+])(?:1[-.\s])?(?:\(\d{3}\)\s?|\d{3}[-.\s])\d{3}[-.\s]\d{4}(?![\d\-])"
        ),
    ),
    Detector(
        "PHONE_NUMBER",
        re.compile(r"(?<![\d.\-+])\d{10}(?![\d\-.]?\d)"),
        validate=phone_context,
    ),
    Detector("US_SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    Detector(
        "CREDIT_CARD",
        re.compile(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"),
        validate=luhn_valid,
    ),
    Detector(
        "IP_ADDRESS",
        re.compile(rf"(?<![\d.])(?:{_OCTET}\.){{3}}{_OCTET}(?!\.?\d)"),
    ),
]


class _Scanner(Protocol):
    async def scan(self, text: str) -> list[Finding]: ...


class _RegexScanner:
    def __init__(self, patterns: list[Pattern], *, include_defaults: bool = True) -> None:
        self._detectors: list[Detector] = list(_DEFAULT_DETECTORS) if include_defaults else []
        self._detectors.extend(Detector(p.name, re.compile(p.regex)) for p in patterns)

    def _scan_sync(self, text: str) -> list[Finding]:
        return scan_detectors(text, self._detectors)

    async def scan(self, text: str) -> list[Finding]:
        return await run_scan(self._scan_sync, text)


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
        # The NLP pipeline is not documented as thread-safe: one dedicated worker
        # runs every scan, so callers wait on the loop, not in shared pool threads.
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="dendrux-presidio")
        self._pending = 0
        self._pending_lock = threading.Lock()
        for p in extra_patterns:
            recognizer = PatternRecognizer(
                supported_entity=p.name,
                patterns=[PresidioPattern(name=p.name, regex=p.regex, score=1.0)],
            )
            self._analyzer.registry.add_recognizer(recognizer)

    def _scan_sync(self, text: str) -> list[Finding]:
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

    def _release(self, _future: object) -> None:
        with self._pending_lock:
            self._pending -= 1

    async def scan(self, text: str) -> list[Finding]:
        """Scan on the dedicated worker, with at most ``PRESIDIO_MAX_PENDING`` in flight.

        NLP analysis stalls the loop even on small inputs, so it is always
        off-loop. When the queue is full the scan is reported ``UNSCANNABLE``
        (the engine then fails closed) instead of queueing without bound.
        """
        with self._pending_lock:
            if self._pending >= PRESIDIO_MAX_PENDING:
                return [Finding(entity_type=UNSCANNABLE, start=0, end=0, score=1.0, text="")]
            self._pending += 1
        try:
            future = self._executor.submit(self._scan_sync, text)
        except BaseException:
            self._release(None)
            raise
        future.add_done_callback(self._release)
        # Cancelling the waiter must not cancel a queued concurrent future:
        # ThreadPoolExecutor retains that work item and its text until dequeue.
        # Keep its reservation until the worker physically drains it.
        return await asyncio.shield(asyncio.wrap_future(future))


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
        restore: Which tools get real values back in place of placeholders
            (``action="redact"``). ``"all"`` (default) restores into every
            tool call (the 0.4.1 behaviour). ``"allowlisted"`` restores only
            into tools marked ``@tool(receives_pii=...)`` (and only the
            entity types they list), so a prompt-injected tool call cannot
            turn a placeholder into a real value for an arbitrary tool.
            ``_restore_chosen`` records whether it was passed explicitly.
    """

    def __init__(
        self,
        *,
        action: Literal["redact", "block", "warn"] = "redact",
        engine: Engine = "regex",
        extra_patterns: list[Pattern] | None = None,
        include_defaults: bool = True,
        restore: RestoreMode | _UnsetType = _RESTORE_NOT_CHOSEN,
    ) -> None:
        if action not in ("redact", "block", "warn"):
            raise ValueError(f"Invalid action: {action!r}. Must be 'redact', 'block', or 'warn'.")
        if engine not in ("regex", "presidio"):
            raise ValueError(f"Invalid engine: {engine!r}. Must be 'regex' or 'presidio'.")
        self._restore_chosen: bool = not isinstance(restore, _UnsetType)
        if isinstance(restore, _UnsetType):
            restore = "all"
        if restore not in ("all", "allowlisted"):
            raise ValueError(f"Invalid restore: {restore!r}. Must be 'all' or 'allowlisted'.")
        self.action: Literal["redact", "block", "warn"] = action
        self.engine: Engine = engine
        self.restore: RestoreMode = restore

        if engine == "regex":
            self._scanner: _Scanner = _RegexScanner(
                extra_patterns or [], include_defaults=include_defaults
            )
        else:
            # Presidio ships built-in recognizers for the five canonical
            # entities (and 13 more). include_defaults is a regex-only
            # knob here — Presidio's defaults are always active.
            self._scanner = _PresidioScanner(extra_patterns or [])

    async def scan(self, text: str) -> list[Finding]:
        return await self._scanner.scan(text)
