"""Shared regex detector plumbing for the built-in guardrails.

A :class:`Detector` is a compiled regex plus an optional validator that sees
the match and the surrounding text (Luhn for cards, context keywords for
phones and AWS secrets). Every built-in expression is written so that one
scan costs O(len(text)): repeats are bounded, and a prefilter skips
expressions whose anchor character is absent.

Scans of large text run in a worker thread (:func:`run_scan`) so a long scan
interleaves with other tasks instead of stalling the event loop. This does
not parallelise: ``re`` holds the GIL, which is why linear time matters.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from dendrux.guardrails._protocol import Finding

if TYPE_CHECKING:
    from collections.abc import Callable

# Scans over this many characters run in a worker thread.
THREAD_SCAN_THRESHOLD = 32 * 1024


@dataclass(frozen=True)
class Detector:
    """A named regex with an optional match validator.

    ``prefilter`` is a literal that must occur in the text for the regex to
    be worth running (``"@"`` for e-mail); it is a cheap ``in`` check.
    """

    name: str
    regex: re.Pattern[str]
    validate: Callable[[re.Match[str], str], bool] | None = None
    prefilter: str | None = None


def scan_detectors(text: str, detectors: list[Detector]) -> list[Finding]:
    """Run every detector over ``text``; findings sorted by start offset."""
    findings: list[Finding] = []
    for det in detectors:
        if det.prefilter is not None and det.prefilter not in text:
            continue
        for match in det.regex.finditer(text):
            if det.validate is not None and not det.validate(match, text):
                continue
            findings.append(
                Finding(
                    entity_type=det.name,
                    start=match.start(),
                    end=match.end(),
                    score=1.0,
                    text=match.group(),
                )
            )
    findings.sort(key=lambda f: f.start)
    return findings


async def run_scan(
    fn: Callable[[str], list[Finding]], text: str, *, force_thread: bool = False
) -> list[Finding]:
    """Call the synchronous scan ``fn``; in a worker thread when ``text`` is large."""
    if force_thread or len(text) > THREAD_SCAN_THRESHOLD:
        return await asyncio.to_thread(fn, text)
    return fn(text)


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------


def luhn_valid(match: re.Match[str], text: str) -> bool:
    """True when the digits of the match pass the Luhn checksum."""
    digits = [int(c) for c in match.group() if c.isdigit()]
    if len(digits) < 12:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


_PHONE_CONTEXT = re.compile(
    r"(?i)\b(?:phone|tel|telephone|mobile|cell|fax|call|whatsapp|sms|contact)\b[^\d]{0,12}$"
)
_PHONE_CONTEXT_WINDOW = 40


def phone_context(match: re.Match[str], text: str) -> bool:
    """A bare digit run counts as a phone number only right after a phone keyword."""
    start = match.start()
    window = text[max(0, start - _PHONE_CONTEXT_WINDOW) : start]
    return _PHONE_CONTEXT.search(window) is not None


def e164_digit_count(match: re.Match[str], text: str) -> bool:
    """E.164 allows 8 to 15 digits in total (country code included)."""
    n = sum(1 for c in match.group() if c.isdigit())
    return 8 <= n <= 15


_AWS_SECRET_CONTEXT = re.compile(r"(?i)(?:aws|secret|access[_\-\s]?key)")
_AWS_ACCESS_KEY_ID = re.compile(r"AKIA[0-9A-Z]{16}")
_AWS_CONTEXT_BEFORE = 48
_AWS_PAIR_WINDOW = 200


def aws_secret_context(match: re.Match[str], text: str) -> bool:
    """A 40-character base64-like run is an AWS secret key only with evidence.

    Evidence is an ``AKIA...`` access key id within a short distance (the
    usual id/secret pair) or a context keyword just before it. Without this,
    git SHAs and other 40-character tokens read as secrets.
    """
    start, end = match.span()
    if _AWS_SECRET_CONTEXT.search(text[max(0, start - _AWS_CONTEXT_BEFORE) : start]):
        return True
    window = text[max(0, start - _AWS_PAIR_WINDOW) : end + _AWS_PAIR_WINDOW]
    return _AWS_ACCESS_KEY_ID.search(window) is not None
