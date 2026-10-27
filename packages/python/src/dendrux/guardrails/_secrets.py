"""Secret detection guardrail — regex-based secret/credential detection.

Detects API keys, AWS credentials, and private key headers.
Default action is "block" — secrets should not reach the LLM.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Literal

from dendrux.guardrails._detectors import (
    Detector,
    aws_secret_context,
    run_scan,
    scan_detectors,
)

if TYPE_CHECKING:
    from dendrux.guardrails._protocol import Finding, Pattern

_SECRET_DETECTORS: list[Detector] = [
    Detector("AWS_ACCESS_KEY", re.compile(r"AKIA[0-9A-Z]{16}"), prefilter="AKIA"),
    # A 40-character run also matches git SHAs and other tokens: it counts as an
    # AWS secret key only next to an AKIA id or a context keyword.
    Detector(
        "AWS_SECRET_KEY",
        re.compile(r"(?<![A-Za-z0-9/+])(?=[A-Za-z0-9/+])[A-Za-z0-9/+=]{40}(?![A-Za-z0-9/+=])"),
        validate=aws_secret_context,
    ),
    Detector(
        "GENERIC_API_KEY",
        re.compile(
            r"(?i)(?:api[_\-]?key|apikey|token|secret[_\-]?key)[=:\s]+[\"']?[A-Za-z0-9_\-]{20,}[\"']?"
        ),
    ),
    Detector("PRIVATE_KEY", re.compile(r"-----BEGIN\s+(?:RSA\s+)?PRIVATE KEY-----")),
]


class SecretDetection:
    """Regex-based secret/credential detection guardrail.

    Detects AWS keys, API keys, and private key headers.
    Default action is "block" — secrets should not reach the LLM.

    A 40-character token counts as an AWS secret key only next to an
    ``AKIA...`` access key id or after a context word (``aws``, ``secret``,
    ``access key``), so git SHAs and hashes do not read as secrets.

    Args:
        action: What the framework does with findings.
            "block" (default) terminates the run.
            "redact" replaces secrets with placeholders.
            "warn" logs but doesn't modify.
        extra_patterns: Additional Pattern objects for custom secret types.
        include_defaults: Include built-in patterns. Set False to use
            only extra_patterns.
    """

    def __init__(
        self,
        *,
        action: Literal["redact", "block", "warn"] = "block",
        extra_patterns: list[Pattern] | None = None,
        include_defaults: bool = True,
    ) -> None:
        if action not in ("redact", "block", "warn"):
            raise ValueError(f"Invalid action: {action!r}. Must be 'redact', 'block', or 'warn'.")
        self.action: Literal["redact", "block", "warn"] = action
        self._detectors: list[Detector] = list(_SECRET_DETECTORS) if include_defaults else []
        if extra_patterns:
            self._detectors.extend(Detector(p.name, re.compile(p.regex)) for p in extra_patterns)

    def _scan_sync(self, text: str) -> list[Finding]:
        return scan_detectors(text, self._detectors)

    async def scan(self, text: str) -> list[Finding]:
        """Scan text for secret patterns."""
        return await run_scan(self._scan_sync, text)
