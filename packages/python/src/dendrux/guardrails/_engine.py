"""GuardrailEngine — stateful per-run guardrail orchestrator.

Owns pii_mapping, placeholder generation, and action application.
The loop calls the engine at integration points; the engine is
stateful (pii_mapping grows), serializable (for pause/resume),
and testable in isolation.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

from dendrux.guardrails._protocol import Finding

if TYPE_CHECKING:
    from dendrux.guardrails._protocol import Guardrail

logger = logging.getLogger(__name__)

_MAX_RECURSION_DEPTH = 50
_MAX_BLOCK_SNIPPET = 80
_PLACEHOLDER_RE = re.compile(r"<<[A-Z][A-Z_]*_\d+>>")


def _block_message(guardrail: object, finding: Finding, *, where: str = "") -> str:
    """Format a block error string with the matched span included.

    The matched text is truncated and ``repr()``-ed so attacker-controlled
    payloads don't smuggle control characters into logs / dashboards.
    """
    name = type(guardrail).__name__
    snippet = finding.text
    if len(snippet) > _MAX_BLOCK_SNIPPET:
        snippet = snippet[:_MAX_BLOCK_SNIPPET] + "..."
    suffix = f" in {where}" if where else ""
    return f"Guardrail {name!r} blocked: {finding.entity_type} matched {snippet!r}{suffix}"


class GuardrailEngine:
    """Per-run guardrail orchestrator.

    Manages the list of guardrails, the run-scoped pii_mapping
    (forward and reverse), and placeholder counters per entity type.

    The loop calls:
      - scan_incoming(text) before provider.complete()
      - scan_outgoing(text) after LLM response
      - deanonymize(params) before tool execution

    Args:
        guardrails: List of Guardrail instances to apply.
        pii_mapping: Existing mapping to restore from (pause/resume).
    """

    def __init__(
        self,
        guardrails: list[Guardrail],
        pii_mapping: dict[str, str] | None = None,
    ) -> None:
        self._guardrails = list(guardrails)
        # Forward: <<EMAIL_ADDRESS_1>> → real value
        self._pii_mapping: dict[str, str] = dict(pii_mapping) if pii_mapping else {}
        # Reverse: real value → <<EMAIL_ADDRESS_1>>
        self._reverse: dict[str, str] = {v: k for k, v in self._pii_mapping.items()}
        # Counters: EMAIL_ADDRESS → next N
        self._counters: dict[str, int] = {}
        # Rebuild counters from existing mapping
        for placeholder in self._pii_mapping:
            # Parse <<TYPE_N>> to get TYPE and N
            inner = placeholder[2:-2]  # strip << and >>
            parts = inner.rsplit("_", 1)
            if len(parts) == 2 and parts[1].isdigit():
                entity_type = parts[0]
                n = int(parts[1])
                self._counters[entity_type] = max(self._counters.get(entity_type, 0), n + 1)

    def _next_placeholder(self, entity_type: str) -> str:
        """Generate the next placeholder for an entity type."""
        n = self._counters.get(entity_type, 1)
        self._counters[entity_type] = n + 1
        return f"<<{entity_type}_{n}>>"

    def _get_or_create_placeholder(self, entity_type: str, real_value: str) -> str:
        """Get existing placeholder for a value, or create a new one."""
        if real_value in self._reverse:
            return self._reverse[real_value]
        placeholder = self._next_placeholder(entity_type)
        self._pii_mapping[placeholder] = real_value
        self._reverse[real_value] = placeholder
        return placeholder

    async def scan_incoming(self, text: str) -> tuple[str, list[Finding], str | None]:
        """Scan text going to the LLM. Apply actions.

        Returns:
            (possibly_modified_text, all_findings, block_error_or_none)

        If a guardrail with action="block" finds something, returns
        the block error message. The caller should terminate the run.
        """
        all_findings: list[Finding] = []

        for guardrail in self._guardrails:
            findings = _deoverlap(await guardrail.scan(text))
            if not findings:
                continue
            all_findings.extend(findings)

            if guardrail.action == "block":
                return (
                    text,
                    all_findings,
                    _block_message(guardrail, findings[0]),
                )

            if guardrail.action == "redact":
                text = self._apply_redaction(text, findings)
            # warn: findings collected but text unchanged

        return text, all_findings, None

    async def scan_outgoing(
        self,
        text: str,
        tool_call_params: list[dict[str, Any]] | None = None,
    ) -> tuple[list[Finding], str | None]:
        """Detection-only scan of LLM output.

        DB is ground truth: the runner persists raw. Placeholders are
        applied at the *next* scan_incoming, not here. Outgoing's job
        is to enforce block policies and emit detection events.

        Returns:
            (findings, block_error_or_none)
        """
        all_findings: list[Finding] = []

        for guardrail in self._guardrails:
            findings = _deoverlap(await guardrail.scan(text))
            if findings:
                all_findings.extend(findings)
                if guardrail.action == "block":
                    return (
                        all_findings,
                        _block_message(guardrail, findings[0], where="LLM output"),
                    )

        if tool_call_params:
            for params in tool_call_params:
                p_findings, p_block = await _scan_params_for_findings(params, self._guardrails)
                all_findings.extend(p_findings)
                if p_block is not None:
                    return all_findings, p_block

        return all_findings, None

    def _apply_redaction(self, text: str, findings: list[Finding]) -> str:
        """Replace findings in text with placeholders. Right-to-left to preserve offsets.

        Findings must already be de-overlapped (done at scan level).
        """
        sorted_findings = sorted(findings, key=lambda f: f.start, reverse=True)
        for finding in sorted_findings:
            placeholder = self._get_or_create_placeholder(finding.entity_type, finding.text)
            text = text[: finding.start] + placeholder + text[finding.end :]
        return text

    def deanonymize(self, params: dict[str, Any]) -> dict[str, Any]:
        """Replace <<PLACEHOLDER>> values with real values in tool call params.

        Walks the dict recursively. Returns a new dict (does not mutate input).
        If a placeholder is not in the mapping (corrupted by LLM), it passes
        through unchanged — the caller emits a warning event.
        """
        result = _deanonymize_value(params, self._pii_mapping)
        return result  # type: ignore[no-any-return]

    def get_pii_mapping(self) -> dict[str, str]:
        """Return the current pii_mapping for persistence."""
        return dict(self._pii_mapping)

    @property
    def has_guardrails(self) -> bool:
        return len(self._guardrails) > 0


def _deoverlap(findings: list[Finding]) -> list[Finding]:
    """Remove overlapping findings, keeping the longer (more specific) match.

    When a credit card number like 4111111111111111 matches both PHONE
    (first 10 digits) and CREDIT_CARD (all 16), the CREDIT_CARD match
    wins because it's longer. Works for any scanner, not just regex.
    """
    if len(findings) <= 1:
        return findings
    # Sort by start position, then by length descending (longer wins)
    sorted_f = sorted(findings, key=lambda f: (f.start, -(f.end - f.start)))
    result: list[Finding] = [sorted_f[0]]
    for f in sorted_f[1:]:
        prev = result[-1]
        if f.start < prev.end:
            # Overlapping — keep the one that covers more
            if (f.end - f.start) > (prev.end - prev.start):
                result[-1] = f
            # else: prev is longer or equal, skip f
        else:
            result.append(f)
    return result


def deanonymize_text(text: str | None, mapping: dict[str, str]) -> tuple[str | None, list[str]]:
    """Reverse pii_mapping for user-facing text (e.g., RunResult.answer).

    Returns (deanonymized_text, unmapped_placeholders). Placeholders shaped
    like ``<<ENTITY_TYPE_N>>`` that survive substitution (LLM hallucinated
    a placeholder we never registered) are listed in the second element so
    the caller can emit a governance event. They pass through unchanged.

    The DB-stores-raw invariant is unaffected: this is a return-path
    transform on the value handed to the dev, not a mutation of anything
    persisted.
    """
    if text is None or not mapping:
        return text, []
    result = _deanonymize_value(text, mapping)
    if not isinstance(result, str):
        return result, []
    unmapped = sorted({p for p in _PLACEHOLDER_RE.findall(result) if p not in mapping})
    return result, unmapped


def _deanonymize_value(value: Any, mapping: dict[str, str], _depth: int = 0) -> Any:
    """Recursively replace placeholder strings with real values."""
    if _depth > _MAX_RECURSION_DEPTH:
        return value
    if isinstance(value, str):
        if value in mapping:
            return mapping[value]
        result = value
        for placeholder, real_value in mapping.items():
            if placeholder in result:
                result = result.replace(placeholder, real_value)
        return result
    if isinstance(value, dict):
        return {k: _deanonymize_value(v, mapping, _depth + 1) for k, v in value.items()}
    if isinstance(value, list):
        return [_deanonymize_value(item, mapping, _depth + 1) for item in value]
    return value


async def _scan_params_for_findings(
    params: dict[str, Any],
    guardrails: list[Guardrail],
    _depth: int = 0,
) -> tuple[list[Finding], str | None]:
    """Recursively collect findings from params without mutating them.

    The runner persists raw params; redaction for the LLM happens at the
    next scan_incoming. This walk is detection-only — needed here so
    ``action="block"`` can abort the run before side-effecting tools fire.

    Returns (findings, block_error_or_none).
    """
    if _depth > _MAX_RECURSION_DEPTH:
        return [], None

    all_findings: list[Finding] = []
    for key, value in params.items():
        leaf_findings, block = await _scan_value_for_findings(value, key, guardrails, _depth + 1)
        all_findings.extend(leaf_findings)
        if block is not None:
            return all_findings, block

    return all_findings, None


async def _scan_value_for_findings(
    value: Any,
    key: str,
    guardrails: list[Guardrail],
    _depth: int = 0,
) -> tuple[list[Finding], str | None]:
    """Walk a single value, collecting findings from every string leaf."""
    if _depth > _MAX_RECURSION_DEPTH:
        return [], None

    if isinstance(value, str):
        return await _scan_leaf_for_findings(value, key, guardrails)

    if isinstance(value, dict):
        all_findings: list[Finding] = []
        for k, v in value.items():
            findings, block = await _scan_value_for_findings(v, k, guardrails, _depth + 1)
            all_findings.extend(findings)
            if block is not None:
                return all_findings, block
        return all_findings, None

    if isinstance(value, list):
        all_findings = []
        for item in value:
            findings, block = await _scan_value_for_findings(item, key, guardrails, _depth + 1)
            all_findings.extend(findings)
            if block is not None:
                return all_findings, block
        return all_findings, None

    return [], None


async def _scan_leaf_for_findings(
    value: str,
    key: str,
    guardrails: list[Guardrail],
) -> tuple[list[Finding], str | None]:
    """Detection-only scan of a single string leaf.

    Uses both value-only and ``key=value`` context scans so patterns like
    GENERIC_API_KEY (which need the key name for detection) still fire.
    """
    all_findings: list[Finding] = []
    key_prefix_len = len(key) + 1

    for guardrail in guardrails:
        value_findings = await guardrail.scan(value)
        context_findings = await guardrail.scan(f"{key}={value}")

        projected: list[Finding] = []
        for cf in context_findings:
            val_start = max(0, cf.start - key_prefix_len)
            val_end = cf.end - key_prefix_len
            if val_end <= 0:
                continue
            val_text = value[val_start:val_end]
            if not val_text:
                continue
            projected.append(
                Finding(
                    entity_type=cf.entity_type,
                    start=val_start,
                    end=min(val_end, len(value)),
                    score=cf.score,
                    text=val_text,
                )
            )

        merged = _deoverlap(value_findings + projected)
        if not merged:
            continue
        all_findings.extend(merged)

        if guardrail.action == "block":
            return (
                all_findings,
                _block_message(guardrail, merged[0], where="tool call params"),
            )

    return all_findings, None
