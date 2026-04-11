"""GuardrailEngine — stateful per-run guardrail orchestrator.

Owns pii_mapping, placeholder generation, and action application.
The loop calls the engine at integration points; the engine is
stateful (pii_mapping grows), serializable (for pause/resume),
and testable in isolation.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from dendrux.guardrails._protocol import Finding

if TYPE_CHECKING:
    from dendrux.guardrails._protocol import Guardrail

logger = logging.getLogger(__name__)


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
        # Forward: <<EMAIL_1>> → real value
        self._pii_mapping: dict[str, str] = dict(pii_mapping) if pii_mapping else {}
        # Reverse: real value → <<EMAIL_1>>
        self._reverse: dict[str, str] = {v: k for k, v in self._pii_mapping.items()}
        # Counters: EMAIL → next N
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
                entity = findings[0].entity_type
                return (
                    text,
                    all_findings,
                    (f"Guardrail '{type(guardrail).__name__}' blocked: {entity} detected"),
                )

            if guardrail.action == "redact":
                text = self._apply_redaction(text, findings)
            # warn: findings collected but text unchanged

        return text, all_findings, None

    async def scan_outgoing(
        self,
        text: str,
        tool_call_params: list[dict[str, Any]] | None = None,
    ) -> tuple[str, list[dict[str, Any]] | None, list[Finding], str | None, bool]:
        """Scan LLM output (response text + tool call params).

        Returns:
            (possibly_modified_text, possibly_modified_params, findings,
             block_error, params_were_redacted)
        """
        all_findings: list[Finding] = []
        params_redacted = False

        # Scan response text
        for guardrail in self._guardrails:
            findings = _deoverlap(await guardrail.scan(text))
            if not findings:
                continue
            all_findings.extend(findings)

            if guardrail.action == "block":
                entity = findings[0].entity_type
                return (
                    text,
                    tool_call_params,
                    all_findings,
                    (
                        f"Guardrail '{type(guardrail).__name__}' blocked: "
                        f"{entity} detected in LLM output"
                    ),
                    False,
                )

            if guardrail.action == "redact":
                text = self._apply_redaction(text, findings)

        # Scan tool call params — recursive per-leaf scanning
        if tool_call_params:
            for params in tool_call_params:
                p_findings, p_block, p_changed = await _scan_params_recursive(
                    params, self._guardrails, self
                )
                all_findings.extend(p_findings)
                if p_changed:
                    params_redacted = True
                if p_block is not None:
                    return (
                        text,
                        tool_call_params,
                        all_findings,
                        p_block,
                        False,
                    )

        return text, tool_call_params, all_findings, None, params_redacted

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


def _deanonymize_value(value: Any, mapping: dict[str, str]) -> Any:
    """Recursively replace placeholder strings with real values."""
    if isinstance(value, str):
        # Check if the entire string is a placeholder
        if value in mapping:
            return mapping[value]
        # Check for placeholders embedded in text
        result = value
        for placeholder, real_value in mapping.items():
            if placeholder in result:
                result = result.replace(placeholder, real_value)
        return result
    if isinstance(value, dict):
        return {k: _deanonymize_value(v, mapping) for k, v in value.items()}
    if isinstance(value, list):
        return [_deanonymize_value(item, mapping) for item in value]
    return value


async def _scan_params_recursive(
    params: dict[str, Any],
    guardrails: list[Guardrail],
    engine: GuardrailEngine,
) -> tuple[list[Finding], str | None, bool]:
    """Recursively scan and redact tool call params.

    For each string leaf:
      1. Scan the raw value (catches emails, phones, AWS keys, etc.)
      2. Scan key=value context (catches GENERIC_API_KEY, token= patterns)
      3. Project key-context findings back to value span for redaction
      4. De-overlap merged findings
      5. Apply redaction against the original value

    Returns (all_findings, block_error_or_none, params_were_changed).
    """
    all_findings: list[Finding] = []
    changed = False

    for key, value in params.items():
        new_val, leaf_findings, block, did_change = await _scan_value_recursive(
            value,
            key,
            guardrails,
            engine,
        )
        all_findings.extend(leaf_findings)
        if block is not None:
            return all_findings, block, changed
        if did_change:
            changed = True
            params[key] = new_val

    return all_findings, None, changed


async def _scan_value_recursive(
    value: Any,
    key: str,
    guardrails: list[Guardrail],
    engine: GuardrailEngine,
) -> tuple[Any, list[Finding], str | None, bool]:
    """Scan a single value recursively.

    Returns (new_value, findings, block_error, changed).
    """
    if isinstance(value, str):
        new_val, findings, block = await _scan_leaf(value, key, guardrails, engine)
        return new_val, findings, block, new_val != value
    if isinstance(value, dict):
        all_findings: list[Finding] = []
        changed = False
        for k, v in value.items():
            new_v, findings, block, did_change = await _scan_value_recursive(
                v,
                k,
                guardrails,
                engine,
            )
            all_findings.extend(findings)
            if block is not None:
                return value, all_findings, block, changed
            if did_change:
                changed = True
                value[k] = new_v
        return value, all_findings, None, changed
    if isinstance(value, list):
        all_findings = []
        changed = False
        for i, item in enumerate(value):
            new_item, findings, block, did_change = await _scan_value_recursive(
                item,
                key,
                guardrails,
                engine,
            )
            all_findings.extend(findings)
            if block is not None:
                return value, all_findings, block, changed
            if did_change:
                changed = True
                value[i] = new_item
        return value, all_findings, None, changed
    return value, [], None, False


async def _scan_leaf(
    value: str,
    key: str,
    guardrails: list[Guardrail],
    engine: GuardrailEngine,
) -> tuple[str, list[Finding], str | None]:
    """Scan a single string leaf with value-only and key-context scans.

    Value-only catches patterns that don't need key context (emails, phones,
    AWS access keys). Key-context catches patterns like GENERIC_API_KEY that
    match ``api_key=<value>``. Key-context findings are projected back to
    value-relative offsets for correct redaction.

    Returns (possibly_redacted_value, findings, block_error_or_none).
    """
    all_findings: list[Finding] = []
    key_prefix_len = len(key) + 1  # "key="

    for guardrail in guardrails:
        # 1. Scan raw value
        value_findings = await guardrail.scan(value)

        # 2. Scan key=value context for key-dependent patterns
        context_text = f"{key}={value}"
        context_findings = await guardrail.scan(context_text)

        # 3. Project context findings to value span, keep only those
        #    that overlap with the value portion
        projected: list[Finding] = []
        for cf in context_findings:
            val_start = max(0, cf.start - key_prefix_len)
            val_end = cf.end - key_prefix_len
            if val_end <= 0:
                continue  # finding is entirely in the key portion
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

        # 4. Merge and de-overlap
        merged = _deoverlap(value_findings + projected)
        if not merged:
            continue
        all_findings.extend(merged)

        # 5. Apply action
        if guardrail.action == "block":
            entity = merged[0].entity_type
            return (
                value,
                all_findings,
                (
                    f"Guardrail '{type(guardrail).__name__}' blocked: "
                    f"{entity} detected in tool call params"
                ),
            )

        if guardrail.action == "redact":
            value = engine._apply_redaction(value, merged)

    return value, all_findings, None
