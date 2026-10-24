"""GuardrailEngine — stateful per-run guardrail orchestrator.

Owns pii_mapping, placeholder generation, and action application.
The loop calls the engine at integration points; the engine is
stateful (pii_mapping grows), serializable (for pause/resume),
and testable in isolation.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from dendrux.guardrails._detectors import THREAD_SCAN_THRESHOLD
from dendrux.guardrails._protocol import UNSCANNABLE, Finding, scans_images, scans_text

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from dendrux.guardrails._protocol import Guardrail
    from dendrux.types import Message, ToolDef

logger = logging.getLogger(__name__)

_MAX_RECURSION_DEPTH = 50
# Text or params past these limits are not scanned; see ``_unscannable``.
MAX_SCAN_CHARS = 4 * 1024 * 1024
_MAX_SCAN_NODES = 10_000
# Distinct redacted values per run. Past it a redacting scan is unscannable.
MAX_MAPPING_ENTRIES = 20_000
# Key text added to a ``key=value`` context scan (enough for ``api_key=...``).
_KEY_CONTEXT_CHARS = 64
_MAX_TOOLS_PER_PLACEHOLDER = 16
_MAX_KINDS_IN_MESSAGE = 10
_PLACEHOLDER_RE = re.compile(r"<<[A-Z][A-Z_]*_\d+>>")
_CANDIDATE_RE = re.compile(r"<<[^<>]{1,128}>>")
# Provenance of placeholders (``PII(restore="allowlisted")`` only), persisted
# next to the mapping. Never a real placeholder.
_ORIGINS_KEY = "__dendrux_origins__"
RESTORE_SKIPPED_CODE = "pii_restore_skipped"

_notice_lock = threading.Lock()
_restore_notice_logged = False


def _block_message(
    guardrail: object,
    findings: list[Finding],
    *,
    where: str = "",
    location: str = "",
) -> str:
    """Format a block error: guardrail, finding kinds and counts, and where.

    Never includes matched text: an error string reaches events, spans, logs
    and ``RunResult.error``, and what a guardrail matched is by definition
    something that should not travel there.
    """
    name = type(guardrail).__name__
    counts = Counter(f.entity_type for f in findings)
    shown = list(counts.items())[:_MAX_KINDS_IN_MESSAGE]
    kinds = ", ".join(f"{n} {kind}" for kind, n in shown)
    if len(counts) > len(shown):
        kinds += f", +{len(counts) - len(shown)} more kinds"
    parts = [f"Guardrail {name!r} blocked: {kinds}"]
    if where:
        parts.append(f"in {where}")
    if location:
        parts.append(f"at {location}")
    return " ".join(parts)


def _unscannable() -> Finding:
    """Finding raised when content exceeded a scan limit (fail closed)."""
    return Finding(entity_type=UNSCANNABLE, start=0, end=0, score=1.0, text="")


def _message_location(msg: Message, index: int | None) -> str:
    bits = [f"role={msg.role.value}"]
    if msg.call_id:
        bits.append(f"tool_call_id={msg.call_id}")
    head = f"message {index}" if index is not None else "message"
    return f"{head} ({', '.join(bits)})"


def _has_unscannable(findings: list[Finding]) -> bool:
    return any(f.entity_type == UNSCANNABLE for f in findings)


def _unscannable_outcome(
    guardrail: Guardrail,
    all_findings: list[Finding],
    *,
    where: str,
    location: str = "",
) -> str | None:
    """Apply one guardrail's action to content it could not scan (fail closed).

    Records a single ``UNSCANNABLE`` finding; returns the block error under
    ``block`` and ``redact`` (nothing can be redacted in unscanned content),
    None under ``warn``.
    """
    if not _has_unscannable(all_findings):
        all_findings.append(_unscannable())
    if guardrail.action in ("block", "redact"):
        return _block_message(guardrail, [_unscannable()], where=where, location=location)
    return None


def _entity_of(placeholder: str) -> str:
    """``<<EMAIL_ADDRESS_1>>`` -> ``EMAIL_ADDRESS``."""
    return placeholder[2:-2].rsplit("_", 1)[0]


def _note_restore_default(guardrails: Iterable[object]) -> None:
    """Log once per process that PII placeholders are restored into every tool."""
    global _restore_notice_logged
    from dendrux.guardrails._pii import PII

    if _restore_notice_logged:
        return
    for g in guardrails:
        if isinstance(g, PII) and g.action == "redact" and not g._restore_chosen:
            with _notice_lock:
                if _restore_notice_logged:
                    return
                _restore_notice_logged = True
            logger.info(
                "PII redaction is on and restore= was not chosen: placeholders are restored "
                "into every tool call (restore='all'). If this agent reads untrusted content "
                "(web pages, e-mail, documents), consider PII(restore='allowlisted') with "
                "@tool(receives_pii=True) on the tools that need real values. This is "
                "informational; your choice stands."
            )
            return


@dataclass(frozen=True)
class RestoreOutcome:
    """Result of restoring placeholders into one tool call's params.

    ``skipped`` maps entity type to the number of placeholders left in place
    because the tool is not allowed to receive that kind of PII.
    """

    params: dict[str, Any]
    skipped: dict[str, int] = field(default_factory=dict)

    def event_data(self, tool_name: str, call_id: str | None) -> dict[str, Any]:
        """Payload of ``guardrail.restore_skipped``: counts and kinds only."""
        return {
            "tool_name": tool_name,
            "call_id": call_id,
            "entities": dict(self.skipped),
            "count": sum(self.skipped.values()),
            "code": RESTORE_SKIPPED_CODE,
        }


@dataclass(frozen=True)
class Origin:
    """Where text came from, for ``restore="allowlisted"`` provenance.

    ``trusted``: user, system and developer-supplied context. ``tool``: a tool
    result (``name`` is the tool). ``model``: assistant output, which never
    grants trust.
    """

    kind: Literal["trusted", "tool", "model"]
    name: str | None = None


TRUSTED = Origin("trusted")
MODEL = Origin("model")


def tool_origin(name: str) -> Origin:
    """Origin of a tool result produced by the tool ``name``."""
    return Origin("tool", name)


class GuardrailEngine:
    """Per-run guardrail orchestrator.

    Manages the list of guardrails, the run-scoped pii_mapping
    (forward and reverse), and placeholder counters per entity type.

    The loop calls:
      - scan_incoming(text) before provider.complete()
      - scan_outgoing(text) after LLM response
      - deanonymize(params, tool_def=...) before tool execution

    Content past the scan limits (text over ``MAX_SCAN_CHARS``; tool-call
    params nested deeper than 50 levels or with more than 10,000 nodes) is
    never silently skipped: it yields an ``UNSCANNABLE`` finding that blocks
    the run under ``block`` and ``redact`` and is reported under ``warn``.

    Args:
        guardrails: List of Guardrail instances to apply.
        pii_mapping: Existing mapping to restore from (pause/resume).
    """

    def __init__(
        self,
        guardrails: list[Guardrail],
        pii_mapping: dict[str, str] | None = None,
        *,
        run_id: str | None = None,
    ) -> None:
        self._guardrails = list(guardrails)
        # Image-only guardrails (``ImageGuardrail`` without ``scan``) are
        # handled by the request view; text scanning skips them.
        # Objects with neither method keep the v0.3.0 failure (AttributeError).
        self._text_guardrails: list[Guardrail] = [
            g for g in self._guardrails if scans_text(g) or not scans_images(g)
        ]
        raw = dict(pii_mapping) if pii_mapping else {}
        origins_raw = raw.pop(_ORIGINS_KEY, None)
        # Legacy runs may already exceed the cap. Keep every historical value
        # available for redaction/restoration, but admit no new distinct value.
        self._at_mapping_cap = len(raw) >= MAX_MAPPING_ENTRIES
        if len(raw) > MAX_MAPPING_ENTRIES:
            logger.warning(
                "Resumed PII mapping exceeds cap: run_id=%s count=%d cap=%d",
                run_id,
                len(raw),
                MAX_MAPPING_ENTRIES,
            )
        # Forward: <<EMAIL_ADDRESS_1>> → real value
        self._pii_mapping: dict[str, str] = raw
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
        # ``restore="allowlisted"`` (any PII guardrail asking for it wins).
        self._allowlisted = any(getattr(g, "restore", "all") == "allowlisted" for g in guardrails)
        # Provenance, tracked only under ``allowlisted``. A placeholder may be
        # restored into a tool only when it is trusted or that tool produced it;
        # a placeholder with no record (legacy or corrupt state, or minted from
        # model output) is restored nowhere until its origin is seen again.
        self._trusted: set[str] = set()
        self._tool_origins: dict[str, set[str]] = {}
        if self._allowlisted:
            self._load_origins(origins_raw)
        _note_restore_default(self._guardrails)

    @property
    def restore_mode(self) -> str:
        """``"allowlisted"`` or ``"all"`` (the 0.4.1 behaviour)."""
        return "allowlisted" if self._allowlisted else "all"

    def _next_placeholder(self, entity_type: str) -> str:
        """Generate the next placeholder for an entity type."""
        n = self._counters.get(entity_type, 1)
        self._counters[entity_type] = n + 1
        return f"<<{entity_type}_{n}>>"

    def _get_or_create_placeholder(
        self, entity_type: str, real_value: str, origin: Origin = TRUSTED
    ) -> str:
        """Get existing placeholder for a value, or create a new one.

        Under ``allowlisted`` every sighting records where the value came from:
        trusted text trusts it, a tool result ties it to that tool, and model
        output records nothing. Records only ever grow.
        """
        placeholder = self._reverse.get(real_value)
        if placeholder is None:
            placeholder = self._next_placeholder(entity_type)
            self._pii_mapping[placeholder] = real_value
            self._reverse[real_value] = placeholder
        if self._allowlisted:
            self._record_origin(placeholder, origin)
        return placeholder

    def _record_origin(self, placeholder: str, origin: Origin) -> None:
        if origin.kind == "trusted":
            self._trusted.add(placeholder)
        elif origin.kind == "tool" and origin.name:
            tools = self._tool_origins.setdefault(placeholder, set())
            if len(tools) < _MAX_TOOLS_PER_PLACEHOLDER:
                tools.add(origin.name)

    def _load_origins(self, raw: object) -> None:
        """Read persisted provenance defensively; anything unreadable is dropped.

        Dropping is the restrictive direction: a placeholder without a record
        is restored nowhere until a trusted or tool message mints it again.
        """
        if not isinstance(raw, str):
            return
        try:
            data = json.loads(raw)
        except ValueError:
            return
        if not isinstance(data, dict):
            return
        for placeholder, value in data.items():
            if not isinstance(placeholder, str) or placeholder not in self._pii_mapping:
                continue
            if value is True:
                self._trusted.add(placeholder)
            elif isinstance(value, list):
                tools = {t for t in value if isinstance(t, str) and t}
                if tools:
                    self._tool_origins[placeholder] = set(
                        sorted(tools)[:_MAX_TOOLS_PER_PLACEHOLDER]
                    )

    async def scan_incoming(
        self,
        text: str,
        *,
        origin: Origin = TRUSTED,
        location: str = "",
    ) -> tuple[str, list[Finding], str | None]:
        """Scan text going to the LLM. Apply actions.

        ``origin`` says where the text came from (a tool result, model output,
        or trusted text) and feeds ``restore="allowlisted"`` provenance;
        ``location`` describes where the text sits (used in block errors).

        Returns:
            (possibly_modified_text, all_findings, block_error_or_none)

        If a guardrail with action="block" finds something, returns
        the block error message. The caller should terminate the run.
        """
        all_findings: list[Finding] = []
        oversize = len(text) > MAX_SCAN_CHARS

        for guardrail in self._text_guardrails:
            raw_findings = [] if oversize else await guardrail.scan(text)
            if oversize or _has_unscannable(raw_findings):
                block = _unscannable_outcome(
                    guardrail, all_findings, where="text over the scan limit", location=location
                )
                if block is not None:
                    return text, all_findings, block
                continue
            findings = _deoverlap(raw_findings)
            if not findings:
                continue
            all_findings.extend(findings)

            if guardrail.action == "block":
                return (
                    text,
                    all_findings,
                    _block_message(guardrail, findings, location=location),
                )

            if guardrail.action == "redact":
                new_values = {f.text for f in findings if f.text not in self._reverse}
                if new_values and len(self._pii_mapping) + len(new_values) > MAX_MAPPING_ENTRIES:
                    block = _unscannable_outcome(
                        guardrail,
                        all_findings,
                        where="text with too many distinct values to redact",
                        location=location,
                    )
                    if block is not None:
                        return text, all_findings, block
                    continue
                text = await self._redact(text, findings, origin)
                self._at_mapping_cap = len(self._pii_mapping) >= MAX_MAPPING_ENTRIES
            # warn: findings collected but text unchanged

        return text, all_findings, None

    async def scan_incoming_message(
        self, msg: Message, *, index: int | None = None
    ) -> tuple[Message, list[Finding], str | None]:
        """Scan one outgoing-to-LLM message; parts-aware.

        Text-only messages scan ``content`` exactly like :meth:`scan_incoming`.
        Multimodal messages scan each text part separately; redaction rebuilds
        the parts (image parts untouched) so ``content`` stays the projection.

        ``index`` is the message's position in the request; it only feeds the
        location in block errors. Tool-result text is scanned with the tool as
        its origin.

        Returns:
            (possibly_replaced_message, all_findings, block_error_or_none)
        """
        from dataclasses import replace

        from dendrux.types import Role

        if msg.role == Role.TOOL:
            origin = tool_origin(msg.name or "")
        elif msg.role == Role.ASSISTANT:
            origin = MODEL
        else:
            origin = TRUSTED
        location = _message_location(msg, index)

        if msg.parts is None:
            cleaned, findings, block_err = await self.scan_incoming(
                msg.content, origin=origin, location=location
            )
            if block_err is None and cleaned != msg.content:
                # replace() preserves the origin envelope (kind/placement/source).
                msg = replace(msg, content=cleaned)
            return msg, findings, block_err

        from dendrux.types import TextPart

        all_findings: list[Finding] = []
        new_parts = list(msg.parts)
        changed = False
        for part_index, part in enumerate(msg.parts):
            if not isinstance(part, TextPart):
                continue
            cleaned, findings, block_err = await self.scan_incoming(
                part.text, origin=origin, location=f"{location}, part {part_index}"
            )
            all_findings.extend(findings)
            if block_err is not None:
                return msg, all_findings, block_err
            if cleaned != part.text:
                changed = True
                new_parts[part_index] = TextPart(cleaned)
        if changed:
            parts = tuple(new_parts)
            msg = replace(msg, parts=parts, content=_projection(parts))
        return msg, all_findings, None

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
        oversize = len(text) > MAX_SCAN_CHARS

        for guardrail in self._text_guardrails:
            raw_findings = [] if oversize else await guardrail.scan(text)
            if oversize or _has_unscannable(raw_findings):
                block = _unscannable_outcome(
                    guardrail, all_findings, where="LLM output over the scan limit"
                )
                if block is not None:
                    return all_findings, block
                continue
            findings = _deoverlap(raw_findings)
            if findings:
                all_findings.extend(findings)
                if guardrail.action == "block":
                    return (
                        all_findings,
                        _block_message(guardrail, findings, where="LLM output"),
                    )

        if tool_call_params:
            for call_index, params in enumerate(tool_call_params):
                p_findings, p_block = await _scan_params_for_findings(
                    params, self._text_guardrails, where=f"tool call params (call {call_index})"
                )
                all_findings.extend(p_findings)
                if p_block is not None:
                    return all_findings, p_block

        return all_findings, None

    async def _redact(self, text: str, findings: list[Finding], origin: Origin) -> str:
        """Redact in a worker thread when the text is large (see ``_apply_redaction``)."""
        if len(text) > THREAD_SCAN_THRESHOLD:
            return await asyncio.to_thread(self._apply_redaction, text, findings, origin)
        return self._apply_redaction(text, findings, origin)

    def _apply_redaction(self, text: str, findings: list[Finding], origin: Origin = TRUSTED) -> str:
        """Replace findings in text with placeholders in one linear pass.

        Findings must already be de-overlapped (done at scan level).
        Placeholders are numbered right to left, as in 0.4.1.
        """
        ordered = sorted(findings, key=lambda f: f.start)
        placeholders = [
            self._get_or_create_placeholder(f.entity_type, f.text, origin)
            for f in reversed(ordered)
        ]
        placeholders.reverse()
        pieces: list[str] = []
        pos = 0
        for finding, placeholder in zip(ordered, placeholders, strict=True):
            pieces.append(text[pos : finding.start])
            pieces.append(placeholder)
            pos = finding.end
        pieces.append(text[pos:])
        return "".join(pieces)

    def deanonymize(
        self, params: dict[str, Any], *, tool_def: ToolDef | None = None
    ) -> dict[str, Any]:
        """Replace <<PLACEHOLDER>> values with real values in tool call params.

        Walks the dict recursively. Returns a new dict (does not mutate input).
        If a placeholder is not in the mapping (corrupted by LLM), it passes
        through unchanged — the caller emits a warning event.

        With ``PII(restore="allowlisted")`` only ``tool_def.receives_pii``
        decides what is restored (no ``tool_def``: nothing). Use
        :meth:`restore_params` to also learn what was left in place.
        """
        return self.restore_params(params, tool_def=tool_def).params

    def restore_params(
        self, params: dict[str, Any], *, tool_def: ToolDef | None = None
    ) -> RestoreOutcome:
        """:meth:`deanonymize` plus a count of placeholders deliberately left in place.

        ``restore="all"`` (the default) restores every placeholder exactly as
        in 0.4.1 and never reports skips. Under ``"allowlisted"`` a placeholder
        is restored only when the tool's ``receives_pii`` covers its entity
        type and, for placeholders minted from a tool result, the tool is the
        one that produced it. The caller emits ``guardrail.restore_skipped``
        from :meth:`RestoreOutcome.event_data` when ``skipped`` is non-empty.
        """
        if not self._allowlisted:
            result = _restore_value(params, self._pii_mapping, None, None)
            return RestoreOutcome(params=result)
        skipped: dict[str, int] = {}
        allowed = self._allowed_for(tool_def)
        result = _restore_value(params, self._pii_mapping, allowed, skipped)
        return RestoreOutcome(params=result, skipped=skipped)

    def _allowed_for(self, tool_def: ToolDef | None) -> Callable[[str], bool]:
        if tool_def is None:
            return _deny_all
        receives = tool_def.receives_pii
        if receives is True:
            entities: frozenset[str] | None = None
        elif isinstance(receives, frozenset) and receives:
            entities = receives
        else:
            return _deny_all
        tool_name = tool_def.name
        trusted = self._trusted
        tool_origins = self._tool_origins

        def allowed(placeholder: str) -> bool:
            if placeholder not in trusted and tool_name not in tool_origins.get(placeholder, ()):
                return False
            return entities is None or _entity_matches(_entity_of(placeholder), entities)

        return allowed

    def get_pii_mapping(self) -> dict[str, str]:
        """Return the current pii_mapping for persistence."""
        mapping = dict(self._pii_mapping)
        if self._allowlisted:
            record: dict[str, Any] = {p: sorted(t) for p, t in self._tool_origins.items()}
            record.update(dict.fromkeys(self._trusted, True))
            if record:
                mapping[_ORIGINS_KEY] = json.dumps(record, sort_keys=True)
        return mapping

    @property
    def has_guardrails(self) -> bool:
        return len(self._guardrails) > 0


def _deny_all(placeholder: str) -> bool:
    return False


def _entity_matches(entity: str, allowed: frozenset[str]) -> bool:
    """Exact entity type, or a leading word of it (``EMAIL`` covers ``EMAIL_ADDRESS``).

    Case-insensitive, like the upper-case entity names themselves.
    """
    entity = entity.upper()
    for name in allowed:
        upper = name.upper()
        if entity == upper or entity.startswith(f"{upper}_"):
            return True
    return False


def has_placeholder(value: Any, _depth: int = 0) -> bool:
    """True when a string, or any string nested in ``value``, holds a ``<<TYPE_N>>`` placeholder."""
    if _depth > _MAX_RECURSION_DEPTH:
        return False
    if isinstance(value, str):
        return _PLACEHOLDER_RE.search(value) is not None
    if isinstance(value, dict):
        return any(has_placeholder(v, _depth + 1) for v in value.values())
    if isinstance(value, list):
        return any(has_placeholder(v, _depth + 1) for v in value)
    return False


def _projection(parts: tuple[Any, ...]) -> str:
    from dendrux.types import TextPart

    return "\n\n".join(p.text for p in parts if isinstance(p, TextPart))


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
    if _ORIGINS_KEY in mapping:
        mapping = {k: v for k, v in mapping.items() if k != _ORIGINS_KEY}
    result = _restore_value(text, mapping, None, None)
    if not isinstance(result, str):
        return result, []
    unmapped = sorted({p for p in _PLACEHOLDER_RE.findall(result) if p not in mapping})
    return result, unmapped


def _restore_value(
    value: Any,
    mapping: dict[str, str],
    allowed: Callable[[str], bool] | None,
    skipped: dict[str, int] | None,
    _depth: int = 0,
) -> Any:
    """Recursively replace placeholder strings with real values.

    ``allowed`` (None: everything) decides per placeholder; the ones it
    refuses stay in place and are counted by entity type in ``skipped``.
    """
    if _depth > _MAX_RECURSION_DEPTH:
        return value
    if isinstance(value, str):
        if value in mapping:
            if allowed is None or allowed(value):
                return mapping[value]
            _count_skip(skipped, value, 1)
            return value
        if "<<" not in value:
            return value

        def swap(match: re.Match[str]) -> str:
            placeholder = match.group()
            real_value = mapping.get(placeholder)
            if real_value is None:
                return placeholder
            if allowed is None or allowed(placeholder):
                return real_value
            _count_skip(skipped, placeholder, 1)
            return placeholder

        return _CANDIDATE_RE.sub(swap, value)
    if isinstance(value, dict):
        return {
            k: _restore_value(v, mapping, allowed, skipped, _depth + 1) for k, v in value.items()
        }
    if isinstance(value, list):
        return [_restore_value(item, mapping, allowed, skipped, _depth + 1) for item in value]
    return value


def _count_skip(skipped: dict[str, int] | None, placeholder: str, n: int) -> None:
    if skipped is not None:
        entity = _entity_of(placeholder)
        skipped[entity] = skipped.get(entity, 0) + n


@dataclass
class _Leaves:
    """Scannable strings of a params tree, plus why the walk stopped early (if it did).

    Each item is ``(key, text, with_context)``: dict keys are leaves of their
    own (``with_context`` False); values are scanned alone and as ``key=value``.
    """

    items: list[tuple[str, str, bool]] = field(default_factory=list)
    limit: str | None = None


def _scalar_text(value: int | float) -> str | None:
    """Text of a number for scanning, or None when it is absurdly long."""
    try:
        return str(value)
    except ValueError:  # int with more digits than the interpreter will convert
        return None


def _collect_leaves(params: dict[str, Any]) -> _Leaves:
    """Walk ``params`` lazily, bounded in depth, nodes and characters.

    Nothing is expanded ahead of the budget: containers are iterated one
    element at a time, every element (and every key) is counted as it is
    reached, and the walk stops the moment a limit is hit.
    """
    out = _Leaves()
    nodes = 0
    chars = 0
    # Frames: (iterator, is_dict, key for list items, depth of the frame's values).
    stack: list[tuple[Any, bool, str, int]] = [(iter(params.items()), True, "", 1)]
    while stack:
        it, is_dict, ctx_key, depth = stack[-1]
        try:
            item = next(it)
        except StopIteration:
            stack.pop()
            continue
        if is_dict:
            key, value = item
            if not isinstance(key, str):
                key = str(key)
            chars += len(key)
        else:
            key, value = ctx_key, item
        nodes += 1
        if nodes > _MAX_SCAN_NODES:
            out.limit = out.limit or "node count"
            break
        if chars > MAX_SCAN_CHARS:
            out.limit = out.limit or "size"
            break
        if is_dict and key:
            out.items.append((key, key, False))
        if isinstance(value, str):
            text: str | None = value
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            text = _scalar_text(value)
            if text is None:
                out.limit = out.limit or "size"
        else:
            text = None
        if text is not None:
            chars += len(text)
            if chars > MAX_SCAN_CHARS:
                out.limit = out.limit or "size"
                break
            out.items.append((key, text, True))
        elif isinstance(value, dict | list):
            if depth + 1 > _MAX_RECURSION_DEPTH:
                out.limit = out.limit or "nesting depth"
            elif isinstance(value, dict):
                stack.append((iter(value.items()), True, key, depth + 1))
            else:
                stack.append((iter(value), False, key, depth + 1))
    return out


async def _scan_params_for_findings(
    params: dict[str, Any],
    guardrails: list[Guardrail],
    *,
    where: str = "tool call params",
) -> tuple[list[Finding], str | None]:
    """Collect findings from params without mutating them.

    The runner persists raw params; redaction for the LLM happens at the
    next scan_incoming. This walk is detection-only — needed here so
    ``action="block"`` can abort the run before side-effecting tools fire.

    Keys, strings and numbers are scanned. Params nested too deeply, with
    too many nodes, or too large are not scanned past the limit: that yields
    an ``UNSCANNABLE`` finding which blocks under ``block`` and ``redact``
    (fail closed).

    Returns (findings, block_error_or_none).
    """
    leaves = _collect_leaves(params)
    all_findings: list[Finding] = []
    for key, text, with_context in leaves.items:
        leaf_findings, block = await _scan_leaf_for_findings(
            text, key, guardrails, where=where, with_context=with_context
        )
        all_findings.extend(leaf_findings)
        if block is not None:
            return all_findings, block

    if leaves.limit is not None:
        for guardrail in guardrails:
            block = _unscannable_outcome(
                guardrail, all_findings, where=f"{where}, over the {leaves.limit} limit"
            )
            if block is not None:
                return all_findings, block

    return all_findings, None


async def _scan_leaf_for_findings(
    value: str,
    key: str,
    guardrails: list[Guardrail],
    *,
    where: str = "tool call params",
    with_context: bool = True,
) -> tuple[list[Finding], str | None]:
    """Detection-only scan of a single string leaf.

    Uses both value-only and ``key=value`` context scans so patterns like
    GENERIC_API_KEY (which need the key name for detection) still fire. Only
    the last characters of the key join the context string, so a long key
    cannot multiply the scan cost.
    """
    all_findings: list[Finding] = []
    key_tail = key[-_KEY_CONTEXT_CHARS:]
    key_prefix_len = len(key_tail) + 1

    for guardrail in guardrails:
        value_findings = await guardrail.scan(value)
        context_findings = await guardrail.scan(f"{key_tail}={value}") if with_context else []
        if _has_unscannable(value_findings) or _has_unscannable(context_findings):
            block = _unscannable_outcome(guardrail, all_findings, where=where)
            if block is not None:
                return all_findings, block
            continue

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
                _block_message(guardrail, merged, where=where),
            )

    return all_findings, None
