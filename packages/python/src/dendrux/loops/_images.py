"""Per-request image view: ``prepare_request`` and the loop-side commit helpers.

``prepare_request`` turns canonical history into the request the provider
sees: catalog-canonical refs, placeholders in place of images that must not
be sent, verified bytes on the wire copy. It only reads (catalog, blob bytes,
capability, guardrail scanners, describe callables); every state change comes
back as an :class:`~dendrux.media.ImageViewDelta` plus governance event specs
that the loop applies and records (:class:`ImageCallSite`).
"""

from __future__ import annotations

import copy
import inspect
import logging
import threading
import weakref
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal

from dendrux.errors import (
    BlobIntegrityError,
    BlobNotFoundError,
    ImageError,
    ImageLimitError,
    ImagePersistenceUnsupportedError,
    ImageRejectedError,
    UnsupportedModalityError,
)
from dendrux.guardrails._protocol import ImageScanInput, guardrail_identity, scans_images
from dendrux.loops._helpers import notify_governance, record_governance
from dendrux.media._ids import placeholder, placeholder_rank
from dendrux.media._ingest import check_image_policy
from dendrux.media._policy import ImageDescribeRequest, ImagePolicy
from dendrux.media._prepared import (
    GovernanceEvent,
    ImageStats,
    ImageViewDelta,
    PreparedRequest,
    SlotInfo,
)
from dendrux.media._tokens import estimate_image_tokens
from dendrux.types import (
    GovernanceEventType,
    GuardrailVerdict,
    ImagePart,
    ImageViewState,
    Message,
    Role,
    TextPart,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from dendrux.guardrails._protocol import Finding
    from dendrux.llm.base import LLMProvider
    from dendrux.loops.base import LoopNotifier, LoopRecorder
    from dendrux.media._runtime import ImageRuntime
    from dendrux.media._store import BlobInfo
    from dendrux.types import ContentPart, GuardrailVerdictKey, ImageKey, ImageRef

logger = logging.getLogger(__name__)

Capability = Literal["yes", "no", "unknown"]

# Warning keys recorded in ImageViewState.warned (once per run).
WARN_DROP_WITH_THINKING = "drop_with_thinking"
WARN_IMAGE_UNSCANNED = "guardrail.image_unscanned"

_DESCRIBE_MAX_CHARS = 2000


# ---------------------------------------------------------------------------
# Learned capability (fallback copy; providers keep their own cache too)
# ---------------------------------------------------------------------------

_learned_lock = threading.Lock()
_learned_no: weakref.WeakKeyDictionary[Any, set[tuple[str, str, str]]] = weakref.WeakKeyDictionary()


def _capability_key(provider: LLMProvider, model: str) -> tuple[str, str, str]:
    keyer = getattr(provider, "image_capability_key", None)
    if callable(keyer):
        key = keyer(model)
        return (str(key[0]), str(key[1]), str(key[2]))
    return (type(provider).__name__, "", model)


def learn_image_unsupported(provider: LLMProvider, model: str) -> None:
    """Remember that this provider instance refused images for ``model`` (process lifetime).

    Built-in adapters learn this themselves when they classify the refusal;
    this also tells the provider (``learn_image_unsupported``) when it exposes
    that method, and keeps a per-instance fallback for custom providers.
    """
    learner = getattr(provider, "learn_image_unsupported", None)
    if callable(learner):
        learner(model)
    key = _capability_key(provider, model)
    with _learned_lock:
        try:
            _learned_no.setdefault(provider, set()).add(key)
        except TypeError:  # pragma: no cover — provider not weak-referenceable
            return


def _is_learned_no(provider: LLMProvider, model: str) -> bool:
    key = _capability_key(provider, model)
    with _learned_lock:
        try:
            return key in _learned_no.get(provider, ())
        except TypeError:  # pragma: no cover
            return False


async def effective_capability(
    provider: LLMProvider, model: str, policy: ImagePolicy
) -> Capability:
    """Override → learned ``"no"`` → ``provider.image_input_support(model)`` (§5.6)."""
    override = policy.capability_overrides.get(model)
    if override in ("yes", "no"):
        return override
    learned = getattr(provider, "learned_image_support", None)
    if (callable(learned) and learned(model) == "no") or _is_learned_no(provider, model):
        return "no"
    support = getattr(provider, "image_input_support", None)
    if callable(support):
        cap: Capability = await support(model)
    else:
        cap = "unknown" if provider.capabilities.supports_multimodal else "no"
    if cap == "unknown" and policy.unknown_capability == "error":
        return "no"
    return cap


def provider_family(provider: LLMProvider, model: str) -> str | None:
    """Token-estimate family: ``"anthropic"``, ``"openai"``, ``"openrouter"`` or None."""
    keyer = getattr(provider, "image_capability_key", None)
    if callable(keyer):
        surface = str(keyer(model)[0])
        return {"anthropic": "anthropic", "chat": "openai", "responses": "openai"}.get(
            surface, "openrouter" if surface == "openrouter" else None
        )
    name = type(provider).__name__.lower()
    if "openrouter" in name:
        return "openrouter"
    if "anthropic" in name:
        return "anthropic"
    if "openai" in name:
        return "openai"
    return None


def _capability_source(provider: LLMProvider, model: str, policy: ImagePolicy) -> str:
    """``UnsupportedModalityError.source`` for a capability ``"no"``."""
    if policy.capability_overrides.get(model) == "no":
        return "capability"
    source = getattr(provider, "image_capability_source", None)
    if callable(source):
        value = source()
        if isinstance(value, str):
            return value
    return "capability"


def _provider_name(provider: LLMProvider) -> str:
    return type(provider).__name__


# ---------------------------------------------------------------------------
# prepare_request
# ---------------------------------------------------------------------------


def has_images(messages: Sequence[Message]) -> bool:
    """Whether any message carries image parts."""
    return any(m.parts is not None for m in messages)


@dataclass
class _Slot:
    slot: int
    message_index: int
    part_index: int
    part: ImagePart
    key: ImageKey
    origin: Literal["user", "tool"]
    tool_name: str | None
    new: bool
    info: BlobInfo | None = None
    reason: str | None = None  # placeholder reason, None = visible

    @property
    def image_id(self) -> str:
        return self.part.ref.image_id


@dataclass
class _Build:
    """Mutable working set of one ``prepare_request`` call."""

    slots: list[_Slot]
    state: ImageViewState
    new_reasons: dict[int, str] = field(default_factory=dict)
    described: dict[ImageKey, str] = field(default_factory=dict)
    verdicts: dict[GuardrailVerdictKey, GuardrailVerdict] = field(default_factory=dict)
    epoch: int | None = None
    dropped_through: int | None = None
    warned: list[str] = field(default_factory=list)
    events: list[GovernanceEvent] = field(default_factory=list)
    view_visible: int = 0

    def place(self, slot: _Slot, reason: str) -> None:
        """Placeholder ``slot`` with ``reason`` unless it already has one."""
        if slot.reason is not None:
            return
        slot.reason = reason
        self.new_reasons[slot.slot] = reason

    def visible(self) -> list[_Slot]:
        return [s for s in self.slots if s.reason is None]

    def described_text(self, key: ImageKey) -> str | None:
        text = self.state.described.get(key)
        return text if text is not None else self.described.get(key)


def _enumerate_slots(messages: Sequence[Message], state: ImageViewState, scope: str) -> list[_Slot]:
    slots: list[_Slot] = []
    for mi, msg in enumerate(messages):
        if msg.parts is None or msg.role == Role.SYSTEM:
            continue
        origin: Literal["user", "tool"] = "tool" if msg.role == Role.TOOL else "user"
        for pi, part in enumerate(msg.parts):
            if not isinstance(part, ImagePart):
                continue
            index = len(slots)
            slots.append(
                _Slot(
                    slot=index,
                    message_index=mi,
                    part_index=pi,
                    part=part,
                    key=(scope, part.ref.sha256),
                    origin=origin,
                    tool_name=msg.name if origin == "tool" else None,
                    new=index >= state.sent_through,
                )
            )
    return slots


def _check_descriptor(ref: ImageRef, info: BlobInfo) -> None:
    if (
        ref.sha256 != info.sha256
        or ref.media_type != info.media_type
        or ref.byte_length != info.byte_length
        or ref.width != info.width
        or ref.height != info.height
    ):
        raise BlobIntegrityError(ref.image_id, "descriptor")


async def _read_bytes(runtime: ImageRuntime, info: BlobInfo, cache: dict[str, bytes]) -> bytes:
    data = cache.get(info.sha256)
    if data is None:
        if runtime.resolver is None:
            raise ImagePersistenceUnsupportedError("state_store")
        data = await runtime.resolver.read(runtime.tenant_scope, info)
        cache[info.sha256] = data
    return data


def _eligible(build: _Build, policy: ImagePolicy, last_assistant: int) -> list[int]:
    assert policy.drop is not None
    scope_all = policy.drop.scope == "all"
    return sorted(
        s.slot
        for s in build.slots
        if s.message_index < last_assistant and (scope_all or s.origin == "tool")
    )


def _advance_drop(build: _Build, policy: ImagePolicy, eligible: list[int], *, forced: bool) -> bool:
    """Advance ``dropped_through`` (hysteresis, or forced); True when it moved."""
    assert policy.drop is not None
    keep, interval = policy.drop.keep_recent, policy.drop.interval
    current = build.dropped_through if build.dropped_through is not None else 0
    current = max(current, build.state.dropped_through)
    visible = [s for s in eligible if s >= current]
    threshold = keep if forced else keep + interval - 1
    if len(visible) <= threshold:
        return False
    new_d = max(current, visible[len(visible) - keep])
    if new_d <= current:
        return False
    build.dropped_through = new_d
    build.epoch = build.state.epoch + 1
    return True


def _apply_drops(build: _Build, eligible: list[int]) -> None:
    d = max(build.state.dropped_through, build.dropped_through or 0)
    if d <= 0:
        return
    eligible_set = {s for s in eligible if s < d}
    for slot in build.slots:
        if slot.slot in eligible_set:
            build.place(slot, "drop")


def _cap_violation(
    build: _Build,
    policy: ImagePolicy,
    provider: LLMProvider,
    model: str,
    family: str | None,
) -> ImageLimitError | None:
    visible = build.visible()
    if not visible:
        return None
    if len(visible) > policy.max_images_per_request:
        return ImageLimitError(
            "request_count", limit=policy.max_images_per_request, actual=len(visible)
        )
    total = sum(s.info.byte_length for s in visible if s.info is not None)
    if total > policy.max_request_image_bytes:
        return ImageLimitError("request_bytes", limit=policy.max_request_image_bytes, actual=total)
    limit = policy.max_estimated_image_tokens_per_request
    if limit is not None:
        tokens = _estimate(visible, family, model)
        if tokens is not None and tokens > limit:
            return ImageLimitError("estimated_tokens", limit=limit, actual=tokens)
    rule_fn = getattr(provider, "many_image_rule", None)
    rule = rule_fn(model) if callable(rule_fn) else None
    if rule is not None:
        max_count, max_side = rule
        if len(visible) > max_count:
            long_side = max(max(s.part.ref.width, s.part.ref.height) for s in visible)
            if long_side > max_side:
                return ImageLimitError("many_image_dimension", limit=max_side, actual=long_side)
    return None


def _estimate(visible: list[_Slot], family: str | None, model: str) -> int | None:
    if family is None:
        return None
    total = 0
    for s in visible:
        est = estimate_image_tokens(
            s.part.ref, provider_family=family, model=model, detail=s.part.detail
        )
        if est is None:
            return None
        total += est
    return total


async def _call_describe(describe: Any, request: ImageDescribeRequest) -> str:
    out = describe(request)
    if inspect.isawaitable(out):
        out = await out
    if not isinstance(out, str):
        raise TypeError(
            f"ImagePolicy.on_unsupported describe callable must return str, "
            f"got {type(out).__name__}"
        )
    return out.strip()[:_DESCRIBE_MAX_CHARS]


def _render(slot: _Slot, build: _Build) -> str:
    assert slot.reason is not None
    if slot.reason == "described":
        return placeholder("described", slot.image_id, build.described_text(slot.key) or "")
    return placeholder(slot.reason, slot.image_id)  # type: ignore[arg-type]


def _projection(parts: Sequence[ContentPart]) -> str:
    return "\n\n".join(p.text for p in parts if isinstance(p, TextPart))


def _rebuild(
    messages: list[Message], build: _Build, data: dict[str, bytes]
) -> tuple[list[Message], list[Message]]:
    by_message: dict[int, list[_Slot]] = {}
    for slot in build.slots:
        by_message.setdefault(slot.message_index, []).append(slot)
    semantic = list(messages)
    wire = list(messages)
    for mi, slots in by_message.items():
        msg = messages[mi]
        assert msg.parts is not None
        sem_parts: list[ContentPart] = list(msg.parts)
        wire_parts: list[ContentPart] = list(msg.parts)
        for slot in slots:
            if slot.reason is not None:
                text = TextPart(_render(slot, build))
                sem_parts[slot.part_index] = text
                wire_parts[slot.part_index] = text
            else:
                assert slot.info is not None
                ref = slot.info.to_ref()
                sem_parts[slot.part_index] = ImagePart(ref=ref, detail=slot.part.detail)
                wire_parts[slot.part_index] = ImagePart(
                    ref=ref, detail=slot.part.detail, data=data[slot.key[1]]
                )
        semantic[mi] = _with_parts(msg, sem_parts)
        wire[mi] = _with_parts(msg, wire_parts)
    return semantic, wire


def _with_parts(msg: Message, parts: list[ContentPart]) -> Message:
    if any(isinstance(p, ImagePart) for p in parts):
        parts_t = tuple(parts)
        return replace(msg, parts=parts_t, content=_projection(parts_t))
    return replace(msg, parts=None, content=_projection(parts))


async def prepare_request(
    messages: list[Message],
    *,
    state: ImageViewState,
    policy: ImagePolicy,
    provider: LLMProvider,
    model: str,
    runtime: ImageRuntime | None,
    guardrails: Sequence[Any],
    iteration: int,
    force_tombstones: Sequence[ImageKey] = (),
) -> PreparedRequest:
    """Build one request's image view (§5.1). Reads only; never writes or emits.

    Without images in ``messages`` it returns ``semantic is wire is messages``
    and an empty delta. Otherwise the steps run in the frozen order: catalog
    metadata, current policy, capability, rejected, drop policy, request caps,
    rehydration, image guardrails; placeholders are sticky per slot.

    Raises:
        BlobNotFoundError, BlobIntegrityError: a referenced blob is missing or
            does not match its persisted descriptor.
        InvalidImageError, ImageLimitError: a new image fails the current
            policy, or the request caps cannot be met.
        UnsupportedModalityError: a new image for a model without image input.
        ImagePersistenceUnsupportedError: images present without a runtime.
    """
    del iteration  # part of the frozen signature; the view does not depend on it
    if not has_images(messages):
        return PreparedRequest(
            semantic=messages,
            wire=messages,
            slots=(),
            new_slots=(),
            stats=ImageStats(count=0, bytes=0, estimated_tokens=0),
            guardrail_findings=None,
            events=(),
            delta=ImageViewDelta(),
            block_error=None,
        )
    if runtime is None:
        raise ImagePersistenceUnsupportedError("loop")
    catalog = runtime.catalog
    if catalog is None:
        raise ImagePersistenceUnsupportedError("state_store")
    scope = runtime.tenant_scope

    # Step 1: enumerate slots.
    build = _Build(slots=_enumerate_slots(messages, state, scope), state=state)
    for slot in build.slots:
        sticky = state.placeholders.get(slot.slot)
        if sticky is not None:
            slot.reason = sticky

    # Step 2: authoritative metadata (one query).
    shas = sorted({s.key[1] for s in build.slots})
    current = await catalog.get_current(scope, shas)
    for slot in build.slots:
        info = current.get(slot.key[1])
        if info is None:
            raise BlobNotFoundError(slot.image_id)
        _check_descriptor(slot.part.ref, info)
        slot.info = info
        if info.state in ("purging", "purged") and slot.reason != "purged":
            slot.reason = "purged"
            build.new_reasons[slot.slot] = "purged"
        elif info.state != "live" and slot.reason is None:
            raise BlobNotFoundError(slot.image_id)

    # Step 3: current policy re-check on catalog metadata.
    for slot in build.slots:
        if slot.reason is not None:
            continue
        info = slot.info
        assert info is not None
        try:
            check_image_policy(
                policy,
                media_type=info.media_type,
                frame_count=info.frame_count,
                byte_length=info.byte_length,
                width=info.width,
                height=info.height,
            )
        except ImageError:
            if slot.new:
                raise
            build.place(slot, "policy")

    data: dict[str, bytes] = {}

    # Step 4: capability.
    pending = build.visible()
    if pending:
        cap = await effective_capability(provider, model, policy)
        if cap == "no":
            mode = policy.on_unsupported
            if mode == "error":
                new_pending = [s for s in pending if s.new]
                if new_pending:
                    raise UnsupportedModalityError(
                        provider=_provider_name(provider),
                        model=model,
                        image_count=len(new_pending),
                        origin=new_pending[0].origin,
                        source=_capability_source(provider, model, policy),
                    )
                for slot in pending:
                    build.place(slot, "unsupported")
            elif mode == "placeholder":
                for slot in pending:
                    build.place(slot, "unsupported")
            else:
                for slot in pending:
                    if build.described_text(slot.key) is None:
                        assert slot.info is not None
                        raw = await _read_bytes(runtime, slot.info, data)
                        build.described[slot.key] = await _call_describe(
                            mode,
                            ImageDescribeRequest(
                                image_id=slot.image_id,
                                ref=slot.info.to_ref(),
                                origin=slot.origin,
                                tool_name=slot.tool_name,
                                data=raw,
                            ),
                        )
                    build.place(slot, "described")

    # Step 5: rejected tombstones.
    rejected = set(state.rejected) | set(force_tombstones)
    for slot in build.visible():
        if slot.key in rejected:
            build.place(slot, "rejected")

    # Step 6: drop policy (hysteresis).
    last_assistant = max(
        (i for i, m in enumerate(messages) if m.role == Role.ASSISTANT), default=-1
    )
    eligible: list[int] = []
    if policy.drop is not None:
        eligible = _eligible(build, policy, last_assistant)
        _advance_drop(build, policy, eligible, forced=False)
        _apply_drops(build, eligible)
        if (
            getattr(provider, "thinking_enabled", False) is True
            and WARN_DROP_WITH_THINKING not in state.warned
        ):
            logger.warning(
                "ImageDropPolicy with extended thinking: dropping images can invalidate "
                "the provider's thinking-block prefix."
            )
            build.warned.append(WARN_DROP_WITH_THINKING)

    # Step 7: request caps (forced prune when a drop policy exists).
    family = provider_family(provider, model)
    violation = _cap_violation(build, policy, provider, model, family)
    if (
        violation is not None
        and policy.drop is not None
        and _advance_drop(build, policy, eligible, forced=True)
    ):
        _apply_drops(build, eligible)
        violation = _cap_violation(build, policy, provider, model, family)
    if violation is not None:
        raise violation
    if build.dropped_through is not None:
        build.view_visible = sum(1 for s in eligible if s >= build.dropped_through)

    # Step 8: rehydrate visible bytes (verified by the resolver).
    for slot in build.visible():
        assert slot.info is not None
        await _read_bytes(runtime, slot.info, data)

    # Step 9: image guardrails.
    findings_out, block_error = await _scan_images(build, guardrails, runtime, data, scope)

    # Step 10: delta + events.
    semantic, wire = _rebuild(messages, build, data)
    visible = build.visible()
    stats = ImageStats(
        count=len(visible),
        bytes=sum(s.info.byte_length for s in visible if s.info is not None),
        estimated_tokens=_estimate(visible, family, model),
    )
    _collect_events(build, scope)
    delta = ImageViewDelta(
        epoch=build.epoch,
        dropped_through=build.dropped_through,
        described=dict(build.described),
        guardrail_verdicts=dict(build.verdicts),
        placeholders=dict(build.new_reasons),
        warned=tuple(build.warned),
    )
    slots = tuple(
        SlotInfo(
            slot=s.slot,
            message_index=s.message_index,
            part_index=s.part_index,
            key=s.key,
            image_id=s.image_id,
            origin=s.origin,
            tool_name=s.tool_name,
            visible=s.reason is None,
            placeholder=s.reason,
        )
        for s in build.slots
    )
    return PreparedRequest(
        semantic=semantic,
        wire=wire,
        slots=slots,
        new_slots=tuple(s.slot for s in build.slots if s.new),
        stats=stats,
        guardrail_findings=findings_out,
        events=tuple(build.events),
        delta=delta,
        block_error=block_error,
    )


async def _scan_images(
    build: _Build,
    guardrails: Sequence[Any],
    runtime: ImageRuntime,
    data: dict[str, bytes],
    scope: str,
) -> tuple[dict[str, Any] | None, str | None]:
    if not guardrails:
        return None, None
    visible = build.visible()
    image_guards = [g for g in guardrails if scans_images(g)]
    if not image_guards:
        if not visible:
            return None, None
        if WARN_IMAGE_UNSCANNED not in build.state.warned:
            build.warned.append(WARN_IMAGE_UNSCANNED)
            build.events.append(
                GovernanceEvent(
                    GovernanceEventType.GUARDRAIL_IMAGE_UNSCANNED.value, {"count": len(visible)}
                )
            )
        return {"count": len(visible), "scanned": False}, None
    verdict_rows: list[dict[str, Any]] = []
    block_error: str | None = None
    detected: list[str] = []
    detected_count = 0
    for slot in visible:
        assert slot.info is not None
        for guard in image_guards:
            gid, gver = guardrail_identity(guard)
            vkey: GuardrailVerdictKey = (
                scope,
                slot.key[1],
                gid,
                gver,
                slot.origin,
                slot.tool_name or "",
            )
            verdict = build.state.guardrail_verdicts.get(vkey) or build.verdicts.get(vkey)
            if verdict is None:
                found: list[Finding] = await guard.scan_image(
                    ImageScanInput(
                        image_id=slot.image_id,
                        ref=slot.info.to_ref(),
                        origin=slot.origin,
                        tool_name=slot.tool_name,
                        data=data[slot.key[1]],
                    )
                )
                if found:
                    entities = tuple(sorted({f.entity_type for f in found}))
                    verdict = GuardrailVerdict(action=guard.action, entities=entities)
                    detected_count += len(found)
                    detected.extend(e for e in entities if e not in detected)
                else:
                    verdict = GuardrailVerdict(action="pass")
                build.verdicts[vkey] = verdict
            verdict_rows.append(
                {"image_id": slot.image_id, "guardrail_id": gid, "action": verdict.action}
            )
            if verdict.action == "block" and block_error is None:
                block_error = (
                    f"Guardrail {gid!r} blocked image {slot.image_id}: "
                    f"{', '.join(verdict.entities) or 'flagged'}"
                )
            elif verdict.action == "redact":
                build.place(slot, "guardrail")
    if detected_count:
        build.events.append(
            GovernanceEvent(
                "guardrail.detected",
                {
                    "direction": "incoming_image",
                    "findings_count": detected_count,
                    "entities": detected,
                },
            )
        )
    findings = {"count": len(visible), "scanned": True, "verdicts": verdict_rows}
    return findings, block_error


_REASON_ORDER = ("purged", "policy", "unsupported", "described", "rejected", "drop", "guardrail")


def _collect_events(build: _Build, scope: str) -> None:
    """Prepend view/degraded/state/warning events (guardrail events already queued)."""
    head: list[GovernanceEvent] = []
    if build.dropped_through is not None:
        assert build.epoch is not None
        head.append(
            GovernanceEvent(
                GovernanceEventType.IMAGE_VIEW_CHANGED.value,
                {
                    "epoch": build.epoch,
                    "dropped_through": build.dropped_through,
                    "visible": build.view_visible,
                },
            )
        )
    by_reason: dict[str, list[int]] = {}
    for slot_index, reason in sorted(build.new_reasons.items()):
        by_reason.setdefault(reason, []).append(slot_index)
    for reason in sorted(by_reason, key=lambda r: (placeholder_rank(r), _REASON_ORDER.index(r))):
        slots = by_reason[reason]
        head.append(
            GovernanceEvent(
                GovernanceEventType.IMAGE_DEGRADED.value,
                {"reason": reason, "count": len(slots), "slots": slots},
            )
        )
    if build.described or build.verdicts or build.new_reasons:
        head.append(
            GovernanceEvent(
                GovernanceEventType.IMAGE_STATE.value,
                {
                    "tenant_scope": scope,
                    "described": [
                        {"sha256": k[1], "text": text}
                        for k, text in sorted(build.described.items())
                    ],
                    "guardrail_verdicts": [
                        {
                            "sha256": k[1],
                            "guardrail_id": k[2],
                            "guardrail_version": k[3],
                            "origin": k[4],
                            "tool_name": k[5],
                            "action": v.action,
                            "entities": list(v.entities),
                        }
                        for k, v in sorted(build.verdicts.items())
                    ],
                    "placeholders": [
                        {"slot": slot, "reason": reason}
                        for slot, reason in sorted(build.new_reasons.items())
                    ],
                },
            )
        )
    if WARN_DROP_WITH_THINKING in build.warned:
        head.append(
            GovernanceEvent(
                GovernanceEventType.IMAGE_WARNING.value, {"warning": WARN_DROP_WITH_THINKING}
            )
        )
    build.events[:0] = head


# ---------------------------------------------------------------------------
# Commit helpers (loop side)
# ---------------------------------------------------------------------------


def apply_delta(state: ImageViewState, delta: ImageViewDelta) -> None:
    """Merge ``delta`` into ``state``: first-wins maps, monotonic counters, sticky slots."""
    if delta.epoch is not None:
        state.epoch = max(state.epoch, delta.epoch)
    if delta.dropped_through is not None:
        state.dropped_through = max(state.dropped_through, delta.dropped_through)
    for key, text in delta.described.items():
        state.described.setdefault(key, text)
    for vkey, verdict in delta.guardrail_verdicts.items():
        state.guardrail_verdicts.setdefault(vkey, verdict)
    for slot, reason in delta.placeholders.items():
        if slot not in state.placeholders or reason == "purged":
            state.placeholders[slot] = reason
    for warning in delta.warned:
        if warning not in state.warned:
            state.warned.append(warning)


def release_wire(prepared: PreparedRequest) -> PreparedRequest:
    """Drop the byte-carrying wire messages once the provider call is over.

    Only ``semantic`` (refs, no bytes) is ever recorded; nothing keeps verified
    bytes alive after the call (not the loop frame, not a traceback).
    """
    if prepared.wire is prepared.semantic:
        return prepared
    return replace(prepared, wire=prepared.semantic)


def pause_image_view(state: ImageViewState, history: Sequence[Message]) -> ImageViewState | None:
    """``PauseState.image_view``: a copy once the run has seen an image, else None."""
    if state == ImageViewState() and not has_images(history):
        return None
    return copy.deepcopy(state)


def tool_result_images_dropped(result_images: Sequence[Any], message: Message) -> None:
    """Guard: a custom strategy must keep image parts of a tool result."""
    if result_images and not message.images:
        raise TypeError(
            "strategy dropped image parts: format_tool_result() must keep a tool "
            "result's images (build the message with dendrux.types.tool_result_message)"
        )


@dataclass(frozen=True)
class RetryDecision:
    """What the loop does after a failed provider call on an image-bearing request."""

    action: Literal["propagate", "retry", "raise"]
    tombstones: tuple[ImageKey, ...] = ()
    error: BaseException | None = None
    control: Any = None


@dataclass
class ImageCallSite:
    """Image state and commit protocol for one loop run (all four call sites).

    Owns the run's live :class:`ImageViewState`; :meth:`prepare` runs
    ``prepare_request`` and commits its delta and events in order.
    """

    state: ImageViewState
    policy: ImagePolicy
    runtime: ImageRuntime | None
    provider: LLMProvider
    model: str
    recorder: LoopRecorder | None
    notifier: LoopNotifier | None
    run_id: str
    warnings: list[str]

    async def prepare(
        self,
        messages: list[Message],
        *,
        iteration: int,
        guardrails: Sequence[Any] = (),
        force_tombstones: Sequence[ImageKey] = (),
    ) -> PreparedRequest:
        """``prepare_request`` + steps 2-3 of the commit protocol (delta, then events)."""
        prepared = await prepare_request(
            messages,
            state=self.state,
            policy=self.policy,
            provider=self.provider,
            model=self.model,
            runtime=self.runtime,
            guardrails=guardrails,
            iteration=iteration,
            force_tombstones=force_tombstones,
        )
        if prepared.slots:
            apply_delta(self.state, prepared.delta)
            for event in prepared.events:
                await self.emit(event.event_type, event.data, iteration)
        return prepared

    async def emit(self, event_type: str, data: dict[str, Any], iteration: int) -> None:
        """Record (fail-closed) then notify (best effort) one governance event."""
        await record_governance(self.recorder, self.run_id, event_type, iteration, data)
        await notify_governance(
            self.notifier, self.run_id, event_type, iteration, data, warnings=self.warnings
        )

    def mark_sent(self, prepared: PreparedRequest) -> PreparedRequest:
        """Commit step 6: the request's slots now count as history.

        Returns ``prepared`` with the wire copy released (see :func:`release_wire`).
        """
        if prepared.slots:
            self.state.sent_through = max(self.state.sent_through, len(prepared.slots))
        return release_wire(prepared)

    def findings(self, prepared: PreparedRequest, base: dict[str, Any]) -> dict[str, Any] | None:
        """``guardrail_findings`` for the LLM record, with the image section merged."""
        if prepared.guardrail_findings is not None:
            base = {**base, "images": prepared.guardrail_findings}
        return base or None

    async def on_failure(
        self,
        exc: BaseException,
        prepared: PreparedRequest,
        *,
        iteration: int,
        retried: bool,
        visible_deltas: int,
        control_reader: Any,
    ) -> RetryDecision:
        """§5.7 / §5.1 commit step 7. Call outside the ``except`` block.

        ``control_reader`` is an async callable returning the run control
        (cancel/pause flags) or None; it is read only right before a retry.
        A ``"raise"`` decision carries a fresh, cause-free error; a cancel or
        pause seen before the retry returns ``"propagate"`` with ``control`` set
        (read it with :func:`control_stop`) so the loop can honour it.
        """
        if isinstance(exc, ImageRejectedError) and prepared.slots:
            return await self._on_rejected(
                exc,
                prepared,
                iteration=iteration,
                retried=retried,
                visible_deltas=visible_deltas,
                control_reader=control_reader,
            )
        if (
            isinstance(exc, UnsupportedModalityError)
            and exc.source == "provider_response"
            and prepared.slots
            and not retried
            and visible_deltas == 0
            and not any(prepared.slots[s].visible for s in prepared.new_slots)
        ):
            learn_image_unsupported(self.provider, self.model)
            control = await control_reader()
            if _stop_requested(control):
                return RetryDecision("propagate", control=control)
            return RetryDecision("retry")
        return RetryDecision("propagate")

    async def _on_rejected(
        self,
        exc: ImageRejectedError,
        prepared: PreparedRequest,
        *,
        iteration: int,
        retried: bool,
        visible_deltas: int,
        control_reader: Any,
    ) -> RetryDecision:
        candidates = rejection_candidates(exc, prepared)
        if not candidates:
            return RetryDecision(
                "raise",
                error=_rejected(exc, image_ids=(), located=False, retried=retried),
            )
        keys = [slot.key for slot in candidates]
        unique_keys = list(dict.fromkeys(keys))
        tool_only = all(
            all(s.origin == "tool" for s in prepared.slots if s.visible and s.key == key)
            for key in unique_keys
        )
        eligible = tool_only and not retried and visible_deltas == 0
        control = None
        if eligible:
            control = await control_reader()
        will_retry = eligible and not _stop_requested(control)
        for key in unique_keys:
            if key not in self.state.rejected:
                self.state.rejected.append(key)
        seen: set[ImageKey] = set()
        for slot in candidates:
            if slot.key in seen:
                continue
            seen.add(slot.key)
            await self.emit(
                GovernanceEventType.IMAGE_REJECTED.value,
                {
                    "image_id": slot.image_id,
                    "sha256": slot.key[1],
                    "tenant_scope": slot.key[0],
                    "provider": exc.provider,
                    "model": exc.model,
                    "status_code": exc.status_code,
                    "located": exc.located,
                    "retried": will_retry,
                },
                iteration,
            )
        ids = tuple(dict.fromkeys(slot.image_id for slot in candidates))
        if will_retry:
            return RetryDecision("retry", tombstones=tuple(unique_keys))
        if eligible and _stop_requested(control):
            return RetryDecision("propagate", control=control)
        return RetryDecision(
            "raise", error=_rejected(exc, image_ids=ids, located=exc.located, retried=retried)
        )


def control_stop(decision: RetryDecision) -> Any | None:
    """The cancel/pause control object when ``decision`` defers to it, else None."""
    return decision.control


def _stop_requested(control: Any) -> bool:
    return control is not None and bool(
        getattr(control, "cancel_requested", False) or getattr(control, "pause_requested", False)
    )


def _rejected(
    exc: ImageRejectedError,
    *,
    image_ids: tuple[str, ...],
    located: bool,
    retried: bool,
) -> ImageRejectedError:
    return ImageRejectedError(
        provider=exc.provider,
        model=exc.model,
        image_ids=image_ids,
        located=located,
        retried=retried,
        status_code=exc.status_code,
        provider_code=exc.provider_code,
    )


def rejection_candidates(exc: ImageRejectedError, prepared: PreparedRequest) -> list[SlotInfo]:
    """Visible slots to tombstone for a provider rejection (§5.7, N-10)."""
    visible = [s for s in prepared.slots if s.visible]
    if exc.located and exc.image_ids:
        wanted = set(exc.image_ids)
        located = [s for s in visible if s.image_id in wanted]
        if located:
            return located
    new = set(prepared.new_slots)
    new_visible = [s for s in visible if s.slot in new]
    if new_visible:
        return new_visible
    return [s for s in visible if s.origin == "tool"]


async def no_control() -> None:
    """Control reader for loops without a state store."""
    return None
