"""Project submitted text before its claim and preserve continuation provenance."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from dendrux.guardrails._engine import _entry_engine
from dendrux.runtime._entry_guardrails import _entry_projection
from dendrux.types import Message, Role, content_part_to_dict

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping
    from contextlib import AbstractContextManager

    from dendrux.agent import Agent
    from dendrux.guardrails import GuardrailEngine
    from dendrux.loops._helpers import _BlockedObservationError
    from dendrux.runtime.state import StateStore
    from dendrux.types import ToolCall


async def _continuation_projection(
    agent: Agent, store: StateStore, run_id: str
) -> AbstractContextManager[Any]:
    from dendrux.runtime._blocked_content import _claim_call_scope

    if _entry_engine(run_id) is not None:
        return _claim_call_scope(run_id)
    projection = _entry_projection(agent, run_id)
    if projection.engine is not None:
        from dendrux.guardrails import GuardrailEngine

        projection.engine = GuardrailEngine(
            agent.guardrails or [], pii_mapping=await store.get_pii_mapping(run_id), run_id=run_id
        )
        projection.engine._retain_allowed_findings = True
    return _bound_call(projection, run_id)


@contextmanager
def _bound_call(projection: Any, run_id: str) -> Iterator[None]:
    from dendrux.runtime._blocked_content import _claim_call_scope

    with _claim_call_scope(run_id), projection.bind(run_id):
        yield


@dataclass(frozen=True)
class _SubmissionBlock:
    """The first block in a submission, kept as safe text only (no content, no matches)."""

    error: str
    counts: dict[str, int]
    role: Role
    name: str | None
    call_id: str | None
    offset: int

    @classmethod
    def of(cls, error: str, findings: Any, message: Message, offset: int) -> _SubmissionBlock:
        from collections import Counter

        counts = getattr(findings, "entity_counts", None)
        if counts is None:
            counts = Counter(finding.entity_type for finding in findings)
        return cls(error, dict(counts), message.role, message.name, message.call_id, offset)

    def location_message(self) -> Message:
        return Message(self.role, "", name=self.name, call_id=self.call_id)


@dataclass
class _SubmissionProjection:
    messages: list[Message] = field(default_factory=list)
    block: _BlockedObservationError | None = None

    def retain(
        self,
        original: Message,
        message: Message,
        findings: list[Any],
        error: str | None,
        *,
        inject: bool = True,
    ) -> None:
        if inject:
            self.messages.append(message if error is not None else original)
        if error is not None and self.block is None:
            from dendrux.loops._helpers import _BlockedObservationError
            from dendrux.types import ToolResult

            self.block = _BlockedObservationError(
                message,
                findings,
                error,
                ToolResult(
                    name=message.name or "", call_id=message.call_id or "", payload=message.content
                ),
            )


async def _project_submission(
    run_id: str,
    payload: dict[str, Any] | None,
    *,
    projection: _SubmissionProjection | None = None,
    denied_calls: Mapping[str, ToolCall] | None = None,
) -> tuple[dict[str, Any] | None, _SubmissionBlock | None]:
    """Project each governed text in a submission once.

    ``denied_calls`` (reclaim) maps pending call IDs to their calls: a denial's reason is
    then projected in the tool-result form recovery sends, so recovery's own projection of
    that message reuses this verdict (see :func:`_project_reason`).
    """
    engine = _entry_engine(run_id)
    if engine is None or payload is None:
        return payload, None
    clean = dict(payload)
    tool_block = await _project_submitted_tools(engine, clean, projection)
    input_block = await _project_submitted_input(engine, clean, projection)
    decision_block = await _project_submitted_decisions(
        engine, clean, projection, denied_calls or {}
    )
    return clean, tool_block or input_block or decision_block


async def _project_submitted_tools(
    engine: GuardrailEngine, clean: dict[str, Any], projection: _SubmissionProjection | None
) -> _SubmissionBlock | None:
    from dataclasses import replace

    from dendrux.loops._helpers import _project_tool_message
    from dendrux.runtime.submit import deserialize_tool_results, serialize_tool_results
    from dendrux.types import tool_result_message

    if "submitted_tool_results" not in clean:
        return None
    results = deserialize_tool_results(clean["submitted_tool_results"])
    block = None
    for index, result in enumerate(results):
        original = tool_result_message(result)
        message, findings, error = await _project_tool_message(engine, result, original)
        if projection is not None:
            projection.retain(original, message, findings, error)
        if error is not None:
            results[index] = replace(result, payload=message.content, error=None, meta={}, notes=())
            block = block or _SubmissionBlock.of(error, findings, original, index)
    clean["submitted_tool_results"] = serialize_tool_results(results)
    return block


async def _project_submitted_input(
    engine: GuardrailEngine, clean: dict[str, Any], projection: _SubmissionProjection | None
) -> _SubmissionBlock | None:
    from dendrux.runtime.runner import _stored_parts

    if "submitted_user_input" not in clean:
        return None
    original = Message(
        Role.USER,
        clean["submitted_user_input"],
        parts=_stored_parts(clean.get("submitted_user_parts")),
    )
    message, findings, error = await engine.project_incoming_message(original)
    if projection is not None:
        projection.retain(original, message, findings, error)
    if error is None:
        return None
    clean["submitted_user_input"] = message.content
    if message.parts is not None:
        clean["submitted_user_parts"] = [content_part_to_dict(p) for p in message.parts]
    # The reply follows any results submitted with it in the resumed history.
    offset = len(clean.get("submitted_tool_results") or ())
    return _SubmissionBlock.of(error, findings, original, offset)


async def _project_submitted_decisions(
    engine: GuardrailEngine,
    clean: dict[str, Any],
    projection: _SubmissionProjection | None,
    denied_calls: Mapping[str, ToolCall],
) -> _SubmissionBlock | None:
    if "submitted_decisions" not in clean:
        return None
    decisions = []
    block = None
    for index, decision in enumerate(clean["submitted_decisions"]):
        copy = dict(decision)
        if copy.get("reason") is not None:
            original, message, findings, error = await _project_reason(engine, copy, denied_calls)
            if projection is not None:
                projection.retain(original, message, findings, error, inject=False)
            if error is not None:
                copy["reason"] = message.content
                block = block or _SubmissionBlock.of(error, findings, original, index)
        decisions.append(copy)
    clean["submitted_decisions"] = decisions
    return block


async def _project_reason(
    engine: GuardrailEngine, decision: dict[str, Any], denied_calls: Mapping[str, ToolCall]
) -> tuple[Message, Message, list[Any], str | None]:
    """Project a decision reason; a denial of a known call in the tool-result form
    recovery sends, with the default reason recovery substitutes for an empty one. When
    JSON escaping changes the reason, its raw text is scanned as well."""
    denied = None if decision.get("approved", True) else denied_calls.get(decision["call_id"])
    if denied is None:
        original = Message(Role.USER, decision["reason"])
        return original, *(await engine.project_incoming_message(original))
    from dendrux.loops._helpers import _project_tool_message
    from dendrux.runtime._reclaim import _recovery_denial
    from dendrux.runtime.submit import DEFAULT_DENIAL_REASON
    from dendrux.types import ToolResult

    reason = decision["reason"] or DEFAULT_DENIAL_REASON
    original = _recovery_denial(denied, reason)
    result = ToolResult(name=denied.name, call_id=denied.id, payload=original.content, error=reason)
    return original, *(await _project_tool_message(engine, result, original))


def _submission_evidence(store: StateStore, run_id: str, blocked: bool) -> Any:
    from dendrux.runtime._blocked_content import _blocked_evidence

    return _blocked_evidence(store, run_id) if blocked else nullcontext()


async def _add_projected_note(
    agent: Agent, store: StateStore, run_id: str, add_note: Any, **kwargs: Any
) -> bool:
    from dendrux.runtime._blocked_content import _blocked_evidence

    projection = _entry_projection(agent, run_id)
    message = await projection.project(Message(Role.USER, kwargs["text"]), 1)
    if projection.error is not None:
        kwargs.update(text=message.content, metadata=None)
    async with _blocked_evidence(store, run_id) if projection.error is not None else nullcontext():
        return bool(await add_note(run_id, **kwargs))


async def _continuation_mapping(store: StateStore, run_id: str) -> dict[str, str] | None:
    engine = _entry_engine(run_id)
    return engine.get_pii_mapping() if engine is not None else await store.get_pii_mapping(run_id)
