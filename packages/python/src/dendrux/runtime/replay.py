"""Run replay — expand earlier runs into the message prefix of a new run.

Internal module behind ``agent.run(runs=..., replay_last=..., replay_reasoning=...)``.

The dev owns the conversation (an ordered list of run ids); Dendrux rebuilds
each run from its own persisted traces:

- runs inside the ``replay_last`` window replay in full (user message,
  assistant text + tool calls, tool results, optionally reasoning blocks);
- runs outside it replay as question + answer only.

Reasoning follows the prefix rule: every run records a prefix descriptor at
creation (``agent_runs.meta["dendrux.prefix"]``); a run's reasoning is kept
only while its descriptor equals the descriptor of what precedes it in the
current request. From the first mismatch onward reasoning is stripped (a
trailing strip, which every vendor accepts).

Phase-2 hook: ``expand_replay`` assembles each run's turn in exactly one place
(``_turn_for``); a persisted compaction artifact covering a span of runs would
substitute there.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal

from dendrux.context_blocks import _fold_first
from dendrux.types import (
    REASONING_MODEL_META_KEY,
    REPLAYED_META_KEY,
    Message,
    ReplaySummary,
    Role,
    ToolCall,
    generate_ulid,
)

if TYPE_CHECKING:
    from dendrux.context_blocks import ContextBlock
    from dendrux.llm.base import LLMProvider
    from dendrux.loops.base import Loop
    from dendrux.runtime.state import RunRecord, StateStore, TraceRecord
    from dendrux.types import ToolDef

logger = logging.getLogger(__name__)

PREFIX_META_KEY = "dendrux.prefix"
REPLAY_META_KEY = "dendrux.replay"

# Trace-meta keys on a run's own user message, recorded only when what went on
# the wire differs from the raw ``user_input`` (folded context blocks).
WIRE_CONTENT_KEY = "wire_content"
STABLE_CONTEXT_KEY = "stable_context"
PLACEMENT_KEY = "placement"

_TERMINAL_STATUSES = frozenset({"success", "error", "cancelled", "max_iterations"})

ReplayLast = int | Literal["all"]


# ---------------------------------------------------------------------------
# Request + validation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReplayRequest:
    """Validated replay knobs for one run."""

    runs: tuple[str, ...]
    replay_last: ReplayLast
    replay_reasoning: bool

    def fingerprint(self) -> dict[str, Any]:
        """Knobs as they enter the idempotency fingerprint."""
        return {
            "runs": list(self.runs),
            "replay_last": self.replay_last,
            "replay_reasoning": self.replay_reasoning,
        }


def validate_replay_args(
    runs: list[str] | None,
    replay_last: ReplayLast,
    replay_reasoning: bool,
    *,
    loop: Loop,
    context: list[ContextBlock] | None,
) -> ReplayRequest | None:
    """Synchronously validate the replay knobs. Returns None when ``runs`` is empty."""
    if replay_last != "all" and (
        isinstance(replay_last, bool) or not isinstance(replay_last, int) or replay_last < 0
    ):
        raise ValueError(f"replay_last must be a non-negative int or 'all', got {replay_last!r}")
    if not isinstance(replay_reasoning, bool):
        raise ValueError(f"replay_reasoning must be a bool, got {replay_reasoning!r}")
    if not runs:
        return None
    if isinstance(runs, str) or not all(isinstance(r, str) and r for r in runs):
        raise ValueError("runs must be a list of run id strings")

    seen: set[str] = set()
    for run_id in runs:
        if run_id in seen:
            from dendrux.errors import ReplayRunError

            raise ReplayRunError(run_id, "duplicate", "it appears more than once in runs")
        seen.add(run_id)

    from dendrux.loops.single import SingleCall

    if isinstance(loop, SingleCall) and replay_last != 0:
        raise ValueError(
            "runs= with replay_last != 0 is not supported on SingleCall: it has no tools "
            "slot to pair replayed tool history with. Use replay_last=0 (question + "
            "answer only) or a ReAct loop."
        )
    if context and any(b.placement == "stable" for b in context):
        raise ValueError(
            "context= blocks with placement='stable' cannot be combined with runs=: the "
            "first replayed run already carries the stable band it was created with, and "
            "refolding it would double it and break the cached prefix. Pass dynamic "
            "blocks only."
        )
    return ReplayRequest(
        runs=tuple(runs), replay_last=replay_last, replay_reasoning=replay_reasoning
    )


@dataclass
class LoadedRun:
    """A replayable run: its row plus its persisted traces."""

    record: RunRecord
    traces: list[TraceRecord]


async def load_replay_runs(
    store: StateStore, request: ReplayRequest, *, tenant_id: str | None
) -> list[LoadedRun]:
    """Load and validate every run in ``request.runs``. Raises ``ReplayRunError``."""
    from dendrux.errors import ReplayRunError

    loaded: list[LoadedRun] = []
    for run_id in request.runs:
        record = await store.get_run(run_id)
        if record is None:
            raise ReplayRunError(run_id, "not_found", "run not found")
        if record.tenant_id != tenant_id:
            raise ReplayRunError(
                run_id, "tenant_mismatch", "it belongs to a different tenant than this call"
            )
        if record.status not in _TERMINAL_STATUSES:
            raise ReplayRunError(
                run_id,
                "not_terminal",
                f"it is still '{record.status}'; only terminal runs can be replayed",
            )
        traces = await store.get_traces(run_id)
        loaded.append(LoadedRun(record=record, traces=sorted(traces, key=lambda t: t.order_index)))
    return loaded


# ---------------------------------------------------------------------------
# Prefix descriptor
# ---------------------------------------------------------------------------


def _hash(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, ensure_ascii=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def provider_id(provider: LLMProvider) -> str:
    """Provider class name plus base URL when the client exposes one."""
    name = type(provider).__name__
    base_url = getattr(getattr(provider, "_client", None), "base_url", None)
    return f"{name}@{base_url}" if base_url else name


def tool_hashes(tool_defs: list[ToolDef]) -> dict[str, str]:
    """Per-tool hash over what the provider receives (name, description, schema)."""
    return {
        td.name: _hash(
            {
                "name": td.name,
                "description": td.description,
                "parameters": td.parameters,
                "strict": td.strict,
            }
        )
        for td in sorted(tool_defs, key=lambda t: t.name)
    }


@dataclass(frozen=True)
class CurrentPrefix:
    """The parts of a request's prefix that bind replayed reasoning."""

    provider: str
    model: str
    system_hash: str
    tools_hash: str
    tools: dict[str, str] = field(default_factory=dict)


def current_prefix(
    agent: Any,
    provider: LLMProvider,
    loop: Loop,
    provider_kwargs: dict[str, Any] | None,
) -> CurrentPrefix:
    """Descriptor inputs for this run. Call after MCP discovery."""
    model = (provider_kwargs or {}).get("model") or provider.model
    tools = tool_hashes(agent.get_all_tool_defs(loop=loop))
    return CurrentPrefix(
        provider=provider_id(provider),
        model=model,
        system_hash=_hash(agent.get_system_prompt(loop=loop)),
        tools_hash=_hash(tools),
        tools=tools,
    )


def prefix_descriptor(current: CurrentPrefix, seed: dict[str, Any]) -> dict[str, Any]:
    """The ``meta["dendrux.prefix"]`` record for a new run."""
    return {
        "provider": current.provider,
        "model": current.model,
        "system_hash": current.system_hash,
        "tools_hash": current.tools_hash,
        "tools": dict(current.tools),
        "seed": seed,
    }


def history_seed(history: list[Message]) -> dict[str, Any]:
    """Seed for a run created with ``history=`` (or ``{}`` without)."""
    if not history:
        return {}
    return {"history_hash": _hash([[m.role.value, m.content] for m in history])}


def _seed_key(seed: Any) -> str:
    seed = seed if isinstance(seed, dict) else {}
    return _hash(
        {
            "expansion": seed.get("expansion") or [],
            "history_hash": seed.get("history_hash"),
            "retry_of": seed.get("retry_of"),
        }
    )


def _differing_fields(
    recorded: Any, current: CurrentPrefix, current_seed: list[dict[str, Any]]
) -> list[str]:
    if not isinstance(recorded, dict):
        return ["predates-metadata"]
    fields: list[str] = []
    if recorded.get("provider") != current.provider:
        fields.append("cross-vendor")
    if recorded.get("model") != current.model:
        fields.append("model")
    if recorded.get("system_hash") != current.system_hash:
        fields.append("system")
    if recorded.get("tools_hash") != current.tools_hash:
        fields.append("tools")
    if _seed_key(recorded.get("seed")) != _seed_key({"expansion": current_seed}):
        fields.append("seed")
    return fields


# ---------------------------------------------------------------------------
# Trace → Message rebuild (shared with retry)
# ---------------------------------------------------------------------------


def rebuild_messages(traces: list[TraceRecord], *, reasoning_for: str | None) -> list[Message]:
    """Rebuild ``Message`` objects from persisted traces.

    ``reasoning_for`` names the model reasoning blocks may be replayed to:
    blocks are kept when their producing model matches (or is unrecorded);
    ``None`` strips all reasoning. A run's user message is rebuilt as it was
    sent on the wire (folded context included).
    """
    messages: list[Message] = []
    for trace in sorted(traces, key=lambda t: t.order_index):
        try:
            role = Role(trace.role)
        except ValueError:
            continue

        meta = trace.meta or {}
        msg_meta: dict[str, Any] = {}
        msg_kwargs: dict[str, Any] = {"role": role, "content": trace.content}

        if role == Role.USER:
            if isinstance(meta.get(WIRE_CONTENT_KEY), str):
                msg_kwargs["content"] = meta[WIRE_CONTENT_KEY]
            if meta.get(PLACEMENT_KEY) == "stable":
                msg_kwargs["placement"] = "stable"

        if role == Role.TOOL:
            msg_kwargs["name"] = meta.get("tool_name", "unknown")
            msg_kwargs["call_id"] = meta.get("call_id", generate_ulid())
            if meta.get("is_error"):
                msg_meta["is_error"] = True

        if role == Role.ASSISTANT:
            if "tool_calls" in meta:
                msg_kwargs["tool_calls"] = [
                    ToolCall(
                        name=tc["name"],
                        params=tc.get("params") or {},
                        id=tc.get("id", generate_ulid()),
                        provider_tool_call_id=tc.get("provider_tool_call_id"),
                        parse_error=tc.get("parse_error"),
                    )
                    for tc in meta["tool_calls"]
                ]
            if meta.get("phase"):
                msg_meta["phase"] = meta["phase"]
            producer = meta.get(REASONING_MODEL_META_KEY)
            if (
                reasoning_for is not None
                and meta.get("reasoning_blocks")
                and producer in (None, reasoning_for)
            ):
                msg_kwargs["reasoning_blocks"] = meta["reasoning_blocks"]
                if producer:
                    msg_meta[REASONING_MODEL_META_KEY] = producer

        if msg_meta:
            msg_kwargs["meta"] = msg_meta
        messages.append(Message(**msg_kwargs))
    return messages


def _pair_tool_calls(messages: list[Message]) -> list[Message]:
    """Drop tool calls without a result (and results without a call)."""
    answered = {m.call_id for m in messages if m.role == Role.TOOL}
    issued: set[str] = set()
    out: list[Message] = []
    for msg in messages:
        if msg.role == Role.ASSISTANT and msg.tool_calls:
            kept = [tc for tc in msg.tool_calls if tc.id in answered]
            issued.update(tc.id for tc in kept)
            if len(kept) != len(msg.tool_calls):
                if not kept and not msg.content:
                    continue
                msg = replace(msg, tool_calls=kept or None)
        elif msg.role == Role.TOOL and msg.call_id not in issued:
            continue
        out.append(msg)
    return out


def _has_reasoning(traces: list[TraceRecord]) -> bool:
    return any(t.role == "assistant" and (t.meta or {}).get("reasoning_blocks") for t in traces)


def _first_user_trace(traces: list[TraceRecord]) -> TraceRecord | None:
    return next((t for t in traces if t.role == "user"), None)


def _raw_input(run: LoadedRun) -> str:
    data = run.record.input_data or {}
    value = data.get("input")
    if isinstance(value, str):
        return value
    first = _first_user_trace(run.traces)
    return first.content if first is not None else ""


def _text_turn(run: LoadedRun) -> list[Message]:
    first = _first_user_trace(run.traces)
    stable = (first.meta or {}).get(STABLE_CONTEXT_KEY) if first is not None else None
    raw = _raw_input(run)
    if isinstance(stable, str) and stable:
        messages = [Message(role=Role.USER, content=_fold_first(stable, raw), placement="stable")]
    else:
        messages = [Message(role=Role.USER, content=raw)]
    if run.record.answer:
        messages.append(
            Message(role=Role.ASSISTANT, content=run.record.answer, meta={"phase": "final_answer"})
        )
    return messages


def _full_turn(run: LoadedRun, *, reasoning_for: str | None) -> list[Message]:
    messages = _pair_tool_calls(rebuild_messages(run.traces, reasoning_for=reasoning_for))
    if not messages or messages[0].role != Role.USER:
        messages.insert(0, Message(role=Role.USER, content=_raw_input(run)))
    return messages


def _mark_replayed(messages: list[Message]) -> list[Message]:
    return [
        replace(m, meta={**m.meta, REPLAYED_META_KEY: True}) if m.role == Role.ASSISTANT else m
        for m in messages
    ]


# ---------------------------------------------------------------------------
# Expansion
# ---------------------------------------------------------------------------


@dataclass
class ReplayPlan:
    """The expanded prefix plus what to record about it."""

    messages: list[Message]
    summary: ReplaySummary
    seed: dict[str, Any]
    meta: dict[str, Any]


def _full_window(n: int, replay_last: ReplayLast) -> int:
    return n if replay_last == "all" else min(int(replay_last), n)


def expand_replay(
    loaded: list[LoadedRun],
    request: ReplayRequest,
    *,
    current: CurrentPrefix,
    guardrails: bool = False,
) -> ReplayPlan:
    """Expand ``loaded`` runs into messages per the window and the prefix rule."""
    warnings: list[str] = []
    replay_reasoning = request.replay_reasoning
    if replay_reasoning and guardrails:
        replay_reasoning = False
        warnings.append(
            "replay_reasoning forced off: guardrails re-anonymise replayed content, which "
            "changes the prefix that reasoning is bound to"
        )

    n = len(loaded)
    first_full = n - _full_window(n, request.replay_last)
    expansion: list[dict[str, Any]] = []
    messages: list[Message] = []
    runs_full: list[str] = []
    runs_text: list[str] = []
    kept: list[str] = []
    dropped: list[str] = []
    first_dropped: str | None = None

    for k, run in enumerate(loaded):
        run_id = run.record.id
        if k < first_full:
            runs_text.append(run_id)
            expansion.append({"run_id": run_id, "mode": "text", "reasoning": False})
            messages.extend(_text_turn(run))
            continue

        runs_full.append(run_id)
        keep = False
        if replay_reasoning and _has_reasoning(run.traces):
            recorded = (run.record.meta or {}).get(PREFIX_META_KEY)
            fields = _differing_fields(recorded, current, expansion)
            if first_dropped is not None and not fields:
                fields = [f"follows dropped run {first_dropped}"]
            if fields:
                dropped.append(run_id)
                first_dropped = first_dropped or run_id
                warnings.append(_drop_warning(run_id, fields))
            else:
                keep = True
                kept.append(run_id)

        turn = _full_turn(run, reasoning_for=current.model if keep else None)
        warnings.extend(_tool_warnings(run, turn, current))
        messages.extend(_mark_replayed(turn))
        expansion.append({"run_id": run_id, "mode": "full", "reasoning": keep})

    if messages:
        messages[-1] = replace(messages[-1], placement="stable")

    for warning in warnings:
        logger.warning("Run replay: %s", warning)

    summary = ReplaySummary(
        runs_full=runs_full,
        runs_text=runs_text,
        reasoning_kept=kept,
        reasoning_dropped=dropped,
        warnings=warnings,
    )
    seed = {**request.fingerprint(), "expansion": expansion}
    meta = {**request.fingerprint(), "reasoning_effective": replay_reasoning, **summary.to_dict()}
    return ReplayPlan(messages=messages, summary=summary, seed=seed, meta=meta)


def _drop_warning(run_id: str, fields: list[str]) -> str:
    if fields == ["predates-metadata"]:
        return f"run {run_id}: reasoning dropped (run predates replay metadata)"
    if len(fields) == 1 and fields[0].startswith("follows dropped run"):
        return f"run {run_id}: reasoning dropped ({fields[0]}; reasoning strips are trailing)"
    return (
        f"run {run_id}: reasoning dropped ({', '.join(fields)} changed since the run was created)"
    )


def _tool_warnings(run: LoadedRun, turn: list[Message], current: CurrentPrefix) -> list[str]:
    names = sorted({tc.name for m in turn if m.tool_calls for tc in m.tool_calls})
    if not names:
        return []
    recorded = (run.record.meta or {}).get(PREFIX_META_KEY)
    recorded_tools = recorded.get("tools") if isinstance(recorded, dict) else None
    absent = [n for n in names if n not in current.tools]
    changed = [
        n
        for n in names
        if n in current.tools
        and isinstance(recorded_tools, dict)
        and n in recorded_tools
        and recorded_tools[n] != current.tools[n]
    ]
    out: list[str] = []
    if absent:
        out.append(
            f"run {run.record.id}: replayed tool call(s) {', '.join(absent)} not in the "
            "current tool set"
        )
    if changed:
        out.append(
            f"run {run.record.id}: replayed tool call(s) {', '.join(changed)} have a changed schema"
        )
    return out


def build_seed(plan: ReplayPlan | None, history: list[Message]) -> dict[str, Any]:
    """Seed for the new run's own prefix descriptor."""
    seed: dict[str, Any] = dict(plan.seed) if plan is not None else {}
    seed.update(history_seed(history))
    return seed


def user_trace_message(
    user_input: str,
    seeded_history: list[Message],
    stable_band: str,
    context: list[ContextBlock] | None = None,
) -> Message:
    """The current user message as recorded to react_traces.

    Content stays the raw ``user_input``; when folded context changed what
    went on the wire, the wire form is kept in meta so a later replay can
    send it byte-for-byte.
    """
    from dataclasses import asdict

    from dendrux.skills._session import current as current_session

    current = seeded_history[-1]
    meta: dict[str, Any] = {}
    session = current_session()
    if session and session.explicit:
        meta["dendrux.skills.context"] = [
            asdict(b) for b in context or [] if b.placement == "dynamic" or len(seeded_history) == 1
        ]
    if current.content != user_input:
        meta[WIRE_CONTENT_KEY] = current.content
    if len(seeded_history) == 1 and current.placement == "stable" and stable_band:
        meta[PLACEMENT_KEY] = "stable"
        meta[STABLE_CONTEXT_KEY] = stable_band
    return Message(role=Role.USER, content=user_input, meta=meta)


async def record_run_prefix(
    store: StateStore | None,
    run_id: str,
    *,
    current: CurrentPrefix,
    seed: dict[str, Any],
    plan: ReplayPlan | None,
) -> None:
    """Merge the prefix descriptor (and replay summary) into ``agent_runs.meta``.

    Best-effort: a store without ``update_run_meta`` or a failed write only
    means later replays treat this run as predating replay metadata.
    """
    if store is None:
        return
    update = getattr(store, "update_run_meta", None)
    if update is None:
        return
    patch: dict[str, Any] = {PREFIX_META_KEY: prefix_descriptor(current, seed)}
    if plan is not None:
        patch[REPLAY_META_KEY] = plan.meta
    try:
        await update(run_id, patch)
    except Exception:
        logger.warning("Failed to record prefix descriptor for run %s", run_id, exc_info=True)
