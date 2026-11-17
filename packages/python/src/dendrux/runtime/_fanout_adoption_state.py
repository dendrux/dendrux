"""Recognize a claimed v4 fan-out adoption and measure its resolved prefix (0.8).

Pure decision logic: no store, no I/O, no clock. The bind read-back, the
adoption loop and recovery all ask the same two questions through this module,
so "which rows are already adopted" has one definition:

- :func:`fanout_adoption` — is this claimed checkpoint a v4 adoption, and for
  which calls, in call order?
- :func:`resolved_prefix` — how many of those calls already have a committed
  observation?

A row is *resolved* when its outcome is ``completed`` or ``failed`` and its
content digest is set. A failed child report is a committed observation, never
a row to redo. ``unknown`` outcomes, gaps and identity mismatches are refused as
``ManagedStateError(reason="binding_conflict")``; nothing is guessed or repaired.
Errors carry the run id and the reason code only, never stored text.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

from dendrux.errors import ManagedStateError

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

FANOUT_ADOPTION_KEY = "managed_fanout_adoption"
"""Key of the adoption marker in a claim descriptor and of the ids-only claim payload."""
FANOUT_ADOPTION_VERSION = 1
MIN_FANOUT = 2
MAX_FANOUT = 32
"""Bounds on the calls of one v4 fan-out wait (the wait format's own limits)."""

ROW_STARTED = "started"
"""Intent-row outcome: the call is admitted and has no observation yet."""
ROW_COMPLETED = "completed"
"""Intent-row outcome: a successful child report was committed."""
ROW_FAILED = "failed"
"""Intent-row outcome: an unsuccessful child report was committed (still resolved)."""
RESOLVED_OUTCOMES = frozenset({ROW_COMPLETED, ROW_FAILED})

_SUBMISSION_KEY = "submission"
_WAIT_KEY = "managed_wait"  # dendrux.runtime.state.MANAGED_BINDING_KEY
_CONFLICT = "binding_conflict"


class FanoutBinding(NamedTuple):
    """One call of a fan-out wait and the child run bound to it."""

    tool_call_id: str
    child_run_id: str


class IntentRow(NamedTuple):
    """The fields of a persisted tool-intent row that the prefix check reads.

    ``child_run_id`` is the row's ``meta.managed.child_run_id`` (None for a row
    that is not a managed call).
    """

    tool_call_id: str
    outcome: str
    content_digest: str | None
    child_run_id: str | None


def adoption_marker() -> dict[str, int]:
    """A new copy of the marker a v4 adoption claim descriptor carries."""
    return {"v": FANOUT_ADOPTION_VERSION}


def adoption_payload(call_ids: Iterable[str]) -> dict[str, dict[str, object]]:
    """The ids-only claim payload of a v4 adoption: call ids in call order, no reports."""
    return {FANOUT_ADOPTION_KEY: {"v": FANOUT_ADOPTION_VERSION, "calls": list(call_ids)}}


def fanout_adoption(state: object, *, run_id: str | None = None) -> tuple[str, ...] | None:
    """The call ids, in call order, of the v4 adoption a claimed checkpoint holds.

    ``state`` is the run's checkpoint (``loop_state``); it is only read. Returns
    None unless it holds both a v4 binding map and a submission marked exactly
    ``{"v": 1}``: a parked wait, a v3 adoption and an ordinary run are all None.

    Raises:
        ManagedStateError: ``binding_conflict`` when the marked submission's
            call ids are not exactly the binding map's calls, each once, or the
            map is not a fan-out's: 2..32 calls, each bound to its own child.
    """
    bindings = _v4_bindings(state)
    submission = _mapping(state).get(_SUBMISSION_KEY)
    if bindings is None or not _is_marker(_mapping(submission).get(FANOUT_ADOPTION_KEY)):
        return None
    ids = _mapping(submission).get("call_ids")
    if not isinstance(ids, list) or not all(isinstance(i, str) for i in ids):
        raise ManagedStateError(reason=_CONFLICT, run_id=run_id)
    if set(ids) != set(bindings) or len(ids) != len(bindings) or not _is_fanout_map(bindings):
        raise ManagedStateError(reason=_CONFLICT, run_id=run_id)
    return tuple(ids)


def resolved_prefix(
    rows: Iterable[IntentRow],
    bindings: Sequence[tuple[str, str]],
    *,
    run_id: str | None = None,
) -> int:
    """How many leading calls of a fan-out wait already have a committed observation.

    ``bindings`` are ``(tool_call_id, child_run_id)`` in call order. ``rows`` are
    the parent iteration's intent rows in any order; rows of other calls are
    ignored. Returns k−1 where rows ``1..k−1`` are resolved and rows ``k..N``
    are ``started`` (0 = nothing adopted yet, N = fully adopted).

    Raises:
        ManagedStateError: ``binding_conflict`` for a missing or duplicate row, a
            row bound to another child, a resolved row without a digest, a
            resolved row after a ``started`` one, any other outcome (including
            ``unknown``), or empty or repeating bindings.
    """
    by_call = _rows_by_call(rows, bindings, run_id)
    resolved = 0
    for position, (call_id, child_run_id) in enumerate(bindings):
        row = by_call.get(call_id)
        if row is None or row.child_run_id != child_run_id:
            raise ManagedStateError(reason=_CONFLICT, run_id=run_id)
        if row.outcome == ROW_STARTED:
            continue
        # Resolved rows must form a prefix: none may follow a started row.
        if not _is_resolved(row) or resolved != position:
            raise ManagedStateError(reason=_CONFLICT, run_id=run_id)
        resolved += 1
    return resolved


def _is_resolved(row: IntentRow) -> bool:
    return row.outcome in RESOLVED_OUTCOMES and bool(row.content_digest)


def _rows_by_call(
    rows: Iterable[IntentRow], bindings: Sequence[tuple[str, str]], run_id: str | None
) -> dict[str, IntentRow]:
    """The wait's rows keyed by call id; refuses repeated bindings and repeated rows."""
    calls = {call_id for call_id, _ in bindings}
    children = {child_run_id for _, child_run_id in bindings}
    if not bindings or len(calls) != len(bindings) or len(children) != len(bindings):
        raise ManagedStateError(reason=_CONFLICT, run_id=run_id)
    by_call: dict[str, IntentRow] = {}
    for row in rows:
        if row.tool_call_id not in calls:
            continue
        if row.tool_call_id in by_call:
            raise ManagedStateError(reason=_CONFLICT, run_id=run_id)
        by_call[row.tool_call_id] = row
    return by_call


def _is_fanout_map(bindings: Mapping[object, object]) -> bool:
    """A v4 wait binds 2..32 calls, each to a different child run id."""
    children = list(bindings.values())
    return (
        MIN_FANOUT <= len(children) <= MAX_FANOUT
        and all(isinstance(child, str) for child in children)
        and len(set(children)) == len(children)
    )


def _mapping(value: object) -> Mapping[object, object]:
    """``value`` when it is a JSON object, else an empty one."""
    return value if isinstance(value, dict) else {}


def _is_marker(value: object) -> bool:
    """Exactly ``{"v": 1}`` (a bool is not a version number)."""
    if not isinstance(value, dict) or set(value) != {"v"}:
        return False
    version = value["v"]
    return type(version) is int and version == FANOUT_ADOPTION_VERSION


def _v4_bindings(state: object) -> Mapping[object, object] | None:
    """The checkpoint's v4 ``{call_id: child_run_id}`` map, or None for any other shape."""
    wait = _mapping(state).get(_WAIT_KEY)
    if not isinstance(wait, dict) or type(wait.get("v")) is not int or wait["v"] != 1:
        return None
    bindings = wait.get("bindings")
    return bindings if isinstance(bindings, dict) and bindings else None
