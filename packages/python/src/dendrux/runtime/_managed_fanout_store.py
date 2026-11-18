"""The optional fan-out store contract (0.8): bind 2..32 managed children at once, read them back.

A separate Protocol with its own typed requests and results, so existing store protocols and
0.7 custom stores are unchanged (interface segregation). A store opts in by declaring
``StoreCapabilities.managed_fanout``; ``dendrux.runtime._capabilities.fanout_support`` is the one
discovery helper. The types are internal and frozen. The parent wait is read only through its
stored JSON (spec §4.1): ``schema_version`` 4 with a ``managed_fanout`` envelope.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal, Protocol

from dendrux.runtime._fanout_adoption_state import MAX_FANOUT
from dendrux.runtime.state import (
    _MANAGED_RESERVED_KEYS,
    MANAGED_BINDING_KEY,
    MANAGED_HEADER_KEY,
    ClaimResult,
    EventWrite,
    LoopStatePatch,
    TraceWrite,
    _json_copy,
)
from dendrux.types import _ULID_PATTERN, MANAGED_PROTOCOL_VERSION, MANAGED_WAIT_REASON, PauseState

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from dendrux.runtime.state import TenantScope
    from dendrux.types import UsageStats

FANOUT_WAIT_SCHEMA_VERSION = 4
"""Pause ``schema_version`` of a fan-out wait (a single-child wait stays 3)."""
FANOUT_ENVELOPE_KEY = "managed_fanout"
"""Pause key of a fan-out wait's ordered binding envelope."""
MAX_CHILD_READ = MAX_FANOUT
"""Most child ids one ``get_managed_children`` call reads."""
_ENVELOPE_KEYS = frozenset({"v", "n", "bindings"})
_BINDING_KEYS = frozenset({"tool_call_id", "child_run_id"})


@dataclass(frozen=True)
class ManagedChildSpec:
    """Internal (0.8): one call of a fan-out batch and the fresh child bound to it."""

    tool_call_id: str
    row_id: str
    provider_tool_call_id: str | None
    tool_name: str
    params: dict[str, Any] | None
    child_run_id: str
    child_agent_name: str
    child_path: tuple[str, ...]
    child_input_data: dict[str, Any] | None
    child_loop_state: Mapping[str, Any]
    intent_event: EventWrite
    child_seed_traces: tuple[TraceWrite, ...] = ()
    child_model: str | None = None
    child_strategy: str | None = None
    child_meta: dict[str, Any] | None = None
    child_pii_mapping: dict[str, str] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "child_path", tuple(self.child_path))
        object.__setattr__(self, "child_seed_traces", tuple(self.child_seed_traces))
        for name in ("row_id", "child_run_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not _ULID_PATTERN.fullmatch(value):
                raise ValueError(f"{name} must be a run ULID")
        if not isinstance(self.tool_call_id, str) or not 0 < len(self.tool_call_id) <= 26:
            raise ValueError("tool_call_id must be a stable call id")
        if not self.child_path or any(
            not isinstance(p, str) or not p or len(p) > 255 for p in self.child_path
        ):
            raise ValueError("child_path must name the registered child")
        if MANAGED_HEADER_KEY in self.child_loop_state:
            raise ValueError("The store writes the child's managed header")
        if not isinstance(self.intent_event, EventWrite) or any(
            not isinstance(trace, TraceWrite) for trace in self.child_seed_traces
        ):
            raise ValueError("intent_event and child_seed_traces must be typed writes")

    def detached(self) -> ManagedChildSpec:
        """A copy sharing no mutable value with the caller or with any sibling."""
        seeds = tuple(
            replace(
                trace,
                meta=_json_copy(trace.meta),
                blob_refs=tuple(trace.blob_refs) if trace.blob_refs is not None else None,
            )
            for trace in self.child_seed_traces
        )
        return replace(
            self,
            params=_json_copy(self.params),
            child_input_data=_json_copy(self.child_input_data),
            child_loop_state=_json_copy(dict(self.child_loop_state)),
            child_meta=_json_copy(self.child_meta),
            child_pii_mapping=_json_copy(self.child_pii_mapping),
            child_seed_traces=seeds,
            intent_event=replace(self.intent_event, data=_json_copy(self.intent_event.data)),
        )


@dataclass(frozen=True)
class BindManagedChildrenRequest:
    """Internal (0.8): one atomic bind of a fan-out batch, as typed data only.

    ``children`` are in call order. The store re-checks ``2 <= len(children) <= max_parallel
    <= 32`` (``ManagedStateError(reason="fanout_limit")``) before writing anything.
    """

    tenant_id: str | None
    root_run_id: str
    parent_run_id: str
    parent_claim_epoch: int
    iteration_index: int
    parent_wait: PauseState
    parent_patch: LoopStatePatch
    children: tuple[ManagedChildSpec, ...]
    max_parallel: int
    parent_iteration_count: int | None = None
    parent_total_usage: UsageStats | None = None
    max_delegation_depth: int | None = None
    encoded_wait: dict[str, Any] | None = field(default=None, init=False, compare=False)
    """The wait's stored JSON; set only on the :meth:`detached` copy (encoded exactly once)."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "children", tuple(self.children))
        self._check_identity()
        self._check_children()
        if type(self.max_parallel) is not int:
            raise ValueError("max_parallel must be an int")
        if self.max_delegation_depth is not None and (
            type(self.max_delegation_depth) is not int or self.max_delegation_depth < 1
        ):
            raise ValueError("max_delegation_depth must be a positive int or None")

    def detached(self) -> BindManagedChildrenRequest:
        """A validated copy sharing no mutable data with the caller (taken before any await).

        The wait is encoded once from its own copy. The persisted pause, both binding copies
        and the returned generation all derive from that one value (``encoded_wait``).

        Raises:
            ValueError: the wait is not a fan-out wait binding exactly these children, in order.
        """
        wait = copy.deepcopy(self.parent_wait)
        encoded = _json_copy(wait.to_dict())
        bindings = wait_bindings(encoded)
        if bindings != tuple((c.tool_call_id, c.child_run_id) for c in self.children):
            raise ValueError("parent_wait must bind exactly these children, in call order")
        usage = self.parent_total_usage
        plan = replace(
            self,
            parent_wait=wait,
            parent_patch=LoopStatePatch(
                remove_pending=tuple(self.parent_patch.remove_pending),
                set_phase=self.parent_patch.set_phase,
                set_fields=_json_copy(dict(self.parent_patch.set_fields)),
                set_call_phase=_json_copy(dict(self.parent_patch.set_call_phase)),
            ),
            children=tuple(child.detached() for child in self.children),
            parent_total_usage=replace(usage) if usage is not None else None,
        )
        object.__setattr__(plan, "encoded_wait", encoded)
        return plan

    def _check_identity(self) -> None:
        for name in ("root_run_id", "parent_run_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not _ULID_PATTERN.fullmatch(value):
                raise ValueError(f"{name} must be a run ULID")
        if type(self.parent_claim_epoch) is not int or self.parent_claim_epoch < 0:
            raise ValueError("parent_claim_epoch must be a non-negative int")
        if self.tenant_id is not None and (
            not isinstance(self.tenant_id, str) or not self.tenant_id
        ):
            raise ValueError("tenant_id must be a non-empty string or None")
        if not isinstance(self.parent_wait, PauseState):
            raise ValueError("parent_wait must be a PauseState")

    def _check_children(self) -> None:
        if any(not isinstance(child, ManagedChildSpec) for child in self.children):
            raise ValueError("children must be ManagedChildSpec values")
        for name in ("tool_call_id", "child_run_id", "row_id"):
            values = [getattr(child, name) for child in self.children]
            if len(set(values)) != len(values):
                raise ValueError(f"Each child needs its own {name}")
        runs = {child.child_run_id for child in self.children}
        if {self.root_run_id, self.parent_run_id} & runs:
            raise ValueError("A child run id must be fresh")
        calls = {child.tool_call_id for child in self.children}
        if _MANAGED_RESERVED_KEYS & set(self.parent_patch.set_fields) or (
            calls & set(self.parent_patch.remove_pending)
        ):
            raise ValueError("parent_patch must not change the bindings or their pending calls")


@dataclass(frozen=True)
class BoundChild:
    """Internal (0.8): one bound call; ``child_claim`` only for the confirmed creator."""

    tool_call_id: str
    child_run_id: str
    child_claim: ClaimResult | None


@dataclass(frozen=True)
class BindManagedChildrenResult:
    """Internal (0.8): the bound batch in call order.

    ``created`` iff every child carries its execution token; ``existing`` iff none does.
    """

    outcome: Literal["created", "existing"]
    parent_wait_id: str | None
    children: tuple[BoundChild, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "children", tuple(self.children))
        claims = [isinstance(child.child_claim, ClaimResult) for child in self.children]
        # Execution authority goes to the confirmed creator only, never a read-back.
        if not claims or (
            (self.outcome == "created" and not all(claims))
            or (self.outcome == "existing" and any(claims))
            or self.outcome not in ("created", "existing")
        ):
            raise ValueError("Only a created batch carries every child's execution token")


@dataclass(frozen=True)
class ManagedChildRow:
    """Internal (0.8): one child's status projection. No history; raw JSON is not validated."""

    run_id: str
    parent_run_id: str | None
    tenant_id: str | None
    status: str
    header: Any
    """The raw managed header JSON; the caller validates it."""
    binding: Any
    """The raw checkpoint ``managed_wait`` JSON (v3 or v4), or None."""
    loop: str | None
    pause_id: str | None
    pause_reason: str | None


class ManagedFanoutStore(Protocol):
    """The two methods a store declaring ``managed_fanout`` adds to the managed contract."""

    async def bind_managed_children(
        self, request: BindManagedChildrenRequest
    ) -> BindManagedChildrenResult:
        """Bind every child of a batch, or none, in one transaction (spec §5.3).

        Root lock, then the parent row lock. An established batch is read back as
        ``existing`` without any execution token; a partial or conflicting one raises
        ``ManagedStateError(reason="binding_conflict")``. Otherwise admission runs once for
        the batch (``fanout_limit``; ``count + N`` against ``max_managed_runs``), the parent
        parks at the v4 wait with its epoch retired, and N intents, N children, their seeds
        and N intent events are written. Only the caller whose private nonce reads back gets
        ``created`` with N claims in call order.
        """
        ...

    async def get_managed_children(
        self, parent_run_id: str, child_run_ids: Sequence[str], *, tenant_id: TenantScope
    ) -> list[ManagedChildRow]:
        """Project up to 32 children of ``parent_run_id`` in one fresh primary-key read.

        Rows are scoped to the parent and the explicit tenant scope, returned in input order,
        with missing ids omitted. More than 32 ids (or repeated ids) raise ``ValueError``.
        No history is read and no lock is taken; each call opens its own writer session.
        """
        ...


def wait_bindings(data: object) -> tuple[tuple[str, str], ...]:
    """The ``(tool_call_id, child_run_id)`` pairs of a stored fan-out wait, in call order.

    Reads only the §4.1 JSON shape this store relies on: version 4, ``managed_wait`` reason,
    a pause ULID, the ``managed_fanout`` envelope and pending calls in the same order. The
    full strict decode belongs to the wait codec.

    Raises:
        ValueError: anything else.
    """
    if not isinstance(data, dict) or type(data.get("schema_version")) is not int:
        raise ValueError("Not a fan-out wait")
    envelope = data.get(FANOUT_ENVELOPE_KEY)
    pause_id = data.get("pause_id")
    if (
        data["schema_version"] != FANOUT_WAIT_SCHEMA_VERSION
        or data.get("pause_reason") != MANAGED_WAIT_REASON
        or MANAGED_BINDING_KEY in data
        or not isinstance(pause_id, str)
        or not _ULID_PATTERN.fullmatch(pause_id)
        or not isinstance(envelope, dict)
        or set(envelope) != _ENVELOPE_KEYS
        or type(envelope["v"]) is not int
        or envelope["v"] != MANAGED_PROTOCOL_VERSION
        or not isinstance(envelope["bindings"], list)
        or envelope["n"] != len(envelope["bindings"])
        or not all(isinstance(b, dict) and set(b) == _BINDING_KEYS for b in envelope["bindings"])
    ):
        raise ValueError("Not a fan-out wait")
    bindings = tuple((b["tool_call_id"], b["child_run_id"]) for b in envelope["bindings"])
    calls = data.get("pending_tool_calls")
    if not isinstance(calls, list) or [
        call.get("id") if isinstance(call, dict) else None for call in calls
    ] != [call_id for call_id, _ in bindings]:
        raise ValueError("Not a fan-out wait")
    return bindings


def checkpoint_binding(encoded_wait: dict[str, Any]) -> dict[str, Any]:
    """The checkpoint ``managed_wait`` record of a stored fan-out wait (spec §4.1)."""
    return {
        "v": MANAGED_PROTOCOL_VERSION,
        "pause_id": encoded_wait["pause_id"],
        "bindings": dict(wait_bindings(encoded_wait)),
    }
