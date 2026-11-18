"""Keep each confirmed fan-out claim in exactly one caller's custody."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from dendrux._meta_keys import CANCEL_PERSISTED_META_KEY
from dendrux.runtime._run_events import _persist_task_cancellation
from dendrux.runtime.state import claimed_store

if TYPE_CHECKING:
    from collections.abc import Sequence

    from dendrux.runtime._managed_fanout_store import BindManagedChildrenResult
    from dendrux.runtime.state import StateStore

EntryState = Literal["untransferred", "accepted", "driver_owned", "finalized", "unresolved"]


class FanoutLedger:
    """Custody of a bind result; the driver alone closes and finalizes it.

    ``accept`` and ``close`` never suspend. Indices always refer to bind call order.
    An existing read-back grants no custody. Accepted claims belong to their owner
    tasks; only claims returned by ``close`` may be finalized by the driver.
    """

    def __init__(self, bound: BindManagedChildrenResult) -> None:
        self._children = bound.children if bound.outcome == "created" else ()
        self._states: list[EntryState] = ["untransferred"] * len(self._children)
        self._closed = False

    @property
    def states(self) -> tuple[EntryState, ...]:
        """A read-only snapshot for custody diagnostics, in call order."""
        return tuple(self._states)

    def accept(self, i: int) -> bool:
        """Transfer entry i to its owner before that task's first await."""
        if self._closed or self._states[i] != "untransferred":
            return False
        self._states[i] = "accepted"
        return True

    def close(self) -> list[int]:
        """Close custody and return newly driver-owned indices, exactly once."""
        self._closed = True
        owned = [i for i, state in enumerate(self._states) if state == "untransferred"]
        for i in owned:
            self._states[i] = "driver_owned"
        return owned

    async def finalize(self, store: StateStore, driver_owned: Sequence[int]) -> None:
        """Finalize driver claims once, inside the caller's retained cleanup.

        Mark uncertainty before I/O: neither a failed nor an interrupted attempt
        may ever obtain a fresh epoch or retry an uncertain cancellation write.
        This method does not join owners or touch their generators or contexts.
        """
        for i in driver_owned:
            if self._states[i] != "driver_owned":
                continue
            self._states[i] = "unresolved"
            child = self._children[i]
            claim = child.child_claim
            if claim is None:
                continue
            result = await _persist_task_cancellation(
                claimed_store(store, child.child_run_id, claim), child.child_run_id
            )
            if result.meta.get(CANCEL_PERSISTED_META_KEY) is True:
                self._states[i] = "finalized"
