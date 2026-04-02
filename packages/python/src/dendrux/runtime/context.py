"""Delegation context — automatic parent-child run linking.

Propagates run ancestry through nested agent.run() calls via
contextvars. When a tool function calls agent.run() for a sub-agent,
the child run automatically knows its parent's run_id and delegation
level — zero developer code needed.

Linking is best-effort:
  - Both persisted + same store → parent_run_id set on child
  - Parent ephemeral or different store → link skipped, warning emitted
  - Delegation always works regardless — linking is observability, not control

Store identity: compared by database URL (via store_identity property),
not by Python object identity. Two stores pointing at the same database
match even if they are different objects. Custom StateStore implementations
can define a store_identity property; otherwise falls back to object id.

Limitation: automatic linking works for in-process nested async calls.
Distributed/worker boundaries require explicit propagation (future work).
"""

from __future__ import annotations

import logging
from contextvars import ContextVar, Token
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Context dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DelegationContext:
    """Runtime context for a running agent — set by the runner, read by child runs.

    Attributes:
        run_id: This run's ID.
        delegation_level: Logical nesting depth (0 for root).
        persisted: Whether this run was written to a state store.
        store_identity: Comparable identity for the state store (database URL
            for SQLAlchemy stores, str(id()) for custom stores). None if not
            persisted.
        warned_mismatches: Mismatch types already warned about for this
            parent run. Used to deduplicate warnings when an orchestrator
            fans out to many child runs. Mutable set on a frozen dataclass —
            frozen prevents field reassignment, not mutation of the set.
    """

    run_id: str
    delegation_level: int = 0
    persisted: bool = False
    store_identity: str | None = None
    warned_mismatches: set[str] = field(default_factory=set)

    def __hash__(self) -> int:
        return hash(self.run_id)


# ---------------------------------------------------------------------------
# ContextVar — the propagation mechanism
# ---------------------------------------------------------------------------

_current_delegation: ContextVar[DelegationContext | None] = ContextVar(
    "dendrux_delegation", default=None
)


def get_delegation_context() -> DelegationContext | None:
    """Read the current delegation context (None if this is a root run)."""
    return _current_delegation.get()


def set_delegation_context(ctx: DelegationContext) -> Token[DelegationContext | None]:
    """Set the delegation context for the current async task.

    Returns a token for resetting in finally blocks.
    """
    return _current_delegation.set(ctx)


def reset_delegation_context(token: Token[DelegationContext | None]) -> None:
    """Reset the delegation context to its previous value."""
    _current_delegation.reset(token)


# ---------------------------------------------------------------------------
# Store identity — extract comparable identity from any StateStore
# ---------------------------------------------------------------------------


def get_store_identity(store: object | None) -> str | None:
    """Extract a comparable identity from a state store.

    Resolution order:
      1. store.store_identity property (SQLAlchemyStateStore returns DB URL)
      2. Fallback: str(id(store)) for custom stores without the property

    Returns None if store is None.
    """
    if store is None:
        return None
    identity = getattr(store, "store_identity", None)
    if identity is not None:
        return str(identity)
    return str(id(store))


# ---------------------------------------------------------------------------
# Linking logic — called by the runner when creating a child run
# ---------------------------------------------------------------------------


def resolve_parent_link(
    parent_ctx: DelegationContext | None,
    child_store: object | None,
) -> tuple[str | None, int]:
    """Determine parent_run_id and delegation_level for a new run.

    Returns:
        (parent_run_id, delegation_level) to pass to create_run().
        parent_run_id is None when linking is not safe.
    """
    if parent_ctx is None:
        # Root run — no parent
        return None, 0

    delegation_level = parent_ctx.delegation_level + 1

    if child_store is None:
        # Child is ephemeral — no DB row, no linking needed
        return None, delegation_level

    if not parent_ctx.persisted:
        # Case 2: parent ephemeral, child persisted
        if "ephemeral_parent" not in parent_ctx.warned_mismatches:
            logger.warning(
                "Delegation link skipped for child of run %s: "
                "parent run is not persisted.",
                parent_ctx.run_id,
            )
            parent_ctx.warned_mismatches.add("ephemeral_parent")
        return None, delegation_level

    child_identity = get_store_identity(child_store)
    if parent_ctx.store_identity != child_identity:
        # Case 4: different stores
        if "different_store" not in parent_ctx.warned_mismatches:
            logger.warning(
                "Delegation link skipped for child of run %s: "
                "parent and child do not share the same state store.",
                parent_ctx.run_id,
            )
            parent_ctx.warned_mismatches.add("different_store")
        return None, delegation_level

    # Case 1: same store, both persisted — link
    return parent_ctx.run_id, delegation_level
