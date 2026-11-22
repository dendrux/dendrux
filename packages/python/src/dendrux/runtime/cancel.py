"""Cancel a persisted run and its subagents without building an Agent."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, cast

from dendrux import _codes
from dendrux.errors import TenantScopeRequiredError
from dendrux.runtime import _cancel, sweep
from dendrux.runtime._entry_checks import require_boolean
from dendrux.runtime._run_events import _shielded_cleanup
from dendrux.runtime._tenant_checks import _check_tenant_value
from dendrux.runtime.state import _Unscoped, unclaimed_store
from dendrux.types import ULID_CHARS, Budget, _validate_identifier

if TYPE_CHECKING:
    from dendrux.runtime.state import StateStore
    from dendrux.types import RunResult

logger = logging.getLogger(__name__)
_UNSCOPED_REFUSED = "cancel_run() requires tenant_id=<str or None>; UNSCOPED is not accepted."


async def cancel_run(
    run_id: str,
    *,
    tenant_id: str | None,
    database_url: str | None = None,
    state_store: StateStore | None = None,
    cascade: bool = True,
    max_nodes: int | None = None,
) -> RunResult:
    """Cancel ``run_id`` and its subagents without an ``Agent`` or a provider.

    This is the same operation as :meth:`dendrux.Agent.cancel_run`: a paused run is
    finalized ``CANCELLED`` at once; a running run gets a durable request and stops at
    its next checkpoint, in whichever process owns it. There is no task registry here,
    so in-flight model or tool calls are not interrupted immediately. A run that
    finishes before observing the request completes normally.

    Args:
        run_id: The run to cancel. A managed subagent run is refused; cancel its root.
        tenant_id: Required. The run's tenant, or ``None`` for an untenanted run. The
            cascade stays inside this tenant. ``UNSCOPED`` is refused.
        database_url: Open a temporary store for this call, then dispose it.
        state_store: Use this store. It stays owned by the caller and is never closed.
            Prefer a shared store for frequent calls, such as a server route.
        cascade: Also cancel legacy (delegated) descendants. Managed subagents are
            always included.
        max_nodes: Root-inclusive traversal bound. Default
            ``Budget.max_subtree_nodes`` (10,000). Reaching it sets
            ``meta["cascade_truncated"]``. There is no cursor: a repeat call with the
            same bound visits the same runs, so raise ``max_nodes`` to cover the rest.

    Returns:
        The same :class:`RunResult` shape as ``Agent.cancel_run``.

    Raises:
        ValueError: Not exactly one of ``database_url`` and ``state_store``, or an
            invalid ``tenant_id``, ``cascade`` or ``max_nodes``.
        TenantScopeRequiredError: ``tenant_id`` is ``UNSCOPED``.
        RunNotFoundError: ``run_id`` does not exist within ``tenant_id``.
        ManagedChildControlError: ``run_id`` is a managed subagent run.
    """
    sweep._validate_source(database_url, state_store, None)
    _validate_identifier(run_id, "run_id", maximum=ULID_CHARS)
    require_boolean(cascade, "cascade")
    bound = _resolve_bound(max_nodes)
    _require_tenant(run_id, tenant_id)
    if database_url is None:
        # _validate_source guarantees a caller store here; strip any execution claim so
        # no bound epoch reaches the canceller's writes.
        store = cast("StateStore", unclaimed_store(state_store))
        return await _cancel.cancel_run(
            store, run_id, tenant_id=tenant_id, cascade=cascade, max_nodes=bound
        )
    store, engine = await sweep._create_temp_store(database_url)
    try:
        return await _cancel.cancel_run(
            store, run_id, tenant_id=tenant_id, cascade=cascade, max_nodes=bound
        )
    finally:
        # Finish disposal even if the caller cancels meanwhile, then re-raise that cancel.
        await _shielded_cleanup(engine.dispose)


def _resolve_bound(max_nodes: object) -> int:
    if max_nodes is None:
        return Budget.max_subtree_nodes
    if isinstance(max_nodes, bool) or not isinstance(max_nodes, int) or max_nodes < 1:
        raise ValueError("max_nodes must be a positive integer or None.")
    return max_nodes


def _require_tenant(run_id: str, tenant_id: object) -> None:
    """Refuse the administrative ``UNSCOPED`` scope, then validate the tenant value."""
    if isinstance(tenant_id, _Unscoped):
        _codes.log_code(logger, _codes.TENANT_SCOPE_REQUIRED, run_id=run_id, entry="cancel_run")
        raise TenantScopeRequiredError("cancel_run", message=_UNSCOPED_REFUSED)
    _check_tenant_value("cancel_run", tenant_id, new_run=True)
