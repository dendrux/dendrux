"""Read the admitted root while a managed continuation stream is opening."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

from dendrux._lifecycle import PAUSE_STATUSES
from dendrux.errors import ManagedStateError, RunNotFoundError
from dendrux.runtime._pending import _accepts_tenant, _PendingReader
from dendrux.runtime._run_errors import _public_meta
from dendrux.runtime._run_pending import (
    _build_cached_result,
    _cached_record_result,
    _recovery_info,
    attach_pendings,
)
from dendrux.types import RunStatus

if TYPE_CHECKING:
    from datetime import datetime

    from dendrux.runtime._managed_pending import ManagedHop
    from dendrux.runtime.state import RunRecord, StateStore, TenantScope
    from dendrux.types import RunResult

    _Identity = tuple[str, str | None, str, datetime | None]

_RootView = Callable[["StateStore", str], Awaitable["RunResult"]]
root_reader: ContextVar[_RootView] = ContextVar("managed_root_reader", default=_build_cached_result)


class _AdmittedPendingReader(_PendingReader):
    def __init__(self, store: StateStore, tenant: str | None):
        super().__init__(store)
        self._tenant = tenant

    async def _read_hop(self, run_id: str, tenant_id: TenantScope) -> ManagedHop | None:
        # The ordinary v3 walk distinguishes foreign from missing children. A refusal only
        # reads the admitted owner's rows, including every intermediate hop.
        return await super()._read_hop(run_id, self._tenant)


def _supports_scoped_view(store: Any) -> bool:
    # Partial or older adapters can have some projections but fall back to ID-only
    # reads for others. The SQL hop, leaf and display readers must all keep the scope.
    sql = ("_session_factory", "json_get", "pause_projection")
    scoped = (
        "get_run",
        "get_pending_pause_summary",
        "get_active_pause_sequence",
        "get_approval_params",
        "get_managed_children",
        "get_pii_mapping",
        "get_replayed_tool_calls",
    )
    return all(callable(getattr(store, name, None)) for name in sql) and all(
        _accepts_tenant(getattr(store, name, None)) for name in scoped
    )


async def _current_record(
    store: Any, run_id: str, tenant: str | None, identity: _Identity
) -> RunRecord:
    current: RunRecord | None = await store.get_run(run_id, tenant_id=tenant)
    if (
        current is None
        or (current.id, current.tenant_id, current.agent_name, current.created_at) != identity
    ):
        raise RunNotFoundError(run_id)
    return current


async def _admitted_view(
    store: Any, snapshot: RunResult, tenant: str | None, identity: _Identity
) -> RunResult:
    if not _supports_scoped_view(store):
        return snapshot
    reader = _AdmittedPendingReader(store, tenant)
    run_id = snapshot.run_id
    try:
        view = await reader.facing(run_id, tenant_id=tenant)
    except ManagedStateError:
        # A move between the hop and its envelope can look like an invalid binding.
        current = await _current_record(store, run_id, tenant, identity)
        if current.status in PAUSE_STATUSES:
            raise
        view = None
    mapping = await store.get_pii_mapping(run_id, tenant_id=tenant)
    current = await _current_record(store, run_id, tenant, identity)
    recovery = None
    if current.last_reclaimed_at is not None:
        recovery = await _recovery_info(store, current)
        current = await _current_record(store, run_id, tenant, identity)
    result = _cached_record_result(current)
    if mapping:
        result.meta = _public_meta({"pii_mapping": mapping})
    result.recovery = recovery
    if view is not None and result.status in PAUSE_STATUSES:
        result.status = RunStatus(view.status)
        attach_pendings(result, view.pendings)
    return result


@contextmanager
def admitted_root(store: StateStore, record: RunRecord) -> Iterator[None]:
    # RunRecord is store-owned and may be mutable. Capture its identity and a detached
    # summary before routing awaits; legacy stores need no further reads on refusal.
    run_id, tenant = record.id, record.tenant_id
    identity = (run_id, tenant, record.agent_name, record.created_at)
    snapshot = _cached_record_result(record)
    owner = asyncio.current_task()
    previous = root_reader.get()

    async def read(request_store: StateStore, requested_id: str) -> RunResult:
        if request_store is store and requested_id == run_id and asyncio.current_task() is owner:
            return await _admitted_view(store, snapshot, tenant, identity)
        return await previous(request_store, requested_id)

    token = root_reader.set(read)
    try:
        yield
    finally:
        root_reader.reset(token)
