"""Read the admitted root while a managed continuation stream is opening."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from dendrux import _codes
from dendrux._lifecycle import PAUSE_STATUSES
from dendrux.errors import ManagedStateError, PersistenceError, RunNotFoundError
from dendrux.runtime._managed_pending import FacingView, is_managed_wait, needs_approval_display
from dendrux.runtime._pending import _accepts_tenant, _own_view, _PendingReader
from dendrux.runtime._run_errors import _public_meta, _safe_error
from dendrux.runtime._run_pending import (
    _cached_record_result,
    _recovery_info,
    attach_pendings,
)
from dendrux.types import RunEvent, RunEventType, RunStatus

if TYPE_CHECKING:
    from datetime import datetime

    from dendrux.runtime._managed_pending import ManagedHop
    from dendrux.runtime.state import RunRecord, StateStore, TenantScope
    from dendrux.types import RunResult

    _Identity = tuple[str, str | None, str, datetime | None]


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


class ProjectionUnavailableError(PersistenceError):
    def __init__(self, run_id: str):
        _codes.log_code(logging.getLogger(__name__), _codes.RESULT_READ_FAILED, run_id=run_id)
        super().__init__(retryable=True, code="result_read_failed")


def _require_reads(store: Any, run_id: str, *methods: str) -> None:
    if not all(_accepts_tenant(getattr(store, name, None)) for name in methods):
        raise ProjectionUnavailableError(run_id)


def _client_calls_complete(calls: Any) -> bool:
    return (
        isinstance(calls, list)
        and bool(calls)
        and all(
            isinstance(call, dict)
            and all(isinstance(call.get(key), str) and call[key] for key in ("id", "name"))
            for call in calls
        )
    )


class _PublicPendingReader(_PendingReader):
    async def facing(self, run_id: str, *, tenant_id: TenantScope) -> FacingView:
        summary_method = self._summary_method()
        if not _accepts_tenant(summary_method):
            raise ProjectionUnavailableError(run_id)
        summary = await summary_method(run_id, tenant_id=tenant_id)
        if summary is None:
            raise RunNotFoundError(run_id)
        if is_managed_wait(summary.status, summary.pause):
            # The legacy tree walk can require ID-only hop and envelope reads.
            raise ProjectionUnavailableError(run_id)
        if summary.status == RunStatus.WAITING_CLIENT_TOOL and not _client_calls_complete(
            summary.pause.get("pending_tool_calls")
        ):
            raise ProjectionUnavailableError(run_id)
        if needs_approval_display(summary):
            _require_reads(self._state, run_id, "get_active_pause_sequence", "get_approval_params")
        pending = await self._pending_of(run_id, tenant_id, summary)
        if summary.status in PAUSE_STATUSES and (pending is None or not pending.pause_id):
            raise ProjectionUnavailableError(run_id)
        return _own_view(summary.status, pending)


class AdmittedRoot:
    """A stream opening’s immutable root identity and its detached refusal summary."""

    def __init__(self, store: StateStore, record: RunRecord) -> None:
        self.store: Any = store
        self.identity = (record.id, record.tenant_id, record.agent_name, record.created_at)
        self.snapshot = _cached_record_result(record)

    async def refusal(self) -> RunResult:
        if not _supports_scoped_view(self.store):
            return self.snapshot
        return await self._result(_AdmittedPendingReader(self.store, self.identity[1]))

    async def refusal_event(self, exc: Exception) -> RunEvent:
        result = await self.refusal()
        result.error = _safe_error(exc)
        result.meta["error_type"] = type(exc).__name__
        return RunEvent(
            type=RunEventType.RUN_ERROR, run_id=result.run_id, run_result=result, error=result.error
        )

    async def latest(self) -> RunResult:
        if _supports_scoped_view(self.store):
            return await self._result(_AdmittedPendingReader(self.store, self.identity[1]))
        _require_reads(self.store, self.identity[0], "get_run", "get_pii_mapping")
        return await self._result(_PublicPendingReader(self.store))

    async def _result(self, reader: _PendingReader) -> RunResult:
        store, identity = self.store, self.identity
        run_id, tenant, _, _ = identity
        current = await _current_record(store, run_id, tenant, identity)
        for _ in range(2):
            revision = _revision(current)
            try:
                result, current = await self._read(reader, current)
            except ProjectionUnavailableError:
                current = await _current_record(store, run_id, tenant, identity)
                if _revision(current) == revision:
                    raise
                continue
            if result is not None and _revision(current) == revision:
                return result
        raise ProjectionUnavailableError(run_id)

    async def _read(
        self, reader: _PendingReader, current: RunRecord
    ) -> tuple[RunResult | None, RunRecord]:
        store, identity = self.store, self.identity
        run_id, tenant, _, _ = identity
        view = await self._facing(reader, current)
        mapping = await store.get_pii_mapping(run_id, tenant_id=tenant)
        current = await _current_record(store, run_id, tenant, identity)
        recovery = None
        if current.last_reclaimed_at is not None:
            _require_reads(store, run_id, "get_replayed_tool_calls")
            recovery = await _recovery_info(store, current)
            current = await _current_record(store, run_id, tenant, identity)
        if view is not None and not view.managed and view.status != current.status:
            return None, current
        result = _cached_record_result(current)
        if mapping:
            result.meta = _public_meta({"pii_mapping": mapping})
        result.recovery = recovery
        if view is not None and result.status in PAUSE_STATUSES:
            result.status = RunStatus(view.status)
            attach_pendings(result, view.pendings)
        return result, current

    async def _facing(self, reader: _PendingReader, current: RunRecord) -> FacingView | None:
        if current.status not in PAUSE_STATUSES:
            return None
        run_id, tenant, _, _ = self.identity
        try:
            return await reader.facing(run_id, tenant_id=tenant)
        except ManagedStateError:
            # A move between the hop and its envelope can look like an invalid binding.
            current = await _current_record(self.store, run_id, tenant, self.identity)
            if current.status in PAUSE_STATUSES:
                raise
            return None


def _revision(record: RunRecord) -> tuple[Any, ...]:
    return record.status, record.claim_epoch, record.status_changed_at, record.updated_at
