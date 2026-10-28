"""Custom ``StateStore`` fixtures for the 0.6 capability work (phase 0c).

Today ``StoreCapabilities`` does not exist in ``dendrux``. The declaration used by
fixtures (c) and (e) is the plain frozen dataclass ``StoreCapabilities`` defined here, with
the field set frozen in ``owner-docs/build-0.6.0/contracts.md`` section 2. Phase 1 switches
the fixtures to the real type by replacing the dataclass below with
``from dendrux.runtime.state import StoreCapabilities`` (one import line); nothing else in
this module reads its fields by anything but name.

Every fixture wraps a real ``SQLAlchemyStateStore`` (usually over in-memory SQLite) and
forwards each call, so the whole conformance matrix can run through it; it records every
call as ``(method, keyword names)`` in ``calls`` so phase 1 can assert that a legacy store
never receives a keyword argument that did not exist in 0.5.

====  ==========================================  =======================================
id    class                                       what it is
====  ==========================================  =======================================
(a)   ``LegacyInMemoryStore``                      the 0.5 protocol methods and exactly
                                                   their 0.5 signatures, no ``capabilities``
                                                   attribute, no optional methods; a
                                                   custom store written against 0.4/0.5
(b)   ``UndeclaredSameMethodsStore``               the same method names and the *current*
                                                   signatures of ``SQLAlchemyStateStore``
                                                   (they follow it into 0.6), no
                                                   declaration: must be treated as legacy
(c)   ``FencedButRejectsEpochStore``               declares ``fenced_writes=True`` but its
                                                   write methods have the 0.5 signatures
                                                   (no ``claim_epoch``): construction-time
                                                   failure in phase 1
(d)   ``KwargsLegacyStore``                        every protocol method is
                                                   ``(*args, **kwargs)``, no declaration:
                                                   legacy (a permissive signature must not
                                                   count as capable)
(e)   ``PartialFencedRetryStore``                  declares ``fenced_writes`` and
                                                   ``retry_ids`` only; write methods accept
                                                   ``claim_epoch`` and the minted-id
                                                   keywords (dropped when the wrapped store
                                                   does not know them yet): format 1, fenced
====  ==========================================  =======================================

Run through the conformance suite now: (a) and (d). Wired in phase 1: (b), (c), (e)
(``tests/conformance/test_store_fixtures.py`` pins their construction and surface today).
"""

from __future__ import annotations

import functools
import inspect
from dataclasses import dataclass
from typing import Any

from dendrux.runtime.state import UNSCOPED, SQLAlchemyStateStore


@dataclass(frozen=True)
class StoreCapabilities:
    """Fixture-side copy of the 0.6 declaration (contracts.md section 2)."""

    version: int = 1
    fenced_writes: bool = False
    retry_ids: bool = False
    checkpointing: bool = False
    atomic_completion: bool = False
    recovery: bool = False
    notify: bool = False


#: Every method of the 0.5.0 ``StateStore`` protocol, in definition order.
PROTOCOL_0_5: tuple[str, ...] = (
    "create_run",
    "save_trace",
    "save_tool_call",
    "save_usage",
    "save_llm_interaction",
    "get_llm_interactions",
    "finalize_run",
    "finalize_run_if_status_in",
    "request_cancel",
    "is_cancel_requested",
    "get_run_control",
    "request_pause",
    "try_pause_run",
    "pause_with_event",
    "finalize_with_event",
    "add_note",
    "get_notes",
    "mark_notes_delivered",
    "get_next_event_sequence",
    "pause_run",
    "get_pause_state",
    "get_pii_mapping",
    "claim_paused_run",
    "submit_and_claim",
    "get_run",
    "get_traces",
    "get_tool_calls",
    "save_run_event",
    "get_run_events",
    "list_runs",
    "count_runs",
    "touch_progress",
    "sweep_stale_runs",
    "sweep_abandoned_runs",
    "get_delegation_info",
    "get_subtree",
    "subtree_usage",
    "run_usage",
)


#: Claim methods whose win used to return the epoch ``int`` (0.6 returns ``ClaimResult``).
CLAIM_METHODS = ("claim_paused_run", "submit_and_claim")


def legacy_claim_shape(result: Any) -> Any:
    """What a pre-0.6 store returns for a claim: the epoch ``int``, or ``False`` for a lost CAS.

    Works on today's ``int`` and on 0.6's ``ClaimResult`` (which carries ``claim_epoch``).
    """
    if result is False or isinstance(result, int):
        return result
    return result.claim_epoch


class _Forwarding:
    """Shared plumbing: wrap ``inner``, record ``(method, keyword names)`` per call.

    ``claim_shape``: ``"int"`` makes the claim methods return the pre-0.6 shape (epoch ``int``
    or ``False``), as a legacy store written against 0.5 does; ``"real"`` returns whatever the
    wrapped store returns (a lost CAS is ``False`` in both shapes). Fixtures (a) and (d), the
    legacy ones, use ``"int"``; (b), (c) and (e) use ``"real"``.
    """

    claim_shape = "real"

    def __init__(self, inner: SQLAlchemyStateStore) -> None:
        self._inner = inner
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    @property
    def wrapped(self) -> SQLAlchemyStateStore:
        return self._inner

    def kwarg_names(self, method: str) -> set[str]:
        """Keyword names ever passed to ``method`` (for 'no new keyword reached me' checks)."""
        return {k for name, keys in self.calls if name == method for k in keys}

    async def _forward(self, method: str, /, *args: Any, **kwargs: Any) -> Any:
        self.calls.append((method, tuple(kwargs)))
        result = await getattr(self._inner, method)(*args, **kwargs)
        if self.claim_shape == "int" and method in CLAIM_METHODS:
            return legacy_claim_shape(result)
        return result


class _Protocol050(_Forwarding):
    """The 0.5.0 protocol methods, exact 0.5 signatures, forwarding to the wrapped store."""

    # --- generated from the 0.5.0 protocol; edit only to follow a deliberate 0.5 change ---
    async def create_run(
        self,
        run_id: Any,
        agent_name: Any,
        *,
        input_data: Any = None,
        model: Any = None,
        strategy: Any = None,
        parent_run_id: Any = None,
        delegation_level: Any = 0,
        tenant_id: Any = None,
        meta: Any = None,
        idempotency_key: Any = None,
        idempotency_fingerprint: Any = None,
        retry_of_run_id: Any = None,
        blob_refs: Any = None,
        blob_refs_scope: Any = None,
    ) -> Any:
        return await self._forward(
            "create_run",
            run_id=run_id,
            agent_name=agent_name,
            input_data=input_data,
            model=model,
            strategy=strategy,
            parent_run_id=parent_run_id,
            delegation_level=delegation_level,
            tenant_id=tenant_id,
            meta=meta,
            idempotency_key=idempotency_key,
            idempotency_fingerprint=idempotency_fingerprint,
            retry_of_run_id=retry_of_run_id,
            blob_refs=blob_refs,
            blob_refs_scope=blob_refs_scope,
        )

    async def save_trace(
        self,
        run_id: Any,
        role: Any,
        content: Any,
        *,
        order_index: Any,
        meta: Any = None,
        blob_refs: Any = None,
        blob_refs_scope: Any = None,
    ) -> Any:
        return await self._forward(
            "save_trace",
            run_id=run_id,
            role=role,
            content=content,
            order_index=order_index,
            meta=meta,
            blob_refs=blob_refs,
            blob_refs_scope=blob_refs_scope,
        )

    async def save_tool_call(
        self,
        run_id: Any,
        *,
        tool_call_id: Any,
        provider_tool_call_id: Any,
        tool_name: Any,
        target: Any,
        params: Any,
        result_payload: Any,
        success: Any,
        duration_ms: Any,
        iteration_index: Any,
        error_message: Any = None,
        meta: Any = None,
        blob_refs: Any = None,
        blob_refs_scope: Any = None,
    ) -> Any:
        return await self._forward(
            "save_tool_call",
            run_id=run_id,
            tool_call_id=tool_call_id,
            provider_tool_call_id=provider_tool_call_id,
            tool_name=tool_name,
            target=target,
            params=params,
            result_payload=result_payload,
            success=success,
            duration_ms=duration_ms,
            iteration_index=iteration_index,
            error_message=error_message,
            meta=meta,
            blob_refs=blob_refs,
            blob_refs_scope=blob_refs_scope,
        )

    async def save_usage(
        self,
        run_id: Any,
        *,
        iteration_index: Any,
        usage: Any,
        model: Any = None,
        provider: Any = None,
        duration_ms: Any = None,
        meta: Any = None,
        usage_id: Any = None,
    ) -> Any:
        return await self._forward(
            "save_usage",
            run_id=run_id,
            iteration_index=iteration_index,
            usage=usage,
            model=model,
            provider=provider,
            duration_ms=duration_ms,
            meta=meta,
            usage_id=usage_id,
        )

    async def save_llm_interaction(
        self,
        run_id: Any,
        *,
        iteration_index: Any,
        usage: Any,
        model: Any = None,
        provider: Any = None,
        duration_ms: Any = None,
        semantic_request: Any = None,
        semantic_response: Any = None,
        provider_request: Any = None,
        provider_response: Any = None,
        guardrail_findings: Any = None,
        interaction_id: Any = None,
    ) -> Any:
        return await self._forward(
            "save_llm_interaction",
            run_id=run_id,
            iteration_index=iteration_index,
            usage=usage,
            model=model,
            provider=provider,
            duration_ms=duration_ms,
            semantic_request=semantic_request,
            semantic_response=semantic_response,
            provider_request=provider_request,
            provider_response=provider_response,
            guardrail_findings=guardrail_findings,
            interaction_id=interaction_id,
        )

    async def get_llm_interactions(
        self,
        run_id: Any,
        *,
        iteration_index: Any = None,
        limit: Any = None,
        offset: Any = 0,
        before: Any = None,
    ) -> Any:
        return await self._forward(
            "get_llm_interactions",
            run_id=run_id,
            iteration_index=iteration_index,
            limit=limit,
            offset=offset,
            before=before,
        )

    async def finalize_run(
        self,
        run_id: Any,
        *,
        status: Any,
        answer: Any = None,
        error: Any = None,
        iteration_count: Any = 0,
        total_usage: Any = None,
        expected_current_status: Any = None,
        pii_mapping: Any = None,
        claim_epoch: Any = None,
        error_info: Any = None,
        failure_reason: Any = None,
    ) -> Any:
        return await self._forward(
            "finalize_run",
            run_id=run_id,
            status=status,
            answer=answer,
            error=error,
            iteration_count=iteration_count,
            total_usage=total_usage,
            expected_current_status=expected_current_status,
            pii_mapping=pii_mapping,
            claim_epoch=claim_epoch,
            error_info=error_info,
            failure_reason=failure_reason,
        )

    async def finalize_run_if_status_in(
        self,
        run_id: Any,
        *,
        status: Any,
        allowed_current_statuses: Any,
        answer: Any = None,
        error: Any = None,
        iteration_count: Any = None,
        total_usage: Any = None,
        pii_mapping: Any = None,
        failure_reason: Any = None,
        claim_epoch: Any = None,
    ) -> Any:
        return await self._forward(
            "finalize_run_if_status_in",
            run_id=run_id,
            status=status,
            allowed_current_statuses=allowed_current_statuses,
            answer=answer,
            error=error,
            iteration_count=iteration_count,
            total_usage=total_usage,
            pii_mapping=pii_mapping,
            failure_reason=failure_reason,
            claim_epoch=claim_epoch,
        )

    async def request_cancel(self, run_id: Any) -> Any:
        return await self._forward("request_cancel", run_id=run_id)

    async def is_cancel_requested(self, run_id: Any) -> Any:
        return await self._forward("is_cancel_requested", run_id=run_id)

    async def get_run_control(self, run_id: Any) -> Any:
        return await self._forward("get_run_control", run_id=run_id)

    async def request_pause(self, run_id: Any, *, reason: Any = None) -> Any:
        return await self._forward("request_pause", run_id=run_id, reason=reason)

    async def try_pause_run(
        self,
        run_id: Any,
        *,
        status: Any,
        pause_data: Any,
        iteration_count: Any = None,
        pii_mapping: Any = None,
        total_usage: Any = None,
        claim_epoch: Any = None,
    ) -> Any:
        return await self._forward(
            "try_pause_run",
            run_id=run_id,
            status=status,
            pause_data=pause_data,
            iteration_count=iteration_count,
            pii_mapping=pii_mapping,
            total_usage=total_usage,
            claim_epoch=claim_epoch,
        )

    async def pause_with_event(
        self,
        run_id: Any,
        *,
        status: Any,
        pause_data: Any,
        event_data: Any = None,
        iteration_count: Any = None,
        pii_mapping: Any = None,
        total_usage: Any = None,
        claim_epoch: Any = None,
    ) -> Any:
        return await self._forward(
            "pause_with_event",
            run_id=run_id,
            status=status,
            pause_data=pause_data,
            event_data=event_data,
            iteration_count=iteration_count,
            pii_mapping=pii_mapping,
            total_usage=total_usage,
            claim_epoch=claim_epoch,
        )

    async def finalize_with_event(
        self,
        run_id: Any,
        *,
        status: Any,
        allowed_current_statuses: Any,
        event_type: Any,
        event_data: Any = None,
        answer: Any = None,
        error: Any = None,
        iteration_count: Any = None,
        total_usage: Any = None,
        pii_mapping: Any = None,
        failure_reason: Any = None,
        claim_epoch: Any = None,
    ) -> Any:
        return await self._forward(
            "finalize_with_event",
            run_id=run_id,
            status=status,
            allowed_current_statuses=allowed_current_statuses,
            event_type=event_type,
            event_data=event_data,
            answer=answer,
            error=error,
            iteration_count=iteration_count,
            total_usage=total_usage,
            pii_mapping=pii_mapping,
            failure_reason=failure_reason,
            claim_epoch=claim_epoch,
        )

    async def add_note(
        self,
        run_id: Any,
        *,
        note_id: Any,
        text: Any,
        metadata: Any = None,
        tenant_id: Any = UNSCOPED,
        max_notes: Any = None,
    ) -> Any:
        return await self._forward(
            "add_note",
            run_id=run_id,
            note_id=note_id,
            text=text,
            metadata=metadata,
            tenant_id=tenant_id,
            max_notes=max_notes,
        )

    async def get_notes(
        self, run_id: Any, *, undelivered_only: Any = False, tenant_id: Any = UNSCOPED
    ) -> Any:
        return await self._forward(
            "get_notes", run_id=run_id, undelivered_only=undelivered_only, tenant_id=tenant_id
        )

    async def mark_notes_delivered(self, run_id: Any, note_ids: Any, *, iteration: Any) -> Any:
        return await self._forward(
            "mark_notes_delivered", run_id=run_id, note_ids=note_ids, iteration=iteration
        )

    async def get_next_event_sequence(self, run_id: Any) -> Any:
        return await self._forward("get_next_event_sequence", run_id=run_id)

    async def pause_run(
        self,
        run_id: Any,
        *,
        status: Any,
        pause_data: Any,
        iteration_count: Any = None,
        pii_mapping: Any = None,
        total_usage: Any = None,
        claim_epoch: Any = None,
    ) -> Any:
        return await self._forward(
            "pause_run",
            run_id=run_id,
            status=status,
            pause_data=pause_data,
            iteration_count=iteration_count,
            pii_mapping=pii_mapping,
            total_usage=total_usage,
            claim_epoch=claim_epoch,
        )

    async def get_pause_state(self, run_id: Any) -> Any:
        return await self._forward("get_pause_state", run_id=run_id)

    async def get_pii_mapping(self, run_id: Any) -> Any:
        return await self._forward("get_pii_mapping", run_id=run_id)

    async def claim_paused_run(
        self,
        run_id: Any,
        *,
        expected_status: Any,
        expected_pause_id: Any = None,
        validate: Any = None,
    ) -> Any:
        return await self._forward(
            "claim_paused_run",
            run_id=run_id,
            expected_status=expected_status,
            expected_pause_id=expected_pause_id,
            validate=validate,
        )

    async def submit_and_claim(
        self,
        run_id: Any,
        *,
        expected_status: Any,
        submitted_data: Any,
        validate: Any = None,
        blob_refs: Any = None,
        blob_refs_scope: Any = None,
    ) -> Any:
        return await self._forward(
            "submit_and_claim",
            run_id=run_id,
            expected_status=expected_status,
            submitted_data=submitted_data,
            validate=validate,
            blob_refs=blob_refs,
            blob_refs_scope=blob_refs_scope,
        )

    async def get_run(self, run_id: Any) -> Any:
        return await self._forward("get_run", run_id=run_id)

    async def get_traces(self, run_id: Any, *, limit: Any = None, offset: Any = 0) -> Any:
        return await self._forward("get_traces", run_id=run_id, limit=limit, offset=offset)

    async def get_tool_calls(
        self,
        run_id: Any,
        *,
        iteration_index: Any = None,
        limit: Any = None,
        offset: Any = 0,
        before: Any = None,
    ) -> Any:
        return await self._forward(
            "get_tool_calls",
            run_id=run_id,
            iteration_index=iteration_index,
            limit=limit,
            offset=offset,
            before=before,
        )

    async def save_run_event(
        self,
        run_id: Any,
        *,
        event_type: Any,
        iteration_index: Any = 0,
        correlation_id: Any = None,
        data: Any = None,
    ) -> Any:
        return await self._forward(
            "save_run_event",
            run_id=run_id,
            event_type=event_type,
            iteration_index=iteration_index,
            correlation_id=correlation_id,
            data=data,
        )

    async def get_run_events(
        self,
        run_id: Any,
        *,
        after_sequence_index: Any = None,
        limit: Any = None,
        event_types: Any = None,
        before: Any = None,
    ) -> Any:
        return await self._forward(
            "get_run_events",
            run_id=run_id,
            after_sequence_index=after_sequence_index,
            limit=limit,
            event_types=event_types,
            before=before,
        )

    async def list_runs(
        self,
        *,
        limit: Any = 50,
        offset: Any = 0,
        tenant_id: Any = UNSCOPED,
        status: Any = None,
        agent_name: Any = None,
        parent_run_id: Any = None,
        started_after: Any = None,
        started_before: Any = None,
        metadata_filter: Any = None,
    ) -> Any:
        return await self._forward(
            "list_runs",
            limit=limit,
            offset=offset,
            tenant_id=tenant_id,
            status=status,
            agent_name=agent_name,
            parent_run_id=parent_run_id,
            started_after=started_after,
            started_before=started_before,
            metadata_filter=metadata_filter,
        )

    async def count_runs(
        self,
        *,
        tenant_id: Any = UNSCOPED,
        status: Any = None,
        agent_name: Any = None,
        parent_run_id: Any = None,
        started_after: Any = None,
        started_before: Any = None,
        metadata_filter: Any = None,
    ) -> Any:
        return await self._forward(
            "count_runs",
            tenant_id=tenant_id,
            status=status,
            agent_name=agent_name,
            parent_run_id=parent_run_id,
            started_after=started_after,
            started_before=started_before,
            metadata_filter=metadata_filter,
        )

    async def touch_progress(self, run_id: Any) -> Any:
        return await self._forward("touch_progress", run_id=run_id)

    async def sweep_stale_runs(
        self, older_than: Any, *, batch_size: Any = 100, max_runs: Any = 10000
    ) -> Any:
        return await self._forward(
            "sweep_stale_runs", older_than=older_than, batch_size=batch_size, max_runs=max_runs
        )

    async def sweep_abandoned_runs(
        self, older_than: Any, *, batch_size: Any = 100, max_runs: Any = 10000
    ) -> Any:
        return await self._forward(
            "sweep_abandoned_runs", older_than=older_than, batch_size=batch_size, max_runs=max_runs
        )

    async def get_delegation_info(
        self, run_id: Any, *, max_rows: Any = 1000, max_depth: Any = 32
    ) -> Any:
        return await self._forward(
            "get_delegation_info", run_id=run_id, max_rows=max_rows, max_depth=max_depth
        )

    async def get_subtree(self, run_id: Any, *, tenant_id: Any = UNSCOPED) -> Any:
        return await self._forward("get_subtree", run_id=run_id, tenant_id=tenant_id)

    async def subtree_usage(self, run_id: Any) -> Any:
        return await self._forward("subtree_usage", run_id=run_id)

    async def run_usage(self, run_id: Any) -> Any:
        return await self._forward("run_usage", run_id=run_id)


class LegacyInMemoryStore(_Protocol050):
    """(a) A custom store written against the 0.5 protocol: no declaration, no extras.

    "In-memory" because the wrapped store is the in-memory SQLite one the unit suite uses;
    the point of the fixture is the surface (exactly the 0.5 protocol signatures, nothing
    optional such as ``release_run_claim`` or ``update_run_meta``), not the storage.
    """

    claim_shape = "int"


class FencedButRejectsEpochStore(_Protocol050):
    """(c) Declares ``fenced_writes`` but its insert methods take no ``claim_epoch``."""

    capabilities = StoreCapabilities(fenced_writes=True)


class KwargsLegacyStore(_Forwarding):
    """(d) Same names, every method ``(*args, **kwargs)``, no declaration."""

    claim_shape = "int"


def _kwargs_method(name: str) -> Any:
    async def method(self: _Forwarding, *args: Any, **kwargs: Any) -> Any:
        return await self._forward(name, *args, **kwargs)

    method.__name__ = name
    return method


for _name in PROTOCOL_0_5:
    setattr(KwargsLegacyStore, _name, _kwargs_method(_name))


class UndeclaredSameMethodsStore(_Forwarding):
    """(b) Same method names *and* current signatures as ``SQLAlchemyStateStore``, undeclared."""


def _wrapped_method(name: str) -> Any:
    target = getattr(SQLAlchemyStateStore, name)

    @functools.wraps(target)
    async def method(self: _Forwarding, *args: Any, **kwargs: Any) -> Any:
        return await self._forward(name, *args, **kwargs)

    return method


for _name in PROTOCOL_0_5:
    setattr(UndeclaredSameMethodsStore, _name, _wrapped_method(_name))


#: Keyword parameters (0.6) that fixture (e) accepts on its write methods.
_PARTIAL_EXTRAS: dict[str, tuple[str, ...]] = {
    "save_trace": ("claim_epoch", "trace_id"),
    "save_tool_call": ("claim_epoch", "row_id"),
    "save_usage": ("claim_epoch", "usage_id"),
    "save_llm_interaction": ("claim_epoch", "interaction_id"),
    "save_run_event": ("claim_epoch", "event_id"),
    "add_note": ("claim_epoch",),
    "mark_notes_delivered": ("claim_epoch",),
    "touch_progress": ("claim_epoch",),
}


class PartialFencedRetryStore(_Forwarding):
    """(e) Declares ``fenced_writes`` and ``retry_ids`` only; other methods are the 0.5 ones."""

    capabilities = StoreCapabilities(fenced_writes=True, retry_ids=True)

    async def _forward(self, method: str, /, *args: Any, **kwargs: Any) -> Any:
        self.calls.append((method, tuple(kwargs)))
        target = getattr(self._inner, method)
        accepted = inspect.signature(target).parameters
        if not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in accepted.values()):
            kwargs = {k: v for k, v in kwargs.items() if k in accepted}
        return await target(*args, **kwargs)


def _partial_method(name: str) -> Any:
    target = getattr(SQLAlchemyStateStore, name)
    extras = _PARTIAL_EXTRAS.get(name, ())
    base = inspect.signature(target)
    params = list(base.parameters.values())
    for extra in extras:
        if extra not in base.parameters:
            params.append(inspect.Parameter(extra, inspect.Parameter.KEYWORD_ONLY, default=None))
    # keyword-only parameters must follow positional ones; var-keyword stays last
    params.sort(key=lambda p: (p.kind, 0))
    signature = base.replace(parameters=params)

    async def method(self: _Forwarding, *args: Any, **kwargs: Any) -> Any:
        return await self._forward(name, *args, **kwargs)

    method.__name__ = name
    method.__signature__ = signature  # type: ignore[attr-defined]
    return method


for _name in PROTOCOL_0_5:
    setattr(PartialFencedRetryStore, _name, _partial_method(_name))

#: Fixture id -> class, in the order the brief lists them.
STORE_FIXTURES: dict[str, type[_Forwarding]] = {
    "a_legacy_in_memory": LegacyInMemoryStore,
    "b_undeclared_same_methods": UndeclaredSameMethodsStore,
    "c_fenced_rejects_epoch": FencedButRejectsEpochStore,
    "d_kwargs_legacy": KwargsLegacyStore,
    "e_partial_fenced_retry": PartialFencedRetryStore,
}
