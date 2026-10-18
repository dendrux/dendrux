"""Public read facade over persisted agent-run state.

``RunStore`` is the stable contract developers use to build dashboards,
monitoring, billing, and any other observation layer on top of Dendrux.

It is read-only for run data. Write paths remain on
``runtime.state.SQLAlchemyStateStore``; the one exception is image blob
maintenance through ``RunStore.blobs`` (``purge()`` / ``collect_garbage()``).
Response types are frozen dataclasses defined in this module; runtime read
records stay internal and are converted at the facade boundary so runtime
internals can evolve without breaking the public contract.

Notes on the public dataclasses
-------------------------------
The dataclasses below are ``frozen=True`` — top-level fields cannot be
reassigned — but nested values (``dict``, ``list``) are normal mutable
JSON objects. Mutating them mutates the in-memory copy the caller
received, not the database. Treat them as read-only snapshots.

Engine ownership
----------------
``RunStore.from_database_url(url)`` creates and owns an engine; call
``await store.close()`` (or use ``async with``) to dispose it. The
``RunStore.from_engine(engine)`` constructor leaves engine ownership
with the caller and ``close()`` is a no-op for the engine in that case.

Example
-------
::

    from dendrux.store import RunStore

    async with RunStore.from_database_url("sqlite+aiosqlite:///dendrux.db") as store:
        async for event in store.stream_events(run_id):
            print(event.event_type, event.sequence_index)
"""

from __future__ import annotations

import asyncio
import functools
import inspect
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from dendrux.db._sqlite_sync import create_sqlite_schema
from dendrux.errors import BlobNotFoundError
from dendrux.media._ids import image_id
from dendrux.media._runtime import tenant_scope_for
from dendrux.runtime._blob_catalog import BlobServices, CatalogBlobAccessor
from dendrux.runtime.state import (
    UNSCOPED,
    SQLAlchemyStateStore,
    StateStore,
    TenantScope,
    _Unscoped,
)
from dendrux.types import ReplaySummary

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from datetime import datetime

    from sqlalchemy.ext.asyncio import AsyncEngine

    from dendrux.media import BlobStore


class RunNotFoundError(Exception):
    """Raised when streaming is requested for a non-existent run."""


# ---------------------------------------------------------------------------
# Public response dataclasses — frozen, slotted, stable contract
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RunSummary:
    """A run row suitable for list views.

    ``replay`` is set on runs created with ``agent.run(runs=...)``: which
    earlier runs were replayed (``runs_full`` + ``runs_text``, in order),
    and what reasoning was kept or dropped.
    """

    run_id: str
    agent_name: str
    status: str
    created_at: datetime | None
    updated_at: datetime | None
    iteration_count: int
    total_input_tokens: int
    total_output_tokens: int
    total_cache_read_tokens: int
    total_cache_creation_tokens: int
    total_cost_usd: float | None
    model: str | None
    parent_run_id: str | None
    delegation_level: int
    total_reasoning_tokens: int = 0
    replay: ReplaySummary | None = None


@dataclass(frozen=True, slots=True)
class RunUsage:
    """Token and cost totals summed live from ``llm_interactions``.

    Live means running and paused runs are counted, unlike the finalized
    ``total_*`` fields on :class:`RunSummary`. ``cost_usd`` is ``None`` when
    any call was unpriced (``unpriced_calls > 0``); ``known_cost_usd`` is
    the sum over the priced calls only. Cache and reasoning counts treat
    providers that did not report them as zero.
    """

    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    reasoning_tokens: int
    cost_usd: float | None
    known_cost_usd: float
    unpriced_calls: int
    llm_calls: int


@dataclass(frozen=True, slots=True)
class SubtreeNode:
    """One run in a subtree: its summary, depth below the root, and own live usage."""

    run: RunSummary
    depth: int
    usage: RunUsage


@dataclass(frozen=True, slots=True)
class RunSubtree:
    """A run, all its descendants, and live usage rolled up across them.

    ``nodes`` is in breadth-first order with the root first (``depth`` 0).
    ``usage`` is the sum over every node, so it matches the sum of the
    per-node ``usage`` values.
    """

    root_run_id: str
    nodes: list[SubtreeNode]
    usage: RunUsage

    @property
    def root(self) -> SubtreeNode:
        return self.nodes[0]


@dataclass(frozen=True, slots=True)
class RunDetail:
    """A single run's full detail — summary fields plus answer and error."""

    run_id: str
    agent_name: str
    status: str
    created_at: datetime | None
    updated_at: datetime | None
    iteration_count: int
    total_input_tokens: int
    total_output_tokens: int
    total_cache_read_tokens: int
    total_cache_creation_tokens: int
    total_cost_usd: float | None
    model: str | None
    strategy: str | None
    parent_run_id: str | None
    delegation_level: int
    input_data: dict[str, Any] | None
    answer: str | None
    error: str | None
    failure_reason: str | None
    total_reasoning_tokens: int = 0
    cancel_requested: bool = False
    replay: ReplaySummary | None = None
    pause_requested: bool = False


@dataclass(frozen=True, slots=True)
class RunNote:
    """A note submitted to a run with ``Agent.submit_note``.

    ``delivered_at`` / ``delivered_iteration`` are None until the runner
    delivered it to the model. A note still undelivered on a finished run
    was never seen by the model. ``metadata`` is the app data stored with
    the note (never shown to the model).
    """

    note_id: str
    run_id: str
    text: str
    metadata: Any
    created_at: datetime | None
    delivered_at: datetime | None
    delivered_iteration: int | None


@dataclass(frozen=True, slots=True)
class StoredEvent:
    """One row from the durable run_events log."""

    sequence_index: int
    iteration_index: int
    event_type: str
    correlation_id: str | None
    data: dict[str, Any] | None
    created_at: datetime | None


@dataclass(frozen=True, slots=True)
class LLMCall:
    """One LLM round-trip recorded during a run.

    ``id`` is the interaction's stable identifier. The matching
    ``llm.completed`` event in ``run_events`` carries it as
    ``StoredEvent.correlation_id``, so an event stream can be joined 1:1
    to full LLM payloads — mirroring ``tool.completed.correlation_id`` →
    ``ToolInvocation.tool_call_id``.
    """

    id: str
    iteration: int
    provider: str | None
    model: str | None
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cache_read_input_tokens: int | None
    cache_creation_input_tokens: int | None
    cost_usd: float | None
    duration_ms: int | None
    provider_request: dict[str, Any] | None
    provider_response: dict[str, Any] | None
    created_at: datetime | None
    reasoning_tokens: int | None = None
    reasoning: str | None = None


@dataclass(frozen=True, slots=True)
class ToolInvocation:
    """One tool invocation recorded during a run."""

    iteration: int
    tool_name: str
    tool_call_id: str
    provider_tool_call_id: str | None
    target: str
    params: dict[str, Any] | None
    result: dict[str, Any] | None
    success: bool
    error: str | None
    duration_ms: int | None
    created_at: datetime | None


@dataclass(frozen=True, slots=True)
class TraceEntry:
    """One entry in the conversation trace — user / assistant / tool message.

    ``phase`` is ``"commentary"`` or ``"final_answer"`` on assistant entries
    (None for other roles and for rows written before phases were recorded).
    """

    role: str
    content: str
    order_index: int
    meta: dict[str, Any] | None
    created_at: datetime | None
    phase: str | None = None


@dataclass(frozen=True, slots=True)
class PausePair:
    """A pause/resume cycle derived from ``run_events``.

    Never reads ``AgentRun.pause_data`` — that column is execution state,
    potentially unredacted, and cleared on finalize. Public pause history
    is built from the ``run.paused`` and ``run.resumed`` events only.
    """

    pause_sequence_index: int
    pause_at: datetime | None
    resume_sequence_index: int | None
    resume_at: datetime | None
    reason: str
    pending_tool_calls: list[dict[str, Any]]
    submitted_results: list[dict[str, Any]]
    user_input: str | None


# ---------------------------------------------------------------------------
# RunStore — the facade
# ---------------------------------------------------------------------------


class _InitOnFirstUse:
    """Wrap a state store so ``init`` runs, awaited, before its first call.

    ``init`` is idempotent, so concurrent first calls may each run it. A
    failed ``init`` is not remembered: the next call retries it. After
    :meth:`close`, calls raise; an ``init`` that finishes after close
    disposes the engine again (``dispose``) and raises.
    """

    def __init__(self, inner: Any, init: Any, dispose: Any) -> None:
        self._inner = inner
        self._init = init
        self._dispose = dispose
        self._done = False
        self.closed = False

    async def _ensure(self) -> None:
        from sqlalchemy.exc import InvalidRequestError

        if self.closed:
            raise InvalidRequestError("This RunStore is closed.")
        if self._done:
            return
        await self._init()
        if self.closed:
            await self._dispose()
            raise InvalidRequestError("This RunStore was closed during initialization.")
        self._done = True

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._inner, name)
        if not inspect.iscoroutinefunction(attr):
            return attr

        @functools.wraps(attr)
        async def guarded(*args: Any, **kwargs: Any) -> Any:
            await self._ensure()
            return await attr(*args, **kwargs)

        return guarded


class RunStore:
    """Read-only public facade over persisted run state.

    "Read-only" refers to run data: the store never writes runs, events or
    notes. It does write image blob state, and only through ``blobs``:
    ``blobs.purge()`` erases an image's bytes and ``blobs.collect_garbage()``
    deletes unreferenced blobs (and finishes interrupted purges/deletes).
    ``read_image(run_id, sha256)`` reads an image a run references.
    ``blob_store=`` must be the same external store the agents use, when
    they use one (rows record only ``storage="external"``, not which store).
    ``from_database_url`` on a SQLite URL does bring the *schema* of
    an older Dendrux SQLite file up to date once, before its first query
    (additive ``ALTER TABLE ... ADD COLUMN`` / index creation only, the same
    upgrade ``Agent`` runs), and creates the tables in a new file.

    Instantiate via ``RunStore.from_database_url(...)`` for the common case,
    or ``RunStore.from_engine(engine)`` when you already own a SQLAlchemy
    async engine. The primary constructor accepts any ``StateStore`` so
    alternative backends can be injected in tests.

    Engine lifecycle: ``from_database_url`` creates and owns the engine;
    ``close()`` (or ``async with``) disposes it. ``from_engine`` leaves
    engine ownership with the caller; ``close()`` is then a no-op.
    """

    def __init__(self, state_store: StateStore, *, blob_store: BlobStore | None = None) -> None:
        self._state = state_store
        self._owned_engine: AsyncEngine | None = None
        self._blob_services = BlobServices.for_state_store(state_store, blob_store=blob_store)

    @classmethod
    def from_engine(cls, engine: AsyncEngine, *, blob_store: BlobStore | None = None) -> RunStore:
        """Build a RunStore over an engine the caller owns and disposes."""
        return cls(SQLAlchemyStateStore(engine), blob_store=blob_store)

    @classmethod
    def from_database_url(
        cls, database_url: str, *, blob_store: BlobStore | None = None
    ) -> RunStore:
        """Build a RunStore from a SQLAlchemy async URL.

        The engine is created and owned by this RunStore. Call
        ``await store.close()`` or use ``async with`` to dispose it. For a
        SQLite URL the schema is brought up to date once before the first
        query (see the class docstring); an older file that cannot be
        upgraded raises :class:`~dendrux.errors.SqliteSchemaSyncError` from
        that first call.
        """
        from sqlalchemy.ext.asyncio import create_async_engine

        engine = create_async_engine(database_url)
        store: Any = SQLAlchemyStateStore(engine)
        if database_url.startswith("sqlite"):
            store = _InitOnFirstUse(
                store, functools.partial(create_sqlite_schema, engine), engine.dispose
            )
        instance = cls(store, blob_store=blob_store)
        instance._owned_engine = engine
        return instance

    @property
    def blobs(self) -> CatalogBlobAccessor:
        """Image blob reads and maintenance (``read``, ``info``, ``references``,
        ``purge``, ``collect_garbage``); ``tenant_id`` is the authorisation boundary."""
        return self._blob_services.accessor

    async def read_image(
        self, run_id: str, sha256: str, *, tenant_id: TenantScope = UNSCOPED
    ) -> bytes:
        """Verified bytes of an image the run references.

        The run must exist (and belong to ``tenant_id`` when given) and have
        a recorded reference to ``sha256`` in its own tenant scope.

        Tenant defaults differ on purpose, and match the rest of ``RunStore``:
        ``read_image`` is run-scoped, so omitting ``tenant_id`` (``UNSCOPED``)
        trusts the caller and reads in *the run's own* tenant scope (like
        ``get_run``); pass ``tenant_id`` to enforce a tenant. The blob-scoped
        ``blobs.read(sha256)`` has no run to take a scope from, so its default
        ``tenant_id=None`` means the untenanted scope only (``"-"``), never
        "any tenant"; pass the tenant explicitly for tenanted blobs.

        Raises:
            BlobNotFoundError: unknown run, other tenant, no reference, or the
                image is not live (absence and mismatch are indistinguishable).
            BlobIntegrityError: the stored bytes failed verification.
        """
        services = self._blob_services
        accessor = services.accessor
        catalog = services.catalog
        record = await self._state.get_run(run_id)
        missing = BlobNotFoundError(image_id(sha256 if isinstance(sha256, str) else ""))
        if record is None or (
            not isinstance(tenant_id, _Unscoped) and record.tenant_id != tenant_id
        ):
            raise missing
        run_scope = tenant_scope_for(record.tenant_id)
        if catalog is None:
            accessor._require_catalog()
            raise missing  # pragma: no cover - _require_catalog raised
        if not await catalog.run_references(run_scope, run_id, sha256):
            raise missing
        return await accessor.read(sha256, tenant_id=record.tenant_id)

    async def close(self) -> None:
        """Dispose any engine this store owns. No-op for caller-owned engines."""
        if isinstance(self._state, _InitOnFirstUse):
            self._state.closed = True
        if self._owned_engine is not None:
            await self._owned_engine.dispose()
            self._owned_engine = None

    async def __aenter__(self) -> RunStore:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    async def list_runs(
        self,
        *,
        status: str | list[str] | None = None,
        agent_name: str | None = None,
        parent_run_id: str | None = None,
        tenant_id: str | None = None,
        started_after: datetime | None = None,
        started_before: datetime | None = None,
        metadata_filter: dict[str, Any] | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[RunSummary]:
        """List runs with SQL-side filtering and pagination.

        ``metadata_filter`` matches runs where every given key/value pair
        is present at the top level of the run's ``metadata`` (the dev
        passes this to ``agent.run(..., metadata=...)``). The canonical
        chatbot pattern is ``metadata_filter={"thread_id": "..."}``;
        keys are restricted to ``[A-Za-z0-9_-]+``, nested paths are not
        supported.
        """
        records = await self._state.list_runs(
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
        return [_run_to_summary(r) for r in records]

    async def count_runs(
        self,
        *,
        status: str | list[str] | None = None,
        agent_name: str | None = None,
        parent_run_id: str | None = None,
        tenant_id: str | None = None,
        started_after: datetime | None = None,
        started_before: datetime | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> int:
        """Count runs matching the same filters as ``list_runs``."""
        return await self._state.count_runs(
            tenant_id=tenant_id,
            status=status,
            agent_name=agent_name,
            parent_run_id=parent_run_id,
            started_after=started_after,
            started_before=started_before,
            metadata_filter=metadata_filter,
        )

    async def get_run(self, run_id: str) -> RunDetail | None:
        """Fetch a single run's detail, or ``None`` if it does not exist."""
        record = await self._state.get_run(run_id)
        if record is None:
            return None
        return _run_to_detail(record)

    async def get_events(
        self,
        run_id: str,
        *,
        after_sequence_index: int | None = None,
        limit: int | None = None,
        event_types: list[str] | None = None,
        before: datetime | None = None,
    ) -> list[StoredEvent]:
        """Read events for a run in ``sequence_index`` order.

        ``after_sequence_index`` is an exclusive cursor. Pass the previous
        batch's last ``sequence_index`` to continue. ``event_types`` keeps
        only those types (an empty list matches nothing). ``before`` is an
        exclusive, timezone-aware upper bound on ``created_at``.
        """
        rows = await self._state.get_run_events(
            run_id,
            after_sequence_index=after_sequence_index,
            limit=limit,
            event_types=event_types,
            before=before,
        )
        return [_event_to_stored(r) for r in rows]

    async def get_subtree(self, run_id: str, *, tenant_id: TenantScope = UNSCOPED) -> RunSubtree:
        """Return a run, all its descendants, and live rolled-up usage.

        ``RunSubtree.usage`` is the authoritative usage: it is summed from
        ``llm_interactions`` at read time (root inclusive), so running and
        paused runs and not-yet-finalized children are counted. The
        ``total_*`` fields on each node's :class:`RunSummary` are the
        persisted snapshots on ``agent_runs`` and may lag or read 0 for
        unfinished runs. The reads are best-effort and not one snapshot
        transaction: a run that is still writing can make nodes and usage
        differ by a call or a child.

        Tenancy: omit ``tenant_id`` for a trusted, unscoped read (all
        descendants). Pass a string to scope to that tenant, or an explicit
        ``None`` to scope to untenanted runs only. The root must be in scope
        (otherwise it reads as missing) and so must every descendant; runs
        outside the scope are excluded together with everything below them.

        Raises :class:`RunNotFoundError` if the run does not exist.
        """
        from dendrux.runtime.state import require_store_method

        record = await require_store_method(self._state, "get_subtree")(run_id, tenant_id=tenant_id)
        if record is None:
            raise RunNotFoundError(f"Run '{run_id}' does not exist.")
        return RunSubtree(
            root_run_id=run_id,
            nodes=[
                SubtreeNode(
                    run=_run_to_summary(n.run), depth=n.depth, usage=_usage_to_public(n.usage)
                )
                for n in record.nodes
            ],
            usage=_usage_to_public(record.usage),
        )

    async def stream_events(
        self,
        run_id: str,
        *,
        after_sequence_index: int | None = None,
        poll_interval_s: float = 0.5,
    ) -> AsyncIterator[StoredEvent]:
        """Stream events as they land in the durable log.

        Raises :class:`RunNotFoundError` before the first yield if
        ``run_id`` does not exist. The caller decides when to stop —
        typically by breaking out of ``async for`` on a terminal
        ``event_type`` or by calling ``aclose()`` on the returned generator.
        """
        run = await self._state.get_run(run_id)
        if run is None:
            raise RunNotFoundError(f"Run '{run_id}' does not exist.")

        cursor = after_sequence_index
        while True:
            rows = await self._state.get_run_events(
                run_id,
                after_sequence_index=cursor,
                limit=100,
            )
            if rows:
                for row in rows:
                    event = _event_to_stored(row)
                    yield event
                    cursor = event.sequence_index
                # If the batch was full, there may be more buffered — keep
                # draining before we sleep.
                if len(rows) == 100:
                    continue
            await asyncio.sleep(poll_interval_s)

    async def get_llm_calls(
        self,
        run_id: str,
        *,
        iteration: int | None = None,
        limit: int | None = None,
        offset: int = 0,
        before: datetime | None = None,
    ) -> list[LLMCall]:
        """Return LLM interactions for a run. ``limit``/``offset`` push
        pagination into SQL. ``before`` is an exclusive, timezone-aware
        upper bound on ``created_at``."""
        rows = await self._state.get_llm_interactions(
            run_id,
            iteration_index=iteration,
            limit=limit,
            offset=offset,
            before=before,
        )
        return [_llm_to_public(r) for r in rows]

    async def get_tool_invocations(
        self,
        run_id: str,
        *,
        iteration: int | None = None,
        limit: int | None = None,
        offset: int = 0,
        before: datetime | None = None,
    ) -> list[ToolInvocation]:
        """Return tool invocations for a run. ``limit``/``offset`` push
        pagination into SQL. ``before`` is an exclusive, timezone-aware
        upper bound on ``created_at``."""
        rows = await self._state.get_tool_calls(
            run_id,
            iteration_index=iteration,
            limit=limit,
            offset=offset,
            before=before,
        )
        return [_tool_to_public(r) for r in rows]

    async def get_traces(
        self,
        run_id: str,
        *,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[TraceEntry]:
        """Return the conversation trace (user / assistant / tool messages)
        for a run, ordered by ``order_index``. ``limit``/``offset`` push
        pagination into SQL."""
        rows = await self._state.get_traces(run_id, limit=limit, offset=offset)
        return [_trace_to_public(r) for r in rows]

    async def get_pii_mapping(self, run_id: str) -> dict[str, str]:
        """Return the PII placeholder → real-value mapping for a run.

        The mapping is the audit key that links raw traces persisted in
        the DB to the placeholder view the LLM provider API actually saw.
        Use it to render the LLM-eye view from raw traces in dashboards
        or exports.

        Empty dict if the run has no mapping (no PII guardrail active,
        no entities detected, or the run does not exist).
        """
        mapping = await self._state.get_pii_mapping(run_id)
        return mapping or {}

    async def get_notes(
        self,
        run_id: str,
        *,
        undelivered_only: bool = False,
        tenant_id: TenantScope = UNSCOPED,
    ) -> list[RunNote]:
        """Inbox notes of a run in delivery order (``created_at``, then id).

        Tenancy: pass ``tenant_id`` (a string, or ``None`` for untenanted
        runs) to require that the run belongs to exactly that tenant;
        otherwise :class:`RunNotFoundError` is raised and nothing is read.
        Omitted means unscoped (trusted single-tenant use): an empty list
        for a run with no notes or that does not exist.
        """
        from dendrux.runtime.state import _Unscoped, require_store_method

        get_notes = require_store_method(self._state, "get_notes")
        if not isinstance(tenant_id, _Unscoped):
            record = await self._state.get_run(run_id)
            if record is None or record.tenant_id != tenant_id:
                raise RunNotFoundError(f"Run '{run_id}' does not exist.")
            rows = await get_notes(run_id, undelivered_only=undelivered_only, tenant_id=tenant_id)
        else:
            rows = await get_notes(run_id, undelivered_only=undelivered_only)
        return [
            RunNote(
                note_id=r.id,
                run_id=r.run_id,
                text=r.text,
                metadata=r.metadata,
                created_at=r.created_at,
                delivered_at=r.delivered_at,
                delivered_iteration=r.delivered_iteration,
            )
            for r in rows
        ]

    async def get_pauses(self, run_id: str) -> list[PausePair]:
        """Derive pause/resume cycles from ``run_events``.

        Pairing is nearest-neighbor: each ``run.paused`` pairs with the
        next ``run.resumed`` in ``sequence_index`` order. An unpaired
        ``run.paused`` at the end of the log represents an active pause
        (``resume_sequence_index`` is ``None``).

        Filters event_type at the SQL layer (only ``run.paused`` and
        ``run.resumed`` rows are fetched). Never reads
        ``AgentRun.pause_data``.
        """
        rows = await self._state.get_run_events(run_id, event_types=["run.paused", "run.resumed"])
        return _derive_pauses(rows)


# ---------------------------------------------------------------------------
# Runtime -> public type converters
# ---------------------------------------------------------------------------


def _run_to_summary(record: Any) -> RunSummary:
    return RunSummary(
        run_id=record.id,
        agent_name=record.agent_name,
        status=record.status,
        created_at=record.created_at,
        updated_at=record.updated_at,
        iteration_count=record.iteration_count,
        total_input_tokens=record.total_input_tokens,
        total_output_tokens=record.total_output_tokens,
        total_cache_read_tokens=record.total_cache_read_tokens,
        total_cache_creation_tokens=record.total_cache_creation_tokens,
        total_cost_usd=record.total_cost_usd,
        model=record.model,
        parent_run_id=record.parent_run_id,
        delegation_level=record.delegation_level,
        total_reasoning_tokens=record.total_reasoning_tokens,
        replay=_replay_of(record),
    )


def _usage_to_public(rollup: Any) -> RunUsage:
    return RunUsage(
        input_tokens=rollup.input_tokens,
        output_tokens=rollup.output_tokens,
        cache_read_tokens=rollup.cache_read_tokens,
        cache_write_tokens=rollup.cache_write_tokens,
        reasoning_tokens=rollup.reasoning_tokens,
        cost_usd=rollup.cost_usd,
        known_cost_usd=rollup.known_cost_usd,
        unpriced_calls=rollup.unpriced_calls,
        llm_calls=rollup.llm_calls,
    )


def _replay_of(record: Any) -> ReplaySummary | None:
    from dendrux.runtime.replay import REPLAY_META_KEY

    data = (getattr(record, "meta", None) or {}).get(REPLAY_META_KEY)
    return ReplaySummary.from_dict(data) if isinstance(data, dict) else None


def _run_to_detail(record: Any) -> RunDetail:
    return RunDetail(
        run_id=record.id,
        agent_name=record.agent_name,
        status=record.status,
        created_at=record.created_at,
        updated_at=record.updated_at,
        iteration_count=record.iteration_count,
        total_input_tokens=record.total_input_tokens,
        total_output_tokens=record.total_output_tokens,
        total_cache_read_tokens=record.total_cache_read_tokens,
        total_cache_creation_tokens=record.total_cache_creation_tokens,
        total_cost_usd=record.total_cost_usd,
        model=record.model,
        strategy=record.strategy,
        parent_run_id=record.parent_run_id,
        delegation_level=record.delegation_level,
        input_data=record.input_data,
        answer=record.answer,
        error=record.error,
        failure_reason=record.failure_reason,
        total_reasoning_tokens=record.total_reasoning_tokens,
        cancel_requested=bool(getattr(record, "cancel_requested", False)),
        replay=_replay_of(record),
        pause_requested=bool(getattr(record, "pause_requested", False)),
    )


def _event_to_stored(record: Any) -> StoredEvent:
    return StoredEvent(
        sequence_index=record.sequence_index,
        iteration_index=record.iteration_index,
        event_type=record.event_type,
        correlation_id=record.correlation_id,
        data=record.data,
        created_at=record.created_at,
    )


def _llm_to_public(record: Any) -> LLMCall:
    return LLMCall(
        id=record.id,
        iteration=record.iteration_index,
        provider=record.provider,
        model=record.model,
        input_tokens=record.input_tokens,
        output_tokens=record.output_tokens,
        total_tokens=record.input_tokens + record.output_tokens,
        cache_read_input_tokens=record.cache_read_input_tokens,
        cache_creation_input_tokens=record.cache_creation_input_tokens,
        cost_usd=record.cost_usd,
        duration_ms=record.duration_ms,
        provider_request=record.provider_request,
        provider_response=record.provider_response,
        created_at=record.created_at,
        reasoning_tokens=record.reasoning_tokens,
        reasoning=(record.semantic_response or {}).get("reasoning"),
    )


def _trace_to_public(record: Any) -> TraceEntry:
    return TraceEntry(
        role=record.role,
        content=record.content,
        order_index=record.order_index,
        meta=record.meta,
        created_at=record.created_at,
        phase=(record.meta or {}).get("phase"),
    )


def _tool_to_public(record: Any) -> ToolInvocation:
    iteration = record.iteration_index if record.iteration_index is not None else 0
    return ToolInvocation(
        iteration=iteration,
        tool_name=record.tool_name,
        tool_call_id=record.tool_call_id,
        provider_tool_call_id=record.provider_tool_call_id,
        target=record.target,
        params=record.params,
        result=record.result,
        success=record.success,
        error=record.error_message,
        duration_ms=record.duration_ms,
        created_at=record.created_at,
    )


def _derive_pauses(events: list[Any]) -> list[PausePair]:
    """Pair ``run.paused`` with the nearest following ``run.resumed``.

    Known limitation (hardening H-007): pairing is order-based, not
    correlation-id-based. An id-based fix will upgrade this function once
    correlation IDs are added to the pause/resume pair.
    """
    pauses: list[PausePair] = []
    pending: Any | None = None
    for ev in events:
        if ev.event_type == "run.paused":
            if pending is not None:
                pauses.append(_build_pause_pair(pending, None))
            pending = ev
        elif ev.event_type == "run.resumed":
            if pending is not None:
                pauses.append(_build_pause_pair(pending, ev))
                pending = None
    if pending is not None:
        pauses.append(_build_pause_pair(pending, None))
    return pauses


def _build_pause_pair(paused: Any, resumed: Any | None) -> PausePair:
    pdata = paused.data or {}
    rdata = (resumed.data or {}) if resumed is not None else {}
    return PausePair(
        pause_sequence_index=paused.sequence_index,
        pause_at=paused.created_at,
        resume_sequence_index=resumed.sequence_index if resumed is not None else None,
        resume_at=resumed.created_at if resumed is not None else None,
        reason=str(pdata.get("status", "")),
        pending_tool_calls=list(pdata.get("pending_tool_calls") or []),
        submitted_results=list(rdata.get("submitted_results") or []),
        user_input=rdata.get("user_input"),
    )


__all__ = [
    "LLMCall",
    "PausePair",
    "RunDetail",
    "RunNote",
    "RunNotFoundError",
    "RunStore",
    "RunSubtree",
    "RunSummary",
    "RunUsage",
    "StoredEvent",
    "SubtreeNode",
    "ToolInvocation",
    "TraceEntry",
]
