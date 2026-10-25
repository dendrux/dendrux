"""Public read facade over persisted agent-run state.

``RunStore`` is the stable contract developers use to build inspection tools,
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
import dataclasses
import functools
import inspect
import re
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from dendrux._sentinel import _OmittedType
from dendrux.db._sqlite_sync import create_sqlite_schema
from dendrux.errors import BlobNotFoundError, RunNotFoundError
from dendrux.media._ids import image_id
from dendrux.media._runtime import tenant_scope_for
from dendrux.runtime._blob_catalog import BlobServices, CatalogBlobAccessor
from dendrux.runtime._pending import (
    _KIND_BY_STATUS as _KIND_BY_STATUS,
)
from dendrux.runtime._pending import (
    PendingKind as PendingKind,
)
from dendrux.runtime._pending import (
    PendingPause as PendingPause,
)
from dendrux.runtime._pending import (
    PendingToolCall as PendingToolCall,
)
from dendrux.runtime._pending import (
    _pending_from as _pending_from,
)
from dendrux.runtime._pending import (
    _PendingReader,
    check_metadata_filter,
    check_read_bool,
    check_read_datetime,
    check_read_int,
    check_read_names,
    check_read_scope,
    check_read_text,
    check_run_id,
)
from dendrux.runtime.state import (
    MAX_LIST_OFFSET,
    UNSCOPED,
    SQLAlchemyStateStore,
    StateStore,
    TenantScope,
    _to_aware_utc,
    _Unscoped,
)
from dendrux.store._follow import (
    PAUSE_STATUSES as PAUSE_STATUSES,
)
from dendrux.store._follow import (
    EventFeed,
    Follow,
    FollowCapacityError,
    FollowEnd,
    FollowHeartbeat,
    FollowItem,
    redact_event,
)
from dendrux.store._follow import (
    approval_calls as approval_calls,
)
from dendrux.store._redact import (
    PAUSE_VIEW_EVENTS,
    PastParams,
    PauseView,
    data_of,
    needs_declared_tools,
    needs_mcp_tools,
    needs_unknown_calls,
    read_declared_tools,
    read_mcp_tools,
    read_pause_view,
    read_unknown_calls,
    view_with,
)
from dendrux.types import ErrorInfo, ReplaySummary

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from datetime import datetime

    from sqlalchemy.ext.asyncio import AsyncEngine

    from dendrux.media import BlobStore


# ``RunNotFoundError`` is re-exported from :mod:`dendrux.errors`: one class
# everywhere since 0.5.0 (its constructor takes the run id).

ERROR_INFO_META_KEY = "dendrux.error_info"
"""``agent_runs.meta`` key holding a provider failure's :class:`ErrorInfo` dict."""

_OMITTED_TENANT = _OmittedType()


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
    tenant_id: str | None = None


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
    """A single run's full detail — summary fields plus answer and error.

    ``error_info`` is set when a provider failure ended the run.
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
    tenant_id: str | None = None
    error_info: ErrorInfo | None = None


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
    pause_id: str | None = None


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
    engine ownership with the caller; ``close()`` still stops the owned event feed.

    Tenancy: every read takes ``tenant_id``. Omitted (``UNSCOPED``) is a
    trusted all-tenant read. A string scopes the read to that tenant and
    ``None`` to untenanted runs; a run outside the scope reads exactly like a
    missing one (``None``, an empty list, or ``RunNotFoundError`` where a
    missing run raises). Scoping is applied in the same SQL query on the
    built-in store; a custom ``StateStore`` whose read methods do not take
    ``tenant_id`` is scoped by reading the run first.
    """

    def __init__(self, state_store: StateStore, *, blob_store: BlobStore | None = None) -> None:
        self._state = state_store
        self._owned_engine: AsyncEngine | None = None
        self._blob_services = BlobServices.for_state_store(state_store, blob_store=blob_store)
        self._event_feed: EventFeed | None = None

    @classmethod
    def from_engine(cls, engine: AsyncEngine, *, blob_store: BlobStore | None = None) -> RunStore:
        """Build a RunStore over an engine the caller owns and disposes."""
        return cls(SQLAlchemyStateStore(engine), blob_store=blob_store)

    @classmethod
    def from_database_url(
        cls,
        database_url: str,
        *,
        blob_store: BlobStore | None = None,
        database_options: dict[str, Any] | None = None,
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

        from dendrux.db._engine_options import engine_options, normalized_url

        parsed = normalized_url(database_url)
        engine = create_async_engine(parsed, **engine_options(parsed, database_options))
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
        check_run_id(run_id)
        check_read_scope(tenant_id)
        services = self._blob_services
        accessor = services.accessor
        catalog = services.catalog
        missing = BlobNotFoundError(image_id(_image_digest(sha256)))
        record = await self._state.get_run(run_id)
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
        """Stop owned event feeds, then dispose only engines this store owns."""
        if self._event_feed is not None:
            await self._event_feed.aclose()
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
        tenant_id: TenantScope = UNSCOPED,
        started_after: datetime | None = None,
        started_before: datetime | None = None,
        metadata_filter: dict[str, Any] | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[RunSummary]:
        """List runs with SQL-side filtering and pagination, newest first.

        ``tenant_id``: omit (``UNSCOPED``) for every tenant, a string for that
        tenant, ``None`` for untenanted runs only (changed in 0.5.0). ``limit``
        is clamped to 1..1000 and ``offset`` may be at most 10,000.
        ``started_after`` / ``started_before`` must be timezone-aware; they are
        compared in UTC.

        ``metadata_filter`` matches runs where every given key/value pair
        is present at the top level of the run's ``metadata`` (the dev
        passes this to ``agent.run(..., metadata=...)``). The canonical
        chatbot pattern is ``metadata_filter={"thread_id": "..."}``;
        keys are restricted to ``[A-Za-z0-9_-]+``, nested paths are not
        supported, and values must be scalars (str, int, float, bool or None).
        """
        _check_list_filters(
            tenant_id,
            status,
            agent_name,
            parent_run_id,
            metadata_filter,
            started_after,
            started_before,
        )
        check_read_int("limit", limit, wide=True, optional=False)
        check_read_int("offset", offset, wide=True, optional=False)
        if offset > MAX_LIST_OFFSET:
            raise ValueError(f"offset must be at most {MAX_LIST_OFFSET}, got {offset}.")
        records = await self._state.list_runs(
            limit=limit,
            offset=offset,
            tenant_id=tenant_id,
            status=status,
            agent_name=agent_name,
            parent_run_id=parent_run_id,
            started_after=_utc(started_after),
            started_before=_utc(started_before),
            metadata_filter=metadata_filter,
        )
        return [_run_to_summary(r) for r in records]

    async def count_runs(
        self,
        *,
        status: str | list[str] | None = None,
        agent_name: str | None = None,
        parent_run_id: str | None = None,
        tenant_id: TenantScope = UNSCOPED,
        started_after: datetime | None = None,
        started_before: datetime | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> int:
        """Count runs matching the same filters as ``list_runs``."""
        _check_list_filters(
            tenant_id,
            status,
            agent_name,
            parent_run_id,
            metadata_filter,
            started_after,
            started_before,
        )
        return await self._state.count_runs(
            tenant_id=tenant_id,
            status=status,
            agent_name=agent_name,
            parent_run_id=parent_run_id,
            started_after=_utc(started_after),
            started_before=_utc(started_before),
            metadata_filter=metadata_filter,
        )

    async def get_run(self, run_id: str, *, tenant_id: TenantScope = UNSCOPED) -> RunDetail | None:
        """Fetch a single run's detail, or ``None`` if it does not exist or is out of scope."""
        record = await self._scoped_run(run_id, tenant_id)
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
        tenant_id: TenantScope = UNSCOPED,
    ) -> list[StoredEvent]:
        """Read events for a run in ``sequence_index`` order.

        ``after_sequence_index`` is an exclusive cursor. Pass the previous
        batch's last ``sequence_index`` to continue. ``limit`` is clamped to
        1..1000 (``None`` reads every event). ``event_types`` keeps only those
        types (an empty list matches nothing). ``before`` is an exclusive,
        timezone-aware upper bound on ``created_at``. Empty when the run is
        outside ``tenant_id``.
        """
        check_read_int("after_sequence_index", after_sequence_index)
        check_read_int("limit", limit, wide=True)
        check_read_names("event_types", event_types)
        check_read_datetime("before", before)
        rows = await self._scoped_read(
            self._state.get_run_events,
            run_id,
            tenant_id,
            after_sequence_index=after_sequence_index,
            limit=limit,
            event_types=event_types,
            before=before,
        )
        return [_event_to_stored(r) for r in rows or []]

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

        check_run_id(run_id)
        check_read_scope(tenant_id)
        record = await require_store_method(self._state, "get_subtree")(run_id, tenant_id=tenant_id)
        if record is None:
            raise RunNotFoundError(run_id)
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
        tenant_id: TenantScope | _OmittedType = _OMITTED_TENANT,
    ) -> AsyncIterator[StoredEvent]:
        """Stream raw events as they land in the durable log (0.4 behaviour).

        .. warning::
           Omitting ``tenant_id`` reads **any tenant's** run (it means
           ``UNSCOPED``, kept so 0.4.1 callers do not break) and emits a
           ``DeprecationWarning``. Pass ``tenant_id=`` explicitly.
           For anything served to clients prefer :meth:`follow`: it shares one
           poller per run, ends when the run is terminal, caps followers and
           duration, and redacts payloads. This method does none of that; it
           polls once per caller and never ends on its own.

        Raises :class:`RunNotFoundError` before the first yield if
        ``run_id`` does not exist or is outside ``tenant_id``. The caller
        decides when to stop — typically by breaking out of ``async for`` on
        a terminal ``event_type`` or by calling ``aclose()`` on the returned
        generator. A run's tenant never changes, so the scope is checked once.
        """
        if isinstance(tenant_id, _OmittedType):
            _warn_stream_events_unscoped()
            tenant_id = UNSCOPED
        check_read_int("after_sequence_index", after_sequence_index)
        run = await self._scoped_run(run_id, tenant_id)
        if run is None:
            raise RunNotFoundError(run_id)

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
        tenant_id: TenantScope = UNSCOPED,
    ) -> list[LLMCall]:
        """Return LLM interactions for a run. ``limit``/``offset`` push
        pagination into SQL. ``before`` is an exclusive, timezone-aware
        upper bound on ``created_at``. Empty when the run is outside
        ``tenant_id``."""
        _check_page(iteration, limit, offset, before)
        rows = await self._scoped_read(
            self._state.get_llm_interactions,
            run_id,
            tenant_id,
            iteration_index=iteration,
            limit=limit,
            offset=offset,
            before=before,
        )
        return [_llm_to_public(r) for r in rows or []]

    async def get_tool_invocations(
        self,
        run_id: str,
        *,
        iteration: int | None = None,
        limit: int | None = None,
        offset: int = 0,
        before: datetime | None = None,
        tenant_id: TenantScope = UNSCOPED,
    ) -> list[ToolInvocation]:
        """Return tool invocations for a run. ``limit``/``offset`` push
        pagination into SQL. ``before`` is an exclusive, timezone-aware
        upper bound on ``created_at``. Empty when the run is outside
        ``tenant_id``."""
        _check_page(iteration, limit, offset, before)
        rows = await self._scoped_read(
            self._state.get_tool_calls,
            run_id,
            tenant_id,
            iteration_index=iteration,
            limit=limit,
            offset=offset,
            before=before,
        )
        return [_tool_to_public(r) for r in rows or []]

    async def get_traces(
        self,
        run_id: str,
        *,
        limit: int | None = None,
        offset: int = 0,
        tenant_id: TenantScope = UNSCOPED,
    ) -> list[TraceEntry]:
        """Return the conversation trace (user / assistant / tool messages)
        for a run, ordered by ``order_index``. ``limit``/``offset`` push
        pagination into SQL. Empty when the run is outside ``tenant_id``."""
        _check_page(None, limit, offset)
        rows = await self._scoped_read(
            self._state.get_traces, run_id, tenant_id, limit=limit, offset=offset
        )
        return [_trace_to_public(r) for r in rows or []]

    async def get_pii_mapping(
        self, run_id: str, *, tenant_id: TenantScope = UNSCOPED
    ) -> dict[str, str]:
        """Return the PII placeholder → real-value mapping for a run.

        The mapping is the audit key that links raw traces persisted in
        the DB to the placeholder view the LLM provider API actually saw.
        Use it to render the LLM-eye view from raw traces in inspection tools
        or exports.

        Empty dict if the run has no mapping (no PII guardrail active,
        no entities detected, the run does not exist, or it is outside
        ``tenant_id``).
        """
        return await _PendingReader(self._state).get_pii_mapping(run_id, tenant_id=tenant_id)

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

        check_run_id(run_id)
        check_read_scope(tenant_id)
        check_read_bool("undelivered_only", undelivered_only)
        get_notes = require_store_method(self._state, "get_notes")
        if not isinstance(tenant_id, _Unscoped):
            record = await self._state.get_run(run_id)
            if record is None or record.tenant_id != tenant_id:
                raise RunNotFoundError(run_id)
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

    async def get_pauses(
        self,
        run_id: str,
        *,
        tenant_id: TenantScope = UNSCOPED,
        limit: int = 100,
        include_payloads: bool = False,
    ) -> list[PausePair]:
        """First bounded pause/resume pairs; use get_pause_page for truncation metadata.

        Payloads are hidden as in :meth:`get_pause_page` unless
        ``include_payloads=True`` (``user_input`` is None by default since 0.5.0).
        """
        items, _ = await self.get_pause_page(
            run_id, tenant_id=tenant_id, limit=limit, include_payloads=include_payloads
        )
        return items

    async def get_pause_page(
        self,
        run_id: str,
        *,
        tenant_id: TenantScope = UNSCOPED,
        limit: int = 100,
        include_payloads: bool = False,
    ) -> tuple[list[PausePair], bool]:
        """Bounded pairs (1..1000), plus an honest truncated flag.

        Read a lookahead beyond the maximum paired rows so the last returned
        pause's resume is never lost at the page boundary. Orphan resume events
        can shorten the page; truncated remains true when more rows may exist.

        With ``include_payloads=False`` past pauses' arguments are hidden and
        ``user_input`` (a person's answer) is None.
        """
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 1000:
            raise ValueError("limit must be an int between 1 and 1000.")
        check_read_bool("include_payloads", include_payloads)
        method = self._state.get_run_events
        kwargs: dict[str, Any] = {}
        builtin = "_pause_page" in inspect.signature(method).parameters
        if builtin:
            kwargs["_pause_page"] = True
        row_limit = 2 * limit + 2 if builtin else min(2 * limit + 2, 1000)
        rows = await self._scoped_read(
            method,
            run_id,
            tenant_id,
            event_types=["run.paused", "run.resumed"],
            limit=row_limit,
            **kwargs,
        )
        rows = rows or []
        pairs = _derive_pauses(rows)
        truncated = len(rows) == row_limit or len(pairs) > limit
        if len(rows) == row_limit and pairs and pairs[-1].resume_sequence_index is None:
            pairs.pop()  # unresolved lookahead is not represented as an active pause
        items = []
        past = PastParams()
        pause_view = PauseView(-1, None, {})
        mcp = None
        if pairs:
            pause_view = await read_pause_view(self, run_id, tenant_id, -1)
            if not include_payloads:
                mcp = await read_mcp_tools(self, run_id, tenant_id)
        for pair in pairs[:limit]:
            event = StoredEvent(
                sequence_index=pair.pause_sequence_index,
                iteration_index=0,
                event_type="run.paused",
                correlation_id=None,
                created_at=pair.pause_at,
                data={"status": pair.reason, "pending_tool_calls": pair.pending_tool_calls},
            )
            view = await view_with(
                self,
                run_id,
                event,
                pause_view,
                tenant_id=tenant_id,
                include_payloads=include_payloads,
                past=past,
                mcp_tools=mcp,
            )
            resumed = await view_with(
                self,
                run_id,
                dataclasses.replace(
                    event,
                    event_type="run.resumed",
                    data={"submitted_results": pair.submitted_results},
                ),
                pause_view,
                tenant_id=tenant_id,
                include_payloads=include_payloads,
                past=past,
                mcp_tools=mcp,
            )
            items.append(_pair_view(pair, view, resumed, include_payloads))
        return items, truncated

    async def get_pending(self, run_id: str, *, tenant_id: TenantScope) -> PendingPause | None:
        """What the run is waiting for now, or None when it is not paused.

        ``tenant_id`` is required (keyword-only): a string or ``None`` scopes
        the read, ``UNSCOPED`` reads any tenant and must be passed explicitly.
        Bounded reads on the built-in store: the pause history is never
        loaded. Use it after a reload, a reconnect, a timed-out request or a
        retried idempotent call to rebuild the client's form or tool call.
        Arguments are in the form described on :class:`PendingToolCall`.

        Raises:
            RunNotFoundError: the run does not exist or is outside ``tenant_id``.
        """
        return await _PendingReader(self._state).get_pending(run_id, tenant_id=tenant_id)

    def follow(
        self,
        run_id: str,
        *,
        tenant_id: TenantScope,
        after_sequence_index: int | None = None,
        include_payloads: bool = False,
        until_terminal: bool = True,
        max_duration_s: float | None = 900.0,
    ) -> Follow:
        """A hardened live feed of the run's events; see :meth:`EventFeed.follow`.

        Uses this store's shared :class:`EventFeed` (one poller per watched
        run, at most 500 followers; build your own ``EventFeed(store, ...)``
        for other limits). ``tenant_id`` is required. Events pass through
        :func:`redact_event`: with ``include_payloads=False`` the system
        prompt, provider bodies and past pauses' arguments are hidden, while
        the active pause's arguments stay visible. Ends with a
        :class:`FollowEnd` (``"terminal"`` once the run finished and drained).

        Example::

            async with store.follow(run_id, tenant_id=tenant, after_sequence_index=cursor) as f:
                async for item in f:
                    if isinstance(item, FollowEnd):
                        break
                    if isinstance(item, FollowHeartbeat):
                        continue
                    send(item)
        """
        if self._event_feed is None:
            self._event_feed = EventFeed(self)
        return self._event_feed.follow(
            run_id,
            tenant_id=tenant_id,
            after_sequence_index=after_sequence_index,
            include_payloads=include_payloads,
            until_terminal=until_terminal,
            max_duration_s=max_duration_s,
        )

    async def _status(self, run_id: str, tenant_id: TenantScope = UNSCOPED) -> str | None:
        return await _PendingReader(self._state)._status(run_id, tenant_id)

    async def _active_pause_sequence(
        self, run_id: str, tenant_id: TenantScope = UNSCOPED
    ) -> int | None:
        return await _PendingReader(self._state)._active_pause_sequence(run_id, tenant_id)

    async def _approval_params(
        self, run_id: str, seq: int | None, tenant_id: TenantScope = UNSCOPED
    ) -> dict[str, dict[str, Any]]:
        return await _PendingReader(self._state)._approval_params(run_id, seq, tenant_id)

    async def _event_view(
        self,
        run_id: str,
        event: StoredEvent,
        *,
        tenant_id: TenantScope = UNSCOPED,
        include_payloads: bool = False,
    ) -> StoredEvent:
        view = PauseView(event.sequence_index, None, {})
        if event.event_type in PAUSE_VIEW_EVENTS:
            view = await read_pause_view(self, run_id, tenant_id, event.sequence_index)
        unknown = declared = mcp = None
        if needs_unknown_calls(event, include_payloads):
            unknown = await read_unknown_calls(self, run_id, tenant_id)
        if needs_declared_tools(event, include_payloads):
            declared = await read_declared_tools(self, run_id, tenant_id)
        if needs_mcp_tools(event, include_payloads):
            mcp = await read_mcp_tools(self, run_id, tenant_id)
        return await view_with(
            self,
            run_id,
            event,
            view,
            tenant_id=tenant_id,
            include_payloads=include_payloads,
            past=PastParams(),
            unknown_calls=unknown,
            declared_tools=declared,
            mcp_tools=mcp,
        )

    async def get_event_view(
        self,
        run_id: str,
        event: StoredEvent,
        *,
        tenant_id: TenantScope,
        include_payloads: bool = False,
    ) -> StoredEvent:
        """Return a scoped event view with recorded approval arguments when available.

        For the active approval or input pause, missing model-sent arguments
        are hidden even when the stored execution arguments contain real values.
        """
        check_read_bool("include_payloads", include_payloads)
        if await self._status(run_id, tenant_id) is None:
            raise RunNotFoundError(run_id)
        return await self._event_view(
            run_id, event, tenant_id=tenant_id, include_payloads=include_payloads
        )

    # ------------------------------------------------------------------
    # Tenant scoping
    # ------------------------------------------------------------------

    async def _scoped_run(self, run_id: str, tenant_id: TenantScope) -> Any:
        """The run record within ``tenant_id``, or None (missing and foreign alike)."""
        return await _PendingReader(self._state)._scoped_run(run_id, tenant_id)

    async def _scoped_read(
        self, method: Any, run_id: str, tenant_id: TenantScope, **kwargs: Any
    ) -> Any:
        """Call a per-run read within ``tenant_id``; None when the run is out of scope.

        The built-in store applies the scope as a predicate of the same
        query. A custom store's method without ``tenant_id`` gets a run
        lookup first (a run's tenant never changes, so this cannot race).
        """
        return await _PendingReader(self._state)._scoped_read(method, run_id, tenant_id, **kwargs)


# ---------------------------------------------------------------------------
# Runtime -> public type converters
# ---------------------------------------------------------------------------


_SHA256 = re.compile(r"[0-9a-f]{64}")


def _image_digest(sha256: object) -> str:
    """``sha256`` when it is a digest; otherwise no image can match it."""
    if isinstance(sha256, str) and _SHA256.fullmatch(sha256):
        return sha256
    raise BlobNotFoundError(image_id(""))


def _pair_view(
    pair: PausePair, view: StoredEvent, resumed: StoredEvent, include_payloads: bool
) -> PausePair:
    """``pair`` with its redacted pending calls and results; a person's answer
    only with payloads."""
    shown = data_of(view).get("pending_tool_calls", [])
    if include_payloads:
        return dataclasses.replace(pair, pending_tool_calls=shown)
    return dataclasses.replace(
        pair,
        pending_tool_calls=shown,
        submitted_results=data_of(resumed).get("submitted_results", []),
        user_input=None,
    )


def _check_list_filters(
    tenant_id: object,
    status: object,
    agent_name: object,
    parent_run_id: object,
    metadata_filter: object,
    started_after: object,
    started_before: object,
) -> None:
    """Validate ``list_runs`` / ``count_runs`` filters before any query."""
    check_read_scope(tenant_id)
    check_read_datetime("started_after", started_after)
    check_read_datetime("started_before", started_before)
    check_read_names("status", status, allow_str=True)
    if agent_name is not None:
        check_read_text("agent_name", agent_name)
    if parent_run_id is not None:
        check_read_text("parent_run_id", parent_run_id)
    check_metadata_filter(metadata_filter)


def _check_page(iteration: object, limit: object, offset: object, before: object = None) -> None:
    check_read_datetime("before", before)
    check_read_int("iteration", iteration)
    check_read_int("limit", limit, wide=True)
    check_read_int("offset", offset, wide=True, optional=False)


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
        tenant_id=getattr(record, "tenant_id", None),
    )


def _warn_stream_events_unscoped() -> None:
    # Deduplicated by ``warnings`` per caller location (the ``async for`` site).
    warnings.warn(
        "RunStore.stream_events() was called without tenant_id, so it reads any tenant's "
        "run. Pass tenant_id= (a tenant, None for untenanted runs, or UNSCOPED for a trusted "
        "all-tenant reader), and prefer RunStore.follow() for client-facing feeds.",
        DeprecationWarning,
        stacklevel=3,
    )


def _utc(value: datetime | None) -> datetime | None:
    """Aware datetimes in UTC; naive ones are rejected by the store."""
    if value is None or value.tzinfo is None:
        return value
    return _to_aware_utc(value)


def _error_info_of(record: Any) -> ErrorInfo | None:
    data = (getattr(record, "meta", None) or {}).get(ERROR_INFO_META_KEY)
    return ErrorInfo.from_dict(data) if isinstance(data, dict) else None


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
        tenant_id=getattr(record, "tenant_id", None),
        error_info=_error_info_of(record),
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


def _list_of(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _build_pause_pair(paused: Any, resumed: Any | None) -> PausePair:
    pdata = data_of(paused)
    rdata = data_of(resumed) if resumed is not None else {}
    return PausePair(
        pause_sequence_index=paused.sequence_index,
        pause_at=paused.created_at,
        resume_sequence_index=resumed.sequence_index if resumed is not None else None,
        resume_at=resumed.created_at if resumed is not None else None,
        reason=str(pdata.get("status", "")),
        pending_tool_calls=_list_of(pdata.get("pending_tool_calls")),
        submitted_results=_list_of(rdata.get("submitted_results")),
        user_input=rdata.get("user_input"),
        pause_id=pdata.get("pause_id") if isinstance(pdata.get("pause_id"), str) else None,
    )


__all__ = [
    "EventFeed",
    "Follow",
    "FollowCapacityError",
    "FollowEnd",
    "FollowHeartbeat",
    "FollowItem",
    "LLMCall",
    "PausePair",
    "PendingPause",
    "PendingToolCall",
    "RunDetail",
    "RunNote",
    "RunNotFoundError",
    "RunStore",
    "RunSubtree",
    "RunSummary",
    "RunUsage",
    "StoredEvent",
    "SubtreeNode",
    "TenantScope",
    "ToolInvocation",
    "TraceEntry",
    "UNSCOPED",
    "redact_event",
]
