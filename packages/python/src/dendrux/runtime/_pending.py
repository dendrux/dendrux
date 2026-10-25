"""Shared bounded pending projection below Agent and the public read facade."""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, cast

if TYPE_CHECKING:
    from collections.abc import Mapping

from dendrux.errors import RunNotFoundError
from dendrux.runtime.state import (
    PAUSE_SUMMARY_KEYS,
    UNSCOPED,
    PauseSummary,
    TenantScope,
    _apply_tenant_scope,
    _check_scope,
    _Unscoped,
)

logger = logging.getLogger(__name__)

PAUSE_STATUSES = frozenset(
    {"paused", "waiting_approval", "waiting_client_tool", "waiting_human_input"}
)
ERROR_INFO_META_KEY = "dendrux.error_info"

PendingKind = Literal["client_tool", "approval", "input", "paused"]
_KIND_BY_STATUS: dict[str, PendingKind] = {
    "waiting_client_tool": "client_tool",
    "waiting_approval": "approval",
    "waiting_human_input": "input",
    "paused": "paused",
}


@dataclass(frozen=True, slots=True)
class PendingToolCall:
    """One call the active pause waits on: a client tool to run, or a call to approve.

    ``params``: for a **client-tool** pause, the arguments in the form the
    0.4.1 ``run.paused`` event and ``RunResult.meta["pause_state"]`` carry
    them, which is the form the client tool receives: after guardrail
    de-anonymisation, so with ``PII(restore="all")`` (the default) they hold
    real values. For an **approval** pause, the placeholder form the model
    sent (``<<EMAIL_ADDRESS_1>>``), never de-anonymised values: that is what a
    human approves. If recorded model arguments are unavailable, ``params``
    is None and ``args_hidden`` is True.
    ``target`` is ``"client"`` or ``"server"``.
    ``approval`` is ``{"source": "static" | "hook", "reason": ...}`` for a
    call waiting for approval, else None.
    """

    id: str
    name: str
    target: str | None
    params: dict[str, Any] | None
    approval: dict[str, Any] | None = None
    args_hidden: bool = False


@dataclass(frozen=True, slots=True)
class PendingPause:
    """What a paused run is waiting for (``RunStore.get_pending``).

    ``kind``: ``"client_tool"`` (run ``tool_calls`` and submit results),
    ``"approval"`` (approve or reject ``tool_calls``), ``"input"`` (answer
    ``question``, with suggested ``options``) or ``"paused"`` (a requested
    or budget pause; see ``reason``, ``message``, ``budget_exceeded``).
    ``pause_id`` identifies this pause (None until the runtime records one).
    :meth:`to_dict` gives the ``run.paused`` event shape plus the extra fields.
    """

    run_id: str
    status: str
    kind: PendingKind
    pause_id: str | None
    tool_calls: list[PendingToolCall] = field(default_factory=list)
    question: str | None = None
    options: list[str] = field(default_factory=list)
    reason: str | None = None
    message: str | None = None
    budget_exceeded: list[dict[str, Any]] | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready dict in the ``run.paused`` shape (``pending_tool_calls`` etc.)."""
        return {
            "run_id": self.run_id,
            "status": self.status,
            "kind": self.kind,
            "pause_id": self.pause_id,
            "pending_tool_calls": [
                {
                    "id": c.id,
                    "name": c.name,
                    "target": c.target,
                    "params": c.params,
                    "approval": c.approval,
                    **({"args_hidden": True} if c.args_hidden else {}),
                }
                for c in self.tool_calls
            ],
            "question": self.question,
            "options": list(self.options),
            "reason": self.reason,
            "message": self.message,
            "exceeded": self.budget_exceeded,
        }


def _pending_from(
    run_id: str,
    status: str,
    kind: PendingKind,
    pause: dict[str, Any],
    last_step: dict[str, Any] | None,
) -> PendingPause:
    targets = pause.get("pending_targets") or {}
    gates = pause.get("approval_gates") or {}
    calls = []
    for tc in pause.get("pending_tool_calls") or []:
        if not isinstance(tc, dict) or "id" not in tc:
            continue
        gate = gates.get(tc["id"]) if isinstance(gates, dict) else None
        calls.append(
            PendingToolCall(
                id=str(tc["id"]),
                name=str(tc.get("name", "")),
                target=targets.get(tc["id"]) if isinstance(targets, dict) else None,
                params=None if tc.get("args_hidden") else dict(tc.get("params") or {}),
                args_hidden=bool(tc.get("args_hidden")),
                approval=dict(gate) if isinstance(gate, dict) else None,
            )
        )
    question: str | None = None
    options: list[str] = []
    action = (last_step or {}).get("action") if kind == "input" else None
    if isinstance(action, dict) and action.get("type") == "clarification":
        question = action.get("question")
        options = [str(o) for o in action.get("options") or []]
    exceeded = pause.get("budget_exceeded")
    return PendingPause(
        run_id=run_id,
        status=status,
        kind=kind,
        pause_id=pause.get("pause_id") if isinstance(pause.get("pause_id"), str) else None,
        tool_calls=calls,
        question=question,
        options=options,
        reason=pause.get("pause_reason"),
        message=pause.get("pause_message"),
        budget_exceeded=exceeded if isinstance(exceeded, list) else None,
    )


def approval_calls(
    calls: Any,
    approval_params: Mapping[str, dict[str, Any]] | None,
) -> Any:
    """Use recorded model arguments; never infer them from execution values."""
    if not isinstance(calls, list):
        return calls
    result = []
    for call in calls:
        if not isinstance(call, dict):
            result.append(call)
            continue
        shown = dict(call)
        params = (approval_params or {}).get(str(call.get("id")))
        if params is not None:
            shown.pop("arguments", None)
            shown["params"] = params
        else:
            shown.pop("arguments", None)
            shown.update(params=None, args_hidden=True)
        result.append(shown)
    return result


def _is_postgres_store(state: Any) -> bool:
    engine = getattr(state, "_engine", None)
    dialect = getattr(getattr(engine, "dialect", None), "name", None)
    return dialect == "postgresql" and getattr(state, "_session_factory", None) is not None


def _unconvertible_json(exc: BaseException) -> bool:
    """A Postgres data error raised while converting stored JSON (U+0000, a lone surrogate)."""
    orig = getattr(exc, "orig", None)
    code = getattr(orig, "sqlstate", None) or getattr(orig, "pgcode", None)
    return isinstance(code, str) and code in ("22P02", "22P05")


async def _pg_pause_summary(state: Any, run_id: str, tenant_id: TenantScope) -> PauseSummary | None:
    """What a run waits for, in one bounded statement on Postgres.

    ``pause_data`` is converted to ``jsonb`` once, in a subquery the planner
    cannot inline (``OFFSET 0``), and the summary keys and last step are read
    from that one value. A stored document ``jsonb`` cannot hold (U+0000 or a
    lone surrogate, which the ``json`` column accepts) makes the conversion
    fail; that run is then read whole and summarised in Python, so one bad
    row never breaks a read.
    """
    from sqlalchemy import cast, func, select
    from sqlalchemy.dialects.postgresql import JSONB
    from sqlalchemy.exc import DBAPIError

    from dendrux.db.models import AgentRun, RunEvent

    paused_seq = (
        select(func.max(RunEvent.sequence_index))
        .where(RunEvent.agent_run_id == run_id, RunEvent.event_type == "run.paused")
        .scalar_subquery()
    )
    row_stmt = _apply_tenant_scope(
        select(
            AgentRun.status.label("status"),
            paused_seq.label("seq"),
            AgentRun.pii_mapping.label("pii"),
            cast(AgentRun.pause_data, JSONB).label("doc"),
        ).where(AgentRun.id == run_id),
        tenant_id,
    )
    inner = row_stmt.offset(0).subquery()
    doc = inner.c.doc
    stmt = select(
        inner.c.status,
        inner.c.seq,
        inner.c.pii,
        *[doc[key] for key in PAUSE_SUMMARY_KEYS],
        doc["steps"][-1],
    )
    try:
        async with state._session_factory() as session:
            row = (await session.execute(stmt)).first()
    except DBAPIError as exc:
        if not _unconvertible_json(exc):
            raise
        logger.warning(
            "pause state is not convertible to jsonb; read whole: "
            "run_id=%s code=pause_data_not_jsonb",
            run_id,
            extra={"run_id": run_id, "code": "pause_data_not_jsonb"},
        )
        return await _whole_pause_summary(state, run_id, tenant_id)
    if row is None:
        return None
    status, seq, mapping, *values = row
    pause = {k: v for k, v in zip(PAUSE_SUMMARY_KEYS, values[:-1], strict=True) if v is not None}
    last = values[-1]
    return PauseSummary(
        str(status),
        pause,
        last if isinstance(last, dict) else None,
        seq,
        dict(mapping) if isinstance(mapping, dict) else None,
    )


async def _whole_pause_summary(
    state: Any, run_id: str, tenant_id: TenantScope
) -> PauseSummary | None:
    """The summary from the whole ``pause_data`` document, decoded by the driver."""
    from sqlalchemy import func, select

    from dendrux.db.models import AgentRun, RunEvent

    paused_seq = (
        select(func.max(RunEvent.sequence_index))
        .where(RunEvent.agent_run_id == run_id, RunEvent.event_type == "run.paused")
        .scalar_subquery()
    )
    stmt = _apply_tenant_scope(
        select(AgentRun.status, paused_seq, AgentRun.pii_mapping, AgentRun.pause_data).where(
            AgentRun.id == run_id
        ),
        tenant_id,
    )
    async with state._session_factory() as session:
        row = (await session.execute(stmt)).first()
    if row is None:
        return None
    status, seq, mapping, data = row
    data = data if isinstance(data, dict) else {}
    steps = data.get("steps")
    last = steps[-1] if isinstance(steps, list) and steps else None
    return PauseSummary(
        str(status),
        {k: data[k] for k in PAUSE_SUMMARY_KEYS if k in data and data[k] is not None},
        last if isinstance(last, dict) else None,
        seq,
        dict(mapping) if isinstance(mapping, dict) else None,
    )


def _accepts_tenant(method: Any) -> bool:
    """Require an explicit scope parameter; **kwargs may silently swallow it."""
    try:
        return "tenant_id" in inspect.signature(method).parameters
    except (TypeError, ValueError):
        return False


class _PendingReader:
    """The same read/scoping policy for Agent, runtime and RunStore."""

    def __init__(self, state: Any) -> None:
        self._state = state

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
        _check_scope(tenant_id)
        getter = getattr(self._state, "get_pause_summary", None)
        if getter is not None:
            if _is_postgres_store(self._state):
                summary = await _pg_pause_summary(self._state, run_id, tenant_id)
            else:
                summary = await getter(run_id, tenant_id=tenant_id)
            if summary is None:
                raise RunNotFoundError(run_id)
            status, pause, last_step = summary.status, summary.pause, summary.last_step
        else:  # custom store: scope by lookup, then the full pause state
            record = await self._scoped_run(run_id, tenant_id)
            if record is None:
                raise RunNotFoundError(run_id)
            status = record.status
            full = await self._state.get_pause_state(run_id) or {}
            pause = dict(full)
            steps = full.get("steps") or []
            last_step = steps[-1] if steps else None
        kind = _KIND_BY_STATUS.get(status)
        if kind is None:
            return None
        if kind in ("approval", "input"):
            seq = await self._active_pause_sequence(run_id, tenant_id)
            params = await self._approval_params(run_id, seq, tenant_id)
            pause = {
                **pause,
                "pending_tool_calls": approval_calls(pause.get("pending_tool_calls"), params),
            }
        return _pending_from(run_id, status, kind, pause, last_step)

    async def _status(self, run_id: str, tenant_id: TenantScope = UNSCOPED) -> str | None:
        getter = getattr(self._state, "get_run_status", None)
        if getter is not None:
            return cast("str | None", await getter(run_id, tenant_id=tenant_id))
        record = await self._scoped_run(run_id, tenant_id)
        return str(record.status) if record is not None else None

    async def _active_pause_sequence(
        self, run_id: str, tenant_id: TenantScope = UNSCOPED
    ) -> int | None:
        getter = getattr(self._state, "get_active_pause_sequence", None)
        if getter is not None:
            return cast("int | None", await getter(run_id, tenant_id=tenant_id))
        if await self._status(run_id, tenant_id) not in PAUSE_STATUSES:
            return None
        # Legacy readers without explicit filters cannot promise a bounded
        # history read. Fail closed: approval/input params stay hidden.
        try:
            parameters = inspect.signature(self._state.get_run_events).parameters
        except (TypeError, ValueError):
            return None
        if not {"limit", "event_types"} <= parameters.keys():
            return None
        # Bounded custom-store fallback. Unknown tail means no active exception.
        rows = await self._scoped_read(
            self._state.get_run_events,
            run_id,
            tenant_id,
            event_types=["run.paused", "run.resumed"],
            limit=1000,
        )
        if not rows or len(rows) == 1000 or rows[-1].event_type != "run.paused":
            return None
        return int(rows[-1].sequence_index)

    async def _approval_params(
        self, run_id: str, seq: int | None, tenant_id: TenantScope = UNSCOPED
    ) -> dict[str, dict[str, Any]]:
        getter = getattr(self._state, "get_approval_params", None)
        if getter is None or seq is None:
            return {}  # legacy custom stores fail closed when PII is present
        return cast("dict[str, dict[str, Any]]", await getter(run_id, seq, tenant_id=tenant_id))

    async def _scoped_run(self, run_id: str, tenant_id: TenantScope) -> Any:
        """The run record within ``tenant_id``, or None (missing and foreign alike)."""
        _check_scope(tenant_id)
        if isinstance(tenant_id, _Unscoped):
            return await self._state.get_run(run_id)
        getter = self._state.get_run
        if _accepts_tenant(getter):
            return await getter(run_id, tenant_id=tenant_id)
        record = await getter(run_id)
        return record if record is not None and record.tenant_id == tenant_id else None

    async def _scoped_read(
        self, method: Any, run_id: str, tenant_id: TenantScope, **kwargs: Any
    ) -> Any:
        """Call a per-run read within ``tenant_id``; None when the run is out of scope.

        The built-in store applies the scope as a predicate of the same
        query. A custom store's method without ``tenant_id`` gets a run
        lookup first (a run's tenant never changes, so this cannot race).
        """
        _check_scope(tenant_id)
        if isinstance(tenant_id, _Unscoped):
            return await method(run_id, **kwargs)
        if _accepts_tenant(method):
            return await method(run_id, tenant_id=tenant_id, **kwargs)
        if await self._scoped_run(run_id, tenant_id) is None:
            return None
        return await method(run_id, **kwargs)

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
        mapping = await self._scoped_read(self._state.get_pii_mapping, run_id, tenant_id)
        return mapping or {}


# Keep implementation below Agent without changing public docs or pickle paths.
PendingPause.__module__ = "dendrux.store"
PendingToolCall.__module__ = "dendrux.store"
