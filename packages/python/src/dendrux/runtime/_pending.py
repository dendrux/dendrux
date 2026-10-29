"""Shared bounded pending projection below Agent and the public read facade."""

from __future__ import annotations

import inspect
import logging
import math
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal, cast

from dendrux import _codes

if TYPE_CHECKING:
    from collections.abc import Mapping

from dendrux.errors import RunNotFoundError
from dendrux.runtime.state import (
    PAUSE_SUMMARY_KEYS,
    UNSCOPED,
    PauseSummary,
    TenantScope,
    _apply_tenant_scope,
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


def pause_display_calls(
    pause: Mapping[str, Any], params: Mapping[str, dict[str, Any]] | None
) -> Any:
    calls = pause.get("pending_tool_calls")
    if pause.get("status") != "paused":
        return approval_calls(calls, params)
    targets = pause.get("pending_targets") or {}
    gates = pause.get("approval_gates") or {}
    if not isinstance(calls, list):
        return calls
    return [
        call
        if isinstance(call, dict)
        and targets.get(call.get("id")) == "client"
        and call.get("id") not in gates
        else approval_calls([call], params)[0]
        for call in calls
    ]


# ---------------------------------------------------------------------------
# Read-argument validation. Ids, cursors and filters often come straight from
# a URL: a bad one is a ValueError with a fixed message (never the value, never
# a driver error with SQL text), logged with a stable code.
# ---------------------------------------------------------------------------

MAX_READ_ID_CHARS = 255
"""Longest run id, tenant id or name accepted by a read (the id columns' width)."""
MAX_READ_FILTER_ITEMS = 100
"""Most entries in a read's list filter (``event_types``, ``status``)."""
MAX_METADATA_FILTER_KEYS = 32
_INT32 = (-(2**31), 2**31 - 1)
_INT64 = (-(2**63), 2**63 - 1)
_METADATA_KEY = re.compile(r"[A-Za-z0-9_\-]{1,128}")


def _invalid(argument: str, problem: str) -> ValueError:
    _codes.log_code(logger, _codes.READ_ARGUMENT_INVALID, argument=argument)
    return ValueError(f"{argument} {problem}")


def check_read_text(argument: str, value: object, *, max_chars: int = MAX_READ_ID_CHARS) -> str:
    """A non-empty str of at most ``max_chars`` that a database can store."""
    if not isinstance(value, str) or not value:
        raise _invalid(argument, "must be a non-empty str.")
    if len(value) > max_chars:
        raise _invalid(argument, f"must be at most {max_chars} characters.")
    if "\x00" in value:
        raise _invalid(argument, "must not contain U+0000.")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        raise _invalid(argument, "must not contain unpaired surrogates.") from None
    return value


def check_run_id(run_id: object) -> str:
    return check_read_text("run_id", run_id)


def check_read_scope(tenant_id: object) -> None:
    """``UNSCOPED``, ``None`` or a valid tenant id; never echoes a rejected value."""
    if tenant_id is None or isinstance(tenant_id, _Unscoped):
        return
    if not isinstance(tenant_id, str):
        raise _invalid("tenant_id", "must be a non-empty str, None (untenanted runs) or UNSCOPED.")
    check_read_text("tenant_id", tenant_id)


def check_read_bool(argument: str, value: object) -> bool:
    """Exactly ``True`` or ``False``: a string such as ``"false"`` is refused."""
    if value is not True and value is not False:
        raise _invalid(argument, "must be a bool.")
    return value


def check_read_datetime(argument: str, value: object) -> None:
    """None, or a timezone-aware ``datetime``."""
    if value is None:
        return
    if not isinstance(value, datetime):
        raise _invalid(argument, "must be a timezone-aware datetime or None.")
    if value.tzinfo is None or value.utcoffset() is None:
        raise _invalid(argument, "must be timezone-aware (for example datetime.now(UTC)).")


def check_read_duration(argument: str, value: object, *, optional: bool = True) -> None:
    """A finite positive number of seconds (None passes when ``optional``)."""
    if value is None and optional:
        return
    if not _positive_seconds(value):
        raise _invalid(argument, "must be a positive number of seconds.")


def _positive_seconds(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(value) and value > 0


UNKNOWN_TOOL_PREFIX = "Unknown tool:"
"""How the loop's unknown-tool rejection message starts (its stored error text)."""
MAX_UNKNOWN_TOOL_CALLS = 1000


async def unknown_tool_calls(
    state: Any, run_id: str, tenant_id: TenantScope
) -> frozenset[str] | None:
    """Call ids of the run's calls rejected because the model named no declared tool.

    One indexed read of ``tool_calls`` (ids only). None when it cannot be
    known (a custom store, or more than ``MAX_UNKNOWN_TOOL_CALLS``); callers
    then treat every failed call's name as unknown.
    """
    factory = getattr(state, "_session_factory", None)
    if factory is None:
        return None
    from sqlalchemy import select

    from dendrux.db.models import ToolCallRecord as ToolCall
    from dendrux.runtime.state import _run_in_scope

    stmt = (
        select(ToolCall.tool_call_id)
        .where(
            ToolCall.agent_run_id == run_id,
            _run_in_scope(run_id, tenant_id),
            ToolCall.success.is_(False),
            ToolCall.error_message.startswith(UNKNOWN_TOOL_PREFIX, autoescape=True),
        )
        .limit(MAX_UNKNOWN_TOOL_CALLS + 1)
    )
    async with factory() as session:
        ids = (await session.execute(stmt)).scalars().all()
    if len(ids) > MAX_UNKNOWN_TOOL_CALLS:
        return None
    return frozenset(ids)


def check_read_int(
    argument: str, value: object, *, wide: bool = False, optional: bool = True
) -> None:
    """An int (not bool) that fits the column: int32, or int64 when ``wide``.

    None passes when ``optional``.
    """
    if value is None and optional:
        return
    if isinstance(value, bool) or not isinstance(value, int):
        raise _invalid(argument, "must be an int or None.")
    low, high = _INT64 if wide else _INT32
    if not low <= value <= high:
        raise _invalid(argument, f"must be between {low} and {high}.")


def check_read_bounded(argument: str, value: object, low: int, high: int) -> None:
    """An int (not bool) between ``low`` and ``high``; the message never echoes it."""
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise _invalid(argument, f"must be an int between {low} and {high}.")


def check_read_names(argument: str, values: object, *, allow_str: bool = False) -> None:
    """None, or a list or tuple of at most MAX_READ_FILTER_ITEMS valid strings."""
    if values is None:
        return
    if allow_str and isinstance(values, str):
        check_read_text(argument, values)
        return
    if not isinstance(values, (list, tuple)):
        raise _invalid(argument, "must be a list of str or None.")
    if len(values) > MAX_READ_FILTER_ITEMS:
        raise _invalid(argument, f"must have at most {MAX_READ_FILTER_ITEMS} entries.")
    for value in values:
        check_read_text(argument, value)


def check_metadata_filter(metadata_filter: object) -> None:
    """None, or a dict of top-level keys to scalar JSON values (the documented contract)."""
    if metadata_filter is None:
        return
    if not isinstance(metadata_filter, dict):
        raise _invalid("metadata_filter", "must be a dict or None.")
    if len(metadata_filter) > MAX_METADATA_FILTER_KEYS:
        raise _invalid("metadata_filter", f"must have at most {MAX_METADATA_FILTER_KEYS} keys.")
    for key, value in metadata_filter.items():
        if not isinstance(key, str) or not _METADATA_KEY.fullmatch(key):
            raise _invalid("metadata_filter", "keys must match [A-Za-z0-9_-]{1,128}.")
        if isinstance(value, str):
            check_read_text("metadata_filter", value, max_chars=4096)
        elif isinstance(value, float):
            if not math.isfinite(value):
                raise _invalid("metadata_filter", "values must be finite numbers.")
        elif isinstance(value, int) and not isinstance(value, bool):
            check_read_int("metadata_filter", value, wide=True)
        elif value is not None and not isinstance(value, bool):
            raise _invalid("metadata_filter", "values must be str, int, float, bool or None.")


MAX_DECLARED_TOOL_NAMES = 1000


async def declared_tool_names(
    state: Any, run_id: str, tenant_id: TenantScope
) -> frozenset[str] | None:
    """Tool names the run recorded for calls to declared tools (unknown-tool
    rejections excluded). One indexed read of distinct names; None when it
    cannot be known (a custom store, or more than ``MAX_DECLARED_TOOL_NAMES``)."""
    factory = getattr(state, "_session_factory", None)
    if factory is None:
        return None
    from sqlalchemy import and_, not_, select

    from dendrux.db.models import ToolCallRecord
    from dendrux.runtime.state import _run_in_scope

    rejected = and_(
        ToolCallRecord.success.is_(False),
        ToolCallRecord.error_message.startswith(UNKNOWN_TOOL_PREFIX, autoescape=True),
    )
    stmt = (
        select(ToolCallRecord.tool_name)
        .where(
            ToolCallRecord.agent_run_id == run_id,
            _run_in_scope(run_id, tenant_id),
            not_(rejected),
        )
        .distinct()
        .limit(MAX_DECLARED_TOOL_NAMES + 1)
    )
    async with factory() as session:
        names = (await session.execute(stmt)).scalars().all()
    if len(names) > MAX_DECLARED_TOOL_NAMES:
        return None
    return frozenset(names)


async def pause_window_start(
    state: Any, run_id: str, pause_seq: int, tenant_id: TenantScope
) -> int | None:
    """Index of the boundary before the pause at ``pause_seq``: the last earlier
    ``run.paused``, ``run.resumed`` or terminal event (-1 when none). Events
    after it and before ``pause_seq`` belong to that pause. One indexed read;
    None when it cannot be known (a custom store)."""
    factory = getattr(state, "_session_factory", None)
    if factory is None:
        return None
    from sqlalchemy import func, select

    from dendrux.db.models import RunEvent
    from dendrux.runtime.state import _run_in_scope

    stmt = select(func.coalesce(func.max(RunEvent.sequence_index), -1)).where(
        RunEvent.agent_run_id == run_id,
        _run_in_scope(run_id, tenant_id),
        RunEvent.event_type.in_(
            ["run.paused", "run.resumed", "run.completed", "run.error", "run.cancelled"]
        ),
        RunEvent.sequence_index < pause_seq,
    )
    async with factory() as session:
        return int((await session.execute(stmt)).scalar_one())


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
        _codes.log_code(logger, _codes.PAUSE_DATA_NOT_JSONB, run_id=run_id)
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
        check_run_id(run_id)
        check_read_scope(tenant_id)
        status, pause, last_step = await self._summary(run_id, tenant_id)
        kind = _KIND_BY_STATUS.get(status)
        if kind is None:
            return None
        targets = pause.get("pending_targets") or {}
        gates = pause.get("approval_gates") or {}
        paused_server_calls = kind == "paused" and any(
            isinstance(call, dict)
            and (targets.get(call.get("id")) != "client" or call.get("id") in gates)
            for call in pause.get("pending_tool_calls") or ()
        )
        if kind in ("approval", "input") or paused_server_calls:
            seq = await self._active_pause_sequence(run_id, tenant_id)
            params = await self._approval_params(run_id, seq, tenant_id)
            pause = {
                **pause,
                "pending_tool_calls": pause_display_calls({**pause, "status": status}, params),
            }
        return _pending_from(run_id, status, kind, pause, last_step)

    async def _summary(
        self, run_id: str, tenant_id: TenantScope
    ) -> tuple[str, dict[str, Any], dict[str, Any] | None]:
        """Status, pause keys and last step; raises RunNotFoundError when out of scope."""
        getter = getattr(self._state, "get_pause_summary", None)
        if getter is None:  # custom store: scope by lookup, then the full pause state
            record = await self._scoped_run(run_id, tenant_id)
            if record is None:
                raise RunNotFoundError(run_id)
            full = await self._state.get_pause_state(run_id) or {}
            steps = full.get("steps") or []
            return record.status, dict(full), steps[-1] if steps else None
        if _is_postgres_store(self._state):
            summary = await _pg_pause_summary(self._state, run_id, tenant_id)
        else:
            summary = await getter(run_id, tenant_id=tenant_id)
        if summary is None:
            raise RunNotFoundError(run_id)
        return summary.status, summary.pause, summary.last_step

    async def _status(self, run_id: str, tenant_id: TenantScope = UNSCOPED) -> str | None:
        check_run_id(run_id)
        check_read_scope(tenant_id)
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
        check_run_id(run_id)
        check_read_scope(tenant_id)
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
        check_run_id(run_id)
        check_read_scope(tenant_id)
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
