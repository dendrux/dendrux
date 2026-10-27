"""Size caps at Agent entry points (internal).

Called through :mod:`dendrux.runtime._entry_checks` before any I/O.

- ``Limits.max_input_bytes`` caps the UTF-8 text a caller hands the model:
  ``user_input`` on ``run`` / ``stream`` / ``resume*`` / ``submit_input`` and
  the approval rejection text on ``submit_approval``. Images are excluded
  (they have their own caps). Over the cap: ``ValueError``.
- The checks also bind the Agent's ``Limits`` to its tool call limiter, which
  is how the tool path (result cap, MCP read cap) learns them.

The metadata cap is in :mod:`dendrux.runtime._tenant_checks` and the note cap
in the state store (``add_note(max_notes=)``).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from dendrux.tools._pool import ToolCallLimiter

if TYPE_CHECKING:
    from dendrux.agent import Agent
    from dendrux.runtime._entry_checks import EntryPoint, RunnerEntryPoint
    from dendrux.runtime.state import TenantScope
    from dendrux.types import Limits

logger = logging.getLogger(__name__)

INPUT_SIZE_LIMIT_CODE = "input_size_limit"


def _limits(agent: Agent) -> Limits:
    from dendrux.types import Limits

    limits = getattr(agent, "_limits", None)
    return limits if isinstance(limits, Limits) else Limits()


def _bind_limits(agent: Agent, limits: Limits) -> None:
    limiter = getattr(agent, "_tool_call_limiter", None)
    if isinstance(limiter, ToolCallLimiter) and limiter.limits is not limits:
        limiter.limits = limits


def _text_bytes(value: Any) -> int:
    """UTF-8 bytes of the text in a ``UserContent`` value (images count 0)."""
    if value is None:
        return 0
    if isinstance(value, str):
        return len(value.encode("utf-8"))
    if isinstance(value, (bytes, bytearray)):
        return 0
    try:
        items = list(value)
    except TypeError:
        return 0
    return sum(len(item.encode("utf-8")) for item in items if isinstance(item, str))


def _check_persistable_text(value: Any, field: str) -> None:
    from dendrux.types import _validate_persistable

    if isinstance(value, str) or value is None:
        _validate_persistable(value, field)
    elif isinstance(value, (list, tuple)):
        for item in value:
            if isinstance(item, str):
                _validate_persistable(item, field)


def _check_text(
    agent: Agent,
    entry: str,
    field: str,
    value: Any,
    limit: int,
    run_id: Any,
) -> None:
    _check_persistable_text(value, field)
    size = _text_bytes(value)
    if size <= limit:
        return
    logger.warning(
        "Rejected oversized input (code=%s, entry=%s, field=%s, run_id=%s, bytes=%d, limit=%d)",
        INPUT_SIZE_LIMIT_CODE,
        entry,
        field,
        run_id if isinstance(run_id, str) else None,
        size,
        limit,
    )
    raise ValueError(
        f"Agent '{agent.name}' {field} is {size} bytes of text; the limit is {limit} "
        "(Limits.max_input_bytes)."
    )


def _check_model_text(agent: Agent, entry: str, fields: dict[str, Any], run_id: Any) -> None:
    from dendrux.types import _validate_persistable

    total = 0

    def add(value: Any) -> None:
        nonlocal total
        if isinstance(value, str):
            _validate_persistable(value, "model_input")
            total += len(value.encode("utf-8"))
        elif isinstance(value, (list, tuple)):
            for item in value:
                add(item)

    add(fields.get("user_input"))
    for turn in fields.get("history") or ():
        add(getattr(turn, "content", None))
    for block in fields.get("context") or ():
        add(getattr(block, "content", None))
    if total > _limits(agent).max_input_bytes:
        logger.warning(
            "Rejected oversized model input: run_id=%s code=input_size_limit",
            run_id,
            extra={"run_id": run_id, "code": INPUT_SIZE_LIMIT_CODE},
        )
        raise ValueError("Model input exceeds Limits.max_input_bytes.")


def check_entry(agent: Agent, entry: EntryPoint, *, tenant_id: TenantScope, **fields: Any) -> None:
    """Enforce ``max_input_bytes`` on caller-supplied model-visible text."""
    limits = _limits(agent)
    _bind_limits(agent, limits)
    limit = limits.max_input_bytes
    run_id = fields.get("run_id")
    if "user_input" in fields:
        _check_text(agent, entry, "user_input", fields["user_input"], limit, run_id)
    _check_model_text(agent, entry, fields, run_id)
    if entry == "submit_approval":
        _check_text(agent, entry, "rejection_reason", fields.get("rejection_reason"), limit, run_id)
        decisions = fields.get("decisions")
        if isinstance(decisions, Mapping):
            reasons = [getattr(d, "reason", None) for d in decisions.values()]
            _check_text(
                agent,
                entry,
                "decision reasons",
                [r for r in reasons if isinstance(r, str)],
                limit,
                run_id,
            )


def check_runner_entry(
    agent: Agent,
    entry: RunnerEntryPoint,
    *,
    tenant_id: str | None,
    metadata: dict[str, Any] | None,
    **fields: Any,
) -> None:
    """The same caps for direct runner callers (``dendrux.run(...)``)."""
    limits = _limits(agent)
    _bind_limits(agent, limits)
    if "user_input" in fields:
        _check_text(agent, entry, "user_input", fields["user_input"], limits.max_input_bytes, None)
    _check_model_text(agent, entry, fields, fields.get("run_id"))
