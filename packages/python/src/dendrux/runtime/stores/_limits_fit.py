"""Compare an Agent's configured payload limits with the database's statement capacity."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from dendrux import _codes

if TYPE_CHECKING:
    from dendrux.types import Limits

logger = logging.getLogger(__name__)

# A limit counts raw UTF-8 bytes. Stored as JSON, one control byte can become six (backslash, "u",
# four hex digits) and a non-ASCII character can too; text-protocol SQL escaping can then double
# every backslash and quote. So every Limits-bounded payload is reserved at 6 x 2 = 12 times.
_JSON_EXPANSION = 6
_SQL_ESCAPING = 2
# Column lists, identifiers and the small binds that ride in the same statement.
_STATEMENT_SCAFFOLDING = 64 * 1024


def required_statement_bytes(limits: Limits) -> int:
    """A worst-case estimate of the largest statement the Limits-bounded payloads can produce.

    An early warning only. It cannot see payloads that Limits does not bound (model prompts and
    responses, rewritten history); the server's own limit is the final guard and surfaces as
    StatementTooLargeError when the server answers.
    """
    payload = max(limits.max_input_bytes, limits.max_tool_result_bytes) + limits.max_metadata_bytes
    return _STATEMENT_SCAFFOLDING + payload * _JSON_EXPANSION * _SQL_ESCAPING


async def ensure_limits_fit(store: object, limits: Limits) -> None:
    """Refuse limits whose largest statement the database would reject.

    A store class opts in by defining ``async statement_capacity(self) -> int | None`` (the
    most bytes one statement may carry). A store without it, or with no limit, is accepted.
    The method is looked up on the class, as for the protocol's other methods.
    """
    report = getattr(type(store), "statement_capacity", None)
    if not callable(report):
        return
    capacity = await report(store)
    if capacity is None:
        return
    needed = required_statement_bytes(limits)
    if needed <= capacity:
        return
    _codes.log_code(logger, _codes.STORE_STATEMENT_LIMIT, run_id=None, bytes=needed, limit=capacity)
    raise ValueError(
        f"The configured Limits can produce SQL statements of up to {needed} bytes, but the "
        f"database accepts {capacity} (max_allowed_packet). Lower Limits.max_input_bytes, "
        "Limits.max_tool_result_bytes or Limits.max_metadata_bytes, or raise the server's "
        "max_allowed_packet."
    )


async def ensure_agent_limits_fit(agent: object, store: object | None) -> None:
    """Check ``agent``'s configured limits against ``store`` once per (agent, store).

    Every path that persists for an Agent calls this with the store it is about to use, so the
    check does not depend on which entry point (Agent method or runner function) was taken.
    A refusal is not remembered: the next call checks again.
    """
    limits = getattr(agent, "_limits", None)
    if store is None or limits is None or getattr(agent, "_limits_checked_for", None) is store:
        return
    await ensure_limits_fit(store, limits)
    agent._limits_checked_for = store  # type: ignore[attr-defined]
