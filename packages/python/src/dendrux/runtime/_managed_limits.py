"""Pin a managed root's run cap at creation and recover its saved walk bound."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from dendrux import _codes
from dendrux._lifecycle import STORAGE_FORMAT_CHECKPOINTED
from dendrux.errors import ManagedStateError, PersistenceError
from dendrux.runtime.state import MANAGED_HEADER_KEY, ManagedHeader
from dendrux.types import Limits

if TYPE_CHECKING:
    from dendrux.runtime.state import RunRecord


class ManagedRunLimitError(ManagedStateError):
    """The root cap is exhausted, or bounded membership cannot certify admission."""

    code = "managed_run_limit"

    def __init__(self, *, run_id: str | None = None, reason: str = "run_limit") -> None:
        self.run_id, self.reason = run_id, "run_limit"
        # ManagedStateError normalizes persisted-state reasons and logs that code;
        # admission has its own stable reason, message and log code.
        PersistenceError.__init__(self, code=self.code)
        self.safe_message = "Managed run admission refused by the root run cap."
        self.args = (self.safe_message,)
        _codes.log_code(logging.getLogger(__name__), _codes.MANAGED_RUN_LIMIT, run_id=run_id)


def stamp_root_header(store: Any, run_id: str, *, limits: Limits | None) -> None:
    """Put the root's managed header into the claimed checkpoint before any bind.

    The recorder copies this state, so the header is written with the root's
    first fenced checkpoint and kept by every later one.
    """
    if limits is None:
        return
    state = getattr(store, "loop_state", None)
    if not isinstance(state, dict):
        raise ManagedStateError(reason="storage_format", run_id=run_id)
    state[MANAGED_HEADER_KEY] = ManagedHeader(root_run_id=run_id, path=()).to_dict()
    state["managed_limits"] = {"max_managed_runs": limits.max_managed_runs}


async def _saved_walk_limit(store: Any, root: RunRecord) -> int:
    if root.storage_format != STORAGE_FORMAT_CHECKPOINTED:
        return Limits.max_managed_runs
    state = await store.get_loop_state(root.id) or {}
    config = state.get("managed_limits", {"max_managed_runs": Limits.max_managed_runs})
    limit = config.get("max_managed_runs") if isinstance(config, dict) else None
    if type(limit) is not int or limit < 1:
        raise ManagedStateError(reason="header_invalid", run_id=root.id)
    return limit
