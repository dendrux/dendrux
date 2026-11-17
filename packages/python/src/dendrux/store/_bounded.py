"""Default bounds and compatibility adapters for public RunStore reads."""

from __future__ import annotations

import inspect
import logging
from typing import Any

from dendrux import _codes
from dendrux.errors import BudgetConfigurationError, SubtreeTooLargeError
from dendrux.runtime._bounds import MAX_PAGE, SUBTREE_MAX_NODES_DEFAULT
from dendrux.runtime._pending import check_read_int

logger = logging.getLogger(__name__)


def _page_limit(method: str, limit: int | None, cursor: str) -> int:
    """The page a list read uses: 1,000 when omitted or None; above 1,000 refuses.

    Zero and negative limits pass unchanged; SQL stores read one row for them.
    """
    if limit is None:
        return MAX_PAGE
    if limit > MAX_PAGE:
        _codes.log_code(logger, _codes.READ_ARGUMENT_INVALID, argument="limit")
        raise ValueError(
            f"RunStore.{method}() limit must be at most {MAX_PAGE}; page with {cursor}=."
        )
    return limit


def _subtree_bound(max_nodes: int | None) -> int:
    """The descendant bound a subtree read uses: 10,000 when omitted or None."""
    check_read_int("max_nodes", max_nodes, wide=True)
    if max_nodes is None:
        return SUBTREE_MAX_NODES_DEFAULT
    if max_nodes < 1:
        _codes.log_code(logger, _codes.READ_ARGUMENT_INVALID, argument="max_nodes")
        raise ValueError("max_nodes must be a positive integer or None")
    return max_nodes


def _accepts_max_nodes(method: Any) -> bool:
    try:
        parameter = inspect.signature(method).parameters.get("max_nodes")
    except (TypeError, ValueError):
        return False
    return parameter is not None and parameter.kind in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    )


def _limit_error(run_id: str, max_nodes: int) -> SubtreeTooLargeError:
    _codes.log_code(logger, _codes.SUBTREE_LIMIT, run_id=run_id, limit=max_nodes)
    return SubtreeTooLargeError(run_id, max_nodes)


async def _read_subtree(method: Any, run_id: str, tenant_id: Any, max_nodes: int) -> Any:
    kwargs = {"max_nodes": max_nodes} if _accepts_max_nodes(method) else {}
    try:
        record = await method(run_id, tenant_id=tenant_id, **kwargs)
    except BudgetConfigurationError as exc:
        if exc.reason != "subtree_limit":
            raise
        raise SubtreeTooLargeError(run_id, max_nodes) from exc
    if record is not None and len(record.nodes) - 1 >= max_nodes:
        raise _limit_error(run_id, max_nodes)
    return record
