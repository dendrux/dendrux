"""Migration warnings and compatibility adapters for public read bounds."""

from __future__ import annotations

import inspect
import logging
import warnings
from typing import Any

from dendrux import _codes
from dendrux.errors import BudgetConfigurationError, SubtreeTooLargeError
from dendrux.runtime._bounds import MAX_PAGE

logger = logging.getLogger(__name__)


def _warn(message: str) -> None:
    frame = inspect.currentframe()
    stacklevel = 1
    while frame is not None and frame.f_globals.get("__name__", "").startswith("dendrux."):
        stacklevel += 1
        frame = frame.f_back
    del frame
    warnings.warn(message, FutureWarning, stacklevel=stacklevel)


def _warn_page(method: str, limit: int | None, cursor: str) -> None:
    if limit is None:
        _warn(
            f"RunStore.{method}() reads every row; pass limit= (at most 1,000) "
            f"and page with {cursor}=. In 0.7 the default becomes 1,000."
        )
    elif limit > MAX_PAGE:
        _warn(
            f"RunStore.{method}() result is clamped to 1,000 now; "
            f"pass limit= (at most 1,000) and page with {cursor}=. "
            "A limit above 1,000 will raise in 0.7."
        )


def _trace_page_limit(limit: int | None) -> int | None:
    return None if limit is None else max(1, min(limit, MAX_PAGE))


def _check_subtree_bound(max_nodes: int | None) -> None:
    if max_nodes is None:
        _warn(
            "RunStore.get_subtree() reads every descendant; pass max_nodes=. "
            "In 0.7 the default max_nodes becomes 10,000."
        )
    elif isinstance(max_nodes, bool) or not isinstance(max_nodes, int) or max_nodes < 1:
        raise ValueError("max_nodes must be a positive integer or None")


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


async def _read_subtree(method: Any, run_id: str, tenant_id: Any, max_nodes: int | None) -> Any:
    kwargs = {"max_nodes": max_nodes} if _accepts_max_nodes(method) else {}
    try:
        record = await method(run_id, tenant_id=tenant_id, **kwargs)
    except BudgetConfigurationError as exc:
        if exc.reason != "subtree_limit" or max_nodes is None:
            raise
        raise _limit_error(run_id, max_nodes) from exc
    if record is not None and max_nodes is not None and len(record.nodes) - 1 >= max_nodes:
        raise _limit_error(run_id, max_nodes)
    return record
