"""Project safe runner failures across persistence boundaries."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from dendrux._meta_keys import ERROR_INFO_META_KEY
from dendrux.runtime import _run_contracts
from dendrux.runtime.state import unclaimed_store
from dendrux.types import ErrorInfo

if TYPE_CHECKING:
    from dendrux.runtime.state import StateStore

ERROR_TEXT_CHARS = 500

logger = logging.getLogger("dendrux.runtime.runner")


def _provider_error_info(exc: Exception) -> ErrorInfo | None:
    from dendrux.errors import ProviderError, _database_failure_in

    return (
        ErrorInfo.from_error(exc)
        if isinstance(exc, ProviderError)
        else _database_error_info(exc, _database_failure_in(exc))
    )


def _database_error_info(exc: Exception, failed: bool) -> ErrorInfo | None:
    from dendrux.errors import PersistenceError

    if not failed:
        return None
    return ErrorInfo(
        category="persistence_error",
        retryable=getattr(exc, "retryable", False) is True,
        safe_message=PersistenceError.safe_message,
        error_type="PersistenceError",
    )


def _safe_error(exc: Exception) -> str:
    info = _provider_error_info(exc)
    return info.safe_message if info else _nonprovider_error(exc)


def _nonprovider_error(exc: Exception) -> str:
    from dendrux.errors import PersistenceError, _database_failure_in

    return PersistenceError.safe_message if _database_failure_in(exc) else str(exc)


def _error_payload(exc: Exception) -> dict[str, Any]:
    info = _provider_error_info(exc)
    return {
        "error": _safe_error(exc)[:ERROR_TEXT_CHARS],
        **({"error_info": info.to_dict()} if info else {}),
        **_database_error_code(exc),
    }


def _database_error_code(exc: Exception) -> dict[str, Any]:
    from dendrux.errors import _database_failure_in

    return {"code": "persistence_error"} if _database_failure_in(exc) else {}


def _error_store_kwargs(store: StateStore, exc: Exception) -> dict[str, Any]:
    info = _provider_error_info(exc)
    base = unclaimed_store(store)
    if info and base is not None and _run_contracts._accepts_kwarg(base.finalize_run, "error_info"):
        return {"error_info": info.to_dict()}
    return {}


def _error_info_from_meta(meta: dict[str, Any] | None) -> ErrorInfo | None:
    data = (meta or {}).get(ERROR_INFO_META_KEY)
    return ErrorInfo.from_dict(data) if isinstance(data, dict) else None


def _public_meta(meta: dict[str, Any]) -> dict[str, Any]:
    result = dict(meta)
    result.pop("pause_state", None)
    mapping = result.get("pii_mapping")
    if isinstance(mapping, dict):
        result["pii_mapping"] = {k: v for k, v in mapping.items() if k != "__dendrux_origins__"}
    return result
