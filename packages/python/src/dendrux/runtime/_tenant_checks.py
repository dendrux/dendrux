"""Tenant and run-metadata checks at Agent entry points (internal).

Called through :mod:`dendrux.runtime._entry_checks`, before any I/O:

- strict tenancy (``Agent(strict_tenancy=True)``): an omitted ``tenant_id``
  raises :class:`~dendrux.errors.TenantScopeRequiredError`, except in a
  delegated child, which runs in its parent's tenant;
- ``tenant_id=""`` is rejected (it would share the untenanted scope);
- top-level run-metadata keys starting with ``dendrux.`` are reserved (B8);
- run metadata and note metadata are capped at ``Limits.max_metadata_bytes``
  of compact UTF-8 JSON.

Every rejection logs one line with a stable ``code``.
"""

from __future__ import annotations

import json
import logging
import warnings
from typing import TYPE_CHECKING, Any, TypeVar, cast

from dendrux import _codes
from dendrux._sentinel import _OmittedType
from dendrux.errors import TenantScopeRequiredError
from dendrux.runtime.context import get_delegation_context
from dendrux.runtime.state import _Unscoped
from dendrux.types import Limits, _validate_identifier, _validate_persistable

if TYPE_CHECKING:
    from dendrux.agent import Agent
    from dendrux.runtime._entry_checks import EntryPoint, RunnerEntryPoint
    from dendrux.runtime.state import TenantScope

_T = TypeVar("_T", bound="TenantScope")

logger = logging.getLogger(__name__)

RESERVED_METADATA_PREFIX = "dendrux."
_NEW_RUN_ENTRIES = frozenset({"run", "stream", "retry"})


def resolve_tenant(
    agent: Agent, entry: EntryPoint, tenant_id: _T | _OmittedType, *, default: _T
) -> _T:
    """The ``tenant_id`` an entry point runs with.

    ``tenant_id`` is what the caller passed, or ``_OMITTED``. An omitted
    tenant resolves to ``default`` (``None`` for ``run``/``stream``/``retry``,
    ``UNSCOPED`` for the rest). Under ``strict_tenancy`` it raises
    :class:`TenantScopeRequiredError` instead, unless the call runs inside a
    delegated run (``runtime.context.get_delegation_context``): a new child
    run then inherits the parent's tenant, and any other entry point is
    scoped to the parent's tenant. An omitted tenant on an Agent that did not
    choose ``strict_tenancy`` emits a ``FutureWarning`` (the 0.6 default flip).
    """
    if not isinstance(tenant_id, _OmittedType):
        return tenant_id
    if getattr(agent, "_strict_tenancy", False):
        parent = get_delegation_context()
        if parent is None:
            _codes.log_code(logger, _codes.TENANT_SCOPE_REQUIRED, entry=entry)
            raise TenantScopeRequiredError(entry)
        if entry in _NEW_RUN_ENTRIES:
            return default  # resolve_child_tenant inherits the parent's tenant
        return cast("_T", parent.tenant_id)
    if not getattr(agent, "_strict_tenancy_chosen", True):
        _warn_strict_default()
    return default


def _warn_strict_default() -> None:
    """B19: a ``FutureWarning`` (shown by default), deduplicated per caller location.

    ``stacklevel`` points at the code that called the Agent method:
    this function, ``resolve_tenant``, ``entry_checks``, the Agent method.
    """
    warnings.warn(
        "An Agent method was called without tenant_id on an Agent that did not choose "
        "strict_tenancy. In dendrux 0.6 strict_tenancy defaults to True and this call "
        "raises TenantScopeRequiredError. Pass tenant_id= (None for untenanted runs), or "
        "choose Agent(strict_tenancy=...) explicitly.",
        FutureWarning,
        stacklevel=5,
    )


def check_entry(agent: Agent, entry: EntryPoint, *, tenant_id: TenantScope, **fields: Any) -> None:
    """Validate an entry point's tenant and metadata before any I/O.

    ``tenant_id`` is the resolved tenant. ``fields`` carries what the entry
    point has: ``run_id``, ``metadata`` (run metadata), ``user_input``,
    ``history``, ``context``, ``provider_kwargs`` (run, stream, retry),
    ``tool_results``, ``rejection_reason`` and ``decisions``
    (submit_approval), ``note_text``, ``note_metadata``.
    """
    _check_tenant_value(entry, tenant_id, new_run=entry in _NEW_RUN_ENTRIES)
    _check_entry_fields(fields)
    limit = _metadata_limit(agent)
    if entry in _NEW_RUN_ENTRIES:
        check_run_metadata(fields.get("metadata"), limit=limit, entry=entry)
    if entry == "submit_note":
        _check_size(fields.get("note_metadata"), limit=limit, entry=entry, what="note metadata")


def _check_entry_fields(fields: dict[str, Any]) -> None:
    if fields.get("run_id") is not None:
        _validate_identifier(fields["run_id"], "run_id", maximum=26)
    if fields.get("idempotency_key") is not None:
        _validate_identifier(fields["idempotency_key"], "idempotency_key")
    for field in ("rejection_reason", "note_text", "note_metadata", "reason"):
        _validate_persistable(fields.get(field), field)


def check_runner_entry(
    agent: Agent,
    entry: RunnerEntryPoint,
    *,
    tenant_id: str | None,
    metadata: dict[str, Any] | None,
    **fields: Any,
) -> None:
    """Validate a new run at the runner's entry, before any I/O.

    ``tenant_id`` is the effective tenant, after delegated-child inheritance
    (``resolve_child_tenant``). Also reached by direct ``dendrux.run(...)``
    callers and runs without a store. ``fields``: ``user_input`` (run,
    stream), ``provider_kwargs``.
    """
    _check_tenant_value(entry, tenant_id, new_run=True)
    _check_optional_identifier(fields.get("idempotency_key"), "idempotency_key")
    check_run_metadata(metadata, limit=_metadata_limit(agent), entry=entry)


def _check_optional_identifier(value: Any, name: str) -> None:
    if value is not None:
        _validate_identifier(value, name)


def check_run_metadata(metadata: Any, *, limit: int, entry: str = "run") -> None:
    """Reject reserved ``dendrux.`` keys and metadata over ``limit`` bytes."""
    if metadata is None:
        return
    if not isinstance(metadata, dict):
        raise ValueError(f"metadata must be a dict, got {type(metadata).__name__}.")
    reserved = sorted(
        k for k in metadata if isinstance(k, str) and k.startswith(RESERVED_METADATA_PREFIX)
    )
    if reserved:
        _codes.log_code(logger, _codes.RESERVED_METADATA_KEY, entry=entry)
        raise ValueError(
            f"metadata keys starting with {RESERVED_METADATA_PREFIX!r} are reserved for "
            f"dendrux (count={len(reserved)}). Rename them."
        )
    _validate_persistable(metadata, "metadata")
    _check_size(metadata, limit=limit, entry=entry, what="metadata")


def _check_tenant_value(entry: str, tenant_id: object, *, new_run: bool) -> None:
    if tenant_id is None or (isinstance(tenant_id, _Unscoped) and not new_run):
        return
    if not isinstance(tenant_id, str):
        raise ValueError(
            f"{entry}(): tenant_id must be a non-empty str or None"
            f"{'' if new_run else ' (or UNSCOPED)'}, got {type(tenant_id).__name__}."
        )
    if not tenant_id:
        raise ValueError(
            f"{entry}(): tenant_id='' is not a tenant. Pass a non-empty string, "
            "or None for untenanted runs."
        )
    _validate_identifier(tenant_id, "tenant_id")


def _metadata_limit(agent: Agent) -> int:
    limits = getattr(agent, "_limits", None)
    return (limits if isinstance(limits, Limits) else Limits()).max_metadata_bytes


def _check_size(value: Any, *, limit: int, entry: str, what: str) -> None:
    if value is None:
        return
    try:
        size = len(
            json.dumps(value, default=str, separators=(",", ":"), ensure_ascii=False).encode()
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{what} must be JSON-serializable.") from exc
    if size > limit:
        _codes.log_code(logger, _codes.METADATA_TOO_LARGE, entry=entry, bytes=size, limit=limit)
        raise ValueError(
            f"{what} is {size} bytes of JSON, over the {limit}-byte limit "
            "(Limits.max_metadata_bytes)."
        )
