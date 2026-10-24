"""Tenant and run-metadata checks at Agent entry points (internal).

Called through :mod:`dendrux.runtime._entry_checks`. Inert until the 0.5.0
tenancy work fills it in: strict tenancy (``Agent(strict_tenancy=True)``),
rejecting ``tenant_id=""``, reserved ``dendrux.`` metadata keys, and the
run-metadata size cap (``Limits.max_metadata_bytes``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

from dendrux._sentinel import _OmittedType

if TYPE_CHECKING:
    from dendrux.agent import Agent
    from dendrux.runtime._entry_checks import EntryPoint
    from dendrux.runtime.state import TenantScope

_T = TypeVar("_T", bound="TenantScope")


def resolve_tenant(
    agent: Agent, entry: EntryPoint, tenant_id: _T | _OmittedType, *, default: _T
) -> _T:
    """The ``tenant_id`` an entry point runs with.

    ``tenant_id`` is what the caller passed, or ``_OMITTED``. An omitted
    tenant resolves to ``default`` (``None`` for ``run``/``stream``/``retry``,
    ``UNSCOPED`` for the rest). Runs before any I/O. A delegated child run
    inherits its parent's tenant (``runtime.context.get_delegation_context``),
    so an omitted tenant there is not "missing".
    """
    if isinstance(tenant_id, _OmittedType):
        return default
    return tenant_id


def check_entry(agent: Agent, entry: EntryPoint, *, tenant_id: TenantScope, **fields: Any) -> None:
    """Validate an entry point's arguments before any I/O. Inert for now.

    ``tenant_id`` is the resolved tenant. ``fields`` carries what the entry
    point has: ``run_id``, ``metadata`` (run metadata), ``user_input``,
    ``history``, ``context``, ``tool_results``, ``note_text``,
    ``note_metadata``.
    """


def check_run_metadata(
    agent: Agent, entry: EntryPoint, *, tenant_id: str | None, metadata: dict[str, Any] | None
) -> None:
    """Validate developer ``metadata`` where the runner merges it into ``run_meta``.

    ``tenant_id`` is the effective tenant, after delegated-child inheritance.
    Inert for now.
    """
