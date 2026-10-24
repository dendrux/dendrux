"""Entry-point check dispatcher (internal).

Every public Agent entry point calls :func:`entry_checks` first, and the
runner calls :func:`run_meta_checks` where it merges developer metadata into
``run_meta``. Both dispatch to two separately owned modules:
:mod:`dendrux.runtime._tenant_checks` (tenancy and metadata) and
:mod:`dendrux.runtime._limit_checks` (size and count caps). The modules are
looked up at call time, so tests can monkeypatch their functions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypeAlias, TypeVar

from dendrux.runtime import _limit_checks, _tenant_checks

if TYPE_CHECKING:
    from dendrux._sentinel import _OmittedType
    from dendrux.agent import Agent
    from dendrux.runtime.state import TenantScope

EntryPoint: TypeAlias = Literal[
    "run",
    "stream",
    "retry",
    "resume",
    "resume_stream",
    "submit_tool_results",
    "submit_input",
    "submit_approval",
    "submit_resume",
    "cancel_run",
    "request_pause",
    "submit_note",
]

_T = TypeVar("_T", bound="TenantScope")


def entry_checks(
    agent: Agent,
    entry: EntryPoint,
    tenant_id: _T | _OmittedType,
    *,
    default_tenant: _T,
    **fields: Any,
) -> _T:
    """Resolve the entry point's tenant and run both check modules; return the tenant."""
    resolved = _tenant_checks.resolve_tenant(agent, entry, tenant_id, default=default_tenant)
    _tenant_checks.check_entry(agent, entry, tenant_id=resolved, **fields)
    _limit_checks.check_entry(agent, entry, tenant_id=resolved, **fields)
    return resolved


def run_meta_checks(
    agent: Agent,
    entry: Literal["run", "stream", "retry"],
    *,
    tenant_id: str | None,
    metadata: dict[str, Any] | None,
) -> None:
    """Checks at the runner's ``run_meta`` merge, after tenant inheritance."""
    _tenant_checks.check_run_metadata(agent, entry, tenant_id=tenant_id, metadata=metadata)
    _limit_checks.check_run_metadata(agent, entry, tenant_id=tenant_id, metadata=metadata)
