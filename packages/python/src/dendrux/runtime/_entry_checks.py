"""Entry-point check dispatcher (internal).

Every public Agent entry point calls :func:`entry_checks` first, before any
I/O. The runner's ``run`` / ``retry`` / ``run_stream`` call
:func:`runner_entry_checks` right after the delegated-child tenant is
resolved, before any I/O, so direct ``dendrux.run(...)`` callers, runs without
a store and delegated children are covered too.

Both dispatch to separately owned modules:
:mod:`dendrux.runtime._tenant_checks` (tenancy and metadata),
:mod:`dendrux.runtime._limit_checks` (size and count caps) and, for new runs,
:func:`dendrux.pricing.check_budget_config` (B4). The modules are looked up at
call time, so tests can monkeypatch their functions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypeAlias, TypeVar, cast

from dendrux import pricing
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
RunnerEntryPoint: TypeAlias = Literal["run", "stream", "retry"]

_NEW_RUN_ENTRIES = frozenset({"run", "stream", "retry"})

_T = TypeVar("_T", bound="TenantScope")


def entry_checks(
    agent: Agent,
    entry: EntryPoint,
    tenant_id: _T | _OmittedType,
    *,
    default_tenant: _T,
    **fields: Any,
) -> _T:
    """Resolve the entry point's tenant and run every check module; return the tenant.

    For ``run`` / ``stream`` / ``retry`` it also calls
    ``pricing.check_budget_config`` with the per-call ``model=`` override from
    ``provider_kwargs`` (None when the call uses the provider's model).
    """
    resolved = _tenant_checks.resolve_tenant(agent, entry, tenant_id, default=default_tenant)
    _tenant_checks.check_entry(agent, entry, tenant_id=resolved, **fields)
    _limit_checks.check_entry(agent, entry, tenant_id=resolved, **fields)
    if entry in _NEW_RUN_ENTRIES:
        provider_kwargs = fields.get("provider_kwargs") or {}
        model = provider_kwargs.get("model")
        pricing.check_budget_config(agent, model=model if isinstance(model, str) else None)
    return resolved


def runner_entry_checks(
    agent: Agent,
    entry: RunnerEntryPoint,
    *,
    tenant_id: str | None,
    metadata: dict[str, Any] | None,
    **fields: Any,
) -> None:
    """Checks at the runner's entry, after tenant inheritance and before any I/O.

    ``tenant_id`` is the effective tenant (a delegated child's is its
    parent's). ``fields`` carries ``user_input`` (run, stream) and
    ``provider_kwargs``.
    """
    _tenant_checks.check_runner_entry(
        agent, entry, tenant_id=tenant_id, metadata=metadata, **fields
    )
    _limit_checks.check_runner_entry(agent, entry, tenant_id=tenant_id, metadata=metadata, **fields)
    check_runner_budget(agent, fields.get("provider"), fields.get("provider_kwargs"))


def require_boolean(value: Any, field: str, *, optional: bool = False) -> None:
    """Do not interpret truthy client data as a control-plane decision."""
    if optional and value is None:
        return
    if type(value) is not bool:
        raise ValueError(f"{field} must be a bool")


def resolve_runner_tenant(
    agent: Agent, entry: RunnerEntryPoint, tenant_id: str | None | _OmittedType
) -> str | None:
    return _tenant_checks.resolve_tenant(agent, entry, tenant_id, default=None)


class _BudgetView:
    """Only the pricing policy's data, using the provider of this execution."""

    def __init__(self, agent: Agent, provider: Any) -> None:
        self.budget = agent.budget
        self.pricing = agent.pricing
        self.provider = provider


def check_runner_budget(agent: Agent, provider: Any, kwargs: dict[str, Any] | None) -> None:
    target = (
        agent
        if provider is None or provider is agent.provider
        else cast("Agent", _BudgetView(agent, provider))
    )
    model = (kwargs or {}).get("model")
    pricing.check_budget_config(target, model=model if isinstance(model, str) else None)
