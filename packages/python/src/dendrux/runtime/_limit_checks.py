"""Size and count caps at Agent entry points (internal).

Called through :mod:`dendrux.runtime._entry_checks`. Inert until the 0.5.0
limits work fills it in: ``Limits.max_input_bytes`` for text input and
``Limits.max_notes_per_run`` for notes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dendrux.agent import Agent
    from dendrux.runtime._entry_checks import EntryPoint
    from dendrux.runtime.state import TenantScope


def check_entry(agent: Agent, entry: EntryPoint, *, tenant_id: TenantScope, **fields: Any) -> None:
    """Enforce input caps before any I/O. Inert for now.

    Receives the same ``fields`` as
    :func:`dendrux.runtime._tenant_checks.check_entry`.
    """


def check_run_metadata(
    agent: Agent, entry: EntryPoint, *, tenant_id: str | None, metadata: dict[str, Any] | None
) -> None:
    """Hook at the runner's ``run_meta`` merge. Inert for now."""
