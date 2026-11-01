"""SQLAlchemy relational state stores."""

from __future__ import annotations

from typing import Any


def tenant_scope_predicate(column: Any, tenant_scope: Any) -> Any:
    """Compare an explicit tenant exactly; UNSCOPED permits administrative reads."""
    from sqlalchemy import true

    from dendrux.runtime.state import UNSCOPED, _check_scope

    _check_scope(tenant_scope)
    return (
        true()
        if isinstance(tenant_scope, type(UNSCOPED))
        else column.is_not_distinct_from(tenant_scope)
    )
