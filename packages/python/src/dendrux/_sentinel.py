"""Sentinel type for "not provided" parameters.

Internal module — not part of the public API. Provides a typed sentinel
that distinguishes "caller didn't pass this argument" from ``None`` or
any other explicit value.

Each module that needs a sentinel instantiates its own ``_UnsetType()``
with a descriptive variable name (``_UNSET``, ``_UNSET_DEPTH``) for
readability at the call site. Narrowing uses ``isinstance(_UnsetType)``
because mypy does not narrow on ``is`` for non-enum types.
"""

from __future__ import annotations


class _UnsetType:
    """Typed sentinel — replaces untyped ``object()`` sentinels.

    Using a dedicated type instead of ``Any = object()`` means mypy can
    track where sentinels flow and flag accidental leaks past the
    boundary (e.g. returning ``_UNSET`` when the caller expects ``int | None``).

    Public method signatures use ``@overload`` to hide ``_UnsetType``
    from static checkers — callers see only the clean types (``int | None``,
    etc.). Runtime introspection (``inspect.signature``, ``help()``) still
    shows the implementation signature; the ``_`` prefix signals "internal."

    Usage::

        from dendrux._sentinel import _UnsetType

        _UNSET = _UnsetType()

        # Implementation signature (hidden behind @overload for static checkers)
        def foo(value: int | _UnsetType = _UNSET) -> int:
            if isinstance(value, _UnsetType):
                return 42
            return value
    """

    __slots__ = ()

    def __repr__(self) -> str:
        return "<UNSET>"
