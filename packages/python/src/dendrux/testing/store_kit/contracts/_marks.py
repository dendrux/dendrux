"""The decorator every contract test carries: its report category and what it needs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable

F = TypeVar("F", bound="Callable[..., Any]")


def contract(
    category: str,
    *,
    needs: tuple[str, ...] = (),
    uses: tuple[str, ...] = (),
    format2: bool = False,
) -> Callable[[F], F]:
    """Mark an async contract test.

    ``category`` is the report line it counts toward. ``needs`` lists capability flags that must
    be declared (otherwise the test skips). ``uses`` lists author fixtures the test reaches for
    (``second``, ``reopen``, ``clock``, ``injector``): missing but obliged by the declaration
    fails, missing and optional skips. ``format2`` skips the test when the store creates
    storage-format-1 runs. The marks sit on the function itself so a store author may also
    import a test into their own module.
    """
    kit_mark = pytest.mark.dendrux_kit(category, needs=needs, uses=uses, format2=format2)

    def apply(function: F) -> F:
        return pytest.mark.asyncio(kit_mark(function))  # type: ignore[no-any-return]

    return apply


def basic() -> Callable[[F], F]:
    return contract("basic")
