"""Select the warning under test without counting process-wide warning noise."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable
    from warnings import WarningMessage


def warnings_of(
    captured: Iterable[WarningMessage], category: type[Warning], match: str | None = None
) -> list[WarningMessage]:
    """Return warnings of ``category`` (including subclasses) matching an optional regex."""
    pattern = re.compile(match) if match is not None else None
    return [
        item
        for item in captured
        if issubclass(item.category, category)
        and (pattern is None or pattern.search(str(item.message)) is not None)
    ]
