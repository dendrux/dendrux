"""Test-controlled deadlines without changing the event loop's clock."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType

    import pytest


class ManualTimeouts:
    """Keep matching timeouts dormant until the test expires them after setup."""

    def __init__(
        self, monkeypatch: pytest.MonkeyPatch, module: ModuleType, seconds: float = 10
    ) -> None:
        self.seconds = seconds
        self.deadlines: list[asyncio.Timeout] = []
        self.armed = True
        monkeypatch.setattr(
            module, "asyncio", SimpleNamespace(**(vars(asyncio) | {"timeout": self.timeout}))
        )

    def timeout(self, seconds: float | None) -> asyncio.Timeout:
        if not self.armed or seconds != self.seconds:
            return asyncio.timeout(seconds)
        deadline = asyncio.timeout(None)
        self.deadlines.append(deadline)
        return deadline

    def expire(self) -> None:
        """Fire real asyncio cancellation at the test's chosen boundary."""
        assert self.deadlines, "the product must install a deadline"
        self.armed = False
        for deadline in self.deadlines:
            deadline.reschedule(asyncio.get_running_loop().time())
