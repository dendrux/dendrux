"""Deterministically simulate Python 3.11's wait_for cancellation race."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Awaitable

    import pytest


def simulate_lost_cancel(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make wait_for lose one cancellation, independent of the Python version."""
    lost = False

    async def wait_for(awaitable: Awaitable[Any], timeout: float | None) -> Any:
        nonlocal lost
        try:
            async with asyncio.timeout(timeout):
                return await awaitable
        except asyncio.CancelledError:
            if lost:
                raise
            lost = True
            return True

    monkeypatch.setattr(asyncio, "wait_for", wait_for)
