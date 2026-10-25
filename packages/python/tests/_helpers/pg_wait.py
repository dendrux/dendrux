"""Bounded observation of asynchronous server-side connection retirement."""

import asyncio
import time
from collections.abc import Awaitable, Callable


async def wait_pg_count(
    count: Callable[[], Awaitable[int]], expected: int, *, timeout: float = 10.0
) -> int:
    deadline = time.monotonic() + timeout
    observed: int | None = None
    while True:
        try:
            observed = await asyncio.wait_for(
                count(), timeout=max(0.0, deadline - time.monotonic())
            )
        except TimeoutError:
            raise AssertionError(
                f"PG sessions: observed {observed}, expected {expected} after {timeout}s"
            ) from None
        if observed == expected:
            return observed
        remaining = deadline - time.monotonic()
        assert remaining > 0, (
            f"PG sessions: observed {observed}, expected {expected} after {timeout}s"
        )
        await asyncio.sleep(min(0.05, remaining))
