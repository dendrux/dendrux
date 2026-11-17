"""Readiness gates with bounded waits and early task-failure diagnostics."""

import asyncio

GATE_TIMEOUT = 30


async def wait_for_gate(
    event: asyncio.Event, *, task: asyncio.Task[object] | None = None, timeout: float = GATE_TIMEOUT
) -> None:
    """Wait for readiness without cancelling the caller-owned task.

    A reached gate wins if both the event and run finish in the same loop turn.
    Otherwise propagate an early run failure, or explain an early end or timeout.
    """
    if event.is_set():
        return
    waiter = asyncio.create_task(event.wait())
    try:
        watched = {waiter} if task is None else {waiter, task}
        done, _ = await asyncio.wait(watched, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
        if event.is_set() or (waiter in done and waiter.result()):
            return
        if task is not None and task in done:
            task.result()
            raise AssertionError("run ended before gate was reached")
        raise AssertionError(f"timed out waiting for gate after {timeout:g} seconds")
    finally:
        waiter.cancel()
        await asyncio.gather(waiter, return_exceptions=True)
