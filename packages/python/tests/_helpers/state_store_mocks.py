"""Shared mixins for fake StateStore implementations used in tests.

Each protocol method that test fakes don't care about (cancellation,
event sequencing, etc.) gets a no-op stub here. Compose into your fake
via mixin so that adding a new protocol method only edits one place.
"""

from __future__ import annotations

from typing import Any


class CancellationStubsMixin:
    """No-op stubs for cancellation-related StateStore methods.

    Most fakes don't exercise cancellation, so they can satisfy the
    Protocol with these defaults. Override in tests that need real
    behaviour.
    """

    async def is_cancel_requested(self, run_id: str) -> bool:  # noqa: ARG002
        return False

    async def request_cancel(self, run_id: str) -> bool:  # noqa: ARG002
        return True

    async def get_next_event_sequence(self, run_id: str) -> int:  # noqa: ARG002
        return 0

    async def finalize_run_if_status_in(
        self,
        run_id: str,
        **kwargs: Any,
    ) -> bool:
        """Default fakes treat this as `finalize_run` — override if needed."""
        kwargs.pop("allowed_current_statuses", None)
        kwargs.pop("expected_current_status", None)
        finalize_run = getattr(self, "finalize_run", None)
        if finalize_run is None:
            return True
        return await finalize_run(run_id, **kwargs)
