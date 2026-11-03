"""Provider-call bookkeeping shared by the ReAct and SingleCall loops.

The failure record, the image-retry decision and the priced response are the
same in every loop and mode; only what a stop means differs (ReAct pauses or
cancels, SingleCall honours cancel only), so callers keep that part. Nothing
here drives a provider stream: pulls and closes stay in the caller's task.

This module sits above ``_images`` (which imports ``_helpers``), so these
helpers cannot live in ``_helpers`` without an import cycle.
"""

from __future__ import annotations

import time
import traceback
from typing import TYPE_CHECKING, Any

from dendrux.llm._evidence import persistable_response
from dendrux.loops._helpers import notify_llm_failed, price_response, record_llm_failed
from dendrux.loops._images import control_stop

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from dendrux.agent import Agent
    from dendrux.llm.base import LLMProvider
    from dendrux.loops._images import ImageCallSite, RetryDecision
    from dendrux.loops.base import LoopNotifier, LoopRecorder
    from dendrux.media._prepared import PreparedRequest
    from dendrux.types import LLMResponse


def drop_frame_locals(exc: BaseException) -> None:
    """Clear finished frames' locals in ``exc``'s traceback (they may hold wire bytes)."""
    traceback.clear_frames(exc.__traceback__)


class ProviderCallSite:
    """One run's provider calls: failure records, the retry decision, the priced response."""

    __slots__ = ("agent", "images", "notifier", "pkw", "provider", "recorder", "run_id", "warnings")

    def __init__(
        self,
        images: ImageCallSite,
        *,
        agent: Agent,
        provider: LLMProvider,
        pkw: dict[str, Any],
        recorder: LoopRecorder | None,
        notifier: LoopNotifier | None,
        run_id: str,
        warnings: list[str],
    ) -> None:
        self.images = images
        self.agent = agent
        self.provider = provider
        self.pkw = pkw
        self.recorder = recorder
        self.notifier = notifier
        self.run_id = run_id
        self.warnings = warnings

    async def record_failure(self, exc: BaseException, t0: float, iteration: int) -> None:
        """``llm.failed`` for a provider call that raised: record, then notify."""
        duration_ms = int((time.monotonic() - t0) * 1000)
        await record_llm_failed(self.recorder, self.run_id, iteration, exc, duration_ms=duration_ms)
        await notify_llm_failed(
            self.notifier, self.run_id, iteration, exc, self.warnings, duration_ms=duration_ms
        )

    async def decide(
        self,
        failure: Exception,
        prepared: PreparedRequest,
        iteration: int,
        *,
        retried: bool,
        visible_deltas: int,
        control_reader: Callable[[], Awaitable[Any]],
    ) -> tuple[Any, BaseException | None, RetryDecision]:
        """What a failed call does next; call it outside the ``except`` block.

        Returns ``(stop, error, decision)``. ``stop`` is the cancel or pause
        control seen before a retry (the caller ends the run its own way).
        Otherwise ``error`` is what to raise: ``failure`` itself (its frames'
        locals cleared) or the decision's fresh, cause-free error. With
        neither, the caller retries with ``decision.tombstones``. The caller
        drops its own ``failure`` before raising, so the raised error's
        traceback never keeps the provider failure alive.
        """
        decision = await self.images.on_failure(
            failure,
            prepared,
            iteration=iteration,
            retried=retried,
            visible_deltas=visible_deltas,
            control_reader=control_reader,
        )
        stop = control_stop(decision)
        if stop is not None:
            return stop, None, decision
        if decision.action == "propagate":
            drop_frame_locals(failure)
            return None, failure, decision
        if decision.action == "raise":
            assert decision.error is not None
            return None, decision.error, decision
        return None, None, decision

    def settle(
        self, prepared: PreparedRequest, response: LLMResponse, t0: float
    ) -> tuple[PreparedRequest, int, LLMResponse]:
        """After the provider returns: commit the image view, time the call, price it."""
        prepared = self.images.mark_sent(prepared)
        duration_ms = int((time.monotonic() - t0) * 1000)
        response = persistable_response(response)
        response = price_response(
            response,
            pricing=self.agent.pricing,
            default_model=self.provider.model,
            model=self.pkw.get("model"),
        )
        return prepared, duration_ms, response
