"""Loop-level retry telemetry: a provider retry reaches the run's recorder and notifier.

A real OpenAI Chat adapter on an httpx mock transport answers 503 once, then
200. The SDK retries; the transport hook must turn the 503 into exactly one
``provider.retry`` governance event on the recorder and one on the notifier,
through each loop's telemetry call sites (ReAct and SingleCall, batch and
stream). Pinned before the H8 telemetry seam moved.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

pytest.importorskip("openai", reason="openai extra not installed")

from dendrux import Agent  # noqa: E402
from dendrux.llm.openai import OpenAIProvider  # noqa: E402
from dendrux.loops.base import BaseNotifier, BaseRecorder  # noqa: E402
from dendrux.loops.react import ReActLoop  # noqa: E402
from dendrux.loops.single import SingleCall  # noqa: E402
from dendrux.strategies.native import NativeToolCalling  # noqa: E402
from dendrux.types import RunEventType, RunStatus  # noqa: E402

RUN_ID = "01JH8RETRY0000000000000000"

GovernanceCall = tuple[str, str, int, dict[str, Any], str | None]


class _Recorder(BaseRecorder):
    def __init__(self) -> None:
        self.governance: list[GovernanceCall] = []

    async def on_governance_event(
        self,
        run_id: str,
        event_type: str,
        iteration: int,
        data: dict[str, Any],
        correlation_id: str | None = None,
    ) -> None:
        self.governance.append((run_id, event_type, iteration, data, correlation_id))


class _Notifier(BaseNotifier):
    def __init__(self) -> None:
        self.governance: list[GovernanceCall] = []

    async def on_governance_event(
        self,
        run_id: str,
        event_type: str,
        iteration: int,
        data: dict[str, Any],
        correlation_id: str | None = None,
    ) -> None:
        self.governance.append((run_id, event_type, iteration, data, correlation_id))


def _completion() -> dict[str, Any]:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": "gpt-4o",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "All done."},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
    }


def _sse() -> bytes:
    chunks = [
        {
            "id": "c1",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "gpt-4o",
            "choices": [{"index": 0, "delta": {"content": "All done."}, "finish_reason": None}],
        },
        {
            "id": "c1",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "gpt-4o",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        },
    ]
    body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
    return body.encode()


def _provider(requests: list[httpx.Request]) -> OpenAIProvider:
    provider = OpenAIProvider(model="gpt-4o", api_key="test-key", max_retries=1)

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(503, headers={"retry-after-ms": "1"}, json={"error": "busy"})
        if json.loads(request.content).get("stream"):
            return httpx.Response(
                200, content=_sse(), headers={"content-type": "text/event-stream"}
            )
        return httpx.Response(200, json=_completion())

    # Keep the provider's own client (and its telemetry hooks); swap only the transport.
    provider._client._client._transport = httpx.MockTransport(handler)
    return provider


EXPECTED_DATA = {
    "status_code": 503,
    "endpoint": "/v1/chat/completions",
    "attempt": 1,
    "retry_after": None,
}


def _assert_one_retry(recorder: _Recorder, notifier: _Notifier) -> None:
    retries = [g for g in recorder.governance if g[1] == "provider.retry"]
    assert retries == [(RUN_ID, "provider.retry", 1, EXPECTED_DATA, None)]
    broadcasts = [g for g in notifier.governance if g[1] == "provider.retry"]
    assert broadcasts == retries


@pytest.mark.parametrize("loop_cls", [ReActLoop, SingleCall], ids=["react", "single"])
async def test_batch_retry_reaches_recorder_and_notifier(loop_cls: type) -> None:
    requests: list[httpx.Request] = []
    provider = _provider(requests)
    recorder, notifier = _Recorder(), _Notifier()
    agent = Agent(name="h8", prompt="Be brief.", provider=provider)
    result = await loop_cls().run(
        agent=agent,
        provider=provider,
        strategy=NativeToolCalling(),
        user_input="go",
        run_id=RUN_ID,
        recorder=recorder,
        notifier=notifier,
    )
    assert result.status == RunStatus.SUCCESS
    assert len(requests) == 2
    _assert_one_retry(recorder, notifier)


@pytest.mark.parametrize("loop_cls", [ReActLoop, SingleCall], ids=["react", "single"])
async def test_stream_retry_reaches_recorder_and_notifier(loop_cls: type) -> None:
    requests: list[httpx.Request] = []
    provider = _provider(requests)
    recorder, notifier = _Recorder(), _Notifier()
    agent = Agent(name="h8", prompt="Be brief.", provider=provider)
    events = [
        e
        async for e in loop_cls().run_stream(
            agent=agent,
            provider=provider,
            strategy=NativeToolCalling(),
            user_input="go",
            run_id=RUN_ID,
            recorder=recorder,
            notifier=notifier,
        )
    ]
    assert events[-1].type == RunEventType.RUN_COMPLETED
    assert len(requests) == 2
    _assert_one_retry(recorder, notifier)
