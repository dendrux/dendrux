# ruff: noqa: E501 — expected traces are captured lines, kept whole for diffing
"""Pin ``SingleCall.run`` end to end, captured before H7b moved its provider call out.

Each cell traces the LLM lifecycle records, governance events and the lease
scope (via the stream pins' probe recorder), every provider call with the
telemetry binding and task it saw, and a summary of the result or the error.
Error cells also check that no dendrux traceback frame still holds the
provider failure or the image-bearing wire request.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from pydantic import BaseModel

from dendrux import Agent
from dendrux.errors import ImageRejectedError, UnsupportedModalityError
from dendrux.guardrails import PII, SecretDetection
from dendrux.loops.base import BaseNotifier
from dendrux.loops.single import SingleCall
from dendrux.strategies.native import NativeToolCalling
from dendrux.types import (
    ImageViewState,
    LLMResponse,
    ProviderCapabilities,
    RunResult,
    StructuredOutputValidationError,
    ToolCall,
    UsageStats,
)
from tests.unit.test_guardrail_images import RAW, NsfwScan
from tests.unit.test_h7b_stream_call_pins import (
    ProbeRecorder,
    Trace,
    _completion_record_fails,
    _holders,
    _sent_image_bytes,
)
from tests.unit.test_loop_images import (
    ControlStore,
    ImageLLM,
    make_png,
    make_runtime,
    rejected,
    store_image,
    user_msg,
)


class Out(BaseModel):
    label: str


@pytest.fixture
def trace() -> Trace:
    return Trace()


def _norm(value: Any) -> Any:
    """Sorted lists and keys, so set-built entity lists compare across hash seeds."""
    if isinstance(value, dict):
        return {k: _norm(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return sorted((_norm(v) for v in value), key=lambda v: json.dumps(v, default=str))
    return value


def _data(value: Any) -> str:
    return json.dumps(_norm(value), sort_keys=True, default=str)


class BatchRecorder(ProbeRecorder):
    async def on_governance_event(
        self, run_id: str, event_type: str, iteration: int, data: Any, **kw: Any
    ) -> None:
        self.trace.lines.append(f"gov:{event_type} {_data(data)}")

    async def on_llm_call_completed(
        self, run_id: str, response: LLMResponse, iteration: int, **kw: Any
    ) -> None:
        self.trace.lines.append(
            f"rec.completed{iteration} {response.text!r} "
            f"findings={_data(kw.get('guardrail_findings'))} ms={type(kw.get('duration_ms'))}"
        )


class BatchNotifier(BaseNotifier):
    def __init__(self, trace: Trace) -> None:
        self.trace = trace

    async def on_llm_call_started(self, run_id: str, iteration: int, **kw: Any) -> None:
        self.trace.lines.append(f"not.started{iteration}")

    async def on_llm_call_completed(
        self, run_id: str, response: LLMResponse, iteration: int, **kw: Any
    ) -> None:
        self.trace.lines.append(
            f"not.completed{iteration} {response.text!r} "
            f"findings={_data(kw.get('guardrail_findings'))}"
        )

    async def on_llm_call_failed(
        self, run_id: str, iteration: int, error: BaseException, **kw: Any
    ) -> None:
        self.trace.lines.append(f"not.failed{iteration}:{type(error).__name__}")

    async def on_governance_event(
        self, run_id: str, event_type: str, iteration: int, data: Any, **kw: Any
    ) -> None:
        self.trace.lines.append(f"not.gov:{event_type} {_data(data)}")


class BatchProbe(ImageLLM):
    def __init__(self, script: list[Any], trace: Trace, **kw: Any) -> None:
        super().__init__(script, **kw)
        self.trace = trace
        self.wires: list[Any] = []
        self.capabilities = ProviderCapabilities(
            supports_native_tools=True,
            supports_tool_call_ids=True,
            supports_multimodal=True,
            supports_structured_output=True,
        )

    async def complete(self, messages: Any, tools: Any = None, **kw: Any) -> LLMResponse:
        self.wires.append(messages)
        self.trace.mark(f"call{len(self.wires)}")
        await asyncio.sleep(0)
        return await super().complete(messages, tools, **kw)


def _usage() -> UsageStats:
    return UsageStats(input_tokens=10, output_tokens=2, total_tokens=12)


def _summary(result: RunResult) -> str:
    return (
        f"RESULT {result.status.value} it={result.iteration_count} answer={result.answer!r} "
        f"output={result.output!r} error={result.error!r} steps={len(result.steps)} "
        f"tokens={result.usage.total_tokens} meta={sorted(result.meta)} "
        f"reason={result.meta.get('failure_reason')}"
    )


async def _run(
    tr: Trace,
    script: list[Any],
    *,
    image: bytes | None = None,
    view: ImageViewState | None = None,
    store: Any = None,
    output_type: type[BaseModel] | None = None,
    guardrails: list[Any] | None = None,
    user_input: str = "x",
) -> tuple[BatchProbe, BaseException | None]:
    rt = make_runtime() if image is not None else None
    history = None
    if rt is not None and image is not None:
        history = [user_msg("q", await store_image(rt, image))]
    probe = BatchProbe(script, tr, support="unknown")
    agent = Agent(
        provider=probe,
        prompt="p",
        loop=SingleCall(),
        guardrails=guardrails,
        output_type=output_type,
    )
    tr.consumer = asyncio.current_task()
    try:
        result = await SingleCall().run(
            agent=agent,
            provider=probe,
            strategy=NativeToolCalling(),
            user_input=user_input,
            initial_history=history,
            recorder=BatchRecorder(tr),
            notifier=BatchNotifier(tr),
            image_runtime=rt,
            image_view_state=view if view is not None else ImageViewState(sent_through=1),
            state_store=store,
            output_type=output_type,
        )
    except Exception as exc:  # noqa: BLE001 — the trace records what escaped
        tr.lines.append(f"RAISED {type(exc).__name__}")
        return probe, exc
    tr.lines.append(_summary(result))
    return probe, None


def _unsupported() -> UnsupportedModalityError:
    return UnsupportedModalityError(provider="x", model="claude-test", source="provider_response")


PNG = make_png(seed=7400)


async def test_text(trace: Trace) -> None:
    _, err = await _run(trace, [LLMResponse(text="ok", usage=_usage())])
    assert err is None
    assert trace.lines == TEXT


async def test_structured_output(trace: Trace) -> None:
    _, err = await _run(trace, [LLMResponse(text='{"label": "a"}')], output_type=Out)
    assert err is None
    assert trace.lines == STRUCTURED


async def test_structured_output_rejected(trace: Trace) -> None:
    _, err = await _run(trace, [LLMResponse(text="nope", usage=_usage())], output_type=Out)
    assert isinstance(err, StructuredOutputValidationError)
    assert trace.lines == STRUCTURED_REJECTED


async def test_unexpected_tool_calls(trace: Trace) -> None:
    _, err = await _run(trace, [LLMResponse(tool_calls=[ToolCall(name="t", params={})])])
    assert isinstance(err, RuntimeError) and "unexpected tool_calls" in str(err)
    assert trace.lines == TOOL_CALLS


async def test_generic_error(trace: Trace) -> None:
    boom = RuntimeError("boom")
    _, err = await _run(trace, [boom])
    assert err is boom
    assert trace.lines == GENERIC


async def test_image_retry(trace: Trace) -> None:
    _, err = await _run(trace, [_unsupported(), LLMResponse(text="ok")], image=PNG)
    assert err is None
    assert trace.lines == RETRY


async def test_image_retry_fails_again(trace: Trace) -> None:
    first, second = _unsupported(), _unsupported()
    probe, err = await _run(trace, [first, second], image=PNG)
    assert err is second
    assert _holders(err, [first, *probe.wires]) == []
    assert trace.lines == RETRY_FAILS


async def test_user_image_rejection_raises_fresh(trace: Trace) -> None:
    first = rejected()
    probe, err = await _run(trace, [first], image=PNG, view=ImageViewState())
    assert isinstance(err, ImageRejectedError) and err is not first
    assert err.__context__ is None and err.__cause__ is None
    assert _holders(err, [first, *probe.wires]) == []
    assert trace.lines == REJECTED


@pytest.mark.parametrize("control", ["cancel", "pause"])
async def test_control_before_retry(trace: Trace, control: str) -> None:
    store = ControlStore()
    setattr(store, f"{control}_after", 0)
    _, err = await _run(trace, [_unsupported(), LLMResponse(text="ok")], image=PNG, store=store)
    assert err is None and store.reads == 1
    assert trace.lines == (CANCEL_BEFORE_RETRY if control == "cancel" else RETRY)


async def test_image_guardrail_block(trace: Trace) -> None:
    _, err = await _run(
        trace,
        [LLMResponse(text="never")],
        image=RAW,
        view=ImageViewState(),
        guardrails=[NsfwScan("block")],
    )
    assert err is None
    assert trace.lines == IMAGE_BLOCK


async def test_incoming_redact_and_outgoing_detect(trace: Trace) -> None:
    _, err = await _run(
        trace,
        [LLMResponse(text="reply to bob@example.com")],
        guardrails=[PII(action="redact")],
        user_input="Email jane@example.com",
    )
    assert err is None
    assert trace.lines == REDACT


async def test_incoming_block(trace: Trace) -> None:
    _, err = await _run(
        trace,
        [LLMResponse(text="never")],
        guardrails=[PII(action="redact"), SecretDetection(action="block")],
        user_input="Email jane@example.com with key AKIAIOSFODNN7EXAMPLE",
    )
    assert err is None
    assert trace.lines == INCOMING_BLOCK


async def test_outgoing_block(trace: Trace) -> None:
    _, err = await _run(
        trace,
        [LLMResponse(text="key AKIAIOSFODNN7EXAMPLE", usage=_usage())],
        guardrails=[SecretDetection(action="block")],
    )
    assert err is None
    assert trace.lines == OUTGOING_BLOCK


async def test_success_then_record_error_holds_no_wire(
    trace: Trace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An image call succeeds, then recording it fails: no frame keeps the image request."""
    monkeypatch.setattr(BatchRecorder, "on_llm_call_completed", _completion_record_fails)
    probe, err = await _run(trace, [LLMResponse(text="ok")], image=PNG)
    assert isinstance(err, RuntimeError) and str(err) == "completion record unavailable"
    assert _sent_image_bytes(probe.wires)
    assert _holders(err, probe.wires) == []


# --------------------------------------------------------------------------- expected traces
# Captured from the code before H7b's SingleCall.run change (877d2892).

TEXT = [
    "rec.msg0:user",
    "rec.started1",
    "not.started1",
    "lease+ t1s",
    "call1 t1s",
    "lease- t1s",
    "rec.completed1 'ok' findings=null ms=<class 'int'>",
    "not.completed1 'ok' findings=null",
    "rec.msg1:assistant",
    "RESULT success it=1 answer='ok' output=None error=None steps=0 tokens=12 meta=[] reason=None",
]
STRUCTURED = [
    "rec.msg0:user",
    "rec.started1",
    "not.started1",
    "lease+ t1s",
    "call1 t1s",
    "lease- t1s",
    "rec.completed1 '{\"label\": \"a\"}' findings=null ms=<class 'int'>",
    'not.completed1 \'{"label": "a"}\' findings=null',
    "rec.msg1:assistant",
    "RESULT success it=1 answer='{\"label\": \"a\"}' output=Out(label='a') error=None steps=0 tokens=0 meta=[] reason=None",
]
STRUCTURED_REJECTED = [
    "rec.msg0:user",
    "rec.started1",
    "not.started1",
    "lease+ t1s",
    "call1 t1s",
    "lease- t1s",
    "rec.completed1 'nope' findings=null ms=<class 'int'>",
    "not.completed1 'nope' findings=null",
    "RAISED StructuredOutputValidationError",
]
TOOL_CALLS = [
    "rec.msg0:user",
    "rec.started1",
    "not.started1",
    "lease+ t1s",
    "call1 t1s",
    "lease- t1s",
    "rec.failed1:RuntimeError",
    "not.failed1:RuntimeError",
    "RAISED RuntimeError",
]
GENERIC = [
    "rec.msg0:user",
    "rec.started1",
    "not.started1",
    "lease+ t1s",
    "call1 t1s",
    "lease- t1s",
    "rec.failed1:RuntimeError",
    "not.failed1:RuntimeError",
    "RAISED RuntimeError",
]
RETRY = [
    "rec.started1",
    "not.started1",
    "lease+ t1s",
    "call1 t1s",
    "lease- t1s",
    "rec.failed1:UnsupportedModalityError",
    "not.failed1:UnsupportedModalityError",
    'gov:image.degraded {"count": 1, "reason": "unsupported", "slots": [0]}',
    'not.gov:image.degraded {"count": 1, "reason": "unsupported", "slots": [0]}',
    'gov:image.state {"described": [], "guardrail_verdicts": [], "placeholders": [{"reason": "unsupported", "slot": 0}], "tenant_scope": "t:acme"}',
    'not.gov:image.state {"described": [], "guardrail_verdicts": [], "placeholders": [{"reason": "unsupported", "slot": 0}], "tenant_scope": "t:acme"}',
    "rec.started1",
    "not.started1",
    "lease+ t1s",
    "call2 t1s",
    "lease- t1s",
    "rec.completed1 'ok' findings=null ms=<class 'int'>",
    "not.completed1 'ok' findings=null",
    "rec.msg1:assistant",
    "RESULT success it=1 answer='ok' output=None error=None steps=0 tokens=0 meta=[] reason=None",
]
RETRY_FAILS = [
    "rec.started1",
    "not.started1",
    "lease+ t1s",
    "call1 t1s",
    "lease- t1s",
    "rec.failed1:UnsupportedModalityError",
    "not.failed1:UnsupportedModalityError",
    'gov:image.degraded {"count": 1, "reason": "unsupported", "slots": [0]}',
    'not.gov:image.degraded {"count": 1, "reason": "unsupported", "slots": [0]}',
    'gov:image.state {"described": [], "guardrail_verdicts": [], "placeholders": [{"reason": "unsupported", "slot": 0}], "tenant_scope": "t:acme"}',
    'not.gov:image.state {"described": [], "guardrail_verdicts": [], "placeholders": [{"reason": "unsupported", "slot": 0}], "tenant_scope": "t:acme"}',
    "rec.started1",
    "not.started1",
    "lease+ t1s",
    "call2 t1s",
    "lease- t1s",
    "rec.failed1:UnsupportedModalityError",
    "not.failed1:UnsupportedModalityError",
    "RAISED UnsupportedModalityError",
]
REJECTED = [
    "rec.started1",
    "not.started1",
    "lease+ t1s",
    "call1 t1s",
    "lease- t1s",
    "rec.failed1:ImageRejectedError",
    "not.failed1:ImageRejectedError",
    'gov:image.rejected {"image_id": "img_f59cae83488b", "located": false, "model": "claude-test", "provider": "ImageLLM", "retried": false, "sha256": "f59cae83488b67dcda4688a1a188a08a9c5192c2bf98740049aa45fc884f25b7", "status_code": 400, "tenant_scope": "t:acme"}',
    'not.gov:image.rejected {"image_id": "img_f59cae83488b", "located": false, "model": "claude-test", "provider": "ImageLLM", "retried": false, "sha256": "f59cae83488b67dcda4688a1a188a08a9c5192c2bf98740049aa45fc884f25b7", "status_code": 400, "tenant_scope": "t:acme"}',
    "RAISED ImageRejectedError",
]
CANCEL_BEFORE_RETRY = [
    "rec.started1",
    "not.started1",
    "lease+ t1s",
    "call1 t1s",
    "lease- t1s",
    "rec.failed1:UnsupportedModalityError",
    "not.failed1:UnsupportedModalityError",
    "RESULT cancelled it=0 answer=None output=None error=None steps=0 tokens=0 meta=[] reason=None",
]
IMAGE_BLOCK = [
    'gov:image.state {"described": [], "guardrail_verdicts": [{"action": "block", "entities": ["NSFW"], "guardrail_id": "acme.NsfwScan", "guardrail_version": "1", "origin": "user", "sha256": "ac1690467bb41327ecc71edea9520850fcd9981fec8b318a467abaf3eba37379", "tool_name": ""}], "placeholders": [], "tenant_scope": "t:acme"}',
    'not.gov:image.state {"described": [], "guardrail_verdicts": [{"action": "block", "entities": ["NSFW"], "guardrail_id": "acme.NsfwScan", "guardrail_version": "1", "origin": "user", "sha256": "ac1690467bb41327ecc71edea9520850fcd9981fec8b318a467abaf3eba37379", "tool_name": ""}], "placeholders": [], "tenant_scope": "t:acme"}',
    'gov:guardrail.detected {"direction": "incoming_image", "entities": ["NSFW"], "findings_count": 1}',
    'not.gov:guardrail.detected {"direction": "incoming_image", "entities": ["NSFW"], "findings_count": 1}',
    'gov:guardrail.blocked {"code": "guardrail_blocked", "counts": {"NSFW": 1}, "direction": "incoming_image", "error": "Guardrail \'acme.NsfwScan\' blocked image img_ac1690467bb4: NSFW", "kinds": ["NSFW"], "location": "incoming_image"}',
    'not.gov:guardrail.blocked {"code": "guardrail_blocked", "counts": {"NSFW": 1}, "direction": "incoming_image", "error": "Guardrail \'acme.NsfwScan\' blocked image img_ac1690467bb4: NSFW", "kinds": ["NSFW"], "location": "incoming_image"}',
    "RESULT error it=1 answer=None output=None error=\"Guardrail 'acme.NsfwScan' blocked image img_ac1690467bb4: NSFW\" steps=0 tokens=0 meta=['failure_reason', 'pii_mapping'] reason=guardrail_blocked",
]
REDACT = [
    "rec.msg0:user",
    'gov:guardrail.detected {"direction": "incoming", "entities": ["EMAIL_ADDRESS"], "findings_count": 1}',
    'not.gov:guardrail.detected {"direction": "incoming", "entities": ["EMAIL_ADDRESS"], "findings_count": 1}',
    'gov:guardrail.redacted {"direction": "incoming", "entities": ["EMAIL_ADDRESS"]}',
    'not.gov:guardrail.redacted {"direction": "incoming", "entities": ["EMAIL_ADDRESS"]}',
    "rec.started1",
    "not.started1",
    "lease+ t1s",
    "call1 t1s",
    "lease- t1s",
    'gov:guardrail.detected {"direction": "outgoing", "entities": ["EMAIL_ADDRESS"], "findings_count": 1}',
    'not.gov:guardrail.detected {"direction": "outgoing", "entities": ["EMAIL_ADDRESS"], "findings_count": 1}',
    'rec.completed1 \'reply to bob@example.com\' findings={"incoming": [{"entity_type": "EMAIL_ADDRESS", "score": 1.0}], "outgoing": [{"entity_type": "EMAIL_ADDRESS", "score": 1.0}]} ms=<class \'int\'>',
    'not.completed1 \'reply to bob@example.com\' findings={"incoming": [{"entity_type": "EMAIL_ADDRESS", "score": 1.0}], "outgoing": [{"entity_type": "EMAIL_ADDRESS", "score": 1.0}]}',
    "rec.msg1:assistant",
    "RESULT success it=1 answer='reply to bob@example.com' output=None error=None steps=0 tokens=0 meta=['pii_mapping'] reason=None",
]
INCOMING_BLOCK = [
    "rec.msg0:user",
    'gov:guardrail.blocked {"code": "guardrail_blocked", "counts": {"AWS_ACCESS_KEY": 1, "EMAIL_ADDRESS": 1}, "direction": "incoming", "error": "Guardrail \'SecretDetection\' blocked: 1 AWS_ACCESS_KEY at message 1 (role=user)", "kinds": ["AWS_ACCESS_KEY", "EMAIL_ADDRESS"], "location": "message 1 (role=user)"}',
    'not.gov:guardrail.blocked {"code": "guardrail_blocked", "counts": {"AWS_ACCESS_KEY": 1, "EMAIL_ADDRESS": 1}, "direction": "incoming", "error": "Guardrail \'SecretDetection\' blocked: 1 AWS_ACCESS_KEY at message 1 (role=user)", "kinds": ["AWS_ACCESS_KEY", "EMAIL_ADDRESS"], "location": "message 1 (role=user)"}',
    "RESULT error it=1 answer=None output=None error=\"Guardrail 'SecretDetection' blocked: 1 AWS_ACCESS_KEY at message 1 (role=user)\" steps=0 tokens=0 meta=['failure_reason', 'pii_mapping'] reason=guardrail_blocked",
]
OUTGOING_BLOCK = [
    "rec.msg0:user",
    "rec.started1",
    "not.started1",
    "lease+ t1s",
    "call1 t1s",
    "lease- t1s",
    'gov:guardrail.detected {"direction": "outgoing", "entities": ["AWS_ACCESS_KEY"], "findings_count": 1}',
    'not.gov:guardrail.detected {"direction": "outgoing", "entities": ["AWS_ACCESS_KEY"], "findings_count": 1}',
    'rec.completed1 \'[blocked by guardrail]\' findings={"outgoing": [{"entity_type": "AWS_ACCESS_KEY", "score": 1.0}]} ms=<class \'int\'>',
    'not.completed1 \'[blocked by guardrail]\' findings={"outgoing": [{"entity_type": "AWS_ACCESS_KEY", "score": 1.0}]}',
    'gov:guardrail.blocked {"code": "guardrail_blocked", "counts": {"AWS_ACCESS_KEY": 1}, "direction": "outgoing", "error": "Guardrail \'SecretDetection\' blocked: 1 AWS_ACCESS_KEY in LLM output", "kinds": ["AWS_ACCESS_KEY"], "location": "output"}',
    'not.gov:guardrail.blocked {"code": "guardrail_blocked", "counts": {"AWS_ACCESS_KEY": 1}, "direction": "outgoing", "error": "Guardrail \'SecretDetection\' blocked: 1 AWS_ACCESS_KEY in LLM output", "kinds": ["AWS_ACCESS_KEY"], "location": "output"}',
    "RESULT error it=1 answer=None output=None error=\"Guardrail 'SecretDetection' blocked: 1 AWS_ACCESS_KEY in LLM output\" steps=0 tokens=12 meta=['failure_reason', 'pii_mapping'] reason=guardrail_blocked",
]
