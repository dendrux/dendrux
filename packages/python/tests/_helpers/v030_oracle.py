"""v0.3.0 text-only differential oracle (Dendrux 0.4.0 Track O, U-01 / R-05).

Purpose
-------
Freeze what Dendrux v0.3.0 does for text-only runs so every later 0.4.0 track
can prove, by a byte comparison, that text-only behaviour did not move.

What a "cell" is
----------------
One cell = one ``Rig`` (a provider adapter + configuration) x one mode
(``batch`` or ``stream``). A cell runs every scenario in ``SCENARIOS`` against
the *real* provider adapter whose SDK client methods are replaced by scripted
fakes (no network, no keys). The fakes record the exact keyword arguments the
adapter hands to the vendor SDK, which is the provider request payload.

Each cell yields two sections:

``wire``
    Order-sensitive. Per scenario and leg: the SDK request payloads (as passed
    to ``messages.create``/``messages.stream``/``chat.completions.create``/
    ``responses.create``), a derived cache summary (every ``cache_control``
    marker path, ``prompt_cache_key``, ``prompt_cache_retention``, top-level
    ``cache_control``), the sha256 of the adapter's ``captured_request`` (what
    is persisted as ``llm_interactions.provider_request``), the Dendrux-level
    messages/tools/kwargs the loop handed the provider, the adapter stream
    event sequence, run results, the run-event stream (stream mode) and
    ``PauseState.to_dict()`` for paused runs.

``rows``
    Order-insensitive in dict keys (JSON canonicalised with sorted keys, so
    Postgres and SQLite compare equal). Every row of ``agent_runs`` (incl.
    ``meta`` prefix/seed hashes, ``idempotency_fingerprint``, ``pause_data``),
    ``react_traces``, ``tool_calls``, ``token_usage``, ``llm_interactions``
    (semantic + provider request/response) and ``run_events``.

Plus a ``hashes`` golden (pure-function idempotency fingerprints, history seed,
tool hashes, prefix descriptor) and an R-04 manifest (hashes of the existing
cache-related test files/tests, asserting they stay unchanged).

Normalisation (the ONLY non-identity transforms; everything else is compared
byte for byte)
-------------------------------------------------------------------------
1. ULIDs (26-char Crockford base32 strings) anywhere in a string value are
   replaced with ``<ulid:N>``, numbered by first appearance within a section.
   Applies to run ids, row ids, Dendrux tool-call ids, correlation ids.
2. Keys named ``*_at``, ``timestamp``, ``ts`` and datetime values become
   ``<ts>``.
3. Keys ``duration_ms``, ``latency_ms``, ``elapsed_ms`` (and any ``*_ms``
   holding a number) become ``<ms>``.
4. ``Decimal``/float DB cost columns are rounded to 6 decimals.
5. SDK ``NOT_GIVEN`` sentinels are dropped from request kwargs (they are not
   serialised on the wire; ``captured_request`` drops them the same way).
6. Tools in a parallel group finish in a racy order. Runs of consecutive
   ``tool.completed`` run-event rows (same run + iteration) and of consecutive
   ``tool_result`` stream events are put in call order (scripted provider ids sort
   in call order); the block keeps its original ``sequence_index`` values.
7. Anthropic strict tools (Dendrux 0.4.1, T2 — a deliberate wire change). A tool
   entry the Anthropic adapter sends with ``"strict": true`` is mapped back to
   its v0.3.0 shape (``name``/``description``/``input_schema`` = the stored
   ``ToolDef.parameters``) before it reaches the payload, ``captured_request``
   and rows. Non-strict entries pass through untouched and are still compared
   byte for byte; the strict wire itself is covered by
   ``tests/unit/test_anthropic.py``.
Provider-assigned ids (tool_use ids, message ids) are supplied by this module's
scripts and are therefore deterministic; they are NOT normalised.

Regeneration
------------
Goldens are regenerated only when ``DENDRUX_REGEN_V030_ORACLE=1`` is set AND the
checkout is Dendrux 0.3.0 (read from ``packages/python/pyproject.toml``).
Regeneration against any other version is refused: a candidate must never
re-bless its own output. Tests never regenerate by default.

Import discipline: nothing here may import 0.4.0-only modules, and the module
must keep working against both v0.3.0 and candidate trees (it only uses public
names plus the two hash helpers that the spec itself names).
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import os
import re
import sys
import tomllib
from contextlib import contextmanager
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Literal
from unittest.mock import patch

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Iterator

    from sqlalchemy.ext.asyncio import AsyncEngine

HELPERS_DIR = Path(__file__).resolve().parent
TESTS_DIR = HELPERS_DIR.parent
PACKAGE_DIR = TESTS_DIR.parent  # packages/python
ORACLE_DIR = TESTS_DIR / "fixtures" / "v030_oracle"
REGEN_ENV = "DENDRUX_REGEN_V030_ORACLE"
ORACLE_VERSION = "0.3.0"
ORACLE_SOURCE = "tag v0.3.0 (e479746)"

Mode = Literal["batch", "stream"]
MODES: tuple[Mode, ...] = ("batch", "stream")


# ---------------------------------------------------------------------------
# Regeneration guard + golden IO
# ---------------------------------------------------------------------------


def package_version() -> str:
    """Version declared by this checkout's ``packages/python/pyproject.toml``."""
    with open(PACKAGE_DIR / "pyproject.toml", "rb") as fh:
        return str(tomllib.load(fh)["project"]["version"])


def regen_enabled() -> bool:
    """True when the caller asked for regeneration. Refuses on non-0.3.0 trees."""
    if os.environ.get(REGEN_ENV) != "1":
        return False
    version = package_version()
    if version != ORACLE_VERSION:
        raise RuntimeError(
            f"{REGEN_ENV}=1 refused: this checkout is Dendrux {version}, goldens may only be "
            f"regenerated from v{ORACLE_VERSION} (never bless candidate output)."
        )
    return True


def assert_imports_this_checkout() -> None:
    """Fail loudly when ``dendrux`` resolves to a different checkout than these tests."""
    import dendrux

    src = (PACKAGE_DIR / "src").resolve()
    loaded = Path(dendrux.__file__).resolve()
    if src not in loaded.parents:
        raise AssertionError(
            f"dendrux imported from {loaded}, expected under {src}. "
            "Run with PYTHONPATH=src from packages/python."
        )


def dumps_ordered(obj: Any) -> str:
    """Compact, key-order-preserving JSON: the byte form wire sections compare on."""
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def dumps_canonical(obj: Any) -> str:
    """Compact JSON with sorted keys (DB-section comparison form)."""
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def golden_path(kind: str, name: str) -> Path:
    """``kind`` is a subdirectory (``wire``/``rows``) or ``""`` for files at the oracle root."""
    return (ORACLE_DIR / kind if kind else ORACLE_DIR) / f"{name}.json"


def write_golden(kind: str, name: str, data: Any) -> None:
    path = golden_path(kind, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_one_line_per_key(data), encoding="utf-8")


def _one_line_per_key(data: Any) -> str:
    """Valid JSON, one top-level entry per line (compact values): small, diff-friendly."""
    if not isinstance(data, dict):
        return dumps_ordered(data) + "\n"
    items = [f"{json.dumps(k)}:{dumps_ordered(v)}" for k, v in data.items()]
    return "{\n" + ",\n".join(items) + "\n}\n"


def read_golden(kind: str, name: str) -> Any:
    path = golden_path(kind, name)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing v0.3.0 golden {path}. Goldens are captured once from a v{ORACLE_VERSION} "
            f"checkout with {REGEN_ENV}=1 and committed; tests never regenerate them."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def first_diff(expected: Any, actual: Any, path: str = "$") -> str | None:
    """Path + values of the first structural difference, or None when identical.

    Dict key *order* counts as a difference (wire sections are order-sensitive).
    """
    if type(expected) is not type(actual):
        return f"{path}: type {type(expected).__name__} != {type(actual).__name__}"
    if isinstance(expected, dict):
        e_keys, a_keys = list(expected), list(actual)
        if e_keys != a_keys:
            missing = [k for k in e_keys if k not in actual]
            extra = [k for k in a_keys if k not in expected]
            return (
                f"{path}: keys differ (missing={missing}, extra={extra}, order={e_keys}->{a_keys})"
            )
        for key in e_keys:
            diff = first_diff(expected[key], actual[key], f"{path}.{key}")
            if diff:
                return diff
        return None
    if isinstance(expected, list):
        if len(expected) != len(actual):
            return f"{path}: length {len(expected)} != {len(actual)}"
        for i, (e, a) in enumerate(zip(expected, actual, strict=True)):
            diff = first_diff(e, a, f"{path}[{i}]")
            if diff:
                return diff
        return None
    if expected != actual:
        return f"{path}: {expected!r} != {actual!r}"
    return None


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

_ULID_RE = re.compile(r"(?<![0-9A-Za-z])[0-9A-HJKMNP-TV-Z]{26}(?![0-9A-Za-z])")
_TS_KEYS = frozenset({"timestamp", "ts"})
_MS_KEYS = frozenset({"duration_ms", "latency_ms", "elapsed_ms"})


class Normalizer:
    """One numbering scope for ULIDs; use a fresh instance per section."""

    def __init__(self) -> None:
        self._ulids: dict[str, str] = {}

    def _ulid(self, match: re.Match[str]) -> str:
        value = match.group(0)
        if value not in self._ulids:
            self._ulids[value] = f"<ulid:{len(self._ulids) + 1}>"
        return self._ulids[value]

    def text(self, value: str) -> str:
        return _ULID_RE.sub(self._ulid, value)

    def __call__(self, obj: Any, key: str | None = None) -> Any:
        if isinstance(obj, dict):
            return {self.text(str(k)): self(v, str(k)) for k, v in obj.items()}
        if isinstance(obj, list | tuple):
            return [self(v, key) for v in obj]
        if isinstance(obj, datetime):
            return "<ts>"
        if key is not None and (key.endswith("_at") or key in _TS_KEYS) and obj is not None:
            return "<ts>"
        is_ms_key = key is not None and (key in _MS_KEYS or key.endswith("_ms"))
        if is_ms_key and isinstance(obj, int | float | Decimal) and not isinstance(obj, bool):
            return "<ms>"
        if isinstance(obj, Decimal):
            return round(float(obj), 6)
        if isinstance(obj, float):
            return round(obj, 6)
        if isinstance(obj, str):
            return self.text(obj)
        if isinstance(obj, Enum):
            return self(obj.value, key)
        return obj


def _is_not_given(obj: Any) -> bool:
    return type(obj).__name__ in {"NotGiven", "Omit"}


def jsonable(obj: Any) -> Any:
    """Plain-JSON snapshot of ``obj`` (deep-copied; unknown objects become a type tag)."""
    if _is_not_given(obj):
        return _DROP
    if obj is None or isinstance(obj, bool | int | float | str):
        return obj
    if isinstance(obj, Enum):
        return jsonable(obj.value)
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            snap = jsonable(v)
            if snap is not _DROP:
                out[str(k)] = snap
        return out
    if isinstance(obj, list | tuple):
        return [s for s in (jsonable(v) for v in obj) if s is not _DROP]
    if isinstance(obj, datetime):
        return "<ts>"
    if isinstance(obj, Decimal):
        return float(obj)
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: jsonable(getattr(obj, f.name)) for f in fields(obj)}
    return f"<{type(obj).__name__}>"


_DROP: Any = object()


# ---------------------------------------------------------------------------
# Scripted turns + fake SDK objects
# ---------------------------------------------------------------------------

ToolCallSpec = tuple[str, dict[str, Any], str]  # (tool name, params, provider tool-call id)


@dataclass(frozen=True)
class Turn:
    """One scripted model response, vendor-neutral."""

    text: str | None = None
    calls: tuple[ToolCallSpec, ...] = ()


@dataclass(frozen=True)
class Usage:
    fresh: int
    cache_read: int
    cache_write: int
    out: int

    @property
    def prompt_total(self) -> int:
        return self.fresh + self.cache_read


def usage_for(index: int) -> Usage:
    """Deterministic usage per call index: first call writes the cache, later calls read it."""
    if index == 0:
        return Usage(fresh=100, cache_read=0, cache_write=400, out=20)
    return Usage(fresh=100 + 10 * index, cache_read=400, cache_write=0, out=20 + index)


class Fake(SimpleNamespace):
    """Attribute bag with a ``model_dump`` returning a fixed dict."""

    def __init__(self, _dump: dict[str, Any] | None = None, **kw: Any) -> None:
        super().__init__(**kw)
        self._dump = _dump

    def model_dump(self, **_: Any) -> dict[str, Any]:
        return copy.deepcopy(self._dump) if self._dump is not None else {}


def _split(text: str) -> list[str]:
    if len(text) < 2:
        return [text]
    mid = len(text) // 2
    return [text[:mid], text[mid:]]


# --- Anthropic ---


def anthropic_message(turn: Turn, usage: Usage, index: int, model: str) -> Fake:
    content: list[Any] = []
    dump_content: list[dict[str, Any]] = []
    if turn.text:
        content.append(Fake(type="text", text=turn.text))
        dump_content.append({"type": "text", "text": turn.text})
    for name, params, pid in turn.calls:
        content.append(Fake(type="tool_use", id=pid, name=name, input=params))
        dump_content.append({"type": "tool_use", "id": pid, "name": name, "input": params})
    stop = "tool_use" if turn.calls else "end_turn"
    u = {
        "input_tokens": usage.fresh,
        "output_tokens": usage.out,
        "cache_creation_input_tokens": usage.cache_write,
        "cache_read_input_tokens": usage.cache_read,
    }
    dump = {
        "id": f"msg_oracle_{index}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": dump_content,
        "stop_reason": stop,
        "stop_sequence": None,
        "usage": u,
    }
    return Fake(dump, id=dump["id"], content=content, usage=Fake(**u), stop_reason=stop)


class _AnthropicStream:
    def __init__(self, events: list[Any], final: Any) -> None:
        self._events = events
        self._final = final

    async def __aenter__(self) -> _AnthropicStream:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    def __aiter__(self) -> AsyncIterator[Any]:
        return self._iter()

    async def _iter(self) -> AsyncIterator[Any]:
        for event in self._events:
            yield event

    async def get_final_message(self) -> Any:
        return self._final


def anthropic_stream(turn: Turn, usage: Usage, index: int, model: str) -> _AnthropicStream:
    events: list[Any] = []
    if turn.text:
        events.append(Fake(type="content_block_start", content_block=Fake(type="text")))
        for chunk in _split(turn.text):
            events.append(
                Fake(type="content_block_delta", delta=Fake(type="text_delta", text=chunk))
            )
        events.append(Fake(type="content_block_stop"))
    for name, params, pid in turn.calls:
        events.append(
            Fake(
                type="content_block_start",
                content_block=Fake(type="tool_use", name=name, id=pid),
            )
        )
        for part in _split(json.dumps(params)):
            events.append(
                Fake(
                    type="content_block_delta",
                    delta=Fake(type="input_json_delta", partial_json=part),
                )
            )
        events.append(Fake(type="content_block_stop"))
    events.append(Fake(type="message_delta", usage=Fake(output_tokens_details=None)))
    return _AnthropicStream(events, anthropic_message(turn, usage, index, model))


# --- OpenAI Chat Completions (and OpenRouter) ---


def _chat_usage(usage: Usage, *, openrouter: bool, index: int) -> Fake:
    details: dict[str, Any] = {"cached_tokens": usage.cache_read}
    prompt = usage.prompt_total
    if openrouter:
        details["cache_write_tokens"] = usage.cache_write
        # OpenRouter's prompt_tokens counts cache writes too (0.5.0 partitions
        # them out of fresh input, D-127).
        prompt += usage.cache_write
    extra: dict[str, Any] = {"cost": round(0.0001 * (index + 1), 6)} if openrouter else {}
    dump = {
        "prompt_tokens": prompt,
        "completion_tokens": usage.out,
        "total_tokens": prompt + usage.out,
        "prompt_tokens_details": details,
        **extra,
    }
    return Fake(
        dump,
        prompt_tokens=prompt,
        completion_tokens=usage.out,
        total_tokens=prompt + usage.out,
        prompt_tokens_details=Fake(**details),
        completion_tokens_details=None,
        **extra,
    )


def chat_completion(turn: Turn, usage: Usage, index: int, model: str, *, openrouter: bool) -> Fake:
    tool_calls = [
        Fake(id=pid, function=Fake(name=name, arguments=json.dumps(params)), type="function")
        for name, params, pid in turn.calls
    ]
    finish = "tool_calls" if turn.calls else "stop"
    message = Fake(role="assistant", content=turn.text, tool_calls=tool_calls or None)
    u = _chat_usage(usage, openrouter=openrouter, index=index)
    dump = {
        "id": f"chatcmpl-oracle-{index}",
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "finish_reason": finish,
                "message": {
                    "role": "assistant",
                    "content": turn.text,
                    "tool_calls": [
                        {
                            "id": pid,
                            "type": "function",
                            "function": {"name": name, "arguments": json.dumps(params)},
                        }
                        for name, params, pid in turn.calls
                    ]
                    or None,
                },
            }
        ],
        "usage": u.model_dump(),
    }
    return Fake(dump, choices=[Fake(message=message, finish_reason=finish, index=0)], usage=u)


class _ChunkStream:
    def __init__(self, chunks: list[Any]) -> None:
        self._chunks = chunks

    def __aiter__(self) -> AsyncIterator[Any]:
        return self._iter()

    async def _iter(self) -> AsyncIterator[Any]:
        for chunk in self._chunks:
            yield chunk


def _chunk(delta: Any = None, finish: str | None = None, usage: Any = None) -> Fake:
    choices = (
        []
        if delta is None and finish is None
        else [
            Fake(delta=delta or Fake(content=None, tool_calls=None), finish_reason=finish, index=0)
        ]
    )
    return Fake(choices=choices, usage=usage)


def chat_stream(
    turn: Turn, usage: Usage, index: int, model: str, *, openrouter: bool
) -> _ChunkStream:
    chunks: list[Any] = []
    if turn.text:
        for piece in _split(turn.text):
            chunks.append(_chunk(Fake(content=piece, tool_calls=None)))
    for i, (name, params, pid) in enumerate(turn.calls):
        first, second = _split(json.dumps(params))
        chunks.append(
            _chunk(
                Fake(
                    content=None,
                    tool_calls=[Fake(index=i, id=pid, function=Fake(name=name, arguments=first))],
                )
            )
        )
        chunks.append(
            _chunk(
                Fake(
                    content=None,
                    tool_calls=[Fake(index=i, id=None, function=Fake(name=None, arguments=second))],
                )
            )
        )
    chunks.append(
        _chunk(Fake(content=None, tool_calls=None), "tool_calls" if turn.calls else "stop")
    )
    chunks.append(_chunk(None, None, _chat_usage(usage, openrouter=openrouter, index=index)))
    return _ChunkStream(chunks)


# --- OpenAI Responses ---


def _responses_usage(usage: Usage) -> Fake:
    dump = {
        "input_tokens": usage.prompt_total,
        "output_tokens": usage.out,
        "total_tokens": usage.prompt_total + usage.out,
        "input_tokens_details": {"cached_tokens": usage.cache_read},
    }
    return Fake(
        dump,
        input_tokens=usage.prompt_total,
        output_tokens=usage.out,
        total_tokens=usage.prompt_total + usage.out,
        input_tokens_details=Fake(cached_tokens=usage.cache_read),
        output_tokens_details=None,
    )


def responses_response(turn: Turn, usage: Usage, index: int, model: str) -> Fake:
    output = [
        Fake(
            type="function_call",
            id=f"fc_oracle_{index}_{i}",
            name=name,
            call_id=pid,
            arguments=json.dumps(params),
        )
        for i, (name, params, pid) in enumerate(turn.calls)
    ]
    u = _responses_usage(usage)
    dump = {
        "id": f"resp_oracle_{index}",
        "object": "response",
        "model": model,
        "output_text": turn.text or "",
        "output": [
            {
                "type": "function_call",
                "id": f"fc_oracle_{index}_{i}",
                "name": name,
                "call_id": pid,
                "arguments": json.dumps(params),
            }
            for i, (name, params, pid) in enumerate(turn.calls)
        ],
        "usage": u.model_dump(),
    }
    return Fake(dump, output_text=turn.text or "", output=output, usage=u)


def responses_stream(turn: Turn, usage: Usage, index: int, model: str) -> _ChunkStream:
    events: list[Any] = []
    if turn.text:
        for piece in _split(turn.text):
            events.append(Fake(type="response.output_text.delta", delta=piece, item_id="msg_o"))
    for i, (name, params, pid) in enumerate(turn.calls):
        item_id = f"fc_oracle_{index}_{i}"
        args = json.dumps(params)
        events.append(
            Fake(
                type="response.output_item.added",
                item=Fake(type="function_call", id=item_id, name=name, call_id=pid),
            )
        )
        events.append(
            Fake(type="response.function_call_arguments.delta", item_id=item_id, delta=args)
        )
        events.append(
            Fake(type="response.function_call_arguments.done", item_id=item_id, arguments=args)
        )
    final = responses_response(turn, usage, index, model)
    events.append(Fake(type="response.completed", response=final))
    return _ChunkStream(events)


# ---------------------------------------------------------------------------
# Rigs (provider adapter x configuration)
# ---------------------------------------------------------------------------

Vendor = Literal["anthropic", "chat", "responses"]
_KEY = "sk-oracle-not-a-real-key"


@dataclass(frozen=True)
class Rig:
    name: str
    vendor: Vendor
    factory: Callable[[], Any]
    openrouter: bool = False
    openrouter_model: str | None = None


def _anthropic(**kw: Any) -> Callable[[], Any]:
    def make() -> Any:
        from dendrux.llm.anthropic import AnthropicProvider

        return AnthropicProvider(model="claude-sonnet-4-6", api_key=_KEY, **kw)

    return make


def _chat(**kw: Any) -> Callable[[], Any]:
    def make() -> Any:
        from dendrux.llm.openai import OpenAIProvider

        return OpenAIProvider(model="gpt-4o", api_key=_KEY, **kw)

    return make


def _responses(**kw: Any) -> Callable[[], Any]:
    def make() -> Any:
        from dendrux.llm.openai_responses import OpenAIResponsesProvider

        return OpenAIResponsesProvider(model="gpt-5", api_key=_KEY, **kw)

    return make


def _openrouter(model: str, **kw: Any) -> Callable[[], Any]:
    def make() -> Any:
        from dendrux.llm.openrouter import OpenRouterProvider

        return OpenRouterProvider(model=model, api_key=_KEY, **kw)

    return make


RIGS: tuple[Rig, ...] = (
    Rig("anthropic", "anthropic", _anthropic()),
    Rig("anthropic_ttl1h", "anthropic", _anthropic(cache_ttl="1h")),
    Rig("openai_chat", "chat", _chat()),
    Rig(
        "openai_chat_routing",
        "chat",
        _chat(cache_key_prefix="oracle-customer", prompt_cache_retention="24h"),
    ),
    Rig("openai_chat_compat", "chat", _chat(base_url="http://oracle.invalid/v1")),
    Rig("openai_responses", "responses", _responses()),
    Rig(
        "openrouter_anthropic",
        "chat",
        _openrouter("anthropic/claude-sonnet-4.5"),
        openrouter=True,
        openrouter_model="anthropic/claude-sonnet-4.5",
    ),
    Rig(
        "openrouter_openai",
        "chat",
        _openrouter("openai/gpt-4o"),
        openrouter=True,
        openrouter_model="openai/gpt-4o",
    ),
)
RIGS_BY_NAME = {r.name: r for r in RIGS}


def cell_name(rig: Rig, mode: Mode) -> str:
    return f"{rig.name}__{mode}"


CELLS: tuple[tuple[Rig, Mode], ...] = tuple((r, m) for r in RIGS for m in MODES)


# ---------------------------------------------------------------------------
# Recorder: SDK calls + provider-level calls
# ---------------------------------------------------------------------------


def cache_summary(payload: dict[str, Any]) -> dict[str, Any]:
    """Every cache-relevant field of one request payload, in a reviewable shape."""
    markers: list[dict[str, Any]] = []

    def walk(node: Any, path: str) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                if k == "cache_control":
                    markers.append({"path": f"{path}.{k}", "value": v})
                else:
                    walk(v, f"{path}.{k}")
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{path}[{i}]")

    walk(payload, "$")
    return {
        "markers": markers,
        "marker_count": len(markers),
        "prompt_cache_key": payload.get("prompt_cache_key"),
        "prompt_cache_retention": payload.get("prompt_cache_retention"),
        "extra_body_cache_control": (payload.get("extra_body") or {}).get("cache_control"),
    }


class Recorder:
    def __init__(self) -> None:
        self.legs: list[dict[str, Any]] = []

    def begin_leg(self, label: str) -> dict[str, Any]:
        leg: dict[str, Any] = {"label": label, "sdk_calls": [], "provider_calls": []}
        self.legs.append(leg)
        return leg


def message_view(m: Any) -> dict[str, Any]:
    """Dendrux Message as the loop hands it to a provider (explicit text-era field list)."""
    return {
        "role": m.role.value,
        "content": m.content,
        "name": m.name,
        "call_id": m.call_id,
        "tool_calls": [
            {
                "name": tc.name,
                "params": tc.params,
                "id": tc.id,
                "provider_tool_call_id": tc.provider_tool_call_id,
            }
            for tc in (m.tool_calls or [])
        ]
        or None,
        "kind": m.kind,
        "placement": m.placement,
        "source": m.source,
        "meta": jsonable(m.meta),
        "reasoning": m.reasoning,
        "reasoning_blocks": jsonable(m.reasoning_blocks),
    }


def tool_view(t: Any) -> dict[str, Any]:
    return {
        "name": t.name,
        "description": t.description,
        "parameters": t.parameters,
        "target": t.target.value,
        "strict": t.strict,
    }


def response_view(resp: Any) -> dict[str, Any]:
    """LLMResponse as the adapter returned it. ``provider_request`` hashed, ordered."""
    usage = resp.usage
    return {
        "text": resp.text,
        "tool_calls": [
            {
                "name": tc.name,
                "params": tc.params,
                "id": tc.id,
                "provider_tool_call_id": tc.provider_tool_call_id,
                "parse_error": tc.parse_error,
            }
            for tc in (resp.tool_calls or [])
        ]
        or None,
        "usage": usage_view(usage),
        "reasoning": resp.reasoning,
        "reasoning_blocks": jsonable(resp.reasoning_blocks),
        "model": resp.model,
        "provider_request_sha256": sha256_text(dumps_ordered(jsonable(resp.provider_request))),
    }


def usage_view(usage: Any) -> dict[str, Any]:
    names = (
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "cache_read_input_tokens",
        "cache_creation_input_tokens",
        "reasoning_tokens",
        "cost_usd",
        "cost_source",
        "usage_reported",
    )
    return {n: jsonable(getattr(usage, n, None)) for n in names}


def _call_args_view(messages: Any, tools: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    return {
        "messages": [message_view(m) for m in messages],
        "tools": [tool_view(t) for t in (tools or [])] or None,
        "kwargs": jsonable(kwargs),
    }


def instrument_provider(provider: Any, leg: dict[str, Any]) -> None:
    """Record what the loop passes to ``complete``/``complete_stream`` and what returns."""
    real_complete = provider.complete
    real_stream = provider.complete_stream

    async def complete(messages: Any, tools: Any = None, **kwargs: Any) -> Any:
        entry: dict[str, Any] = {
            "entry": "complete",
            "args": _call_args_view(messages, tools, kwargs),
        }
        leg["provider_calls"].append(entry)
        resp = await real_complete(messages, tools, **kwargs)
        entry["response"] = response_view(resp)
        return resp

    async def complete_stream(messages: Any, tools: Any = None, **kwargs: Any) -> Any:
        entry: dict[str, Any] = {
            "entry": "complete_stream",
            "args": _call_args_view(messages, tools, kwargs),
            "stream_events": [],
        }
        leg["provider_calls"].append(entry)
        async for event in real_stream(messages, tools, **kwargs):
            if event.type.value == "done":
                entry["response"] = response_view(event.raw)
            else:
                entry["stream_events"].append(
                    [event.type.value, event.text, event.tool_name, event.tool_call_id]
                )
            yield event

    provider.complete = complete
    provider.complete_stream = complete_stream


def _v030_anthropic_tool_shape(provider: Any) -> None:
    """Normalisation 7: map strict Anthropic tool entries back to the v0.3.0 shape."""
    real_convert = getattr(provider, "_convert_tools", None)
    if real_convert is None:
        return

    def convert(tools: list[Any], **kw: Any) -> list[dict[str, Any]]:
        out = real_convert(tools, **kw)
        return [
            {"name": e["name"], "description": e["description"], "input_schema": td.parameters}
            if td.strict and e.get("strict") is True
            else e
            for td, e in zip(tools, out, strict=True)
        ]

    provider._convert_tools = convert


def install_fake_sdk(provider: Any, rig: Rig, turns: list[Turn], leg: dict[str, Any]) -> None:
    """Replace the vendor SDK entry points with scripted fakes that record the payload."""
    counter = {"n": 0}

    def next_turn() -> tuple[Turn, Usage, int]:
        index = counter["n"]
        counter["n"] += 1
        if index >= len(turns):
            raise IndexError(f"oracle script exhausted after {len(turns)} turns ({rig.name})")
        return turns[index], usage_for(index), index

    def record(method: str, kwargs: dict[str, Any]) -> None:
        payload = jsonable(copy.deepcopy(kwargs))
        leg["sdk_calls"].append(
            {
                "method": method,
                "payload": payload,
                "payload_sha256": sha256_text(dumps_ordered(payload)),
                "cache": cache_summary(payload),
            }
        )

    if rig.vendor == "anthropic":

        async def create(**kw: Any) -> Any:
            record("messages.create", kw)
            turn, usage, idx = next_turn()
            return anthropic_message(turn, usage, idx, kw["model"])

        def stream(**kw: Any) -> Any:
            record("messages.stream", kw)
            turn, usage, idx = next_turn()
            return anthropic_stream(turn, usage, idx, kw["model"])

        provider._client.messages.create = create
        provider._client.messages.stream = stream
        _v030_anthropic_tool_shape(provider)
    elif rig.vendor == "chat":

        async def chat_create(**kw: Any) -> Any:
            streaming = bool(kw.get("stream"))
            record(
                "chat.completions.create[stream]" if streaming else "chat.completions.create", kw
            )
            turn, usage, idx = next_turn()
            builder = chat_stream if streaming else chat_completion
            return builder(turn, usage, idx, kw["model"], openrouter=rig.openrouter)

        provider._client.chat.completions.create = chat_create
    else:

        async def responses_create(**kw: Any) -> Any:
            streaming = bool(kw.get("stream"))
            record("responses.create[stream]" if streaming else "responses.create", kw)
            turn, usage, idx = next_turn()
            builder = responses_stream if streaming else responses_response
            return builder(turn, usage, idx, kw["model"])

        provider._client.responses.create = responses_create


@contextmanager
def openrouter_catalog(rig: Rig) -> Iterator[None]:
    """Serve a local model catalog so the native-tools guard never touches the network."""
    if not rig.openrouter:
        yield
        return
    from dendrux.llm import openrouter as or_mod

    catalog: dict[str, Any] = {}
    for slug in (rig.openrouter_model, "override-model"):
        entry = or_mod._parse_model_entry(
            {
                "id": slug,
                "name": slug,
                "supported_parameters": ["tools", "tool_choice", "max_tokens"],
                "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
                "pricing": {"prompt": "0.000003", "completion": "0.000015"},
                "context_length": 200000,
            }
        )
        assert entry is not None
        catalog[entry.id] = entry

    async def fake_catalog(_url: str) -> dict[str, Any]:
        return catalog

    with patch.object(or_mod, "_get_catalog", fake_catalog):
        yield


# ---------------------------------------------------------------------------
# Tools + agents
# ---------------------------------------------------------------------------


def _build_tools() -> tuple[Any, Any, Any]:
    from dendrux import tool

    @tool()
    async def lookup(q: str) -> str:
        """Look something up."""
        return f"result for {q}"

    @tool()
    async def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    @tool(target="client")
    async def read_range(sheet: str) -> str:
        """Read a range from the client spreadsheet."""
        return "unused: runs on the client"

    return lookup, add, read_range


@dataclass
class RunOutcome:
    run_id: str
    summary: dict[str, Any]
    events: list[list[Any]] | None = None


def _strip_pause_versioned_fields(pause: dict[str, Any]) -> None:
    """These text-era scenarios pass no per-call provider options."""
    assert pause.pop("schema_version", 1) == 1
    assert pause.pop("provider_kwargs", {}) == {}


def result_view(result: Any) -> dict[str, Any]:
    """Stable subset of RunResult (text-era fields) + pause JSON."""
    view: dict[str, Any] = {
        "run_id": result.run_id,
        "status": result.status.value,
        "answer": result.answer,
        "iteration_count": result.iteration_count,
        "usage": usage_view(result.usage),
        "error": result.error,
        "commentary": list(result.commentary),
    }
    replay = getattr(result, "replay", None)
    view["replay"] = jsonable(replay) if replay is not None else None
    pause = (result.meta or {}).get("pause_state")
    view["pause_state"] = pause.to_dict() if pause is not None else None
    if view["pause_state"] is not None:
        _strip_pause_versioned_fields(view["pause_state"])
        pause_id = view["pause_state"].pop("pause_id", None)
        assert pause_id is None or re.fullmatch(r"[0-9A-HJKMNP-TV-Z]{26}", pause_id)
    return view


def event_view(event: Any) -> list[Any]:
    call = event.tool_call
    return [
        event.type.value,
        event.text,
        event.tool_name,
        event.tool_call_id,
        event.iteration,
        event.phase,
        event.error,
        call.name if call is not None else None,
        call.provider_tool_call_id if call is not None else None,
        event.tool_result.success if event.tool_result is not None else None,
    ]


_TOOL_RESULT_PROVIDER_ID = 8  # index into event_view()


def canonical_concurrent_tool_results(events: list[list[Any]]) -> list[list[Any]]:
    """Put each run of consecutive TOOL_RESULT events (same iteration) in call order.

    Tools in a parallel group finish in a racy order and the loop notifies in completion
    order; that order is not a contract, so the oracle canonicalises it (provider ids in the
    scripts sort in call order).
    """
    out: list[list[Any]] = []
    i = 0
    while i < len(events):
        if events[i][0] != "tool_result":
            out.append(events[i])
            i += 1
            continue
        j = i
        while j < len(events) and events[j][0] == "tool_result" and events[j][4] == events[i][4]:
            j += 1
        out.extend(sorted(events[i:j], key=lambda e: str(e[_TOOL_RESULT_PROVIDER_ID])))
        i = j
    return out


class Ctx:
    """Scenario context: builds agents on the cell's rig and drives batch/stream runs."""

    def __init__(self, rig: Rig, mode: Mode, store: Any) -> None:
        self.rig = rig
        self.mode = mode
        self.store = store
        self.recorder = Recorder()
        self.outcomes: list[dict[str, Any]] = []
        self.lookup, self.add, self.read_range = _build_tools()

    def agent(
        self,
        label: str,
        turns: list[Turn],
        *,
        tools: tuple[Any, ...] = (),
        prompt: str = "You are the oracle assistant.",
        **kw: Any,
    ) -> Any:
        from dendrux import Agent
        from dendrux.pricing import ModelPricing, PriceTable

        provider = self.rig.factory()
        leg = self.recorder.begin_leg(label)
        install_fake_sdk(provider, self.rig, turns, leg)
        instrument_provider(provider, leg)
        return Agent(
            provider=provider,
            prompt=prompt,
            tools=list(tools),
            state_store=self.store,
            name="OracleAgent",
            pricing=PriceTable(
                {"*": ModelPricing(input=3.0, output=15.0, cache_read=0.3, cache_write=3.75)}
            ),
            **kw,
        )

    async def run(self, agent: Any, user_input: str, **kw: Any) -> RunOutcome:
        if self.mode == "batch":
            result = await agent.run(user_input, **kw)
            return self._record("run", RunOutcome(result.run_id, result_view(result)))
        return self._record("run", await self._drain(agent.stream(user_input, **kw)))

    async def resume(self, agent: Any, run_id: str, **kw: Any) -> RunOutcome:
        if self.mode == "batch":
            result = await agent.resume(run_id, **kw)
            return self._record("resume", RunOutcome(result.run_id, result_view(result)))
        return self._record("resume", await self._drain(agent.resume_stream(run_id, **kw)))

    async def _drain(self, stream: Any) -> RunOutcome:
        events: list[list[Any]] = []
        final = None
        async with stream as s:
            async for event in s:
                events.append(event_view(event))
                if event.run_result is not None:
                    final = event.run_result
        assert final is not None, "stream ended without a terminal run_result"
        return RunOutcome(
            final.run_id, result_view(final), canonical_concurrent_tool_results(events)
        )

    def _record(self, kind: str, outcome: RunOutcome) -> RunOutcome:
        entry: dict[str, Any] = {"kind": kind, "result": outcome.summary}
        if outcome.events is not None:
            entry["events"] = outcome.events
        self.outcomes.append(entry)
        return outcome


def pending_call_ids(outcome: RunOutcome) -> list[tuple[str, str]]:
    """(tool-call id, tool name) of the calls a paused run is waiting on."""
    pause = outcome.summary["pause_state"]
    return [(tc["id"], tc["name"]) for tc in pause["pending_tool_calls"]]


# ---------------------------------------------------------------------------
# Scenarios (the R-05 matrix)
# ---------------------------------------------------------------------------


async def sc_plain(ctx: Ctx) -> None:
    agent = ctx.agent("run", [Turn(text="Hello! How can I help?")])
    await ctx.run(agent, "Hi there", idempotency_key="oracle-plain")


async def sc_history(ctx: Ctx) -> None:
    from dendrux.chat import ChatMessage

    agent = ctx.agent("run", [Turn(text="6")])
    await ctx.run(
        agent,
        "And 3+3?",
        history=[ChatMessage.user("What is 2+2?"), ChatMessage.assistant("4")],
    )


async def sc_context_absent(ctx: Ctx) -> None:
    from dendrux.chat import ChatMessage

    agent = ctx.agent("run", [Turn(text="ok")])
    await ctx.run(
        agent,
        "Summarise",
        history=[ChatMessage.user("Hello"), ChatMessage.assistant("Hi")],
        context=None,
    )


async def sc_context_stable_dynamic(ctx: Ctx) -> None:
    from dendrux.chat import ChatMessage
    from dendrux.context_blocks import ContextBlock

    agent = ctx.agent("run", [Turn(text="ok")])
    await ctx.run(
        agent,
        "Summarise",
        history=[ChatMessage.user("Hello"), ChatMessage.assistant("Hi")],
        context=[
            ContextBlock("project spec: always answer in French", kind="spec", placement="stable"),
            ContextBlock("retrieved doc excerpt", kind="doc", source="kb:42"),
        ],
        idempotency_key="oracle-context",
    )


async def sc_context_stable_only(ctx: Ctx) -> None:
    from dendrux.context_blocks import ContextBlock

    agent = ctx.agent("run", [Turn(text="ok")])
    await ctx.run(agent, "Q", context=[ContextBlock("instructions", placement="stable")])


async def sc_context_dynamic_only(ctx: Ctx) -> None:
    from dendrux.context_blocks import ContextBlock

    agent = ctx.agent("run", [Turn(text="ok")])
    await ctx.run(agent, "Q", context=[ContextBlock("doc", kind="doc")])


async def sc_system_tools(ctx: Ctx) -> None:
    agent = ctx.agent(
        "run",
        [Turn(calls=(("lookup", {"q": "x"}, "call_sys_1"),)), Turn(text="Found: result for x")],
        tools=(ctx.lookup, ctx.add),
        prompt="You are the oracle assistant. Use tools when helpful.",
    )
    await ctx.run(agent, "Look up x")


async def sc_server_tool_text(ctx: Ctx) -> None:
    agent = ctx.agent(
        "run",
        [
            Turn(text="Let me check.", calls=(("add", {"a": 2, "b": 3}, "call_add_1"),)),
            Turn(text="2 + 3 = 5"),
        ],
        tools=(ctx.lookup, ctx.add),
    )
    await ctx.run(agent, "What is 2+3?")


async def sc_parallel_tools(ctx: Ctx) -> None:
    agent = ctx.agent(
        "run",
        [
            Turn(
                text="Checking both.",
                calls=(
                    ("lookup", {"q": "alpha"}, "call_par_1"),
                    ("add", {"a": 7, "b": 8}, "call_par_2"),
                    ("lookup", {"q": "beta"}, "call_par_3"),
                ),
            ),
            Turn(text="alpha, 15, beta"),
        ],
        tools=(ctx.lookup, ctx.add),
    )
    await ctx.run(agent, "Do three things")


async def sc_client_tool_pause_resume(ctx: Ctx) -> None:
    from dendrux.types import ToolResult

    first = ctx.agent(
        "run",
        [Turn(calls=(("read_range", {"sheet": "Sheet1"}, "call_cli_1"),))],
        tools=(ctx.lookup, ctx.read_range),
    )
    paused = await ctx.run(first, "Read the sheet", idempotency_key="oracle-pause")
    (call_id, name), *_ = pending_call_ids(paused)
    # A fresh agent + provider on the same store models resume after a process restart.
    second = ctx.agent(
        "resume",
        [Turn(text="The sheet has 3 rows.")],
        tools=(ctx.lookup, ctx.read_range),
    )
    await ctx.resume(
        second,
        paused.run_id,
        tool_results=[
            ToolResult(name=name, call_id=call_id, payload=json.dumps({"rows": 3}), success=True)
        ],
    )


async def sc_mixed_server_client_pause(ctx: Ctx) -> None:
    from dendrux.types import ToolResult

    first = ctx.agent(
        "run",
        [
            Turn(
                text="Working on it.",
                calls=(
                    ("lookup", {"q": "gamma"}, "call_mix_1"),
                    ("read_range", {"sheet": "S2"}, "call_mix_2"),
                ),
            )
        ],
        tools=(ctx.lookup, ctx.read_range),
    )
    paused = await ctx.run(first, "Mixed work")
    (call_id, name), *_ = pending_call_ids(paused)
    second = ctx.agent(
        "resume",
        [
            Turn(calls=(("lookup", {"q": "delta"}, "call_mix_3"),)),
            Turn(text="All done."),
        ],
        tools=(ctx.lookup, ctx.read_range),
    )
    await ctx.resume(
        second,
        paused.run_id,
        tool_results=[
            ToolResult(name=name, call_id=call_id, payload=json.dumps("cells"), success=True)
        ],
    )


async def sc_replay_runs(ctx: Ctx) -> None:
    tools = (ctx.lookup,)
    tenant = {"tenant_id": "tenant-oracle"}
    a = await ctx.run(
        ctx.agent(
            "run-1",
            [Turn(calls=(("lookup", {"q": "one"}, "call_rep_1"),)), Turn(text="answer one")],
            tools=tools,
        ),
        "first question",
        idempotency_key="oracle-replay-1",
        **tenant,
    )
    b = await ctx.run(
        ctx.agent("run-2", [Turn(text="answer two")], tools=tools),
        "second question",
        runs=[a.run_id],
        replay_last="all",
        **tenant,
    )
    await ctx.run(
        ctx.agent(
            "run-3",
            [Turn(calls=(("lookup", {"q": "three"}, "call_rep_3"),)), Turn(text="answer three")],
            tools=tools,
        ),
        "third question",
        runs=[a.run_id, b.run_id],
        replay_last=1,
        **tenant,
    )


async def sc_idempotent_repeat(ctx: Ctx) -> None:
    agent = ctx.agent("run", [Turn(text="once")])
    first = await ctx.run(agent, "Same request", idempotency_key="oracle-idem")
    again = ctx.agent("repeat", [])  # a repeat must not reach the provider at all
    repeated = await ctx.run(again, "Same request", idempotency_key="oracle-idem")
    assert repeated.run_id == first.run_id


SCENARIOS: tuple[tuple[str, Callable[[Ctx], Any]], ...] = (
    ("plain", sc_plain),
    ("history", sc_history),
    ("context_absent", sc_context_absent),
    ("context_stable_dynamic", sc_context_stable_dynamic),
    ("context_stable_only", sc_context_stable_only),
    ("context_dynamic_only", sc_context_dynamic_only),
    ("system_tools", sc_system_tools),
    ("server_tool_text", sc_server_tool_text),
    ("parallel_tools", sc_parallel_tools),
    ("client_tool_pause_resume", sc_client_tool_pause_resume),
    ("mixed_server_client_pause", sc_mixed_server_client_pause),
    ("replay_runs", sc_replay_runs),
    ("idempotent_repeat", sc_idempotent_repeat),
)


# --- Direct adapter scenarios: hand-built Message lists into complete()/complete_stream() ---


def _direct_messages() -> dict[str, tuple[list[Any], list[Any] | None, dict[str, Any]]]:
    from dendrux.types import Message, Role, ToolCall, ToolDef

    lookup_def = ToolDef(
        name="lookup",
        description="Look something up.",
        parameters={
            "type": "object",
            "properties": {"q": {"type": "string"}},
            "required": ["q"],
        },
    )
    add_def = ToolDef(
        name="add",
        description="Add two integers.",
        parameters={
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
        strict=True,
    )
    tc1 = ToolCall(
        name="lookup", params={"q": "alpha"}, id="tc-direct-1", provider_tool_call_id="call_d1"
    )
    tc2 = ToolCall(
        name="add", params={"a": 1, "b": 2}, id="tc-direct-2", provider_tool_call_id="call_d2"
    )
    sys_msg = Message(role=Role.SYSTEM, content="You are the oracle assistant.")
    stable_a = Message(
        role=Role.USER,
        content="STABLE CONTEXT:\nspec\n\nUSER MESSAGE:\nhello",
        placement="stable",
    )
    stable_b = Message(role=Role.USER, content="replayed earlier run", placement="stable")
    base_kw: dict[str, Any] = {"run_id": "direct-run", "cache_key_prefix": "OracleAgent:model"}
    return {
        "simple": ([sys_msg, Message(role=Role.USER, content="hello")], None, dict(base_kw)),
        "no_system": ([Message(role=Role.USER, content="hello")], None, dict(base_kw)),
        "stable_head": (
            [
                sys_msg,
                stable_a,
                Message(role=Role.ASSISTANT, content="ack"),
                Message(role=Role.USER, content="q2"),
            ],
            [lookup_def],
            dict(base_kw),
        ),
        "two_stable_markers": (
            [
                sys_msg,
                stable_a,
                Message(role=Role.ASSISTANT, content="ack"),
                stable_b,
                Message(role=Role.ASSISTANT, content="ack2"),
                Message(role=Role.USER, content="q3"),
            ],
            [lookup_def, add_def],
            dict(base_kw),
        ),
        "parallel_tool_results": (
            [
                sys_msg,
                stable_a,
                Message(role=Role.ASSISTANT, content="calling", tool_calls=[tc1, tc2]),
                Message(
                    role=Role.TOOL,
                    content='"result for alpha"',
                    name="lookup",
                    call_id="tc-direct-1",
                ),
                Message(
                    role=Role.TOOL,
                    content="3",
                    name="add",
                    call_id="tc-direct-2",
                    meta={"is_error": True},
                ),
                Message(role=Role.ASSISTANT, content="done"),
                Message(role=Role.USER, content="next"),
            ],
            [lookup_def, add_def],
            dict(base_kw),
        ),
        "tool_results_last": (
            [
                sys_msg,
                Message(role=Role.USER, content="go"),
                Message(role=Role.ASSISTANT, content="", tool_calls=[tc1]),
                Message(
                    role=Role.TOOL,
                    content='"result for alpha"',
                    name="lookup",
                    call_id="tc-direct-1",
                ),
            ],
            [lookup_def],
            dict(base_kw),
        ),
        "kwarg_overrides": (
            [sys_msg, Message(role=Role.USER, content="hello")],
            [lookup_def],
            {**base_kw, "temperature": 0.2, "max_tokens": 512, "model": "override-model"},
        ),
        "no_cache_prefix": (
            [sys_msg, Message(role=Role.USER, content="hello")],
            None,
            {"run_id": "direct-run"},
        ),
    }


async def run_direct(rig: Rig, mode: Mode) -> dict[str, Any]:
    """Drive the adapter directly (no Agent, no store) over hand-built message lists."""
    out: dict[str, Any] = {}
    with openrouter_catalog(rig):
        for name, (messages, tools, kwargs) in _direct_messages().items():
            provider = rig.factory()
            recorder = Recorder()
            leg = recorder.begin_leg(name)
            install_fake_sdk(provider, rig, [Turn(text="ok")], leg)
            instrument_provider(provider, leg)
            before = copy.deepcopy([message_view(m) for m in messages])
            if mode == "batch":
                await provider.complete(messages, tools, **copy.deepcopy(kwargs))
            else:
                async for _ in provider.complete_stream(messages, tools, **copy.deepcopy(kwargs)):
                    pass
            assert [message_view(m) for m in messages] == before, "adapter mutated caller messages"
            out[name] = {"legs": [leg]}
    return Normalizer()(out)


# ---------------------------------------------------------------------------
# Database dump
# ---------------------------------------------------------------------------

ROW_TABLES: tuple[str, ...] = (
    "agent_runs",
    "react_traces",
    "tool_calls",
    "token_usage",
    "llm_interactions",
    "run_events",
)
_ROW_ORDER: dict[str, tuple[str, ...]] = {
    "react_traces": ("order_index",),
    "tool_calls": ("iteration_index", "provider_tool_call_id"),
    "token_usage": ("iteration_index",),
    "llm_interactions": ("iteration_index",),
    "run_events": ("sequence_index",),
}


async def dump_rows(engine: AsyncEngine) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    """Every row of the persisted tables (deterministically ordered, JSON-safe, un-normalised)
    plus the row count of every table that did not exist in v0.3.0."""
    from sqlalchemy import select

    from dendrux.db.models import Base

    tables = {t.name: t for t in Base.metadata.sorted_tables}
    out: dict[str, list[dict[str, Any]]] = {}
    extra: dict[str, int] = {}
    async with engine.connect() as conn:
        runs = (
            (
                await conn.execute(
                    select(tables["agent_runs"]).order_by(
                        tables["agent_runs"].c.created_at, tables["agent_runs"].c.id
                    )
                )
            )
            .mappings()
            .all()
        )
        rank = {r["id"]: i for i, r in enumerate(runs)}
        out["agent_runs"] = [_row_dict(r) for r in runs]
        for name in ROW_TABLES[1:]:
            rows = [
                _row_dict(r) for r in (await conn.execute(select(tables[name]))).mappings().all()
            ]
            keys = _ROW_ORDER[name]
            rows.sort(
                key=lambda r, keys=keys: (
                    rank.get(r["agent_run_id"], 10**6),
                    *[(r.get(k) is None, r.get(k)) for k in keys],
                )
            )
            out[name] = rows
        provider_ids = {r["tool_call_id"]: r["provider_tool_call_id"] for r in out["tool_calls"]}
        out["run_events"] = canonical_concurrent_tool_events(out["run_events"], provider_ids)
        # Tables that did not exist in v0.3.0 must stay empty for text-only runs.
        for name, table in tables.items():
            if name not in ROW_TABLES:
                extra[name] = len((await conn.execute(select(table))).all())
    return out, extra


def canonical_concurrent_tool_events(
    events: list[dict[str, Any]], provider_ids: dict[str, str | None]
) -> list[dict[str, Any]]:
    """Put each run of consecutive ``tool.completed`` rows (same run + iteration) in call order.

    Parallel tools commit their events in completion order, which races (notably on Postgres).
    The rows keep the block's original ``sequence_index`` values, handed out in canonical order.
    """
    out: list[dict[str, Any]] = []
    i = 0
    while i < len(events):
        first = events[i]
        if first["event_type"] != "tool.completed":
            out.append(first)
            i += 1
            continue
        j = i
        while (
            j < len(events)
            and events[j]["event_type"] == "tool.completed"
            and events[j]["agent_run_id"] == first["agent_run_id"]
            and events[j]["iteration_index"] == first["iteration_index"]
        ):
            j += 1
        block = events[i:j]
        sequence = sorted(e["sequence_index"] for e in block)
        block.sort(key=lambda e: str(provider_ids.get(e["correlation_id"])))
        for event, seq in zip(block, sequence, strict=True):
            out.append({**event, "sequence_index": seq})
        i = j
    return out


def _row_dict(row: Any) -> dict[str, Any]:
    return {k: jsonable(v) if not isinstance(v, datetime) else "<ts>" for k, v in dict(row).items()}


def normalise_rows(rows: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Normalise a ``dump_rows`` result; golden-visible tables only (see compare_rows)."""
    norm = Normalizer()
    out: dict[str, Any] = {}
    for name in ROW_TABLES:
        projected = []
        for row in rows[name]:
            row, problem = _strip_050_additions(name, row)
            assert problem is None, problem
            projected.append(row)
        out[name] = norm(projected)
    # Round-trip through sorted-key JSON so dialect JSON key order cannot matter.
    return json.loads(dumps_canonical(out))


_STOP_REASONS = (
    "end_turn",
    "tool_use",
    "max_tokens",
    "stop_sequence",
    "content_filter",
    "refusal",
    "pause_turn",
    "other",
    None,
)


def _strip_041_additions(table: str, row: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    """Remove the 0.4.1 stop-reason keys (#42) after checking their shape.

    0.4.1 adds, additively: ``agent_runs.meta["dendrux.stop_reason"]``,
    ``llm_interactions.semantic_response["stop_reason"]`` and
    ``"stop_reason"``/``"truncated"`` on ``run.completed`` event data.
    Everything else must still match v0.3.0 byte for byte.
    """
    row = dict(row)
    if table == "agent_runs" and isinstance(row.get("meta"), dict):
        meta = dict(row["meta"])
        if meta.pop("dendrux.stop_reason", None) not in _STOP_REASONS:
            return row, "agent_runs.meta: invalid dendrux.stop_reason"
        row["meta"] = meta
    elif table == "llm_interactions" and isinstance(row.get("semantic_response"), dict):
        sr = dict(row["semantic_response"])
        if sr.pop("stop_reason", None) not in _STOP_REASONS:
            return row, "llm_interactions.semantic_response: invalid stop_reason"
        row["semantic_response"] = sr
    elif (
        table == "run_events"
        and row.get("event_type") == "run.completed"
        and isinstance(row.get("data"), dict)
    ):
        data = dict(row["data"])
        reason = data.pop("stop_reason", None)
        truncated = data.pop("truncated", False)
        if reason not in _STOP_REASONS or truncated != (reason == "max_tokens"):
            return row, "run.completed: inconsistent stop_reason/truncated"
        row["data"] = data
    return row, None


def _strip_050_additions(table: str, row: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    """Validate new integrity fields before comparing the older text wire shape.

    Strip ULIDs before Normalizer allocates identities, so an added pause ID
    cannot renumber the pre-existing run and call IDs. Goldens stay unchanged.
    """
    row = dict(row)
    if table == "agent_runs":
        if "claim_epoch" in row:
            epoch = row.pop("claim_epoch")
            if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
                return row, "agent_runs.claim_epoch: expected a non-negative int"
        if isinstance(row.get("meta"), dict):
            meta = dict(row["meta"])
            governance = meta.pop("dendrux.governance", None)
            if governance is not None and (
                governance.get("version") != 1
                or not re.fullmatch(r"[0-9a-f]{64}", governance.get("digest", ""))
                or set(governance.get("parts", {}))
                != {"guardrails", "before_tool", "deny", "require_approval", "limits", "budget"}
                or not all(
                    re.fullmatch(r"[0-9a-f]{64}", value) for value in governance["parts"].values()
                )
            ):
                return row, "agent_runs.meta: invalid governance digest"
            row["meta"] = meta
        if isinstance(row.get("pause_data"), dict):
            pause = dict(row["pause_data"])
            _strip_pause_versioned_fields(pause)
            pause.pop("pause_id", None)
            pause.pop("status", None)
            pause.pop("dendrux.governance", None)
            row["pause_data"] = pause
    if table == "run_events" and row.get("event_type") == "run.paused":
        data = dict(row["data"] or {})
        pause_id = data.pop("pause_id", None)
        if pause_id is not None and not re.fullmatch(r"[0-9A-HJKMNP-TV-Z]{26}", pause_id):
            return row, "run.paused: invalid pause_id"
        row["data"] = data
    return row, None


def _strip_050_cache_write_prompt(
    table: str, row: dict[str, Any]
) -> tuple[dict[str, Any], str | None]:
    """Undo the realistic OpenRouter wire in a persisted ``provider_response``.

    The fake OpenRouter wire counts cache writes in ``prompt_tokens`` (as the
    real one does; the v0.3.0 goldens did not). 0.5.0 partitions the writes
    out of fresh input (D-127), so the normalised usage still matches the
    goldens; only the raw vendor dump differs, by exactly the writes.
    """
    response = row.get("provider_response") if table == "llm_interactions" else None
    usage = response.get("usage") if isinstance(response, dict) else None
    details = usage.get("prompt_tokens_details") if isinstance(usage, dict) else None
    writes = details.get("cache_write_tokens") if isinstance(details, dict) else None
    if not isinstance(writes, int) or not writes:
        return row, None
    prompt, total = usage.get("prompt_tokens"), usage.get("total_tokens")
    if not isinstance(prompt, int) or not isinstance(total, int) or prompt < writes:
        return row, f"provider_response.usage: prompt_tokens {prompt!r} lacks the cache writes"
    usage = {**usage, "prompt_tokens": prompt - writes, "total_tokens": total - writes}
    return {**row, "provider_response": {**response, "usage": usage}}, None


def compare_rows(expected: dict[str, Any], actual: dict[str, Any]) -> str | None:
    """None when identical. New nullable columns in ``actual`` must be NULL (R-02/P-06).

    The 0.4.1 stop-reason keys and the 0.5.0 ``claim_epoch`` column are
    validated and then ignored (see :func:`_strip_041_additions` and
    :func:`_strip_050_additions`).
    """
    for table in ROW_TABLES:
        exp_rows, act_rows = expected[table], actual[table]
        if len(exp_rows) != len(act_rows):
            return f"{table}: {len(exp_rows)} rows expected, {len(act_rows)} found"
        for i, (e, raw) in enumerate(zip(exp_rows, act_rows, strict=True)):
            a, problem = _strip_041_additions(table, raw)
            if problem:
                return f"{table}[{i}]: {problem}"
            a, problem = _strip_050_additions(table, a)
            if problem:
                return f"{table}[{i}]: {problem}"
            a, problem = _strip_050_cache_write_prompt(table, a)
            if problem:
                return f"{table}[{i}]: {problem}"
            for col in e:
                if col not in a:
                    return f"{table}[{i}].{col}: column missing"
                diff = first_diff(_sorted(e[col]), _sorted(a[col]), f"{table}[{i}].{col}")
                if diff:
                    return diff
            for col in a:
                if col not in e and a[col] is not None:
                    return f"{table}[{i}].{col}: new column carries {a[col]!r} for a text-only run"
    return None


def _sorted(obj: Any) -> Any:
    return json.loads(dumps_canonical(obj))


# ---------------------------------------------------------------------------
# Database lifecycle helpers
# ---------------------------------------------------------------------------


async def create_sqlite_engine() -> AsyncEngine:
    from sqlalchemy import event
    from sqlalchemy.ext.asyncio import create_async_engine

    from dendrux.db.models import Base

    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:", connect_args={"check_same_thread": False}
    )

    @event.listens_for(engine.sync_engine, "connect")
    def _fk(dbapi_conn: Any, _rec: Any) -> None:
        dbapi_conn.execute("PRAGMA foreign_keys = ON")

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    return engine


async def reset_database(engine: AsyncEngine) -> None:
    """Empty every table (TRUNCATE on Postgres, ordered DELETE on SQLite)."""
    from sqlalchemy import text

    from dendrux.db.models import Base

    async with engine.begin() as conn:
        if engine.dialect.name == "postgresql":
            names = ", ".join(t.name for t in Base.metadata.sorted_tables)
            await conn.execute(text(f"TRUNCATE TABLE {names} RESTART IDENTITY CASCADE"))
        else:
            for table in reversed(Base.metadata.sorted_tables):
                await conn.execute(table.delete())


# ---------------------------------------------------------------------------
# Cell runner
# ---------------------------------------------------------------------------


@dataclass
class CellResult:
    wire: dict[str, Any] = field(default_factory=dict)
    rows: dict[str, Any] = field(default_factory=dict)
    extra_tables: dict[str, dict[str, int]] = field(default_factory=dict)


async def run_cell(rig: Rig, mode: Mode, engine: AsyncEngine) -> CellResult:
    """Run every scenario for one (rig, mode) cell on ``engine``; return both sections."""
    from dendrux.runtime.state import SQLAlchemyStateStore

    assert_imports_this_checkout()
    result = CellResult()
    wire: dict[str, Any] = {}
    with openrouter_catalog(rig):
        for name, scenario in SCENARIOS:
            await reset_database(engine)
            ctx = Ctx(rig, mode, SQLAlchemyStateStore(engine))
            await scenario(ctx)
            wire[name] = {"legs": ctx.recorder.legs, "outcomes": ctx.outcomes}
            raw_rows, extra = await dump_rows(engine)
            result.rows[name] = normalise_rows(raw_rows)
            result.extra_tables[name] = extra
    wire["adapter_direct"] = await run_direct(rig, mode)
    for name in [n for n, _ in SCENARIOS]:
        wire[name] = Normalizer()(wire[name])
    result.wire = json.loads(dumps_ordered(wire))
    return result


def compare_wire(expected: dict[str, Any], actual: dict[str, Any]) -> str | None:
    """Order-sensitive structural + byte comparison of a wire section."""
    diff = first_diff(expected, actual)
    if diff:
        return diff
    if dumps_ordered(expected) != dumps_ordered(actual):  # pragma: no cover - defensive
        return "serialised bytes differ"
    return None


# ---------------------------------------------------------------------------
# Pure-function hashes (idempotency fingerprint, history seed, prefix, tool hashes)
# ---------------------------------------------------------------------------


def pure_hashes() -> dict[str, Any]:
    from dendrux.chat import ChatMessage
    from dendrux.context_blocks import ContextBlock
    from dendrux.runtime import replay
    from dendrux.types import Message, Role, ToolDef, compute_idempotency_fingerprint

    hist = [
        Message(role=Role.USER, content="What is 2+2?"),
        Message(role=Role.ASSISTANT, content="4"),
    ]
    stable = ContextBlock("project spec", kind="spec", placement="stable")
    dynamic = ContextBlock("doc excerpt", kind="doc", source="kb:42")
    fp = compute_idempotency_fingerprint
    replay_knobs = {"runs": ["r1", "r2"], "replay_last": 1, "replay_reasoning": True}
    fingerprints = {
        "plain": fp("OracleAgent", "hello"),
        "output_type": fp("OracleAgent", "hello", "Answer"),
        "history": fp("OracleAgent", "hello", history=hist),
        "context_dynamic": fp("OracleAgent", "hello", context=[dynamic]),
        "context_stable_dynamic": fp("OracleAgent", "hello", context=[stable, dynamic]),
        "context_empty": fp("OracleAgent", "hello", context=[]),
        "replay": fp("OracleAgent", "hello", replay=replay_knobs),
        "skills": fp("OracleAgent", "hello", skills={"active": ["a"]}),
        "unicode": fp("OracleAgent", "héllo ☃"),
        "all": fp("OracleAgent", "hello", "Answer", hist, [stable, dynamic], replay_knobs),
    }
    chat_hist = [ChatMessage.user("What is 2+2?"), ChatMessage.assistant("4")]
    seeds = {
        "empty": replay.history_seed([]),
        "history": replay.history_seed(
            [
                Message(role=Role.USER, content=m.content)
                if m.role.value == "user"
                else Message(role=Role.ASSISTANT, content=m.content)
                for m in chat_hist
            ]
        ),
    }
    lookup_def = ToolDef(
        name="lookup",
        description="Look something up.",
        parameters={"type": "object", "properties": {"q": {"type": "string"}}},
    )
    strict_def = ToolDef(
        name="add",
        description="Add.",
        parameters={"type": "object", "properties": {"a": {"type": "integer"}}},
        strict=True,
    )
    tool_hashes = replay.tool_hashes([strict_def, lookup_def])
    current = replay.CurrentPrefix(
        provider="AnthropicProvider@https://api.anthropic.com",
        model="claude-sonnet-4-6",
        system_hash=replay._hash("You are the oracle assistant."),
        tools_hash=replay._hash(tool_hashes),
        tools=tool_hashes,
    )
    return {
        "fingerprints": fingerprints,
        "history_seeds": seeds,
        "tool_hashes": tool_hashes,
        "prefix_descriptor": replay.prefix_descriptor(current, seeds["history"]),
        "seed_key": replay._seed_key(
            {"expansion": [{"mode": "full"}], "history_hash": "abc", "retry_of": None}
        ),
    }


# ---------------------------------------------------------------------------
# R-04: existing cache-related tests stay unchanged
# ---------------------------------------------------------------------------

# (path relative to tests/, selector). "*" = whole file; a tuple = those test functions
# (matched by name anywhere in the module, class methods included) or, for ("lines", a, b),
# every test function overlapping the v0.3.0 line range (cache_regression_constraints.md R-04).
R04_FILES: tuple[tuple[str, Any], ...] = (
    (
        "unit/test_cache_stable_prefix.py",
        (
            # R-04 names these loosely; these are the actual v0.3.0 test names.
            "test_system_prompt_identical_across_skill_reversal",
            "test_system_prompt_lists_skills_alphabetically",
            "test_local_tools_sorted_by_name",
            "test_mcp_discovered_storage_is_sorted",
            "test_get_all_tool_defs_sorted_across_source_order",
        ),
    ),
    ("unit/test_provider_cache_activation.py", "*"),
    ("unit/test_cache_telemetry.py", "*"),
    ("integration/test_cache_persistence.py", "*"),
    ("unit/test_context_blocks.py", "*"),
    ("unit/test_context_blocks_e2e.py", "*"),
    ("unit/test_run_replay.py", ("lines", 342, 457)),
    ("unit/test_openrouter_provider.py", ("lines", 1025, 1078)),
)


def _test_segments(path: Path) -> dict[str, tuple[int, int, str]]:
    source = path.read_text(encoding="utf-8")
    lines = source.splitlines()
    tree = ast.parse(source)
    found: dict[str, tuple[int, int, str]] = {}

    def visit(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                visit(child, f"{prefix}{child.name}.")
            elif isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
                start = min([child.lineno, *[d.lineno for d in child.decorator_list]])
                found[f"{prefix}{child.name}"] = (
                    start,
                    child.end_lineno or child.lineno,
                    "\n".join(lines[start - 1 : child.end_lineno]),
                )

    visit(tree, "")
    return found


def r04_manifest() -> dict[str, str]:
    """sha256 per protected file (whole) or per protected test (source segment)."""
    manifest: dict[str, str] = {}
    for rel, selector in R04_FILES:
        path = TESTS_DIR / rel
        if selector == "*":
            manifest[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
            continue
        segments = _test_segments(path)
        if selector[0] == "lines":
            _, lo, hi = selector
            chosen = {
                n: seg
                for n, seg in segments.items()
                if n.split(".")[-1].startswith("test_") and seg[0] <= hi and seg[1] >= lo
            }
        else:
            chosen = {n: seg for n, seg in segments.items() if n.split(".")[-1] in selector}
            wanted = set(selector)
            if {n.split(".")[-1] for n in chosen} != wanted:
                missing = wanted - {n.split(".")[-1] for n in chosen}
                raise AssertionError(f"{rel}: protected tests not found: {sorted(missing)}")
        for name, (_, _, text) in sorted(chosen.items()):
            manifest[f"{rel}::{name}"] = sha256_text(text)
    return manifest


def environment_info() -> dict[str, Any]:
    import importlib.metadata as md

    info: dict[str, Any] = {"dendrux": package_version(), "python": sys.version.split()[0]}
    for dist in ("anthropic", "openai", "SQLAlchemy"):
        try:
            info[dist] = md.version(dist)
        except md.PackageNotFoundError:  # pragma: no cover
            info[dist] = None
    return info


REQUIRED_SCENARIOS = frozenset(
    {
        "plain",  # plain user/assistant
        "history",
        "context_absent",  # context absent
        "context_stable_dynamic",  # context present (stable + dynamic)
        "system_tools",  # system + tools
        "server_tool_text",  # server tool
        "client_tool_pause_resume",  # client tool, pause/resume (fresh agent on resume)
        "mixed_server_client_pause",
        "parallel_tools",
        "replay_runs",  # runs= replay
    }
)


def manifest() -> dict[str, Any]:
    """Descriptive golden header. Informational only: never compared by the parity tests."""
    return {
        "oracle_version": ORACLE_VERSION,
        "source": ORACLE_SOURCE,
        "environment": environment_info(),
        "cells": [cell_name(r, m) for r, m in CELLS],
        "scenarios": [n for n, _ in SCENARIOS] + ["adapter_direct"],
        "adapter_direct": list(_direct_messages()),
    }
