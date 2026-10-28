"""Conformance harness: run a scripted cell, capture it, project golden and corpus views.

A *cell* is one scenario flow (``scenarios.py``): a sequence of public entry-path calls
(``run``, ``stream``, ``resume``, ``resume_stream``, ``submit_*``, ``cancel_run``, ``retry``,
``sweep``) against scripted providers (no network). The harness records, for the cell:

``steps``
    One entry per entry-path call: the path, the call label, and either the ``RunResult`` view
    (status, answer, error, ``error_info``, pause id presence, ``pending``, usage, ``to_dict()``
    wire shape, ...) or the exception class chain it raised. Streaming calls also record the
    stream event sequence.
``runs``
    Every run the cell created (oldest first), read back through the store's *protocol*
    methods only: the run record, the persisted event-type sequence, the trace role sequence,
    the tool-call rows and the live pause data.
``calls``
    The exact message list (and tools, options) each scripted provider received, in order,
    captured at the provider boundary. This is the oracle for the 0.6 history rebuild.
``rows``
    Every row of every table after the flow (SQLAlchemy stores only).

The golden file (``tests/conformance/golden``) is the normalised projection of ``steps`` and
``runs``. The corpus file (``tests/_fixtures/corpus_050``) is the *raw* recording of ``calls``,
``steps`` and ``rows`` (real ids and timestamps), so 0.6 code can feed the recorded rows to
its rebuild and compare with the recorded provider messages. Both are compared after the same
normalisation (``v030_oracle.Normalizer``: ULIDs numbered by first appearance, timestamps and
durations masked, floats rounded).

Regeneration is refused unless the checkout's ``src/dendrux`` is byte-identical to tag
``v0.5.0``: a candidate never blesses its own output (see ``regen.py``).
"""

from __future__ import annotations

import base64
import hashlib
import json
import subprocess
import tomllib
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from dendrux.agent import Agent
from dendrux.llm.mock import MockLLM
from dendrux.types import LLMResponse, ToolCall, UsageStats
from tests._helpers import v030_oracle as o

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine

HERE = Path(__file__).resolve().parent
TESTS_DIR = HERE.parent
PACKAGE_DIR = TESTS_DIR.parent  # packages/python
GOLDEN_DIR = HERE / "golden"
CORPUS_DIR = TESTS_DIR / "_fixtures" / "corpus_050"
PAUSE_STATUSES = frozenset(
    {"paused", "waiting_approval", "waiting_client_tool", "waiting_human_input"}
)
#: Tables stored in the corpus: the conversation record, not telemetry (usage and LLM payloads).
CORPUS_TABLES = (
    "agent_runs",
    "react_traces",
    "tool_calls",
    "run_events",
    "run_inbox",
    "dendrux_blobs",
    "dendrux_blob_refs",
)
RECORDED_VERSION = "0.5.0"
RECORDED_TAG = "v0.5.0"
REGEN_ENV = "DENDRUX_REGEN_CONFORMANCE_050"


# ---------------------------------------------------------------------------
# Regeneration guard
# ---------------------------------------------------------------------------


def package_version() -> str:
    with open(PACKAGE_DIR / "pyproject.toml", "rb") as fh:
        return str(tomllib.load(fh)["project"]["version"])


def regen_refusal() -> str | None:
    """Why regeneration is not allowed here, or None when it is.

    Allowed only when the package version is 0.5.0 *and* ``src/dendrux`` has no difference
    against tag ``v0.5.0`` (the oracle is 0.5.0 behaviour, never the candidate's own output).
    """
    version = package_version()
    if version != RECORDED_VERSION:
        return f"this checkout is Dendrux {version}; goldens are 0.5.0 behaviour"
    try:
        proc = subprocess.run(  # noqa: S603
            ["git", "diff", "--quiet", RECORDED_TAG, "--", "src/dendrux"],  # noqa: S607
            cwd=PACKAGE_DIR,
            capture_output=True,
            check=False,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return f"cannot compare src/dendrux with {RECORDED_TAG} ({type(exc).__name__})"
    if proc.returncode != 0:
        return f"src/dendrux differs from {RECORDED_TAG} (or the tag is missing)"
    return None


def assert_imports_this_checkout() -> None:
    o.assert_imports_this_checkout()


# ---------------------------------------------------------------------------
# Response factories (deterministic scripts)
# ---------------------------------------------------------------------------


def usage(inp: int = 10, out: int = 5, cost: float | None = None) -> UsageStats:
    return UsageStats(input_tokens=inp, output_tokens=out, total_tokens=inp + out, cost_usd=cost)


def text(body: str, *, inp: int = 10, out: int = 5, cost: float | None = None) -> LLMResponse:
    return LLMResponse(text=body, usage=usage(inp, out, cost))


def calls(
    *specs: tuple[str, dict[str, Any], str],
    body: str | None = None,
    inp: int = 10,
    out: int = 5,
    cost: float | None = None,
) -> LLMResponse:
    """One response requesting tool calls ``(name, params, provider call id)``."""
    return LLMResponse(
        text=body,
        tool_calls=[ToolCall(name=n, params=p, provider_tool_call_id=pid) for n, p, pid in specs],
        usage=usage(inp, out, cost),
    )


# ---------------------------------------------------------------------------
# Raw JSON (recording) and views
# ---------------------------------------------------------------------------


def raw(obj: Any) -> Any:
    """Plain-JSON snapshot keeping real values (ids, ISO timestamps); bytes as base64."""
    if obj is None or isinstance(obj, bool | int | str):
        return obj
    if isinstance(obj, float):
        return obj
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, Enum):
        return raw(obj.value)
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, bytes):
        return {"__bytes_b64__": base64.b64encode(obj).decode("ascii")}
    if isinstance(obj, dict):
        return {str(k): raw(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple | set | frozenset):
        items = sorted(obj, key=repr) if isinstance(obj, set | frozenset) else obj
        return [raw(v) for v in items]
    if hasattr(obj, "__dataclass_fields__") and not isinstance(obj, type):
        return {name: raw(getattr(obj, name)) for name in obj.__dataclass_fields__}
    return f"<{type(obj).__name__}>"


def part_view(part: Any) -> dict[str, Any]:
    kind = type(part).__name__
    if kind == "TextPart":
        return {"type": "text", "text": part.text}
    view: dict[str, Any] = {"type": "image", "ref": raw(part.ref.to_dict()), "detail": part.detail}
    data = getattr(part, "data", None)
    view["data_sha256"] = hashlib.sha256(data).hexdigest() if data is not None else None
    return view


def tool_call_view(tc: Any) -> dict[str, Any]:
    return {
        "id": tc.id,
        "name": tc.name,
        "params": raw(tc.params),
        "provider_tool_call_id": tc.provider_tool_call_id,
        "parse_error": getattr(tc, "parse_error", None),
    }


def message_view(m: Any) -> dict[str, Any]:
    """A Dendrux ``Message`` as a provider received it, every field."""
    return {
        "role": m.role.value,
        "content": m.content,
        "name": m.name,
        "call_id": m.call_id,
        "tool_calls": [tool_call_view(tc) for tc in (m.tool_calls or [])] or None,
        "kind": m.kind,
        "placement": m.placement,
        "source": m.source,
        "meta": raw(m.meta),
        "reasoning": m.reasoning,
        "reasoning_blocks": raw(m.reasoning_blocks),
        "parts": [part_view(p) for p in m.parts] if m.parts else None,
    }


def tool_def_view(t: Any) -> dict[str, Any]:
    return {
        "name": t.name,
        "description": t.description,
        "parameters": raw(t.parameters),
        "target": t.target.value,
        "strict": t.strict,
    }


def exception_chain(exc: BaseException) -> list[str]:
    """Class names of ``type(exc)`` and its bases up to ``Exception`` (never the message)."""
    names = []
    for cls in type(exc).__mro__:
        names.append(cls.__name__)
        if cls is Exception:
            break
    return names


def result_view(result: Any) -> dict[str, Any]:
    return {
        "run_id": result.run_id,
        "status": result.status.value,
        "answer": result.answer,
        "output": raw(result.output),
        "iteration_count": result.iteration_count,
        "usage": raw(result.usage),
        "error": result.error,
        "error_info": raw(result.error_info),
        "stop_reason": result.stop_reason,
        "truncated": result.truncated,
        "pause_id_present": result.pause_id is not None,
        "pause_id": result.pause_id,
        "commentary": list(result.commentary),
        "replay": raw(result.replay),
        "meta": raw(result.meta),
        "pending": raw(result.pending),
        "steps": [type(s.action).__name__ for s in result.steps],
        "wire": raw(result.to_dict()),
    }


def stream_event_view(event: Any) -> list[Any]:
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


_TOOL_RESULT_PID = 8  # index into stream_event_view()


def canonical_stream_events(events: list[list[Any]]) -> list[list[Any]]:
    """Order each block of consecutive ``tool_result`` events by provider call id.

    Tools of one parallel group finish in a racy order; that order is not a contract.
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
        out.extend(sorted(events[i:j], key=lambda e: str(e[_TOOL_RESULT_PID])))
        i = j
    return out


# ---------------------------------------------------------------------------
# Recording provider
# ---------------------------------------------------------------------------


class Recorder:
    """Provider calls of one cell, in order, plus the entry-path step they happened in."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.step = -1


Hook = Callable[[int, dict[str, Any]], Awaitable[None]]


class RecordingProvider(MockLLM):
    """A scripted ``MockLLM`` that records what each call received.

    ``hook(index, kwargs)`` runs before the response is returned (``kwargs`` carries the
    loop's ``run_id``); it may raise to fail the call or touch the store to arrange a race
    such as a cooperative cancel at a known point.
    """

    def __init__(
        self,
        responses: list[LLMResponse],
        *,
        recorder: Recorder,
        label: str,
        hook: Hook | None = None,
        image_support: Any = "yes",
        reports_cost: bool = False,
    ) -> None:
        super().__init__(responses, image_support=image_support)
        self._reports_cost = reports_cost
        self._recorder = recorder
        self._label = label
        self._hook = hook
        self._index = 0

    @property
    def reports_cost(self) -> bool:
        return self._reports_cost

    async def complete(self, messages: Any, tools: Any = None, **kwargs: Any) -> LLMResponse:
        index = self._index
        self._index += 1
        options = {k: v for k, v in kwargs.items() if k not in ("run_id",)}
        self._recorder.calls.append(
            {
                "index": len(self._recorder.calls),
                "provider": self._label,
                "provider_call": index,
                "step": self._recorder.step,
                "run_id": kwargs.get("run_id"),
                "model": self.model,
                "messages": [message_view(m) for m in messages],
                "tools": [tool_def_view(t) for t in (tools or [])],
                "options": raw(options),
            }
        )
        if self._hook is not None:
            await self._hook(index, kwargs)
        return await super().complete(messages, tools, **kwargs)


# ---------------------------------------------------------------------------
# Cell context
# ---------------------------------------------------------------------------


@dataclass
class Ctx:
    """What a scenario flow drives: the store under test and the recorded entry paths."""

    engine: AsyncEngine
    store: Any
    recorder: Recorder = field(default_factory=Recorder)
    steps: list[dict[str, Any]] = field(default_factory=list)
    agents: list[dict[str, Any]] = field(default_factory=list)
    _providers: int = 0

    # -- construction -------------------------------------------------------

    def provider(
        self,
        *responses: LLMResponse,
        hook: Hook | None = None,
        image_support: Any = "yes",
        reports_cost: bool = False,
    ) -> RecordingProvider:
        label = f"p{self._providers}"
        self._providers += 1
        return RecordingProvider(
            list(responses),
            recorder=self.recorder,
            label=label,
            hook=hook,
            image_support=image_support,
            reports_cost=reports_cost,
        )

    def agent(self, provider: Any, *, tools: Any = (), **kwargs: Any) -> Agent:
        kwargs.setdefault("name", "Conf")
        kwargs.setdefault("prompt", "You are the conformance agent.")
        # 0.5 behaviour is pinned with the choice made explicitly (0.6 flips the default).
        kwargs.setdefault("strict_tenancy", False)
        kwargs.setdefault("on_governance_change", "warn")
        assert kwargs["on_governance_change"] in ("warn", "raise")  # exactly these two values
        self.agents.append(
            {
                "name": kwargs["name"],
                "on_governance_change": kwargs["on_governance_change"],
                "strict_tenancy": kwargs["strict_tenancy"],
            }
        )
        return Agent(provider=provider, tools=list(tools), state_store=self.store, **kwargs)

    # -- recording ----------------------------------------------------------

    async def _record(
        self,
        path: str,
        label: str,
        call: Callable[[], Awaitable[Any]],
        *,
        events: list[list[Any]] | None = None,
    ) -> Any:
        self.recorder.step = len(self.steps)
        step: dict[str, Any] = {"path": path, "label": label}
        self.steps.append(step)
        try:
            result = await call()
        except Exception as exc:
            step["raised"] = exception_chain(exc)
            return None
        if events is not None:
            step["stream_events"] = canonical_stream_events(events)
        if hasattr(result, "status") and hasattr(result, "run_id"):
            step["result"] = result_view(result)
            if result.status.value in PAUSE_STATUSES:
                step["at_pause"] = await self._snapshot_pause(result.run_id)
        else:
            step["value"] = raw(result)
        return result

    async def _snapshot_pause(self, run_id: str) -> dict[str, Any]:
        """Pause data and the traces persisted so far: the state a continuation starts from."""
        return {
            "run_id": run_id,
            "pause_state": raw(await self.store.get_pause_state(run_id)),
            "traces": [raw(t) for t in await self.store.get_traces(run_id)],
        }

    async def run(self, agent: Agent, user_input: Any, label: str = "", **kwargs: Any) -> Any:
        return await self._record("run", label, lambda: agent.run(user_input, **kwargs))

    async def retry(self, agent: Agent, run_id: str, label: str = "", **kwargs: Any) -> Any:
        return await self._record("retry", label, lambda: agent.retry(run_id, **kwargs))

    async def resume(self, agent: Agent, run_id: str, label: str = "", **kwargs: Any) -> Any:
        return await self._record("resume", label, lambda: agent.resume(run_id, **kwargs))

    async def submit_tool_results(
        self, agent: Agent, run_id: str, results: Any, label: str = "", **kwargs: Any
    ) -> Any:
        return await self._record(
            "submit_tool_results",
            label,
            lambda: agent.submit_tool_results(run_id, results, **kwargs),
        )

    async def submit_input(
        self, agent: Agent, run_id: str, body: Any, label: str = "", **kwargs: Any
    ) -> Any:
        return await self._record(
            "submit_input", label, lambda: agent.submit_input(run_id, body, **kwargs)
        )

    async def submit_approval(
        self, agent: Agent, run_id: str, label: str = "", **kwargs: Any
    ) -> Any:
        return await self._record(
            "submit_approval", label, lambda: agent.submit_approval(run_id, **kwargs)
        )

    async def cancel_run(self, agent: Agent, run_id: str, label: str = "", **kwargs: Any) -> Any:
        return await self._record("cancel_run", label, lambda: agent.cancel_run(run_id, **kwargs))

    async def sweep(self, label: str = "", **kwargs: Any) -> Any:
        from dendrux import sweep

        return await self._record("sweep", label, lambda: sweep(state_store=self.store, **kwargs))

    async def _drain(self, stream: Any, events: list[list[Any]]) -> Any:
        async with stream as source:
            async for event in source:
                events.append(stream_event_view(event))
        result = source.result
        if result is None:
            raise RuntimeError("stream produced no result")
        return result

    async def stream(self, agent: Agent, user_input: Any, label: str = "", **kwargs: Any) -> Any:
        events: list[list[Any]] = []
        return await self._record(
            "stream",
            label,
            lambda: self._drain(agent.stream(user_input, **kwargs), events),
            events=events,
        )

    async def resume_stream(self, agent: Agent, run_id: str, label: str = "", **kwargs: Any) -> Any:
        events: list[list[Any]] = []
        return await self._record(
            "resume_stream",
            label,
            lambda: self._drain(agent.resume_stream(run_id, **kwargs), events),
            events=events,
        )

    # -- arranging stored state (what a crashed or idle fleet leaves behind) --------

    async def age_runs(self, *run_ids: str, minutes: int = 120) -> None:
        """Move a run's status/progress clocks into the past (sweep thresholds)."""
        from sqlalchemy import update
        from sqlalchemy.ext.asyncio import async_sessionmaker

        from dendrux.db.models import AgentRun

        old = datetime.now(UTC) - timedelta(minutes=minutes)
        factory = async_sessionmaker(self.engine, expire_on_commit=False)
        async with factory() as session:
            for run_id in run_ids:
                await session.execute(
                    update(AgentRun)
                    .where(AgentRun.id == run_id)
                    .values(
                        last_progress_at=old,
                        status_changed_at=old,
                        updated_at=old,
                    )
                )
            await session.commit()

    async def orphan_running_run(self, agent_name: str = "Orphan") -> str:
        """A run row a crashed worker left in ``running`` (created through the store)."""
        from dendrux.types import generate_ulid

        run_id = generate_ulid()
        await self.store.create_run(run_id, agent_name, input_data={"input": "x"}, model="mock")
        await self.store.save_run_event(
            run_id, event_type="run.started", data={"agent_name": agent_name}
        )
        return run_id


# ---------------------------------------------------------------------------
# Read-back
# ---------------------------------------------------------------------------

_PARALLEL_KEY = "provider_tool_call_id"


def _record_view(record: Any) -> dict[str, Any]:
    view = raw(record)
    return view


def _canonical_events(
    events: list[dict[str, Any]], provider_ids: dict[str, Any]
) -> list[dict[str, Any]]:
    """Order each block of consecutive ``tool.completed`` events by provider call id."""
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
            and events[j]["iteration_index"] == first["iteration_index"]
        ):
            j += 1
        block = events[i:j]
        sequence = sorted(e["sequence_index"] for e in block)
        block.sort(key=lambda e: str(provider_ids.get(e.get("correlation_id"))))
        for event, seq in zip(block, sequence, strict=True):
            out.append({**event, "sequence_index": seq})
        i = j
    return out


async def read_runs(store: Any) -> list[dict[str, Any]]:
    """Every run, oldest first, through the store's protocol read methods."""
    records = await store.list_runs(limit=500)
    records = sorted(records, key=lambda r: (r.created_at, r.delegation_level))
    runs = []
    for record in records:
        run_id = record.id
        traces = await store.get_traces(run_id)
        tool_rows = sorted(
            await store.get_tool_calls(run_id),
            key=lambda t: (t.iteration_index, str(t.provider_tool_call_id)),
        )
        provider_ids = {t.tool_call_id: t.provider_tool_call_id for t in tool_rows}
        events = _canonical_events(
            [raw(e) for e in await store.get_run_events(run_id)], provider_ids
        )
        pause = await store.get_pause_state(run_id)
        runs.append(
            {
                "run_id": run_id,
                "record": raw(record),
                "events": events,
                "traces": [raw(t) for t in traces],
                "tool_calls": [raw(t) for t in tool_rows],
                "pause_state": raw(pause),
            }
        )
    return runs


_ROW_ORDER: dict[str, tuple[str, ...]] = {
    "agent_runs": ("created_at", "id"),
    "react_traces": ("order_index",),
    "tool_calls": ("iteration_index", "provider_tool_call_id"),
    "token_usage": ("iteration_index",),
    "llm_interactions": ("iteration_index",),
    "run_events": ("sequence_index",),
    "run_inbox": ("created_at",),
    "dendrux_blobs": ("tenant_scope", "sha256"),
    "dendrux_blob_refs": ("owner_kind", "owner_id", "ordinal"),
}


async def dump_tables(engine: AsyncEngine) -> dict[str, list[dict[str, Any]]]:
    """Every row of every table, run rows first, deterministic order, raw values."""
    from sqlalchemy import select

    from dendrux.db.models import Base

    tables = {t.name: t for t in Base.metadata.sorted_tables}
    out: dict[str, list[dict[str, Any]]] = {}
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
        out["agent_runs"] = [{k: raw(v) for k, v in dict(r).items()} for r in runs]
        for name, table in tables.items():
            if name == "agent_runs":
                continue
            rows = [
                {k: raw(v) for k, v in dict(r).items()}
                for r in (await conn.execute(select(table))).mappings().all()
            ]
            keys = _ROW_ORDER.get(name, ())
            run_key = "agent_run_id" if "agent_run_id" in table.c else "run_id"

            def order(
                r: dict[str, Any], keys: tuple[str, ...] = keys, run_key: str = run_key
            ) -> Any:
                return (
                    rank.get(r.get(run_key), 10**6),
                    *[
                        (r.get(k) is None, str(r.get(k)))
                        if not isinstance(r.get(k), int)
                        else (False, r.get(k))
                        for k in keys
                    ],
                )

            rows.sort(key=order)
            out[name] = rows
    provider_ids = {r["tool_call_id"]: r["provider_tool_call_id"] for r in out["tool_calls"]}
    out["run_events"] = _canonical_event_rows(out["run_events"], provider_ids)
    return out


def _canonical_event_rows(
    events: list[dict[str, Any]], provider_ids: dict[str, Any]
) -> list[dict[str, Any]]:
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


# ---------------------------------------------------------------------------
# Cell execution and projections
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Cell:
    """One scenario flow."""

    id: str
    scenario: str
    flow: Callable[[Ctx], Awaitable[None]]
    paths: tuple[str, ...]  # entry paths the flow exercises (checked against the recording)
    # "blob_catalog": needs the built-in store's blob catalog (images); a protocol-only
    # custom store cannot run it, so the legacy-store variants skip the cell.
    tags: tuple[str, ...] = ()


async def run_cell(cell: Cell, engine: AsyncEngine, store: Any) -> dict[str, Any]:
    """Run ``cell`` on ``store`` (wrapping ``engine``) and return the raw recording."""
    assert_imports_this_checkout()
    ctx = Ctx(engine=engine, store=store)
    await cell.flow(ctx)
    return {
        "id": cell.id,
        "scenario": cell.scenario,
        "dendrux_version": RECORDED_VERSION,
        "steps": ctx.steps,
        "agents": ctx.agents,
        "runs": await read_runs(store),
        "calls": ctx.recorder.calls,
        "rows": await dump_tables(engine),
    }


def _norm(obj: Any) -> Any:
    """Normalise (one ULID numbering scope per document); object key order is not compared.

    Postgres JSONB returns object keys in its own order, so every document is round-tripped
    through sorted-key JSON.
    """
    return json.loads(o.dumps_canonical(o.Normalizer()(obj)))


def _run_golden(run: dict[str, Any]) -> dict[str, Any]:
    record = run["record"]
    keep = (
        "id",
        "agent_name",
        "status",
        "answer",
        "error",
        "iteration_count",
        "model",
        "strategy",
        "parent_run_id",
        "delegation_level",
        "total_input_tokens",
        "total_output_tokens",
        "total_cost_usd",
        "total_cache_read_tokens",
        "total_cache_creation_tokens",
        "total_reasoning_tokens",
        "failure_reason",
        "retry_of_run_id",
        "cancel_requested",
        "pause_requested",
        "tenant_id",
    )
    tool_keys = (
        "tool_call_id",
        "provider_tool_call_id",
        "tool_name",
        "target",
        "params",
        "result_payload",
        "success",
        "iteration_index",
        "error_message",
    )
    return {
        "record": {k: record[k] for k in keep},
        "events": [[e["event_type"], e["iteration_index"]] for e in run["events"]],
        "trace_roles": [t["role"] for t in run["traces"]],
        "tool_calls": [{k: t.get(k) for k in tool_keys} for t in run["tool_calls"]],
        "paused_with_state": run["pause_state"] is not None,
        "pause_state_keys": sorted(run["pause_state"]) if run["pause_state"] else None,
    }


def messages_digest(messages: list[dict[str, Any]]) -> str:
    """Digest of one provider call's complete, normalised message list."""
    return o.sha256_text(o.dumps_canonical(_norm(messages)))


def golden_view(recording: dict[str, Any]) -> dict[str, Any]:
    """The normalised conformance projection: results, events, trace roles, tool-call rows."""
    return _norm(
        {
            "id": recording["id"],
            "scenario": recording["scenario"],
            "steps": [{k: v for k, v in s.items() if k != "at_pause"} for s in recording["steps"]],
            "runs": [_run_golden(r) for r in recording["runs"]],
            "provider_calls": [
                {
                    "provider": c["provider"],
                    "step": c["step"],
                    "n_messages": len(c["messages"]),
                    "tools": [t["name"] for t in c["tools"]],
                    "messages_sha256": messages_digest(c["messages"]),
                }
                for c in recording["calls"]
            ],
        }
    )


def corpus_view(recording: dict[str, Any]) -> dict[str, Any]:
    """The raw recording stored in the corpus (real ids and timestamps, nothing masked)."""
    return {
        "id": recording["id"],
        "scenario": recording["scenario"],
        "dendrux_version": recording["dendrux_version"],
        "agents": recording["agents"],
        "steps": [_step_summary(s) for s in recording["steps"]],
        "calls": recording["calls"],
        "runs": [
            {
                "run_id": r["run_id"],
                "parent_run_id": r["record"]["parent_run_id"],
                "status": r["record"]["status"],
                "pause_state": r["pause_state"],
            }
            for r in recording["runs"]
        ],
        "rows": {name: recording["rows"][name] for name in CORPUS_TABLES},
    }


def _step_summary(step: dict[str, Any]) -> dict[str, Any]:
    result = step.get("result")
    return {
        "path": step["path"],
        "label": step["label"],
        "run_id": result["run_id"] if result else None,
        "status": result["status"] if result else None,
        "raised": step.get("raised"),
        "at_pause": step.get("at_pause"),
    }


def normalised_corpus(view: dict[str, Any]) -> dict[str, Any]:
    return _norm(view)


# ---------------------------------------------------------------------------
# Golden / corpus IO
# ---------------------------------------------------------------------------


def golden_path(cell_id: str, variant: str | None = None) -> Path:
    """Golden of a cell; ``variant`` ("legacy_a", "legacy_d") holds only cells that differ."""
    return (GOLDEN_DIR / variant if variant else GOLDEN_DIR) / f"{cell_id}.json"


def corpus_path(cell_id: str) -> Path:
    return CORPUS_DIR / f"{cell_id}.json"


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_one_line_per_key(data), encoding="utf-8")


def _one_line_per_key(data: Any) -> str:
    """Valid JSON, one top-level entry per line (compact values): small, diff-friendly."""
    if not isinstance(data, dict):
        return o.dumps_ordered(data) + "\n"
    items = [f"{json.dumps(k)}:{o.dumps_ordered(v)}" for k, v in data.items()]
    return "{\n" + ",\n".join(items) + "\n}\n"


def read_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Goldens and the corpus are recorded once from the 0.5.0 sources "
            f"with {REGEN_ENV}=1 (see tests/conformance/regen.py); tests never create them."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def diff(expected: Any, actual: Any) -> str | None:
    return o.first_diff(expected, actual)
