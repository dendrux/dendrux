"""FakeModel: scripted, stateless model answers for tests that need no API key."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import types
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from dendrux.llm.base import LLMProvider
from dendrux.testing._schema_check import SchemaUnsupportedError, TypeCheck, first_mismatch
from dendrux.types import LLMResponse, Message, ProviderCapabilities, Role, ToolCall, UsageStats

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence


class FakeModelScriptError(LookupError):
    """The script has no usable answer for a call.

    The message names the agent, the turn, counts and known agent names only:
    never message text, tool arguments or payloads. ``code`` is a stable reason:
    ``exhausted``, ``no_when_match``, ``invalid_json``, ``schema_mismatch``,
    ``unsupported_schema`` or ``unknown_agent`` (``script`` when none is given).
    """

    def __init__(self, message: str, *, code: str = "script") -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class _Step:
    text: str | None = None
    calls: tuple[tuple[str, dict[str, Any]], ...] = ()
    structured: bool = False


@dataclass(frozen=True)
class _When:
    predicate: Callable[[list[Message]], bool]
    response: _Step | LLMResponse


Entry = _Step | _When | LLMResponse


@dataclass(frozen=True)
class FakeCall:
    """One call the model received, for assertions."""

    turn: int
    messages: tuple[Message, ...]
    tools: tuple[Any, ...] | None
    output_schema: dict[str, Any] | None
    run_id: str | None
    model: str


def say(text: str) -> _Step:
    """A text answer."""
    return _Step(text=text)


def call(tool: str, /, **args: Any) -> _Step:
    """One tool call. A subagent is called as ``call("name", task="...")``."""
    return _Step(calls=((tool, dict(args)),))


def calls(*steps: _Step) -> _Step:
    """Several tool calls in one response (a parallel fan-out)."""
    merged = tuple(c for step in steps for c in step.calls)
    if not merged or any(step.text is not None or not step.calls for step in steps):
        raise TypeError("calls() takes call(...) steps only")
    return _Step(calls=merged)


def structured(value: BaseModel | Mapping[str, Any]) -> _Step:
    """JSON text for an agent with ``output_type``; checked against its schema when used.

    A model is dumped with its field aliases, the keys its output schema expects.
    A model whose validation and serialization aliases differ should be passed as a dict.
    """
    data = (
        value.model_dump(mode="json", by_alias=True)
        if isinstance(value, BaseModel)
        else dict(value)
    )
    return _Step(text=json.dumps(data, sort_keys=True), structured=True)


def when(predicate: Callable[[list[Message]], bool], response: _Step | LLMResponse) -> _When:
    """Answer only when ``predicate(messages)`` is true.

    Combination with turn keying: the entries of a script list are grouped into
    turn slots. A run of consecutive ``when`` entries, plus the one plain entry
    right after it (if any), is a single slot. In that turn the first ``when``
    whose predicate is true answers; otherwise the plain entry answers; with
    neither, the call raises ``FakeModelScriptError``. Plain entries keep one
    slot each. The predicate must be pure: it may run on any process.
    """
    if not callable(predicate):
        raise TypeError("when() needs a callable predicate")
    return _When(predicate, response)


def _slots(entries: Sequence[Entry]) -> tuple[tuple[Entry, ...], ...]:
    slots: list[tuple[Entry, ...]] = []
    group: list[Entry] = []
    for entry in entries:
        if not isinstance(entry, (_Step, _When, LLMResponse)):
            raise TypeError(f"script entries must be say/call/structured/when, got {type(entry)}")
        group.append(entry)
        if not isinstance(entry, _When):
            slots.append(tuple(group))
            group = []
    if group:
        slots.append(tuple(group))
    return tuple(slots)


def _usage(messages: Sequence[Message], text: str | None) -> UsageStats:
    tokens_in = 10 * len(messages)
    tokens_out = 5 + len(text or "") // 4
    return UsageStats(
        input_tokens=tokens_in, output_tokens=tokens_out, total_tokens=tokens_in + tokens_out
    )


def _code_key(code: Any) -> Any:
    consts = [_code_key(c) if hasattr(c, "co_code") else repr(c) for c in code.co_consts]
    return [code.co_code.hex(), consts, list(code.co_names)]


def _callable_key(fn: Any) -> Any:
    """Conservative identity for a predicate: same code and same plain captured values.

    Anything else (callable objects, bound methods, closures over objects) is the same
    only when it is the same object, so a rebuilt script with such a predicate is refused.
    """
    if isinstance(fn, types.FunctionType):
        cells = [c.cell_contents for c in fn.__closure__ or ()]
        if all(isinstance(v, (bool, int, float, str, bytes, type(None))) for v in cells):
            defaults = [fn.__defaults__, fn.__kwdefaults__]
            return ["fn", _code_key(fn.__code__), cells, repr(defaults)]
    owner = getattr(fn, "__self__", None)
    return ["obj", id(owner if owner is not None else fn), getattr(fn, "__name__", "")]


def _response_key(entry: LLMResponse) -> Any:
    fields = {
        f.name: getattr(entry, f.name)
        for f in dataclasses.fields(entry)
        if f.name not in ("tool_calls", "raw")
    }
    fields["tool_calls"] = [(c.name, c.params, c.parse_error) for c in entry.tool_calls or []]
    return fields


def _step_key(entry: Entry) -> Any:
    if isinstance(entry, _Step):
        return ["step", entry.text, entry.structured, [list(c) for c in entry.calls]]
    if isinstance(entry, _When):
        return ["when", _callable_key(entry.predicate), _step_key(entry.response)]
    return ["raw", _response_key(entry)]


def _fingerprint(slots: Mapping[str, tuple[tuple[Entry, ...], ...]]) -> str:
    keyed = {n: [[_step_key(e) for e in slot] for slot in s] for n, s in slots.items()}
    return hashlib.sha256(json.dumps(keyed, sort_keys=True, default=repr).encode()).hexdigest()


def _copy_entry(entry: Entry) -> Entry:
    """Copy the data of an entry; a predicate stays the caller's own object."""
    if isinstance(entry, _When):
        return _When(entry.predicate, _copy_entry(entry.response))  # type: ignore[arg-type]
    return copy.deepcopy(entry)


class FakeModel:
    """A script of model answers, one list per agent name, with no memory.

    The answer to a call depends only on the agent name bound in
    :meth:`provider` and the turn (the number of assistant messages in that
    call's history). A new ``FakeModel`` built from the same script, in
    another process or after a resume, answers the same way.

    The script is deep-copied when the model is built, and every response and
    call record hands out its own copies, so changing a returned value never
    changes the script. Predicates given to ``when`` stay caller-owned and must
    be pure. A scripted raw ``LLMResponse`` must be deep-copyable.
    """

    def __init__(self, script: Mapping[str, Sequence[Entry]], *, model: str = "fake") -> None:
        self._slots = {
            name: _slots([_copy_entry(entry) for entry in entries])
            for name, entries in script.items()
        }
        self._fingerprint = _fingerprint(self._slots)
        self._model = model
        self._calls: dict[str, list[FakeCall]] = {name: [] for name in self._slots}
        self._used: dict[str, set[int]] = {name: set() for name in self._slots}
        self._failures: list[tuple[str, int, str]] = []

    def provider(self, name: str, *, output_type: Any = None) -> FakeProvider:
        """The provider for agent ``name``; an unknown name raises at bind time.

        Pass the agent's ``output_type`` to check every ``structured(...)`` answer with
        the real Pydantic type, as ``SingleCall`` will. Without it, the answer is checked
        against the call's ``output_schema`` by a small checker that refuses (code
        ``unsupported_schema``) any schema keyword it cannot judge, such as ``pattern``.
        """
        if name not in self._slots:
            raise FakeModelScriptError(
                f"FakeModel has no script for agent {name!r}; known agents: {sorted(self._slots)}",
                code="unknown_agent",
            )
        return FakeProvider(self, name, None if output_type is None else TypeCheck(output_type))

    def calls(self, name: str) -> list[FakeCall]:
        """The calls agent ``name`` received through this instance, oldest first.

        Failed attempts are included. Each read returns fresh copies.
        """
        return copy.deepcopy(self._calls[name])

    def script_errors(self) -> list[tuple[str, int, str]]:
        """Every failed script attempt as ``(agent, turn, code)``, oldest first."""
        return list(self._failures)

    def assert_no_script_errors(self, *others: FakeModel) -> None:
        """Fail if any call, here or in ``others``, could not be answered.

        A fan-out child whose script fails is isolated by the runtime and the
        parent can still succeed, so assert this at the end of the test.
        """
        failures = list(self._failures)
        for other in others:
            self._require_same_script(other)
            failures += other._failures
        if failures:
            shown = [f"{agent}@{turn} ({code})" for agent, turn, code in failures]
            raise AssertionError(f"FakeModel script errors: {shown}")

    def assert_all_used(self, *others: FakeModel) -> None:
        """Fail on a script error or on a turn slot that never answered.

        After a simulated second process, pass the first instance as ``others``:
        each instance only sees the calls made through it. Every instance must be
        built from the same script, otherwise the check is refused. A slot counts
        as used when one of its alternatives answered.
        """
        self.assert_no_script_errors(*others)
        unused = []
        for name, slots in self._slots.items():
            used = set(self._used[name])
            for other in others:
                used |= other._used[name]
            unused += [f"{name}@{turn}" for turn in range(len(slots)) if turn not in used]
        if unused:
            raise AssertionError(f"FakeModel entries never used: {unused}")

    def _require_same_script(self, other: FakeModel) -> None:
        if other._fingerprint != self._fingerprint:
            raise AssertionError(
                "FakeModel instances were built from different scripts; "
                "pass only instances built from the same script"
            )

    def _answer(
        self,
        name: str,
        messages: list[Message],
        tools: Sequence[Any] | None,
        output_schema: dict[str, Any] | None,
        run_id: str | None,
        model: str,
        check: TypeCheck | None,
    ) -> LLMResponse:
        turn = sum(1 for m in messages if m.role == Role.ASSISTANT)
        self._calls[name].append(
            FakeCall(
                turn,
                tuple(copy.deepcopy(messages)),
                tuple(copy.deepcopy(tools)) if tools else None,
                copy.deepcopy(output_schema),
                run_id,
                model,
            )
        )
        try:
            response = self._respond(name, turn, messages, output_schema, model, check)
        except FakeModelScriptError as exc:
            self._failures.append((name, turn, exc.code))
            raise
        self._used[name].add(turn)
        return response

    def _respond(
        self,
        name: str,
        turn: int,
        messages: list[Message],
        output_schema: dict[str, Any] | None,
        model: str,
        check: TypeCheck | None,
    ) -> LLMResponse:
        slots = self._slots[name]
        if turn >= len(slots):
            raise FakeModelScriptError(
                f"FakeModel script for agent {name!r} has no entry for turn {turn} "
                f"({len(slots)} entries)",
                code="exhausted",
            )
        entry = self._pick(name, turn, slots[turn], messages)
        return self._build(name, turn, entry, messages, output_schema, model, check)

    @staticmethod
    def _pick(
        name: str, turn: int, slot: tuple[Entry, ...], messages: list[Message]
    ) -> _Step | LLMResponse:
        for entry in slot:
            if not isinstance(entry, _When):
                return entry
            if entry.predicate(messages):
                return entry.response
        raise FakeModelScriptError(
            f"FakeModel script for agent {name!r} at turn {turn}: no when() entry matched "
            f"({len(slot)} entries)",
            code="no_when_match",
        )

    @staticmethod
    def _build(
        name: str,
        turn: int,
        entry: _Step | LLMResponse,
        messages: list[Message],
        output_schema: dict[str, Any] | None,
        model: str,
        check: TypeCheck | None,
    ) -> LLMResponse:
        if isinstance(entry, LLMResponse):
            return dataclasses.replace(copy.deepcopy(entry), model=model)
        if entry.structured:
            _check_structured(name, turn, entry.text, output_schema, check)
        tool_calls = [
            ToolCall(
                name=tool,
                params=copy.deepcopy(args),
                id=f"fake-{name}-{turn}-{i}",
                provider_tool_call_id=f"fake-{name}-{turn}-{i}",
            )
            for i, (tool, args) in enumerate(entry.calls)
        ]
        return LLMResponse(
            text=entry.text,
            tool_calls=tool_calls or None,
            usage=_usage(messages, entry.text),
            model=model,
            stop_reason="tool_use" if tool_calls else "end_turn",
        )


def _check_structured(
    name: str,
    turn: int,
    text: str | None,
    schema: dict[str, Any] | None,
    check: TypeCheck | None,
) -> None:
    head = f"FakeModel scripted structured response for agent {name!r} at turn {turn}"
    if check is not None:
        problem = check.problem(text)
        if problem:
            raise FakeModelScriptError(
                f"{head} does not match output_type {check.label}: {problem}",
                code="schema_mismatch",
            )
        return
    if schema is None:
        return
    try:
        problem = first_mismatch(json.loads(text or ""), schema)
    except ValueError as exc:
        if isinstance(exc, SchemaUnsupportedError):
            raise FakeModelScriptError(
                f"{head} uses the schema keyword {exc.keyword!r}, which cannot be checked; "
                "bind the provider with output_type= to validate it",
                code="unsupported_schema",
            ) from None
        raise FakeModelScriptError(f"{head} is not valid JSON", code="invalid_json") from None
    if problem:
        raise FakeModelScriptError(
            f"{head} does not match the output schema: {problem}", code="schema_mismatch"
        )


class FakeProvider(LLMProvider):
    """The provider of one agent in a :class:`FakeModel` (get it from ``provider(name)``)."""

    capabilities = ProviderCapabilities(
        supports_native_tools=True,
        supports_tool_call_ids=True,
        supports_parallel_tool_calls=True,
        supports_structured_output=True,
    )

    def __init__(self, fake: FakeModel, name: str, check: TypeCheck | None = None) -> None:
        self._fake = fake
        self._name = name
        self._check = check

    @property
    def model(self) -> str:
        """The fake model name (``"fake"`` by default)."""
        return self._fake._model

    async def complete(
        self,
        messages: list[Message],
        tools: list[Any] | None = None,
        *,
        output_schema: dict[str, Any] | None = None,
        run_id: str | None = None,
        cache_key_prefix: str | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Answer from the script by (agent, turn); honours a per-call ``model=``."""
        return self._fake._answer(
            self._name,
            messages,
            tools,
            output_schema,
            run_id,
            kwargs.get("model") or self.model,
            self._check,
        )
