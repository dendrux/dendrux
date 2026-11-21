"""FakeModel: scripted, stateless model answers for tests that need no API key."""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from dendrux.llm.base import LLMProvider
from dendrux.testing._schema_check import first_mismatch
from dendrux.types import LLMResponse, Message, ProviderCapabilities, Role, ToolCall, UsageStats

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence


class FakeModelScriptError(LookupError):
    """The script has no usable answer for a call.

    The message names the agent, the turn, counts and known agent names only:
    never message text, tool arguments or payloads.
    """


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
    """JSON text for an agent with ``output_type``; checked against its schema when used."""
    data = value.model_dump(mode="json") if isinstance(value, BaseModel) else dict(value)
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


class FakeModel:
    """A script of model answers, one list per agent name, with no memory.

    The answer to a call depends only on the agent name bound in
    :meth:`provider` and the turn (the number of assistant messages in that
    call's history). A new ``FakeModel`` built from the same script, in
    another process or after a resume, answers the same way.
    """

    def __init__(self, script: Mapping[str, Sequence[Entry]], *, model: str = "fake") -> None:
        self._slots = {name: _slots(entries) for name, entries in script.items()}
        self._model = model
        self._calls: dict[str, list[FakeCall]] = {name: [] for name in self._slots}
        self._used: dict[str, set[int]] = {name: set() for name in self._slots}

    def provider(self, name: str) -> FakeProvider:
        """The provider for agent ``name``; an unknown name raises at bind time."""
        if name not in self._slots:
            raise FakeModelScriptError(
                f"FakeModel has no script for agent {name!r}; known agents: {sorted(self._slots)}"
            )
        return FakeProvider(self, name)

    def calls(self, name: str) -> list[FakeCall]:
        """The calls agent ``name`` made through this instance, oldest first."""
        return list(self._calls[name])

    def assert_all_used(self, *others: FakeModel) -> None:
        """Fail if a script entry was never used, here or in ``others``.

        After a simulated second process, pass the first instance as ``others``:
        each instance only sees the calls made through it.
        """
        unused = []
        for name, slots in self._slots.items():
            used = set(self._used[name])
            for other in others:
                used |= other._used.get(name, set())
            unused += [f"{name}@{turn}" for turn in range(len(slots)) if turn not in used]
        if unused:
            raise AssertionError(f"FakeModel entries never used: {unused}")

    def _answer(
        self,
        name: str,
        messages: list[Message],
        tools: Sequence[Any] | None,
        output_schema: dict[str, Any] | None,
        run_id: str | None,
        model: str,
    ) -> LLMResponse:
        turn = sum(1 for m in messages if m.role == Role.ASSISTANT)
        slots = self._slots[name]
        if turn >= len(slots):
            raise FakeModelScriptError(
                f"FakeModel script for agent {name!r} has no entry for turn {turn} "
                f"({len(slots)} entries)"
            )
        entry = self._pick(name, turn, slots[turn], messages)
        self._calls[name].append(
            FakeCall(
                turn, tuple(messages), tuple(tools) if tools else None, output_schema, run_id, model
            )
        )
        self._used[name].add(turn)
        return self._build(name, turn, entry, messages, output_schema, model)

    @staticmethod
    def _pick(name: str, turn: int, slot: tuple[Entry, ...], messages: list[Message]) -> Any:
        for entry in slot:
            if not isinstance(entry, _When):
                return entry
            if entry.predicate(messages):
                return entry.response
        raise FakeModelScriptError(
            f"FakeModel script for agent {name!r} at turn {turn}: no when() entry matched "
            f"({len(slot)} entries)"
        )

    @staticmethod
    def _build(
        name: str,
        turn: int,
        entry: _Step | LLMResponse,
        messages: list[Message],
        output_schema: dict[str, Any] | None,
        model: str,
    ) -> LLMResponse:
        if isinstance(entry, LLMResponse):
            return dataclasses.replace(entry, model=model)
        if entry.structured and output_schema is not None:
            _check_structured(name, turn, entry.text, output_schema)
        tool_calls = [
            ToolCall(
                name=tool,
                params=dict(args),
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


def _check_structured(name: str, turn: int, text: str | None, schema: dict[str, Any]) -> None:
    head = f"FakeModel scripted structured response for agent {name!r} at turn {turn}"
    try:
        value = json.loads(text or "")
    except ValueError as exc:
        raise FakeModelScriptError(f"{head} is not valid JSON") from exc
    problem = first_mismatch(value, schema)
    if problem:
        raise FakeModelScriptError(f"{head} does not match the output schema: {problem}")


class FakeProvider(LLMProvider):
    """The provider of one agent in a :class:`FakeModel` (get it from ``provider(name)``)."""

    capabilities = ProviderCapabilities(
        supports_native_tools=True,
        supports_tool_call_ids=True,
        supports_parallel_tool_calls=True,
        supports_structured_output=True,
    )

    def __init__(self, fake: FakeModel, name: str) -> None:
        self._fake = fake
        self._name = name

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
            self._name, messages, tools, output_schema, run_id, kwargs.get("model") or self.model
        )
