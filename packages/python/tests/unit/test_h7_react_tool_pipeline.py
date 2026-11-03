"""Pin the record, notify and governance order of ``react._process_tool_calls``.

Captured from the code before the H7 stage split. Each entry is
``(source, kind, key, len(history))``: ``source`` is ``rec`` (recorder) or
``not`` (notifier), ``key`` is the call id or the governance event, and the
history length shows whether the result message was appended before or
after the hook ran. The stages keep their order: invalid JSON, static deny,
unknown tool, restore, validate, call limits, skill preflight, hook, approval,
dispatch, record.
"""

from __future__ import annotations

from typing import Any

from dendrux import Agent, Skill, tool
from dendrux.guardrails import PII
from dendrux.guardrails._engine import GuardrailEngine
from dendrux.loops.base import BaseNotifier, BaseRecorder
from dendrux.loops.react import ReActLoop, _process_tool_calls
from dendrux.strategies.native import NativeToolCalling
from dendrux.types import AgentStep, Message, ToolCall, ToolDecision

Entry = tuple[str, str, str | None, int]


class _Log:
    def __init__(self) -> None:
        self.entries: list[Entry] = []
        self.history: list[Message] = []

    def add(self, source: str, kind: str, key: str | None) -> None:
        self.entries.append((source, kind, key, len(self.history)))


def _spies(log: _Log) -> tuple[BaseRecorder, BaseNotifier]:
    class Rec(BaseRecorder):
        async def on_message_appended(self, run_id, message, iteration):
            log.add("rec", "msg", message.call_id)

        async def on_tool_started(self, run_id, tool_call, iteration):
            log.add("rec", "tool_started", tool_call.id)

        async def on_tool_completed(self, run_id, tool_call, tool_result, iteration):
            log.add("rec", "tool", tool_call.id)

        async def on_governance_event(
            self, run_id, event_type, iteration, data, correlation_id=None
        ):
            log.add("rec", "gov", f"{event_type}:{correlation_id}")

    class Not(BaseNotifier):
        async def on_message_appended(self, run_id, message, iteration):
            log.add("not", "msg", message.call_id)

        async def on_tool_started(self, run_id, tool_call, iteration):
            log.add("not", "tool_started", tool_call.id)

        async def on_tool_completed(self, run_id, tool_call, tool_result, iteration):
            log.add("not", "tool", tool_call.id)

        async def on_governance_event(
            self, run_id, event_type, iteration, data, correlation_id=None
        ):
            log.add("not", "gov", f"{event_type}:{correlation_id}")

    return Rec(), Not()


@tool(parallel=False)
async def work() -> str:
    return "ok"


@tool(parallel=False)
async def nope() -> str:
    return "never"


@tool(parallel=False)
async def typed(n: int) -> str:
    return str(n)


@tool(parallel=False, max_calls_per_run=1)
async def lim() -> str:
    return "lim"


@tool(parallel=False)
async def hdeny() -> str:
    return "never"


@tool(parallel=False)
async def herr() -> str:
    return "never"


@tool(parallel=False)
async def hsub() -> str:
    return "never"


@tool(parallel=False)
async def gated() -> str:
    return "never"


@tool(parallel=False)
async def hgated() -> str:
    return "never"


@tool(parallel=False)
async def send(to: str) -> str:
    return to


def _hook(ctx: Any) -> ToolDecision | None:
    if ctx.tool_name == "hdeny":
        return ToolDecision.deny("no")
    if ctx.tool_name == "herr":
        raise RuntimeError("boom")
    if ctx.tool_name == "hsub":
        return ToolDecision.substitute({"v": 1})
    if ctx.tool_name == "hgated":
        return ToolDecision.require_approval("check")
    return None


async def _process(agent: Agent, calls: list[ToolCall], log: _Log, **kwargs: Any) -> Any:
    recorder, notifier = _spies(log)
    return await _process_tool_calls(
        step=AgentStep(reasoning=None, action=calls[0], meta={"all_tool_calls": calls}),
        call_counts={},
        lookups=await agent.get_tool_lookups(loop=ReActLoop()),
        strategy=NativeToolCalling(),
        recorder=recorder,
        notifier=notifier,
        run_id="run",
        iteration=1,
        history=log.history,
        warnings=[],
        deny=agent.deny,
        require_approval=agent.require_approval,
        before_tool=getattr(agent, "before_tool", None),
        tenant_id=None,
        **kwargs,
    )


async def test_every_stage_records_in_pinned_order() -> None:
    agent = Agent(
        prompt="p",
        tools=[work, nope, typed, lim, hdeny, herr, hsub],
        deny=["nope"],
        before_tool=_hook,
        skills=[Skill(name="demo", description="d", body="b", files={"references/n.md": "x"})],
    )
    calls = [
        ToolCall("work", id="j", parse_error="bad"),
        ToolCall("use_skill", {"name": "missing"}, id="sj"),
        ToolCall("nope", id="d"),
        ToolCall("ghost", id="u"),
        ToolCall("typed", {"n": "abc"}, id="v"),
        ToolCall("lim", id="l1"),
        ToolCall("lim", id="l2"),
        ToolCall("read_skill_resource", {"skill": "demo", "path": "references/n.md"}, id="s"),
        ToolCall("hdeny", id="hd"),
        ToolCall("herr", id="he"),
        ToolCall("hsub", id="hs"),
        ToolCall("work", id="w1"),
        ToolCall("work", id="w2"),
    ]
    log = _Log()
    outcome = await _process(agent, calls, log)
    assert [tc.id for tc, _ in outcome.all_results] == [
        "j", "sj", "d", "u", "v", "l2", "s", "hd", "he", "hs", "l1", "w1", "w2",
    ]  # fmt: skip
    assert outcome.pending_calls == [] and outcome.pending_approval is False
    assert [m.call_id for m in log.history] == [tc.id for tc, _ in outcome.all_results]
    assert log.entries == [
        # S1 invalid JSON: tool row, then message.
        ("rec", "tool", "j", 0),
        ("not", "tool", "j", 0),
        ("rec", "msg", "j", 1),
        ("not", "msg", "j", 1),
        # S1 invalid skill activation: preflight governance, tool row, message.
        ("rec", "gov", "skill.activation_failed:sj", 1),
        ("not", "gov", "skill.activation_failed:sj", 1),
        ("rec", "tool", "sj", 1),
        ("not", "tool", "sj", 1),
        ("rec", "msg", "sj", 2),
        ("not", "msg", "sj", 2),
        # S2 static deny: governance, then message; no tool row.
        ("rec", "gov", "policy.denied:d", 2),
        ("not", "gov", "policy.denied:d", 2),
        ("rec", "msg", "d", 3),
        ("not", "msg", "d", 3),
        # S3 unknown tool, S5 invalid params, S6 call limit: reject().
        ("rec", "tool", "u", 3),
        ("not", "tool", "u", 3),
        ("rec", "msg", "u", 4),
        ("not", "msg", "u", 4),
        ("rec", "tool", "v", 4),
        ("not", "tool", "v", 4),
        ("rec", "msg", "v", 5),
        ("not", "msg", "v", 5),
        ("rec", "tool", "l2", 5),
        ("not", "tool", "l2", 5),
        ("rec", "msg", "l2", 6),
        ("not", "msg", "l2", 6),
        # S6b skill preflight: the message is appended to history before the
        # tool row is written (unlike reject(), drift D-f).
        ("rec", "tool", "s", 7),
        ("not", "tool", "s", 7),
        ("rec", "msg", "s", 7),
        ("not", "msg", "s", 7),
        # S7 hook deny: governance, message; no tool row.
        ("rec", "gov", "policy.denied:hd", 7),
        ("not", "gov", "policy.denied:hd", 7),
        ("rec", "msg", "hd", 8),
        ("not", "msg", "hd", 8),
        # S7 hook error: hook_error, denied, message.
        ("rec", "gov", "policy.hook_error:he", 8),
        ("not", "gov", "policy.hook_error:he", 8),
        ("rec", "gov", "policy.denied:he", 8),
        ("not", "gov", "policy.denied:he", 8),
        ("rec", "msg", "he", 9),
        ("not", "msg", "he", 9),
        # S7 substitute: governance, tool row, message.
        ("rec", "gov", "policy.substituted:hs", 9),
        ("not", "gov", "policy.substituted:hs", 9),
        ("rec", "tool", "hs", 9),
        ("not", "tool", "hs", 9),
        ("rec", "msg", "hs", 10),
        ("not", "msg", "hs", 10),
        # S10 dispatch: every tool row before any S11 message.
        ("rec", "tool_started", "l1", 10),
        ("not", "tool_started", "l1", 10),
        ("rec", "tool", "l1", 10),
        ("not", "tool", "l1", 10),
        ("rec", "tool_started", "w1", 10),
        ("not", "tool_started", "w1", 10),
        ("rec", "tool", "w1", 10),
        ("not", "tool", "w1", 10),
        ("rec", "tool_started", "w2", 10),
        ("not", "tool_started", "w2", 10),
        ("rec", "tool", "w2", 10),
        ("not", "tool", "w2", 10),
        # S11 record: messages in dispatch order.
        ("rec", "msg", "l1", 11),
        ("not", "msg", "l1", 11),
        ("rec", "msg", "w1", 12),
        ("not", "msg", "w1", 12),
        ("rec", "msg", "w2", 13),
        ("not", "msg", "w2", 13),
    ]


async def test_approval_pause_records_requests_and_returns_early() -> None:
    agent = Agent(
        prompt="p",
        tools=[work, nope, gated, hgated],
        deny=["nope"],
        require_approval=["gated"],
        before_tool=_hook,
    )
    calls = [
        ToolCall("nope", id="d"),
        ToolCall("gated", id="g"),
        ToolCall("hgated", id="hg"),
        ToolCall("work", id="w"),
    ]
    log = _Log()
    outcome = await _process(agent, calls, log)
    assert outcome.pending_approval is True
    assert [tc.id for tc in outcome.pending_calls] == ["g", "hg", "w"]
    assert outcome.approval_gates == {
        "g": {"source": "static", "reason": "requires_approval"},
        "hg": {"source": "hook", "reason": "check"},
    }
    assert [tc.id for tc, _ in outcome.all_results] == ["d"]
    assert log.entries == [
        ("rec", "gov", "policy.denied:d", 0),
        ("not", "gov", "policy.denied:d", 0),
        ("rec", "msg", "d", 1),
        ("not", "msg", "d", 1),
        ("rec", "gov", "approval.requested:g", 1),
        ("not", "gov", "approval.requested:g", 1),
        ("rec", "gov", "approval.requested:hg", 1),
        ("not", "gov", "approval.requested:hg", 1),
    ]


async def test_restore_skipped_precedes_validation_and_dispatch() -> None:
    engine = GuardrailEngine(
        [PII(restore="allowlisted")], pii_mapping={"<<EMAIL_ADDRESS_1>>": "a@example.com"}
    )
    agent = Agent(prompt="p", tools=[send, typed])
    calls = [
        ToolCall("typed", {"n": "<<EMAIL_ADDRESS_1>>"}, id="v"),
        ToolCall("send", {"to": "<<EMAIL_ADDRESS_1>>"}, id="s"),
    ]
    log = _Log()
    outcome = await _process(agent, calls, log, guardrail_engine=engine)
    assert [tc.id for tc, _ in outcome.all_results] == ["v", "s"]
    assert outcome.all_results[1][1].payload == '"<<EMAIL_ADDRESS_1>>"'
    assert log.entries == [
        ("rec", "gov", "guardrail.restore_skipped:v", 0),
        ("not", "gov", "guardrail.restore_skipped:v", 0),
        ("rec", "gov", "guardrail.restore_skipped:s", 0),
        ("not", "gov", "guardrail.restore_skipped:s", 0),
        ("rec", "tool", "v", 0),
        ("not", "tool", "v", 0),
        ("rec", "msg", "v", 1),
        ("not", "msg", "v", 1),
        ("rec", "tool_started", "s", 1),
        ("not", "tool_started", "s", 1),
        ("rec", "tool", "s", 1),
        ("not", "tool", "s", 1),
        ("rec", "msg", "s", 2),
        ("not", "msg", "s", 2),
    ]
