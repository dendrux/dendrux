"""Unit tests for run replay expansion (``dendrux.runtime.replay``)."""

from __future__ import annotations

from typing import Any

import pytest

from dendrux.context_blocks import ContextBlock
from dendrux.errors import ReplayRunError
from dendrux.llm.anthropic import AnthropicProvider
from dendrux.loops.react import ReActLoop
from dendrux.loops.single import SingleCall
from dendrux.runtime.replay import (
    PREFIX_META_KEY,
    STABLE_CONTEXT_KEY,
    WIRE_CONTENT_KEY,
    CurrentPrefix,
    LoadedRun,
    ReplayRequest,
    _hash,
    build_seed,
    expand_replay,
    history_seed,
    prefix_descriptor,
    provider_id,
    rebuild_messages,
    record_run_prefix,
    tool_hashes,
    user_trace_message,
    validate_replay_args,
)
from dendrux.runtime.state import RunRecord, TraceRecord
from dendrux.types import (
    REPLAYED_META_KEY,
    Message,
    ReplaySummary,
    Role,
    ToolCall,
    ToolDef,
    compute_idempotency_fingerprint,
)

THINK = [{"type": "thinking", "thinking": "hmm", "signature": "sig"}]
LOOKUP = ToolDef(name="lookup", description="Look up", parameters={"type": "object"})


def _current(**overrides: Any) -> CurrentPrefix:
    tools = tool_hashes([LOOKUP])
    base: dict[str, Any] = {
        "provider": "AnthropicProvider@https://api.anthropic.com",
        "model": "m1",
        "system_hash": _hash("sys"),
        "tools_hash": _hash(tools),
        "tools": tools,
    }
    base.update(overrides)
    return CurrentPrefix(**base)


def _trace(i: int, role: str, content: str, **meta: Any) -> TraceRecord:
    return TraceRecord(id=f"t{i}", role=role, content=content, order_index=i, meta=meta or None)


def _tool_run(
    run_id: str,
    *,
    descriptor: dict[str, Any] | None,
    reasoning: bool = True,
    answer: str = "done",
    status: str = "success",
    tool: str = "lookup",
) -> LoadedRun:
    rmeta: dict[str, Any] = {"phase": "commentary"}
    if reasoning:
        rmeta.update(reasoning_blocks=THINK, reasoning_model="m1")
    traces = [
        _trace(0, "user", f"q-{run_id}"),
        _trace(
            1,
            "assistant",
            "",
            tool_calls=[
                {
                    "id": f"c-{run_id}",
                    "provider_tool_call_id": f"p-{run_id}",
                    "name": tool,
                    "params": {"q": 1},
                }
            ],
            **rmeta,
        ),
        _trace(2, "tool", "42", call_id=f"c-{run_id}", tool_name=tool),
        _trace(3, "assistant", answer, phase="final_answer"),
    ]
    meta = {PREFIX_META_KEY: descriptor} if descriptor is not None else {}
    record = RunRecord(
        id=run_id,
        agent_name="a",
        status=status,
        input_data={"input": f"q-{run_id}"},
        answer=answer,
        meta=meta,
    )
    return LoadedRun(record=record, traces=traces)


def _conversation(n: int, current: CurrentPrefix, replay_last: Any = "all") -> list[LoadedRun]:
    """n runs, each created with runs=<all previous>, replay_last=<knob>, reasoning kept."""
    runs: list[LoadedRun] = []
    expansion: list[dict[str, Any]] = []
    for i in range(n):
        run_id = f"r{i}"
        seed = {"expansion": list(expansion)} if expansion else {}
        runs.append(_tool_run(run_id, descriptor=prefix_descriptor(current, seed)))
        expansion.append({"run_id": run_id, "mode": "full", "reasoning": True})
    return runs


def _req(runs: list[LoadedRun], replay_last: Any = "all", reasoning: bool = True) -> ReplayRequest:
    return ReplayRequest(
        runs=tuple(r.record.id for r in runs), replay_last=replay_last, replay_reasoning=reasoning
    )


class TestValidateReplayArgs:
    def test_empty_runs_is_no_replay(self):
        assert validate_replay_args(None, 0, True, loop=ReActLoop(), context=None) is None
        assert validate_replay_args([], "all", True, loop=ReActLoop(), context=None) is None

    @pytest.mark.parametrize("bad", [-1, True, "some", 1.5])
    def test_bad_replay_last(self, bad):
        with pytest.raises(ValueError, match="replay_last"):
            validate_replay_args(["r"], bad, True, loop=ReActLoop(), context=None)

    def test_bad_replay_reasoning(self):
        with pytest.raises(ValueError, match="replay_reasoning"):
            validate_replay_args(["r"], 0, "yes", loop=ReActLoop(), context=None)  # type: ignore[arg-type]

    def test_runs_must_be_list_of_ids(self):
        with pytest.raises(ValueError, match="list of run id"):
            validate_replay_args("r1", 0, True, loop=ReActLoop(), context=None)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="list of run id"):
            validate_replay_args(["r1", ""], 0, True, loop=ReActLoop(), context=None)

    def test_duplicate_run_rejected(self):
        with pytest.raises(ReplayRunError) as exc:
            validate_replay_args(["r1", "r1"], 0, True, loop=ReActLoop(), context=None)
        assert exc.value.reason == "duplicate"
        assert isinstance(exc.value, ValueError)

    def test_singlecall_full_replay_rejected_text_allowed(self):
        with pytest.raises(ValueError, match="SingleCall"):
            validate_replay_args(["r1"], 1, True, loop=SingleCall(), context=None)
        req = validate_replay_args(["r1"], 0, True, loop=SingleCall(), context=None)
        assert req == ReplayRequest(runs=("r1",), replay_last=0, replay_reasoning=True)

    def test_stable_context_rejected_dynamic_allowed(self):
        with pytest.raises(ValueError, match="stable"):
            validate_replay_args(
                ["r1"],
                0,
                True,
                loop=ReActLoop(),
                context=[ContextBlock("x", placement="stable")],
            )
        assert validate_replay_args(["r1"], 0, True, loop=ReActLoop(), context=[ContextBlock("x")])


class TestRebuildMessages:
    def test_user_wire_content_and_placement(self):
        msgs = rebuild_messages(
            [_trace(0, "user", "raw", wire_content="WIRE", placement="stable")],
            reasoning_for=None,
        )
        assert msgs[0].content == "WIRE"
        assert msgs[0].placement == "stable"

    def test_tool_meta_phase_and_reasoning_filter(self):
        traces = [
            _trace(0, "user", "q"),
            _trace(
                1,
                "assistant",
                "",
                tool_calls=[{"id": "c", "name": "t"}],
                phase="commentary",
                reasoning_blocks=THINK,
                reasoning_model="m1",
            ),
            _trace(2, "tool", "boom", call_id="c", tool_name="t", is_error=True),
            _trace(3, "weird", "skip"),
        ]
        kept = rebuild_messages(traces, reasoning_for="m1")
        assert [m.role for m in kept] == [Role.USER, Role.ASSISTANT, Role.TOOL]
        assert kept[1].reasoning_blocks == THINK
        assert kept[1].meta == {"phase": "commentary", "reasoning_model": "m1"}
        assert kept[1].tool_calls[0].params == {}
        assert kept[2].meta == {"is_error": True}
        assert rebuild_messages(traces, reasoning_for="other")[1].reasoning_blocks is None
        assert rebuild_messages(traces, reasoning_for=None)[1].reasoning_blocks is None


class TestExpandReplay:
    def test_all_full_keeps_reasoning(self):
        cur = _current()
        runs = _conversation(3, cur)
        plan = expand_replay(runs, _req(runs), current=cur)
        assert plan.summary.runs_full == ["r0", "r1", "r2"]
        assert plan.summary.reasoning_kept == ["r0", "r1", "r2"]
        assert plan.summary.warnings == []
        assistants = [m for m in plan.messages if m.role == Role.ASSISTANT]
        assert all(m.meta.get(REPLAYED_META_KEY) for m in assistants)
        assert sum(1 for m in assistants if m.reasoning_blocks) == 3
        assert plan.messages[-1].placement == "stable"
        assert [m.placement for m in plan.messages[:-1]] == ["dynamic"] * 11
        assert plan.seed["expansion"][-1] == {"run_id": "r2", "mode": "full", "reasoning": True}

    def test_window_text_only_turns(self):
        cur = _current()
        runs = _conversation(3, cur)
        plan = expand_replay(runs, _req(runs, replay_last=1), current=cur)
        assert plan.summary.runs_text == ["r0", "r1"]
        assert plan.summary.runs_full == ["r2"]
        assert [(m.role, m.content) for m in plan.messages[:4]] == [
            (Role.USER, "q-r0"),
            (Role.ASSISTANT, "done"),
            (Role.USER, "q-r1"),
            (Role.ASSISTANT, "done"),
        ]
        # r2 was created after full r0/r1 → its seed no longer matches.
        assert plan.summary.reasoning_dropped == ["r2"]
        assert "seed" in plan.summary.warnings[0]

    def test_replay_last_zero_all_text(self):
        cur = _current()
        runs = _conversation(2, cur)
        plan = expand_replay(runs, _req(runs, replay_last=0), current=cur)
        assert plan.summary.runs_text == ["r0", "r1"]
        assert not any(m.tool_calls for m in plan.messages)

    @pytest.mark.parametrize(
        ("override", "field"),
        [
            ({"model": "m2"}, "model"),
            ({"system_hash": _hash("other")}, "system"),
            ({"tools_hash": "x"}, "tools"),
            ({"provider": "OpenAIResponsesProvider"}, "cross-vendor"),
        ],
    )
    def test_mismatch_strips_trailing_with_named_field(self, override, field):
        created = _current()
        runs = _conversation(2, created)
        plan = expand_replay(runs, _req(runs), current=_current(**override))
        assert plan.summary.reasoning_dropped == ["r0", "r1"]
        assert plan.summary.reasoning_kept == []
        assert len(plan.summary.warnings) == 2
        assert field in plan.summary.warnings[0]
        assert not any(m.reasoning_blocks for m in plan.messages)

    def test_predates_metadata_then_trailing(self):
        cur = _current()
        runs = [_tool_run("old", descriptor=None), *_conversation(1, cur)]
        # The second run was created with "old" full + reasoning ahead of it.
        runs[1].record.meta[PREFIX_META_KEY]["seed"] = {
            "expansion": [{"run_id": "old", "mode": "full", "reasoning": True}]
        }
        plan = expand_replay(runs, _req(runs), current=cur)
        assert plan.summary.reasoning_dropped == ["old", "r0"]
        assert "predates replay metadata" in plan.summary.warnings[0]
        assert "seed" in plan.summary.warnings[1]

    def test_follows_dropped_run_when_own_fields_match(self):
        cur = _current()
        runs = [_tool_run("old", descriptor=None), _tool_run("new", descriptor=None)]
        runs[1] = _tool_run(
            "new",
            descriptor=prefix_descriptor(
                cur, {"expansion": [{"run_id": "old", "mode": "full", "reasoning": False}]}
            ),
        )
        plan = expand_replay(runs, _req(runs), current=cur)
        assert plan.summary.reasoning_dropped == ["old", "new"]
        assert "follows dropped run old" in plan.summary.warnings[1]

    def test_runs_without_reasoning_neither_kept_nor_dropped(self):
        cur = _current()
        runs = [_tool_run("r0", descriptor=None, reasoning=False)]
        plan = expand_replay(runs, _req(runs), current=cur)
        assert plan.summary.reasoning_kept == plan.summary.reasoning_dropped == []
        assert plan.summary.warnings == []

    def test_replay_reasoning_false_no_warning(self):
        cur = _current()
        runs = _conversation(2, cur)
        plan = expand_replay(runs, _req(runs, reasoning=False), current=cur)
        assert plan.summary.reasoning_kept == plan.summary.reasoning_dropped == []
        assert not any(m.reasoning_blocks for m in plan.messages)

    def test_guardrails_force_reasoning_off(self):
        cur = _current()
        runs = _conversation(1, cur)
        plan = expand_replay(runs, _req(runs), current=cur, guardrails=True)
        assert "forced off" in plan.summary.warnings[0]
        assert plan.meta["reasoning_effective"] is False
        assert not any(m.reasoning_blocks for m in plan.messages)

    def test_tool_absent_and_changed_warnings(self):
        cur = _current()
        runs = _conversation(1, cur)
        runs.append(_tool_run("gone", descriptor=None, reasoning=False, tool="vanished"))
        changed = _current(tools={"lookup": "different"})
        plan = expand_replay(runs, _req(runs), current=changed)
        text = " | ".join(plan.summary.warnings)
        assert "lookup have a changed schema" in text
        assert "vanished not in the current tool set" in text
        # Tool calls replay verbatim regardless.
        assert any(m.tool_calls and m.tool_calls[0].name == "vanished" for m in plan.messages)

    def test_trailing_unanswered_tool_call_dropped(self):
        cur = _current()
        run = _tool_run("r0", descriptor=None, reasoning=False, status="max_iterations")
        run.traces.append(
            _trace(4, "assistant", "", tool_calls=[{"id": "dangling", "name": "lookup"}])
        )
        run.traces.append(
            _trace(5, "assistant", "partial", tool_calls=[{"id": "d2", "name": "lookup"}])
        )
        run.traces.append(_trace(6, "tool", "orphan", call_id="nope", tool_name="lookup"))
        plan = expand_replay([run], _req([run]), current=cur)
        assert plan.messages[-1].content == "partial"
        assert plan.messages[-1].tool_calls is None
        assert not any(m.role == Role.TOOL and m.call_id == "nope" for m in plan.messages)

    def test_full_turn_without_user_trace_gets_input(self):
        cur = _current()
        run = _tool_run("r0", descriptor=None, reasoning=False)
        run.traces.pop(0)
        plan = expand_replay([run], _req([run]), current=cur)
        assert plan.messages[0] == Message(role=Role.USER, content="q-r0")

    def test_text_turn_keeps_stable_band_and_skips_missing_answer(self):
        cur = _current()
        run = _tool_run("r0", descriptor=None, answer="")
        run.traces[0] = _trace(
            0,
            "user",
            "q-r0",
            wire_content="WIRE",
            placement="stable",
            **{STABLE_CONTEXT_KEY: "PROJECT RULES"},
        )
        plan = expand_replay([run], _req([run], replay_last=0), current=cur)
        assert len(plan.messages) == 1
        assert plan.messages[0].content.startswith("STABLE CONTEXT:\nPROJECT RULES")
        assert plan.messages[0].content.endswith("USER MESSAGE:\nq-r0")
        assert plan.messages[0].placement == "stable"


class TestSeedsAndTraces:
    def test_history_seed(self):
        assert history_seed([]) == {}
        assert "history_hash" in history_seed([Message(role=Role.USER, content="hi")])

    def test_build_seed_combines(self):
        cur = _current()
        runs = _conversation(1, cur)
        plan = expand_replay(runs, _req(runs), current=cur)
        seed = build_seed(plan, [Message(role=Role.USER, content="hi")])
        assert seed["runs"] == ["r0"] and "history_hash" in seed
        assert build_seed(None, []) == {}

    def test_user_trace_message(self):
        plain = user_trace_message("hi", [Message(role=Role.USER, content="hi")], "")
        assert plain.meta == {}
        folded = user_trace_message(
            "hi", [Message(role=Role.USER, content="STABLE..hi", placement="stable")], "S"
        )
        assert folded.meta == {
            WIRE_CONTENT_KEY: "STABLE..hi",
            "placement": "stable",
            STABLE_CONTEXT_KEY: "S",
        }
        assert folded.content == "hi"

    def test_provider_id_includes_base_url(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
        assert provider_id(AnthropicProvider(model="m")).startswith("AnthropicProvider@https://")

    async def test_record_run_prefix_best_effort(self):
        class Boom:
            async def update_run_meta(self, run_id: str, patch: dict[str, Any]) -> None:
                raise RuntimeError("db down")

        await record_run_prefix(Boom(), "r", current=_current(), seed={}, plan=None)  # type: ignore[arg-type]
        await record_run_prefix(object(), "r", current=_current(), seed={}, plan=None)  # type: ignore[arg-type]
        await record_run_prefix(None, "r", current=_current(), seed={}, plan=None)


class TestPublicShapes:
    def test_replay_summary_roundtrip(self):
        s = ReplaySummary(runs_full=["a"], warnings=["w"])
        assert ReplaySummary.from_dict(s.to_dict()) == s
        assert ReplaySummary.from_dict({}) == ReplaySummary()

    def test_fingerprint_includes_replay_only_when_given(self):
        base = compute_idempotency_fingerprint("a", "hi")
        assert compute_idempotency_fingerprint("a", "hi", replay=None) == base
        one = compute_idempotency_fingerprint(
            "a", "hi", replay={"runs": ["r1"], "replay_last": 0, "replay_reasoning": True}
        )
        two = compute_idempotency_fingerprint(
            "a", "hi", replay={"runs": ["r1"], "replay_last": "all", "replay_reasoning": True}
        )
        assert len({base, one, two}) == 3


class TestAnthropicReplayWire:
    @pytest.fixture
    def provider(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
        return AnthropicProvider(model="m1", thinking=True)

    def _messages(self, replayed: bool) -> list[Message]:
        meta = {"reasoning_model": "m1", "phase": "final_answer"}
        if replayed:
            meta[REPLAYED_META_KEY] = True
        return [
            Message(role=Role.USER, content="q1"),
            Message(
                role=Role.ASSISTANT,
                content="a1",
                meta=meta,
                reasoning_blocks=THINK,
                placement="stable",
            ),
            Message(role=Role.USER, content="q2"),
        ]

    def test_last_replayed_message_gets_breakpoint(self, provider):
        api_kwargs, _ = provider._build_api_kwargs(self._messages(True), None, {})
        msgs = api_kwargs["messages"]
        assert msgs[1]["content"][0]["type"] == "thinking"
        assert "cache_control" not in msgs[1]["content"][0]
        assert "cache_control" in msgs[1]["content"][1]
        assert "cache_control" in msgs[2]["content"][0]
        assert "cache_control" not in str(msgs[0])

    def test_binding_controls_only_with_replayed_reasoning(self, provider):
        api_kwargs, _ = provider._build_api_kwargs(self._messages(True), None, {})
        assert api_kwargs["thinking"]["block_binding"] == {"prefix_mismatch_behavior": "drop_block"}
        assert api_kwargs["extra_headers"]["anthropic-beta"].startswith("thinking-binding")
        plain, _ = provider._build_api_kwargs(self._messages(False), None, {})
        assert "block_binding" not in plain["thinking"]
        assert "extra_headers" not in plain

    def test_tool_result_stable_index_maps_to_merged_message(self, provider):
        msgs = [
            Message(role=Role.USER, content="q"),
            Message(
                role=Role.ASSISTANT,
                content="",
                tool_calls=[ToolCall(name="t", id="c", provider_tool_call_id="p")],
            ),
            Message(role=Role.TOOL, content="r", name="t", call_id="c", placement="stable"),
            Message(role=Role.USER, content="next"),
        ]
        indices: list[int] = []
        _, api = provider._convert_messages(msgs, stable_indices=indices)
        assert indices == [2]
        assert api[2]["content"][0]["type"] == "tool_result"
