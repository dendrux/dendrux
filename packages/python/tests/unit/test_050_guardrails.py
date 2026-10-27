"""0.5.0 T3: guardrail hardening (scan limits, detector quality, safe errors, PII restore)."""

from __future__ import annotations

import asyncio
import inspect
import logging
from typing import Any, Literal

import pytest

from dendrux.agent import Agent
from dendrux.guardrails import PII, GuardrailEngine, SecretDetection
from dendrux.guardrails import _engine as engine_mod
from dendrux.guardrails._engine import deanonymize_text, has_placeholder
from dendrux.guardrails._protocol import UNSCANNABLE, Finding
from dendrux.llm.mock import MockLLM
from dendrux.loops.base import BaseRecorder
from dendrux.loops.react import ReActLoop
from dendrux.strategies.native import NativeToolCalling
from dendrux.tool import get_tool_def, tool
from dendrux.types import LLMResponse, Message, Role, RunStatus, ToolCall, UsageStats

_EMAIL = "alice@example.com"
_AWS = "AKIAIOSFODNN7EXAMPLE"


def _nested(value: object, depth: int) -> dict[str, Any]:
    out: Any = value
    for _ in range(depth):
        out = {"a": out}
    return {"x": out}


# ---------------------------------------------------------------------------
# D-030: no matched text in block errors
# ---------------------------------------------------------------------------


class TestBlockMessagesCarryNoMatchedText:
    async def test_names_guardrail_kind_count(self) -> None:
        engine = GuardrailEngine([PII(action="block")])
        _, findings, block = await engine.scan_incoming("a@example.com and b@example.com")
        assert block is not None
        assert "PII" in block
        assert "2 EMAIL_ADDRESS" in block
        assert "example.com" not in block
        assert all("example.com" not in repr(f) for f in findings)

    async def test_incoming_message_location(self) -> None:
        engine = GuardrailEngine([SecretDetection()])
        msg = Message(Role.TOOL, f"key {_AWS}", name="lookup", call_id="call_1")
        _, _, block = await engine.scan_incoming_message(msg, index=4)
        assert block is not None
        assert "message 4" in block
        assert "role=tool" in block
        assert "tool_call_id=call_1" in block
        assert _AWS not in block

    async def test_location_without_index(self) -> None:
        engine = GuardrailEngine([SecretDetection()])
        _, _, block = await engine.scan_incoming_message(Message(Role.USER, f"key {_AWS}"))
        assert block is not None
        assert "role=user" in block
        assert _AWS not in block

    async def test_outgoing_params_location(self) -> None:
        engine = GuardrailEngine([SecretDetection()])
        _, block = await engine.scan_outgoing("ok", [{"a": "fine"}, {"api_key": _AWS}])
        assert block is not None
        assert "tool call params (call 1)" in block
        assert _AWS not in block

    async def test_finding_repr_hides_text(self) -> None:
        finding = Finding(entity_type="X", start=0, end=3, score=1.0, text="secret-value")
        assert "secret-value" not in repr(finding)
        assert finding.text == "secret-value"

    async def test_run_error_and_events_never_carry_the_secret(self) -> None:
        events: list[dict[str, Any]] = []

        class Spy(BaseRecorder):
            async def on_message_appended(self, run_id, message, iteration):
                pass

            async def on_llm_call_completed(self, run_id, response, iteration, **kw):
                pass

            async def on_tool_completed(self, run_id, tool_call, tool_result, iteration):
                pass

            async def on_governance_event(
                self, run_id, event_type, iteration, data, correlation_id=None
            ):
                events.append({"event_type": event_type, "data": data})

        agent = Agent(prompt="p", tools=[], guardrails=[SecretDetection(action="block")])
        secret = "api_key=sk_live_" + "ABCDEFGHIJKLMNOPQRSTUVWX1234"
        result = await ReActLoop().run(
            agent=agent,
            provider=MockLLM([]),
            strategy=NativeToolCalling(),
            user_input=f"config: {secret}",
            recorder=Spy(),
        )
        assert result.status == RunStatus.ERROR
        assert "sk_live" not in (result.error or "")
        blocked = [e for e in events if e["event_type"] == "guardrail.blocked"]
        assert len(blocked) == 1
        assert "sk_live" not in str(events)


# ---------------------------------------------------------------------------
# D-027: scan limits fail closed
# ---------------------------------------------------------------------------


class TestUnscannableFailsClosed:
    @pytest.mark.parametrize("guardrail_action", ["block", "redact"])
    async def test_deep_params_block_under_block_and_redact(self, guardrail_action: str) -> None:
        engine = GuardrailEngine([PII(action=guardrail_action)])  # type: ignore[arg-type]
        findings, block = await engine.scan_outgoing("ok", [_nested(_EMAIL, 60)])
        assert block is not None
        assert UNSCANNABLE in block
        assert _EMAIL not in block
        assert any(f.entity_type == UNSCANNABLE for f in findings)

    async def test_deep_params_only_reported_under_warn(self) -> None:
        engine = GuardrailEngine([PII(action="warn")])
        findings, block = await engine.scan_outgoing("ok", [_nested(_EMAIL, 60)])
        assert block is None
        assert [f.entity_type for f in findings] == [UNSCANNABLE]

    async def test_depth_limit_boundary_is_still_scanned(self) -> None:
        engine = GuardrailEngine([PII(action="block")])
        _, block = await engine.scan_outgoing("ok", [_nested(_EMAIL, 40)])
        assert block is not None
        assert "EMAIL_ADDRESS" in block
        assert UNSCANNABLE not in block

    async def test_real_finding_wins_over_unscannable(self) -> None:
        engine = GuardrailEngine([PII(action="block")])
        params = {"shallow": _EMAIL, "deep": _nested("x", 60)["x"]}
        _, block = await engine.scan_outgoing("ok", [params])
        assert block is not None
        assert "EMAIL_ADDRESS" in block

    async def test_too_many_nodes_is_unscannable(self) -> None:
        engine = GuardrailEngine([SecretDetection(action="block")])
        params = {"items": ["x"] * (engine_mod._MAX_SCAN_NODES + 5)}
        findings, block = await engine.scan_outgoing("ok", [params])
        assert block is not None
        assert UNSCANNABLE in block

    async def test_oversize_leaf_is_unscannable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(engine_mod, "MAX_SCAN_CHARS", 100)
        engine = GuardrailEngine([PII(action="block")])
        _, block = await engine.scan_outgoing("ok", [{"blob": "x" * 500}])
        assert block is not None
        assert UNSCANNABLE in block

    @pytest.mark.parametrize("action", ["block", "redact"])
    async def test_oversize_text_blocks_incoming(
        self, monkeypatch: pytest.MonkeyPatch, action: str
    ) -> None:
        monkeypatch.setattr(engine_mod, "MAX_SCAN_CHARS", 100)
        engine = GuardrailEngine([PII(action=action)])  # type: ignore[arg-type]
        text, findings, block = await engine.scan_incoming("x" * 500)
        assert block is not None
        assert UNSCANNABLE in block
        assert text == "x" * 500

    async def test_oversize_text_warn_reports_and_continues(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(engine_mod, "MAX_SCAN_CHARS", 100)
        engine = GuardrailEngine([PII(action="warn")])
        _, findings, block = await engine.scan_incoming("x" * 500)
        assert block is None
        assert [f.entity_type for f in findings] == [UNSCANNABLE]

    async def test_oversize_text_blocks_outgoing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(engine_mod, "MAX_SCAN_CHARS", 100)
        engine = GuardrailEngine([SecretDetection(action="block")])
        _, block = await engine.scan_outgoing("x" * 500, None)
        assert block is not None
        assert UNSCANNABLE in block

    async def test_unscannable_ends_run_with_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(engine_mod, "MAX_SCAN_CHARS", 100)
        agent = Agent(prompt="p", tools=[], guardrails=[PII(action="redact")])
        llm = MockLLM([])
        result = await ReActLoop().run(
            agent=agent, provider=llm, strategy=NativeToolCalling(), user_input="x" * 500
        )
        assert result.status == RunStatus.ERROR
        assert llm.calls_made == 0


# ---------------------------------------------------------------------------
# D-029: PII restore
# ---------------------------------------------------------------------------


@tool()
async def plain_tool(to: str) -> str:
    """No PII access."""
    return to


@tool(receives_pii=True)
async def pii_tool(to: str) -> str:
    """Receives any PII."""
    return to


@tool(receives_pii={"EMAIL"})
async def email_tool(to: str) -> str:
    """Receives e-mail only."""
    return to


def _engine(restore: str | None) -> GuardrailEngine:
    if restore is None:
        return GuardrailEngine([PII()])
    return GuardrailEngine([PII(restore=restore)])  # type: ignore[arg-type]


async def _mint(engine: GuardrailEngine) -> None:
    await engine.scan_incoming(f"mail {_EMAIL} phone +1-555-123-4567")


_PARAMS = {"to": "<<EMAIL_ADDRESS_1>>", "note": "call <<PHONE_NUMBER_1>> or <<EMAIL_ADDRESS_1>>"}


class TestRestoreAll:
    @pytest.mark.parametrize("restore", [None, "all"])
    async def test_every_tool_gets_real_values_like_041(self, restore: str | None) -> None:
        engine = _engine(restore)
        await _mint(engine)
        for t in (plain_tool, pii_tool, None):
            td = get_tool_def(t) if t else None
            out = engine.restore_params(_PARAMS, tool_def=td)
            assert out.params == {
                "to": _EMAIL,
                "note": f"call +1-555-123-4567 or {_EMAIL}",
            }
            assert out.skipped == {}
        assert engine.deanonymize(_PARAMS) == engine.deanonymize(_PARAMS, tool_def=None)
        assert engine.restore_mode == "all"

    async def test_mapping_is_exactly_the_041_shape(self) -> None:
        engine = _engine("all")
        await _mint(engine)
        assert engine.get_pii_mapping() == {
            "<<EMAIL_ADDRESS_1>>": _EMAIL,
            "<<PHONE_NUMBER_1>>": "+1-555-123-4567",
        }


class TestRestoreAllowlisted:
    async def test_plain_tool_keeps_placeholders_and_reports_counts(self) -> None:
        engine = _engine("allowlisted")
        await _mint(engine)
        out = engine.restore_params(_PARAMS, tool_def=get_tool_def(plain_tool))
        assert out.params == _PARAMS
        assert out.skipped == {"EMAIL_ADDRESS": 2, "PHONE_NUMBER": 1}
        data = out.event_data("plain_tool", "c1")
        assert data == {
            "tool_name": "plain_tool",
            "call_id": "c1",
            "entities": {"EMAIL_ADDRESS": 2, "PHONE_NUMBER": 1},
            "count": 3,
            "code": "pii_restore_skipped",
        }
        assert _EMAIL not in str(data)

    async def test_receives_pii_true_restores_everything(self) -> None:
        engine = _engine("allowlisted")
        await _mint(engine)
        out = engine.restore_params(_PARAMS, tool_def=get_tool_def(pii_tool))
        assert out.params["to"] == _EMAIL
        assert out.skipped == {}

    async def test_entity_set_restores_only_listed_entities(self) -> None:
        engine = _engine("allowlisted")
        await _mint(engine)
        out = engine.restore_params(_PARAMS, tool_def=get_tool_def(email_tool))
        assert out.params == {
            "to": _EMAIL,
            "note": f"call <<PHONE_NUMBER_1>> or {_EMAIL}",
        }
        assert out.skipped == {"PHONE_NUMBER": 1}

    async def test_exact_entity_name_is_accepted(self) -> None:
        @tool(receives_pii={"EMAIL_ADDRESS"})
        async def exact(to: str) -> str:
            """Exact name."""
            return to

        engine = _engine("allowlisted")
        await _mint(engine)
        assert engine.deanonymize(_PARAMS, tool_def=get_tool_def(exact))["to"] == _EMAIL

    async def test_entity_names_are_case_insensitive(self) -> None:
        @tool(receives_pii={"email"})
        async def lower(to: str) -> str:
            """Lower-case name."""
            return to

        engine = _engine("allowlisted")
        await _mint(engine)
        assert engine.deanonymize(_PARAMS, tool_def=get_tool_def(lower))["to"] == _EMAIL

    async def test_prefix_does_not_match_inside_a_word(self) -> None:
        @tool(receives_pii={"PHONE_NUM"})
        async def partial(to: str) -> str:
            """Partial name."""
            return to

        engine = _engine("allowlisted")
        await _mint(engine)
        assert engine.deanonymize(_PARAMS, tool_def=get_tool_def(partial)) == _PARAMS

    async def test_no_tool_def_restores_nothing(self) -> None:
        engine = _engine("allowlisted")
        await _mint(engine)
        assert engine.deanonymize(_PARAMS) == _PARAMS
        assert engine.restore_params(_PARAMS).skipped["EMAIL_ADDRESS"] == 2

    async def test_whole_value_placeholder_and_nesting(self) -> None:
        engine = _engine("allowlisted")
        await _mint(engine)
        params = {"a": [{"b": "<<EMAIL_ADDRESS_1>>"}], "n": 3}
        assert engine.deanonymize(params, tool_def=get_tool_def(plain_tool)) == params
        out = engine.deanonymize(params, tool_def=get_tool_def(pii_tool))
        assert out == {"a": [{"b": _EMAIL}], "n": 3}

    async def test_unknown_placeholders_pass_through(self) -> None:
        engine = _engine("allowlisted")
        await _mint(engine)
        out = engine.restore_params({"to": "<<EMAIL_ADDRESS_9>>"}, tool_def=get_tool_def(pii_tool))
        assert out.params == {"to": "<<EMAIL_ADDRESS_9>>"}
        assert out.skipped == {}

    async def test_input_is_not_mutated(self) -> None:
        engine = _engine("allowlisted")
        await _mint(engine)
        params = {"to": "<<EMAIL_ADDRESS_1>>"}
        engine.deanonymize(params, tool_def=get_tool_def(pii_tool))
        assert params == {"to": "<<EMAIL_ADDRESS_1>>"}

    async def test_answer_text_for_the_developer_is_still_restored(self) -> None:
        engine = _engine("allowlisted")
        await _mint(engine)
        text, unmapped = deanonymize_text("hi <<EMAIL_ADDRESS_1>>", engine.get_pii_mapping())
        assert text == f"hi {_EMAIL}"
        assert unmapped == []

    async def test_strictest_guardrail_wins(self) -> None:
        engine = GuardrailEngine([PII(restore="all"), PII(restore="allowlisted")])
        assert engine.restore_mode == "allowlisted"


class TestToolResultProvenance:
    async def _engine_with_tool_result(self) -> GuardrailEngine:
        engine = _engine("allowlisted")
        await engine.scan_incoming_message(Message(Role.USER, "hello"), index=0)
        await engine.scan_incoming_message(
            Message(Role.TOOL, f"found {_EMAIL}", name="pii_tool", call_id="c1"), index=1
        )
        return engine

    async def test_restored_only_into_the_tool_that_produced_it(self) -> None:
        engine = await self._engine_with_tool_result()
        params = {"to": "<<EMAIL_ADDRESS_1>>"}

        @tool(receives_pii=True)
        async def other_pii_tool(to: str) -> str:
            """Allowed PII, but not the producer."""
            return to

        assert engine.deanonymize(params, tool_def=get_tool_def(pii_tool)) == {"to": _EMAIL}
        out = engine.restore_params(params, tool_def=get_tool_def(other_pii_tool))
        assert out.params == params
        assert out.skipped == {"EMAIL_ADDRESS": 1}

    async def test_producer_still_needs_to_be_allowed(self) -> None:
        engine = _engine("allowlisted")
        await engine.scan_incoming_message(
            Message(Role.TOOL, f"found {_EMAIL}", name="plain_tool", call_id="c1")
        )
        params = {"to": "<<EMAIL_ADDRESS_1>>"}
        assert engine.deanonymize(params, tool_def=get_tool_def(plain_tool)) == params

    async def test_user_text_with_the_same_value_makes_it_trusted(self) -> None:
        engine = await self._engine_with_tool_result()
        await engine.scan_incoming_message(Message(Role.USER, f"my mail is {_EMAIL}"), index=2)

        @tool(receives_pii=True)
        async def other_pii_tool(to: str) -> str:
            """Allowed PII."""
            return to

        out = engine.deanonymize(
            {"to": "<<EMAIL_ADDRESS_1>>"}, tool_def=get_tool_def(other_pii_tool)
        )
        assert out == {"to": _EMAIL}

    async def test_provenance_survives_pause_and_resume(self) -> None:
        engine = await self._engine_with_tool_result()
        saved = engine.get_pii_mapping()
        assert set(saved) == {"<<EMAIL_ADDRESS_1>>", engine_mod._ORIGINS_KEY}
        resumed = GuardrailEngine([PII(restore="allowlisted")], pii_mapping=saved)

        @tool(receives_pii=True)
        async def other_pii_tool(to: str) -> str:
            """Allowed PII."""
            return to

        params = {"to": "<<EMAIL_ADDRESS_1>>"}
        assert resumed.deanonymize(params, tool_def=get_tool_def(other_pii_tool)) == params
        assert resumed.deanonymize(params, tool_def=get_tool_def(pii_tool)) == {"to": _EMAIL}
        assert resumed.get_pii_mapping() == saved

    async def test_new_placeholders_continue_numbering_after_resume(self) -> None:
        engine = await self._engine_with_tool_result()
        resumed = GuardrailEngine(
            [PII(restore="allowlisted")], pii_mapping=engine.get_pii_mapping()
        )
        text, _, _ = await resumed.scan_incoming("and bob@example.com")
        assert "<<EMAIL_ADDRESS_2>>" in text

    @pytest.mark.parametrize(
        "origins",
        [
            None,
            "not json",
            "[1, 2]",
            '{"<<EMAIL_ADDRESS_1>>": 5}',
            '{"<<EMAIL_ADDRESS_1>>": []}',
            '{"<<EMAIL_ADDRESS_9>>": true}',
            '{"<<EMAIL_ADDRESS_1>>": [1, 2]}',
        ],
    )
    async def test_missing_or_corrupt_provenance_restores_nowhere(
        self, origins: str | None
    ) -> None:
        mapping = {"<<EMAIL_ADDRESS_1>>": _EMAIL}
        if origins is not None:
            mapping[engine_mod._ORIGINS_KEY] = origins
        engine = GuardrailEngine([PII(restore="allowlisted")], pii_mapping=mapping)
        params = {"to": "<<EMAIL_ADDRESS_1>>"}
        out = engine.restore_params(params, tool_def=get_tool_def(pii_tool))
        assert out.params == params
        assert out.skipped == {"EMAIL_ADDRESS": 1}

    async def test_legacy_mapping_is_rebuilt_from_trusted_history(self) -> None:
        engine = GuardrailEngine(
            [PII(restore="allowlisted")], pii_mapping={"<<EMAIL_ADDRESS_1>>": _EMAIL}
        )
        await engine.scan_incoming_message(Message(Role.USER, f"mail {_EMAIL}"), index=0)
        out = engine.deanonymize({"to": "<<EMAIL_ADDRESS_1>>"}, tool_def=get_tool_def(pii_tool))
        assert out == {"to": _EMAIL}

    async def test_legacy_mapping_rescanned_from_a_tool_result_stays_with_that_tool(self) -> None:
        engine = GuardrailEngine(
            [PII(restore="allowlisted")], pii_mapping={"<<EMAIL_ADDRESS_1>>": _EMAIL}
        )
        await engine.scan_incoming_message(
            Message(Role.TOOL, f"found {_EMAIL}", name="source", call_id="c1")
        )

        @tool(receives_pii=True)
        async def sink(to: str) -> str:
            """Allowed PII, not the producer."""
            return to

        params = {"to": "<<EMAIL_ADDRESS_1>>"}
        assert engine.deanonymize(params, tool_def=get_tool_def(sink)) == params

    async def test_partial_provenance_only_covers_the_recorded_placeholders(self) -> None:
        mapping = {
            "<<EMAIL_ADDRESS_1>>": _EMAIL,
            "<<EMAIL_ADDRESS_2>>": "bob@example.com",
            engine_mod._ORIGINS_KEY: '{"<<EMAIL_ADDRESS_1>>": true}',
        }
        engine = GuardrailEngine([PII(restore="allowlisted")], pii_mapping=mapping)
        out = engine.deanonymize(
            {"a": "<<EMAIL_ADDRESS_1>>", "b": "<<EMAIL_ADDRESS_2>>"},
            tool_def=get_tool_def(pii_tool),
        )
        assert out == {"a": _EMAIL, "b": "<<EMAIL_ADDRESS_2>>"}

    async def test_assistant_text_cannot_erase_a_tool_origin_restriction(self) -> None:
        engine = await self._engine_with_tool_result()
        await engine.scan_incoming_message(Message(Role.ASSISTANT, f"it is {_EMAIL}"), index=2)

        @tool(receives_pii=True)
        async def other_pii_tool(to: str) -> str:
            """Allowed PII, not the producer."""
            return to

        params = {"to": "<<EMAIL_ADDRESS_1>>"}
        assert engine.deanonymize(params, tool_def=get_tool_def(other_pii_tool)) == params
        assert engine.deanonymize(params, tool_def=get_tool_def(pii_tool)) == {"to": _EMAIL}

    async def test_model_minted_placeholder_is_restored_nowhere_until_trusted(self) -> None:
        engine = _engine("allowlisted")
        await engine.scan_incoming_message(Message(Role.ASSISTANT, f"it is {_EMAIL}"), index=0)
        params = {"to": "<<EMAIL_ADDRESS_1>>"}
        assert engine.deanonymize(params, tool_def=get_tool_def(pii_tool)) == params
        await engine.scan_incoming_message(Message(Role.USER, f"I am {_EMAIL}"), index=1)
        assert engine.deanonymize(params, tool_def=get_tool_def(pii_tool)) == {"to": _EMAIL}

    async def test_two_tools_that_produced_the_value_may_both_receive_it(self) -> None:
        engine = await self._engine_with_tool_result()
        await engine.scan_incoming_message(
            Message(Role.TOOL, f"also {_EMAIL}", name="other_pii_tool", call_id="c2")
        )

        @tool(receives_pii=True)
        async def other_pii_tool(to: str) -> str:
            """Second producer."""
            return to

        params = {"to": "<<EMAIL_ADDRESS_1>>"}
        assert engine.deanonymize(params, tool_def=get_tool_def(pii_tool)) == {"to": _EMAIL}
        assert engine.deanonymize(params, tool_def=get_tool_def(other_pii_tool)) == {"to": _EMAIL}

    async def test_provenance_survives_resume_for_cross_tool_denial_after_assistant_echo(
        self,
    ) -> None:
        engine = await self._engine_with_tool_result()
        await engine.scan_incoming_message(Message(Role.ASSISTANT, _EMAIL), index=2)
        resumed = GuardrailEngine(
            [PII(restore="allowlisted")], pii_mapping=engine.get_pii_mapping()
        )

        @tool(receives_pii=True)
        async def other_pii_tool(to: str) -> str:
            """Allowed PII, not the producer."""
            return to

        params = {"to": "<<EMAIL_ADDRESS_1>>"}
        assert resumed.deanonymize(params, tool_def=get_tool_def(other_pii_tool)) == params

    async def test_provenance_is_not_tracked_under_all(self) -> None:
        engine = _engine("all")
        await engine.scan_incoming_message(
            Message(Role.TOOL, f"found {_EMAIL}", name="pii_tool", call_id="c1")
        )
        assert engine_mod._ORIGINS_KEY not in engine.get_pii_mapping()
        assert engine.deanonymize(
            {"to": "<<EMAIL_ADDRESS_1>>"}, tool_def=get_tool_def(plain_tool)
        ) == {"to": _EMAIL}

    async def test_origins_key_never_rewrites_answer_text(self) -> None:
        engine = await self._engine_with_tool_result()
        mapping = engine.get_pii_mapping()
        text, _ = deanonymize_text(f"see {engine_mod._ORIGINS_KEY}", mapping)
        assert text == f"see {engine_mod._ORIGINS_KEY}"


class TestHasPlaceholder:
    def test_detects_nested_placeholders(self) -> None:
        assert has_placeholder("x <<EMAIL_ADDRESS_1>> y")
        assert has_placeholder({"a": [{"b": "<<PHONE_NUMBER_2>>"}]})
        assert not has_placeholder({"a": ["plain", 3, None]})
        assert not has_placeholder(_nested("<<EMAIL_ADDRESS_1>>", 80))


class TestRestoreNotice:
    @pytest.fixture(autouse=True)
    def _fresh(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(engine_mod, "_restore_notice_logged", False)

    @staticmethod
    def _notices(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
        return [r for r in caplog.records if "restore=" in r.getMessage()]

    def test_logged_once_when_not_chosen(self, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level(logging.INFO, logger="dendrux.guardrails._engine")
        GuardrailEngine([PII()])
        GuardrailEngine([PII()])
        notices = self._notices(caplog)
        assert len(notices) == 1
        assert notices[0].levelno == logging.INFO
        assert "allowlisted" in notices[0].getMessage()

    @pytest.mark.parametrize("restore", ["all", "allowlisted"])
    def test_silent_when_chosen(self, caplog: pytest.LogCaptureFixture, restore: str) -> None:
        caplog.set_level(logging.INFO, logger="dendrux.guardrails._engine")
        GuardrailEngine([PII(restore=restore)])  # type: ignore[arg-type]
        assert self._notices(caplog) == []

    def test_silent_when_nothing_is_redacted(self, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level(logging.INFO, logger="dendrux.guardrails._engine")
        GuardrailEngine([PII(action="warn"), SecretDetection()])
        GuardrailEngine([PII(action="block")])
        assert self._notices(caplog) == []


# ---------------------------------------------------------------------------
# D-029 end to end: an indirect-injection attempt through the ReAct loop
# ---------------------------------------------------------------------------

_received: list[str] = []


@tool()
async def send_report(recipient: str) -> str:
    """Send the report (not allowed to see PII)."""
    _received.append(recipient)
    return "sent"


@tool(receives_pii=True)
async def crm_update(contact: str) -> str:
    """Update the CRM (allowed to see PII)."""
    _received.append(contact)
    return "updated"


def _usage() -> UsageStats:
    return UsageStats(input_tokens=5, output_tokens=5, total_tokens=10)


def _call(name: str, **params: Any) -> LLMResponse:
    return LLMResponse(
        text=None,
        tool_calls=[ToolCall(name=name, params=params, id=f"call_{name}")],
        usage=_usage(),
    )


async def _run_injection(restore: str | None, tool_name: str) -> RunStatus:
    _received.clear()
    pii = PII() if restore is None else PII(restore=restore)  # type: ignore[arg-type]
    agent = Agent(prompt="p", tools=[send_report, crm_update], guardrails=[pii])
    llm = MockLLM(
        [
            _call(
                tool_name,
                **{"recipient" if tool_name == "send_report" else "contact": "<<EMAIL_ADDRESS_1>>"},
            ),
            LLMResponse(text="done", usage=_usage()),
        ]
    )
    result = await ReActLoop().run(
        agent=agent,
        provider=llm,
        strategy=NativeToolCalling(),
        user_input=f"My email is {_EMAIL}. Summarise the attached page.",
    )
    return result.status


def _call_site_passes_tool_def() -> bool:
    from dendrux.loops import react

    return "tool_def=" in inspect.getsource(react._process_tool_calls)


class TestIndirectInjectionEndToEnd:
    async def test_default_restore_all_still_sends_the_real_value(self) -> None:
        assert await _run_injection(None, "send_report") == RunStatus.SUCCESS
        assert _received == [_EMAIL]

    async def test_allowlisted_blocks_the_unlisted_tool(self) -> None:
        assert await _run_injection("allowlisted", "send_report") == RunStatus.SUCCESS
        assert _received == ["<<EMAIL_ADDRESS_1>>"]

    @pytest.mark.skipif(
        not _call_site_passes_tool_def(),
        reason="needs the T4 call site to pass tool_def= to GuardrailEngine.deanonymize",
    )
    async def test_allowlisted_still_serves_the_listed_tool(self) -> None:
        assert await _run_injection("allowlisted", "crm_update") == RunStatus.SUCCESS
        assert _received == [_EMAIL]

    @pytest.mark.skipif(
        not _call_site_passes_tool_def(),
        reason="needs the T4 call site to pass tool_def= and emit guardrail.restore_skipped",
    )
    async def test_skipped_restore_emits_counts_only(self) -> None:
        events: list[tuple[str, dict[str, Any]]] = []

        class Spy(BaseRecorder):
            async def on_message_appended(self, run_id, message, iteration):
                pass

            async def on_llm_call_completed(self, run_id, response, iteration, **kw):
                pass

            async def on_tool_completed(self, run_id, tool_call, tool_result, iteration):
                pass

            async def on_governance_event(
                self, run_id, event_type, iteration, data, correlation_id=None
            ):
                events.append((event_type, data))

        agent = Agent(prompt="p", tools=[send_report], guardrails=[PII(restore="allowlisted")])
        llm = MockLLM(
            [
                _call("send_report", recipient="<<EMAIL_ADDRESS_1>>"),
                LLMResponse(text="ok", usage=_usage()),
            ]
        )
        await ReActLoop().run(
            agent=agent,
            provider=llm,
            strategy=NativeToolCalling(),
            user_input=f"My email is {_EMAIL}.",
            recorder=Spy(),
        )
        skipped = [d for t, d in events if t == "guardrail.restore_skipped"]
        assert len(skipped) == 1
        assert skipped[0]["tool_name"] == "send_report"
        assert skipped[0]["entities"] == {"EMAIL_ADDRESS": 1}
        assert _EMAIL not in str(events)


# ---------------------------------------------------------------------------
# Params walk: keys, numbers and bounded traversal
# ---------------------------------------------------------------------------


class TestParamsWalkIsBounded:
    async def test_long_key_with_many_items_is_fast(self) -> None:
        scanned = {"chars": 0, "calls": 0}
        budget = 6_000_000

        class _Meter:
            action: Literal["redact", "block", "warn"] = "warn"

            async def scan(self, text: str) -> list[Finding]:
                scanned["chars"] += len(text)
                scanned["calls"] += 1
                if scanned["chars"] > budget:
                    raise AssertionError(f"scanned {scanned['chars']} characters, budget {budget}")
                return []

        engine = GuardrailEngine([_Meter(), SecretDetection(action="block"), PII(action="warn")])
        params = {"k" * 100_000: ["x"] * 2000}
        findings, block = await engine.scan_outgoing("ok", [params])
        assert block is None
        assert findings == []
        # Only the last characters of a key join a value's context scan, so the 100,000
        # character key is not repeated 2,000 times (that would be about 200 million).
        assert scanned["chars"] <= budget

    async def test_oversize_key_is_unscannable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(engine_mod, "MAX_SCAN_CHARS", 1000)
        engine = GuardrailEngine([PII(action="block")])
        for params in (
            {"x" * 2000: "ok"},
            {"x" * 2000: ["a", "b"]},
            {"a": {"x" * 2000: {}}},
            {"a": [{"x" * 2000: 1}]},
        ):
            _, block = await engine.scan_outgoing("ok", [params])
            assert block is not None, params
            assert UNSCANNABLE in block

    async def test_keys_are_scanned_once_as_leaves(self) -> None:
        engine = GuardrailEngine([SecretDetection(action="block")])
        _, block = await engine.scan_outgoing("ok", [{_AWS: 1}])
        assert block is not None
        assert "AWS_ACCESS_KEY" in block
        assert _AWS not in block

    async def test_numbers_are_scanned(self) -> None:
        engine = GuardrailEngine([PII(action="block")])
        _, block = await engine.scan_outgoing("ok", [{"card": 4111111111111111}])
        assert block is not None
        assert "CREDIT_CARD" in block

    async def test_booleans_and_none_are_ignored(self) -> None:
        engine = GuardrailEngine([PII(action="block")])
        findings, block = await engine.scan_outgoing("ok", [{"a": True, "b": None, "c": 1.5}])
        assert (findings, block) == ([], None)

    async def test_number_too_long_to_convert_is_unscannable(self) -> None:
        engine = GuardrailEngine([PII(action="block")])
        _, block = await engine.scan_outgoing("ok", [{"n": 10**5000}])
        assert block is not None
        assert UNSCANNABLE in block

    async def test_wide_list_is_rejected_without_expanding_it(self) -> None:
        import tracemalloc

        wide = [0] * 1_000_000
        tracemalloc.start()
        try:
            leaves = engine_mod._collect_leaves({"x": wide})
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        assert leaves.limit == "node count"
        assert peak < 3_000_000

    async def test_wide_dict_is_rejected_without_expanding_it(self) -> None:
        import tracemalloc

        wide = {str(i): 0 for i in range(300_000)}
        tracemalloc.start()
        try:
            leaves = engine_mod._collect_leaves({"x": wide})
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        assert leaves.limit == "node count"
        assert peak < 3_000_000

    async def test_over_depth_siblings_are_still_counted(self) -> None:
        leaves = engine_mod._collect_leaves({"x": [_nested("a", 60)["x"]] * 30_000})
        assert leaves.limit in ("node count", "nesting depth")

    async def test_scan_cost_with_context_is_bounded_by_the_key_tail(self) -> None:
        engine = GuardrailEngine([SecretDetection(action="block")])
        _, block = await engine.scan_outgoing(
            "ok", [{"a" * 5000 + "api_key": "sk_live_" + "ABCDEFGHIJKLMNOPQRSTUVWX1234"}]
        )
        assert block is not None
        assert "GENERIC_API_KEY" in block


# ---------------------------------------------------------------------------
# Redaction: linear, off the loop, bounded mapping
# ---------------------------------------------------------------------------


def _ips(n: int) -> str:
    return " ".join(f"10.{(i >> 16) & 255}.{(i >> 8) & 255}.{i & 255}" for i in range(n))


class _WorkBudgetError(AssertionError):
    pass


class _Work:
    """Characters copied through slicing, concatenation and ``replace`` on a ``_CopyCountingStr``.

    Operation counts do not depend on machine load, unlike wall-clock time. The
    budget fails fast, so a quadratic implementation stops at its first big copy
    instead of running for minutes.
    """

    chars = 0
    budget = 0

    @classmethod
    def reset(cls, budget: int) -> None:
        cls.chars = 0
        cls.budget = budget

    @classmethod
    def add(cls, n: int) -> None:
        cls.chars += n
        if cls.chars > cls.budget:
            raise _WorkBudgetError(f"copied {cls.chars} characters, budget {cls.budget}")


class _CopyCountingStr(str):
    def __getitem__(self, key: Any) -> _CopyCountingStr:
        out = str.__getitem__(self, key)
        _Work.add(len(out))
        return _CopyCountingStr(out)

    def __add__(self, other: Any) -> _CopyCountingStr:
        out = str.__add__(self, other)
        _Work.add(len(out))
        return _CopyCountingStr(out)

    def __radd__(self, other: str) -> _CopyCountingStr:
        out = other + str(self)
        _Work.add(len(out))
        return _CopyCountingStr(out)

    def replace(self, *args: Any, **kwargs: Any) -> _CopyCountingStr:  # type: ignore[override]
        out = str.replace(self, *args, **kwargs)
        _Work.add(len(out))
        return _CopyCountingStr(out)


class _CountingDict(dict[str, str]):
    """A dict that counts keyed reads and, separately, every element visited by iteration."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.reads = 0
        self.visits = 0
        self.max_visits: int | None = None

    def _visit(self) -> None:
        self.visits += 1
        if self.max_visits is not None and self.visits > self.max_visits:
            raise AssertionError(f"mapping walked element by element ({self.visits} visits)")

    def __getitem__(self, key: str) -> str:
        self.reads += 1
        return super().__getitem__(key)

    def __contains__(self, key: object) -> bool:
        self.reads += 1
        return super().__contains__(key)

    def get(self, key: str, default: Any = None) -> Any:
        self.reads += 1
        return super().get(key, default)

    def __iter__(self) -> Any:
        for key in super().__iter__():
            self._visit()
            yield key

    def items(self) -> Any:  # type: ignore[override]
        for pair in super().items():
            self._visit()
            yield pair

    def values(self) -> Any:  # type: ignore[override]
        for value in super().values():
            self._visit()
            yield value

    def keys(self) -> Any:  # type: ignore[override]
        for key in super().keys():  # noqa: SIM118
            self._visit()
            yield key


def _count_calls(
    monkeypatch: pytest.MonkeyPatch, name: str, limit: int | None = None
) -> dict[str, int]:
    """Wrap ``GuardrailEngine.<name>``, count its calls and fail fast past ``limit``."""
    calls = {"n": 0}
    original = getattr(GuardrailEngine, name)

    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        calls["n"] += 1
        if limit is not None and calls["n"] > limit:
            raise AssertionError(f"{name} called more than {limit} times")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(GuardrailEngine, name, wrapper)
    return calls


def _snapshot(engine: GuardrailEngine) -> dict[str, str]:
    """``get_pii_mapping`` after the walk limit is lifted (reading it walks the mapping)."""
    for table in (engine._pii_mapping, engine._reverse):
        if isinstance(table, _CountingDict):
            table.max_visits = None
    return engine.get_pii_mapping()


def _counting_engine() -> GuardrailEngine:
    engine = GuardrailEngine([PII(action="redact")])
    engine._pii_mapping = _CountingDict(engine._pii_mapping)
    engine._reverse = _CountingDict(engine._reverse)
    engine._pii_mapping.max_visits = engine._reverse.max_visits = 0
    return engine


class TestRedactionIsBounded:
    async def test_mebibyte_of_distinct_ips_fails_closed_quickly(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        raw = _ips(88_000)
        assert len(raw) > 1_000_000
        _Work.reset(len(raw))
        text = _CopyCountingStr(raw)
        created = _count_calls(monkeypatch, "_get_or_create_placeholder", limit=0)
        applied = _count_calls(monkeypatch, "_apply_redaction", limit=0)
        engine = _counting_engine()
        _, findings, block = await engine.scan_incoming(text)
        assert block is not None
        assert UNSCANNABLE in block
        # Failing closed means no redaction work at all, not redaction followed by a check.
        assert created["n"] == 0
        assert applied["n"] == 0
        assert _Work.chars <= len(raw)
        assert len(_snapshot(engine)) == 0

    async def test_many_distinct_values_below_the_cap_redact_in_linear_time(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        raw = _ips(10_000)
        _Work.reset(len(raw))
        text = _CopyCountingStr(raw)
        created = _count_calls(monkeypatch, "_get_or_create_placeholder")
        minted = _count_calls(monkeypatch, "_next_placeholder")
        applied = _count_calls(monkeypatch, "_apply_redaction")
        engine = _counting_engine()
        out, findings, block = await engine.scan_incoming(text)
        assert block is None
        assert len(findings) == 10_000
        assert "10." not in out
        # One pass over the text, one placeholder lookup per finding, no scan of the mapping.
        assert applied["n"] == 1
        assert created["n"] == 10_000
        assert minted["n"] == 10_000
        assert _Work.chars <= len(raw)
        reverse, mapping = engine._reverse, engine._pii_mapping
        assert isinstance(reverse, _CountingDict) and isinstance(mapping, _CountingDict)
        assert reverse.visits == 0 and mapping.visits == 0
        assert reverse.reads <= 4 * 10_000
        assert len(_snapshot(engine)) == 10_000

    async def test_repeated_value_dedupes_to_one_mapping_entry(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        raw = "192.168.1.1 " * 90_000
        _Work.reset(len(raw))
        text = _CopyCountingStr(raw)
        created = _count_calls(monkeypatch, "_get_or_create_placeholder")
        minted = _count_calls(monkeypatch, "_next_placeholder")
        applied = _count_calls(monkeypatch, "_apply_redaction")
        engine = _counting_engine()
        out, findings, block = await engine.scan_incoming(text)
        assert block is None
        assert len(findings) == 90_000
        assert set(out.split()) == {"<<IP_ADDRESS_1>>"}
        # 90,000 sightings: one text pass, one placeholder minted, no scan of the mapping.
        assert applied["n"] == 1
        assert created["n"] == 90_000
        assert minted["n"] == 1
        assert _Work.chars <= len(raw)
        reverse, mapping = engine._reverse, engine._pii_mapping
        assert isinstance(reverse, _CountingDict) and isinstance(mapping, _CountingDict)
        assert reverse.visits == 0 and mapping.visits == 0
        assert reverse.reads <= 4 * 90_000
        assert _snapshot(engine) == {"<<IP_ADDRESS_1>>": "192.168.1.1"}

    async def test_mapping_cap_blocks_under_redact_and_reports_under_warn(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(engine_mod, "MAX_MAPPING_ENTRIES", 5)
        text = _ips(6)
        _, _, block = await GuardrailEngine([PII(action="redact")]).scan_incoming(text)
        assert block is not None
        assert UNSCANNABLE in block
        engine = GuardrailEngine([PII(action="redact"), SecretDetection(action="warn")])
        _, findings, block = await engine.scan_incoming(_ips(5))
        assert block is None
        assert len(engine.get_pii_mapping()) == 5

    async def test_existing_values_do_not_count_against_the_cap(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(engine_mod, "MAX_MAPPING_ENTRIES", 3)
        engine = GuardrailEngine([PII(action="redact")])
        await engine.scan_incoming(_ips(3))
        _, _, block = await engine.scan_incoming(_ips(3))
        assert block is None

    async def test_large_redaction_runs_in_a_thread(self, monkeypatch: pytest.MonkeyPatch) -> None:
        names: list[str] = []
        real = asyncio.to_thread

        async def spy(fn, /, *args, **kwargs):
            names.append(getattr(fn, "__name__", "?"))
            return await real(fn, *args, **kwargs)

        monkeypatch.setattr(asyncio, "to_thread", spy)
        engine = GuardrailEngine([PII(action="redact")])
        await engine.scan_incoming(_ips(5_000))
        assert "_apply_redaction" in names
        names.clear()
        await engine.scan_incoming("mail alice@example.com")
        assert "_apply_redaction" not in names

    async def test_placeholders_are_numbered_right_to_left_like_041(self) -> None:
        engine = GuardrailEngine([PII(action="redact")])
        text, _, _ = await engine.scan_incoming("a@example.com then b@example.com")
        assert text == "<<EMAIL_ADDRESS_2>> then <<EMAIL_ADDRESS_1>>"

    async def test_restore_of_many_placeholders_is_linear(self) -> None:
        engine = GuardrailEngine([PII(action="redact")])
        await engine.scan_incoming(_ips(5_000))
        counting = _CountingDict(engine._pii_mapping)
        counting.max_visits = 0
        engine._pii_mapping = counting
        params = {"items": [f"value=<<IP_ADDRESS_{i}>>!" for i in range(1, 2001)]}
        out = engine.deanonymize(params)
        assert "<<" not in str(out)
        assert all(item.startswith("value=10.") and item.endswith("!") for item in out["items"])
        # One keyed lookup per placeholder (plus the whole-value check); never a walk
        # over the 5,000 mapping entries for each item.
        assert counting.visits == 0
        assert counting.reads <= 2 * 2000


class TestGuardrailReportedUnscannable:
    class _Busy:
        def __init__(self, action: str) -> None:
            self.action = action

        async def scan(self, text: str) -> list[Finding]:
            return [Finding(entity_type=UNSCANNABLE, start=0, end=0, score=1.0, text="")]

    @pytest.mark.parametrize("action", ["block", "redact"])
    async def test_blocks(self, action: str) -> None:
        engine = GuardrailEngine([self._Busy(action)])  # type: ignore[list-item]
        text, findings, block = await engine.scan_incoming("hello")
        assert block is not None
        assert UNSCANNABLE in block
        assert text == "hello"
        _, out_block = await engine.scan_outgoing("hello", [{"a": "b"}])
        assert out_block is not None

    async def test_warn_reports_and_continues(self) -> None:
        engine = GuardrailEngine([self._Busy("warn")])  # type: ignore[list-item]
        _, findings, block = await engine.scan_incoming("hello")
        assert block is None
        assert [f.entity_type for f in findings] == [UNSCANNABLE]

    class _Partial:
        def __init__(self, action: str) -> None:
            self.action = action

        async def scan(self, text: str) -> list[Finding]:
            return [
                Finding("EMAIL_ADDRESS", 0, 3, 1.0, text[:3]),
                Finding(UNSCANNABLE, 0, 0, 1.0, ""),
            ]

    @pytest.mark.parametrize("action", ["redact", "warn", "block"])
    @pytest.mark.parametrize("outgoing", [False, True])
    async def test_mixed_findings_keep_unscannable(self, action: str, outgoing: bool) -> None:
        engine = GuardrailEngine([self._Partial(action)])  # type: ignore[list-item]
        if outgoing:
            findings, block = await engine.scan_outgoing("abc secret")
        else:
            text, findings, block = await engine.scan_incoming("abc secret")
            assert text == "abc secret"
        assert [f.entity_type for f in findings] == [UNSCANNABLE]
        if action == "warn":
            assert block is None
        else:
            assert block is not None and UNSCANNABLE in block


class TestLegacyOversizeMapping:
    async def test_resumed_mapping_keeps_existing_values_and_rejects_new(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        mapping = {f"<<IP_ADDRESS_{i + 1}>>": f"10.0.{i // 256}.{i % 256}" for i in range(20_001)}
        engine = GuardrailEngine([PII(action="redact")], pii_mapping=mapping, run_id="run-legacy")
        assert engine._at_mapping_cap
        assert engine.get_pii_mapping() == mapping
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert len(warnings) == 1
        assert "run-legacy" in warnings[0].message
        assert "20001" in warnings[0].message

        old = mapping["<<IP_ADDRESS_1>>"]
        text, _, block = await engine.scan_incoming(f"old {old}")
        assert block is None
        assert text == "old <<IP_ADDRESS_1>>"
        text, findings, block = await engine.scan_incoming("new 192.168.1.1")
        assert text == "new 192.168.1.1"
        assert block is not None and UNSCANNABLE in block
        assert any(f.entity_type == UNSCANNABLE for f in findings)
        assert engine.get_pii_mapping() == mapping
        assert len([r for r in caplog.records if r.levelname == "WARNING"]) == 1
