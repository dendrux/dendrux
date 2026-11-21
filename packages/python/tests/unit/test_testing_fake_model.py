"""FakeModel: scripted, stateless answers for tests that need no API key."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from dendrux import Agent
from dendrux.loops.single import SingleCall
from dendrux.testing import (
    FakeModel,
    FakeModelScriptError,
    MockLLM,
    call,
    calls,
    say,
    structured,
    when,
)
from dendrux.tool import tool
from dendrux.types import LLMResponse, Message, Role, RunStatus

TENANT = "t1"


class Severity(BaseModel):
    level: str
    reason: str


@tool()
async def issue_refund(amount: int) -> str:
    """Refund an amount."""
    return f"refunded {amount}"


SCRIPT = {
    "lead": [
        calls(call("severity", task="ticket T-1"), call("billing", task="refund?")),
        say("Refund needs approval."),
    ],
    "severity": [structured(Severity(level="high", reason="double charge"))],
    "billing": [call("issue_refund", amount=40), say("Refund queued.")],
}


def build_team(fake: FakeModel, db: str | None = None) -> Agent:
    sev = Agent(
        name="severity",
        description="Rates severity.",
        prompt="p",
        provider=fake.provider("severity"),
        loop=SingleCall(),
        output_type=Severity,
    )
    bill = Agent(
        name="billing",
        description="Handles billing.",
        prompt="p",
        provider=fake.provider("billing"),
        tools=[issue_refund],
        require_approval=["issue_refund"],
    )
    return Agent(
        name="lead",
        prompt="p",
        provider=fake.provider("lead"),
        subagents=[sev, bill],
        database_url=db,
    )


def msgs(*texts: str, assistant: int = 0) -> list:
    out = [Message(role=Role.USER, content=t) for t in texts]
    out += [Message(role=Role.ASSISTANT, content="a") for _ in range(assistant)]
    return out


def file_db(tmp_path) -> str:
    return f"sqlite+aiosqlite:///{tmp_path / 'fake.db'}"


async def test_fanout_with_single_call_child_finishes_without_key(tmp_path):
    script = {
        "lead": [
            calls(call("severity", task="T-1"), call("helper", task="x")),
            say("all done"),
        ],
        "severity": [structured(Severity(level="low", reason="r"))],
        "helper": [say("helped")],
    }
    fake = FakeModel(script)
    sev = Agent(
        name="severity",
        description="d",
        prompt="p",
        provider=fake.provider("severity"),
        loop=SingleCall(),
        output_type=Severity,
    )
    helper = Agent(name="helper", description="d", prompt="p", provider=fake.provider("helper"))
    lead = Agent(
        name="lead",
        prompt="p",
        provider=fake.provider("lead"),
        subagents=[sev, helper],
        database_url=file_db(tmp_path),
    )
    result = await lead.run("go", tenant_id=TENANT)
    assert result.status == RunStatus.SUCCESS
    assert result.answer == "all done"
    fake.assert_all_used()
    assert fake.calls("severity")[0].output_schema is not None
    assert result.usage.total_tokens > 0


async def test_approval_resumes_in_a_second_process(tmp_path):
    db = file_db(tmp_path)
    fake1 = FakeModel(SCRIPT)
    first = await build_team(fake1, db).run("charged twice", tenant_id=TENANT)
    assert first.status == RunStatus.WAITING_APPROVAL

    fake2 = FakeModel(SCRIPT)  # new objects, same script and database
    done = await build_team(fake2, db).submit_approval(
        first.run_id, approved=True, tenant_id=TENANT, expected_pause_id=first.pause_id
    )
    assert done.status == RunStatus.SUCCESS
    assert done.answer == "Refund needs approval."
    fake2.assert_all_used(fake1)


def test_unknown_agent_raises_at_bind_time():
    with pytest.raises(FakeModelScriptError) as info:
        FakeModel({"lead": [say("x")]}).provider("nope")
    assert "'nope'" in str(info.value) and "lead" in str(info.value)
    assert isinstance(info.value, LookupError)


async def test_used_up_script_names_agent_and_turn_without_message_text():
    provider = FakeModel({"lead": [say("one")]}).provider("lead")
    secret = "SENTINEL-MESSAGE-TEXT-91"
    with pytest.raises(FakeModelScriptError) as info:
        await provider.complete(msgs(secret, assistant=1))
    text = str(info.value)
    assert "'lead'" in text and "turn 1" in text
    assert secret not in text


async def test_schema_mismatch_gives_a_clear_error_without_values():
    fake = FakeModel({"a": [structured({"level": 5, "reason": "SENTINEL-VALUE"})]})
    schema = Severity.model_json_schema()
    with pytest.raises(FakeModelScriptError) as info:
        await fake.provider("a").complete(msgs("q"), output_schema=schema)
    assert "does not match the output schema" in str(info.value)
    assert "$.level" in str(info.value)
    assert "SENTINEL-VALUE" not in str(info.value)

    missing = FakeModel({"a": [structured({"level": "x"})]})
    with pytest.raises(FakeModelScriptError, match="missing required property"):
        await missing.provider("a").complete(msgs("q"), output_schema=schema)


async def test_valid_structured_json_is_accepted():
    fake = FakeModel({"a": [structured(Severity(level="x", reason="y"))]})
    out = await fake.provider("a").complete(msgs("q"), output_schema=Severity.model_json_schema())
    assert Severity.model_validate_json(out.text or "") == Severity(level="x", reason="y")


async def test_when_gives_two_copies_of_a_helper_different_answers(tmp_path):
    helper_script = [
        when(lambda m: "alpha" in str(m[-1].content), say("answer-A")),
        when(lambda m: "beta" in str(m[-1].content), say("answer-B")),
        say("fallback"),
    ]
    script = {
        "lead": [calls(call("helper", task="alpha"), call("helper", task="beta")), say("ok")],
        "helper": helper_script,
    }
    fake = FakeModel(script)
    helper = Agent(name="helper", description="d", prompt="p", provider=fake.provider("helper"))
    lead = Agent(
        name="lead",
        prompt="p",
        provider=fake.provider("lead"),
        subagents=[helper],
        database_url=file_db(tmp_path),
    )
    result = await lead.run("go", tenant_id=TENANT)
    assert result.status == RunStatus.SUCCESS
    answers = sorted(
        str(m.content) for c in fake.calls("lead") for m in c.messages if m.role == Role.TOOL
    )
    assert any("answer-A" in a for a in answers) and any("answer-B" in a for a in answers)


async def test_when_slot_rules():
    fake = FakeModel(
        {"a": [when(lambda m: False, say("no")), say("default"), when(lambda m: False, say("n"))]}
    )
    provider = fake.provider("a")
    assert (await provider.complete(msgs("q"))).text == "default"
    with pytest.raises(FakeModelScriptError, match="no when"):
        await provider.complete(msgs("q", assistant=1))


async def test_tool_call_ids_match_across_instances():
    async def ids() -> list[str]:
        out = await FakeModel(SCRIPT).provider("lead").complete(msgs("q"))
        return [c.id for c in out.tool_calls or []]

    first, second = await ids(), await ids()
    assert first == second == ["fake-lead-0-0", "fake-lead-0-1"]


def test_assert_all_used_reports_unused_entries():
    fake = FakeModel({"a": [say("1"), say("2")]})
    with pytest.raises(AssertionError, match="a@0"):
        fake.assert_all_used()


async def test_assert_all_used_passes_when_every_entry_ran():
    fake = FakeModel({"a": [say("1"), say("2")]})
    p = fake.provider("a")
    await p.complete(msgs("q"))
    await p.complete(msgs("q", assistant=1))
    fake.assert_all_used()


async def test_usage_model_override_raw_response_and_call_records():
    raw = LLMResponse(text="raw")
    fake = FakeModel({"a": [say("hello world"), raw]})
    p = fake.provider("a")
    first = await p.complete(msgs("q"), model="other", run_id="r1")
    assert first.model == "other"
    assert first.usage.input_tokens > 0 and first.usage.total_tokens == (
        first.usage.input_tokens + first.usage.output_tokens
    )
    second = await p.complete(msgs("q", assistant=1))
    assert second.text == "raw" and second.model == "fake"
    assert raw.model is None  # the scripted object is not mutated
    records = fake.calls("a")
    assert [r.turn for r in records] == [0, 1] and records[0].run_id == "r1"


async def test_streaming_uses_the_base_fallback():
    p = FakeModel({"a": [say("streamed")]}).provider("a")
    events = [e async for e in p.complete_stream(msgs("q"))]
    assert [e.type.value for e in events][-1] == "done"


def test_mock_llm_is_re_exported_unchanged():
    assert MockLLM.__module__ == "dendrux.llm.mock"
