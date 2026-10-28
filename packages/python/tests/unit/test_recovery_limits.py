"""Recovery entry bounds reject before I/O without echoing provider options."""

import logging

import pytest

from dendrux.agent import Agent
from dendrux.llm.mock import MockLLM
from dendrux.types import Limits, LLMResponse


def test_recovery_limit_defaults():
    limits = Limits()
    assert getattr(limits, "max_pending_calls", None) == 64
    assert getattr(limits, "max_rebuild_bytes", None) == 16 * 1024 * 1024
    assert getattr(limits, "max_rebuild_rows", None) == 20_000


@pytest.mark.parametrize("name", ["max_pending_calls", "max_rebuild_bytes", "max_rebuild_rows"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_recovery_limit_rejects_invalid_values(name, value):
    assert name in Limits.__dataclass_fields__, "recovery bounds must be configurable"
    with pytest.raises(ValueError):
        Limits(**{name: value})


@pytest.mark.parametrize("entry", ["run", "stream", "runner_run", "runner_stream"])
async def test_provider_options_reject_oversized_before_provider(entry, caplog):
    from dendrux.runtime import runner

    provider = MockLLM([LLMResponse(text="unused")])
    agent = Agent(prompt="test", provider=provider)
    # UTF-8 bytes, not code points, determine the bound.
    value = "SENTINEL_OPTION" + "é" * 1100
    with caplog.at_level(logging.WARNING), pytest.raises(ValueError):
        if entry == "run":
            await agent.run("hello", thinking={"value": value})
        elif entry == "stream":
            agent.stream("hello", thinking={"value": value})
        elif entry == "runner_run":
            await runner.run(
                agent, provider=provider, user_input="hello", thinking={"value": value}
            )
        else:
            runner.run_stream(
                agent, provider=provider, user_input="hello", thinking={"value": value}
            )
    assert "SENTINEL_OPTION" not in caplog.text
    assert any(
        getattr(r, "code", None) == "provider_kwargs_too_large" and hasattr(r, "run_id")
        for r in caplog.records
    )


@pytest.mark.parametrize("streaming", [False, True])
async def test_provider_options_realistic_size_passes(streaming):
    agent = Agent(prompt="test", provider=MockLLM([LLMResponse(text="done")]))
    options = {
        "model": "mock",
        "max_tokens": 4096,
        "temperature": 0.5,
        "thinking": {"type": "enabled", "budget_tokens": 1024, "padding": "x" * 1500},
    }
    if streaming:
        stream = agent.stream("hello", **options)
        async for _ in stream:
            pass
        result = stream.result
    else:
        result = await agent.run("hello", **options)
    assert result.answer == "done"


def test_saved_provider_options_checked_before_claim(caplog):
    from dendrux.runtime.submit import submission_validator

    raw = {
        "agent_name": "bounded",
        "pause_id": "generation",
        "provider_kwargs": {"thinking": {"value": "x" * 2100}},
    }
    agent = Agent(name="bounded", prompt="test")
    with caplog.at_level(logging.WARNING), pytest.raises(ValueError):
        submission_validator("run-resume", raw, agent=agent)
    assert any(
        getattr(r, "code", None) == "provider_kwargs_too_large"
        and getattr(r, "run_id", None) == "run-resume"
        for r in caplog.records
    )


def test_saved_provider_options_checked_again_under_claim_lock():
    from dendrux.runtime.submit import submission_validator

    raw = {
        "agent_name": "bounded",
        "pause_id": "generation",
        "provider_kwargs": {"max_tokens": 4096},
    }
    validate = submission_validator("run-resume", raw, agent=Agent(name="bounded", prompt="test"))
    with pytest.raises(ValueError):
        validate({**raw, "provider_kwargs": {"thinking": "x" * 2100}})


def test_provider_options_exact_byte_boundary():
    from dendrux.types import _validate_provider_kwargs

    _validate_provider_kwargs({"x": "a" * 2040})
    with pytest.raises(ValueError):
        _validate_provider_kwargs({"x": "a" * 2041})


def test_provider_options_encoder_failure_is_safe_and_logged(caplog):
    import sys

    from dendrux.types import _validate_provider_kwargs

    digits = sys.get_int_max_str_digits()
    if not digits:
        pytest.skip("Interpreter does not cap integer decimal conversion")
    with caplog.at_level(logging.WARNING), pytest.raises(ValueError):
        _validate_provider_kwargs({"x": 10 ** (digits + 1)}, run_id="run-bounded")
    assert any(
        getattr(r, "code", None) == "provider_kwargs_invalid"
        and getattr(r, "run_id", None) == "run-bounded"
        for r in caplog.records
    )


@pytest.mark.parametrize("name", ["max_pending_calls", "max_rebuild_bytes", "max_rebuild_rows"])
def test_invalid_recovery_configuration_logs_stable_code(name, caplog):
    with caplog.at_level(logging.WARNING), pytest.raises(ValueError):
        Limits(**{name: 0})
    assert any(
        getattr(r, "code", None) == "invalid_limit" and hasattr(r, "run_id") for r in caplog.records
    )


def test_default_recovery_limits_preserve_legacy_governance_digest():
    from dataclasses import make_dataclass

    from dendrux.runtime.submit import governance_snapshot

    agent = Agent(prompt="test")
    current = agent._limits
    fields = ("max_tool_result_bytes", "max_input_bytes", "max_metadata_bytes", "max_notes_per_run")
    legacy = make_dataclass("LegacyLimits", [(f, int) for f in fields])
    agent._limits = legacy(*(getattr(current, f) for f in fields))
    previous = governance_snapshot(agent)
    agent._limits = current
    assert governance_snapshot(agent) == previous
    agent._limits = Limits(max_pending_calls=32)
    assert governance_snapshot(agent) != previous
