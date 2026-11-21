"""A rejected structured answer never puts the model's values on a persisted or logged surface."""

from __future__ import annotations

import logging
from typing import Any

import pytest
from pydantic import BaseModel, field_validator

from dendrux import Agent, SingleCall
from dendrux.llm.mock import MockLLM
from dendrux.types import (
    LLMResponse,
    ProviderCapabilities,
    StructuredOutputValidationError,
    UsageStats,
)
from tests._helpers.log_records import record_text

SENTINEL = "SENTINEL-7f3a91"


class Strict(BaseModel):
    count: int
    label: str

    @field_validator("label")
    @classmethod
    def _no_secret(cls, value: str) -> str:
        raise ValueError(f"rejected {value} {SENTINEL}-validator")


class Plain(BaseModel):
    count: int


class Forbid(BaseModel):
    model_config = {"extra": "forbid"}
    count: int


_CASES: dict[str, tuple[type[BaseModel], str]] = {
    "custom_validator": (Strict, f'{{"count": 1, "label": "{SENTINEL}-label"}}'),
    "type_error": (Plain, f'{{"count": "{SENTINEL}-count"}}'),
    "bad_json": (Plain, f"{{not json {SENTINEL}"),
    "extra_key": (Forbid, f'{{"count": 1, "{SENTINEL}-key": 2}}'),
}


def _agent(model: type[BaseModel], text: str, store: Any) -> Agent:
    llm = MockLLM([LLMResponse(text=text, usage=UsageStats(input_tokens=1, output_tokens=1))])
    llm.capabilities = ProviderCapabilities(supports_structured_output=True)
    return Agent(
        strict_tenancy=False,
        provider=llm,
        prompt="t",
        loop=SingleCall(),
        output_type=model,
        state_store=store,
    )


@pytest.mark.parametrize("case", sorted(_CASES))
async def test_model_values_stay_off_error_run_row_events_and_logs(
    case: str, store: Any, caplog: pytest.LogCaptureFixture
) -> None:
    model, text = _CASES[case]
    caplog.set_level(logging.DEBUG)
    with pytest.raises(StructuredOutputValidationError) as info:
        await _agent(model, text, store).run("hi")
    err = info.value

    assert SENTINEL not in str(err)
    assert SENTINEL not in err.validation_error
    assert err.output_type_name == model.__name__
    # In-memory access for the developer is kept.
    assert err.raw_response == text

    run_id = (await store.list_runs())[0].id
    record = await store.get_run(run_id)
    assert SENTINEL not in (record.error or "")
    for event in await store.get_run_events(run_id):
        if event.event_type.startswith("llm."):
            continue  # the recorded call is the model's own answer, by design
        assert SENTINEL not in repr(event.data), event.event_type
    ours = [r for r in caplog.records if r.name.startswith("dendrux")]
    assert not [r for r in ours if SENTINEL in record_text(r)]
    assert not [r for r in ours if r.exc_info and SENTINEL in str(r.exc_info[1])]


async def test_message_names_error_kinds_and_field_locations() -> None:
    model, text = _CASES["type_error"]
    llm = MockLLM([LLMResponse(text=text, usage=UsageStats())])
    llm.capabilities = ProviderCapabilities(supports_structured_output=True)
    from dendrux.llm.structured import structured_complete
    from dendrux.types import Message, Role

    with pytest.raises(StructuredOutputValidationError) as info:
        await structured_complete(llm, [Message(role=Role.USER, content="x")], model)
    assert "Plain" in str(info.value)
    assert "count" in info.value.validation_error
    assert "int_parsing" in info.value.validation_error
