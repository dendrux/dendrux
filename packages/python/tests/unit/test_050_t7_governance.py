"""Governance fingerprints carry configuration identity across process restarts."""

import json
import os
import subprocess
import sys

from dendrux import Agent, Limits
from dendrux.guardrails import PII
from dendrux.runtime.submit import governance_snapshot


def test_digest_is_stable_across_fresh_processes_and_hash_seeds():
    code = """
import json
from dendrux import Agent
from dendrux.guardrails import PII
from dendrux.runtime.submit import governance_snapshot
agent = Agent(name="stable", prompt="p", guardrails=[PII()], strict_tenancy=False)
print(json.dumps(governance_snapshot(agent), sort_keys=True))
"""
    outputs = [
        subprocess.check_output(
            [sys.executable, "-c", code], env={**os.environ, "PYTHONHASHSEED": seed}, text=True
        )
        for seed in ["1", "2"]
    ]
    assert outputs[0] == outputs[1]
    assert len(json.loads(outputs[0])["digest"]) == 64


def test_custom_fingerprint_and_entity_set_are_stable_and_detect_changes():
    class Custom:
        action = "block"
        entities = {"b", "a"}
        version = "one"

        def governance_fingerprint(self):
            return {"version": self.version}

    one, two = Custom(), Custom()
    first = Agent(name="a", prompt="p", guardrails=[one], strict_tenancy=False)
    second = Agent(name="a", prompt="p", guardrails=[two], strict_tenancy=False)
    assert governance_snapshot(first) == governance_snapshot(second)
    two.version = "two"
    assert governance_snapshot(first) != governance_snapshot(second)


def test_guardrail_action_entities_restore_and_limits_are_part_of_digest():
    a = Agent(
        name="a", prompt="p", guardrails=[PII(action="redact", restore="all")], strict_tenancy=False
    )
    b = Agent(
        name="a",
        prompt="p",
        guardrails=[PII(action="redact", restore="allowlisted")],
        strict_tenancy=False,
    )
    c = Agent(
        name="a",
        prompt="p",
        guardrails=[PII(action="redact", restore="all")],
        limits=Limits(max_notes_per_run=1),
        strict_tenancy=False,
    )
    assert len({governance_snapshot(x)["digest"] for x in [a, b, c]}) == 3


async def test_ephemeral_batch_and_stream_pauses_have_generation_ids():
    from dendrux.llm.mock import MockLLM
    from dendrux.types import LLMResponse, ToolCall
    from tests.integration.test_050_t7_integrity import client_read, consume

    for streaming in [False, True]:
        a = Agent(
            name="ephemeral",
            prompt="p",
            tools=[client_read],
            strict_tenancy=False,
            provider=MockLLM([LLMResponse(tool_calls=[ToolCall("client_read", {})])]),
        )
        if streaming:
            result, _ = await consume(a.stream("go"))
        else:
            result = await a.run("go")
        assert result.pause_id and len(result.pause_id) == 26
        assert result.meta["pause_state"].pause_id == result.pause_id
