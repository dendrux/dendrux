"""Governance observations precede the terminal lifecycle event."""

from dendrux.runtime import runner
from dendrux.types import RunResult, RunStatus
from tests.unit.test_runner_failure_save import store as store


async def test_unmapped_observation_precedes_terminal(store):
    await store.create_run("r", "test")
    await runner._persist_loop_outcome(
        state_store=store,
        run_id="r",
        result=RunResult(
            run_id="r",
            status=RunStatus.SUCCESS,
            answer="<<PERSON_99>>",
            meta={"pii_mapping": {"<<PERSON_1>>": "example"}},
        ),
    )
    events = await store.get_run_events("r", limit=10)
    assert [e.event_type for e in events] == ["guardrail.unmapped_placeholder", "run.completed"]
