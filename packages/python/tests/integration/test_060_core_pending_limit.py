"""The assistant batch cap is applied before any side effect, on both paths."""

import pytest
from sqlalchemy import update

from dendrux.agent import Agent
from dendrux.db.models import DendruxSetting
from dendrux.llm.mock import MockLLM
from dendrux.runtime.state import ClaimResult, claimed_store
from dendrux.tool import tool
from dendrux.types import Limits, LLMResponse, RunStatus, ToolCall


def test_claim_view_retains_format_and_checkpoint():
    from unittest.mock import Mock

    token = ClaimResult(4, 2, {"v": 1}, {"submission_id": "s"})
    view = claimed_store(Mock(), "run", token)
    assert view.storage_format == 2
    assert view.loop_state == token.loop_state
    assert view.submission == token.submission
    assert view.claim_epoch == 4


def test_rebinding_epoch_keeps_token_only_for_same_run():
    from unittest.mock import Mock

    store = Mock()
    token = ClaimResult(4, 2, {"v": 1}, None)
    view = claimed_store(store, "run", token)
    assert claimed_store(view, "run", 4) is view
    other = claimed_store(view, "other", 4)
    assert other is not view
    assert other.storage_format == 1
    assert other.loop_state is None


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize(
    "storage_format,count,cap", [(2, 65, 64), (2, 32, 64), (2, 5, 4), (1, 5, 4), (0, 5, 4)]
)
async def test_pending_limit_before_execution(
    store, engine, streaming, storage_format, count, cap, caplog
):
    async with engine.begin() as conn:
        await conn.execute(
            update(DendruxSetting)
            .where(DendruxSetting.key == "storage_format")
            .values(value=str(storage_format or 1))
        )
    executed = []

    @tool()
    async def echo(value: int) -> int:
        """Echo the number."""
        executed.append(value)
        return value

    provider = MockLLM(
        [
            LLMResponse(
                tool_calls=[ToolCall(name="echo", params={"value": i}) for i in range(count)]
            ),
            LLMResponse(text="done"),
        ]
    )
    agent = Agent(
        name="bounded",
        prompt="test",
        provider=provider,
        state_store=store if storage_format else None,
        tools=[echo],
        limits=Limits(max_pending_calls=cap),
        strict_tenancy=False,
    )
    if streaming:
        stream = agent.stream("go")
        async for _ in stream:
            pass
        result = stream.result
    else:
        result = await agent.run("go")
    if storage_format == 2 and count > cap:
        assert result.status == RunStatus.ERROR
        assert result.meta["failure_reason"] == "pending_limit"
        assert executed == []
        assert await store.get_tool_calls(result.run_id) == []
        assert (await store.get_run(result.run_id)).failure_reason == "pending_limit"
        events = await store.get_run_events(result.run_id)
        assert events[-1].event_type == "run.error"
        assert not any(e.event_type == "tool.started" for e in events)
        assert any(
            getattr(r, "code", None) == "pending_limit"
            and getattr(r, "run_id", None) == result.run_id
            for r in caplog.records
        )
    else:
        assert result.status == RunStatus.SUCCESS
        assert sorted(executed) == list(range(count))
