"""Run the existing consumer-task ownership probe on H5's real Postgres path."""

import pytest

from dendrux.loops import ReActLoop, SingleCall
from tests.integration.test_h0_stream_ownership import check_consumer_context


@pytest.mark.parametrize("engine", ["postgres"], indirect=True)
@pytest.mark.parametrize(
    "entry,loop_type", [("stream", ReActLoop), ("stream", SingleCall), ("resume_stream", ReActLoop)]
)
@pytest.mark.parametrize("ending", ["exhaust", "close", "error", "cancel"])
@pytest.mark.parametrize("storage_format", [1, 2])
@pytest.mark.parametrize("caller", ["top_level", "delegated"])
async def test_consumer_context_on_postgres(
    store, engine, entry, loop_type, ending, storage_format, caller
):
    await check_consumer_context(store, engine, entry, loop_type, ending, storage_format, caller)


@pytest.mark.parametrize("engine", ["postgres"], indirect=True)
@pytest.mark.parametrize("entry", ["stream", "resume_stream"])
@pytest.mark.parametrize("storage_format", [1, 2])
@pytest.mark.parametrize("caller", ["top_level", "delegated"])
async def test_tool_call_on_stream_context_on_postgres(
    store, engine, entry, storage_format, caller
):
    await check_consumer_context(store, engine, entry, ReActLoop, "tool", storage_format, caller)
