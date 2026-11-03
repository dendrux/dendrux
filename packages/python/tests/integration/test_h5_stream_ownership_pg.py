"""Run the existing consumer-task ownership probe on H5's real Postgres path."""

import pytest

from dendrux.loops import ReActLoop, SingleCall
from tests.integration.test_h0_stream_ownership import (
    test_provider_pull_yield_and_close_preserve_consumer_context as check_consumer_context,
)


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
