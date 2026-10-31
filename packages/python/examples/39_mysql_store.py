"""MySQL / MariaDB store: pause on a client tool, resume, finish.

A client-tool run is paused in MySQLStateStore (or MariaDB), then resumed by
submitting the tool result. A mock provider drives it, so no API key is needed.
The example skips cleanly when no server is configured or reachable.

Setup (once), with a MySQL 8.4+ or MariaDB 10.11+ database you can write to:

    pip install "dendrux[mysql]"
    export DENDRUX_MYSQL_URL=mysql+asyncmy://user:password@127.0.0.1:3306/dendrux_example
    DENDRUX_DATABASE_URL="$DENDRUX_MYSQL_URL" dendrux db migrate

Run with:
    python examples/39_mysql_store.py

For the repository's disposable test servers (make test-dbs-up), use
mysql+asyncmy://root:dendrux_test@127.0.0.1:53306/dendrux_example_test.
"""

from __future__ import annotations

import asyncio
import os
import sys

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import create_async_engine

from dendrux import Agent, tool
from dendrux.errors import PersistenceError
from dendrux.llm.mock import MockLLM
from dendrux.store import MySQLStateStore, RunStore
from dendrux.types import LLMResponse, RunStatus, ToolCall, ToolResult

URL_ENV = "DENDRUX_MYSQL_URL"
TENANT = "example-tenant"


@tool(target="client")
async def read_selection() -> str:
    """Read the cells the user has selected in the spreadsheet."""
    return ""


async def main() -> int:
    url = os.environ.get(URL_ENV)
    if not url:
        print(f"Skipped: set {URL_ENV} to a MySQL or MariaDB URL (see the top of this file).")
        return 0

    # One engine per process, created by you. MySQLStateStore requires these two options.
    engine = create_async_engine(
        url,
        isolation_level="READ COMMITTED",
        connect_args={"init_command": "SET time_zone='+00:00'", "charset": "utf8mb4"},
        pool_size=5,
        max_overflow=0,
        pool_pre_ping=True,
    )
    state = MySQLStateStore(engine)
    reads = RunStore.from_engine(engine)
    provider = MockLLM(
        [
            LLMResponse(tool_calls=[ToolCall("read_selection", {})]),
            LLMResponse(text="The selection sums to 42."),
        ]
    )
    try:
        async with Agent(
            name="sheet-helper",
            prompt="Answer questions about the user's selection.",
            provider=provider,
            tools=[read_selection],
            state_store=state,
        ) as agent:
            paused = await agent.run("What is the selection total?", tenant_id=TENANT)
            assert paused.status == RunStatus.WAITING_CLIENT_TOOL
            call = paused.pending.tool_calls[0]
            print(f"Paused run {paused.run_id}: waiting for the client tool {call.name!r}.")

            # Any worker sharing the database could do this part, in another process.
            finished = await agent.submit_tool_results(
                paused.run_id,
                [ToolResult(call.name, call.id, '"6 + 36"')],
                tenant_id=TENANT,
                expected_pause_id=paused.pause_id,
            )
            print(f"Finished: {finished.status.value}: {finished.answer}")

            detail = await reads.get_run(paused.run_id, tenant_id=TENANT)
            assert detail is not None
            print(f"Stored as {detail.status} with {detail.iteration_count} iterations.")
    except (SQLAlchemyError, PersistenceError) as exc:
        print(
            f"Skipped: cannot use the database ({type(exc).__name__}). Check {URL_ENV} and run "
            "'dendrux db migrate' against it."
        )
        return 0
    finally:
        await reads.close()
        await engine.dispose()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
