"""Bridge — mountable HTTP transport for paused-run interaction.

Provides ``bridge(agent)`` which returns a FastAPI sub-app with endpoints
for submitting tool results, clarification input, SSE streaming, polling,
and cancellation. The bridge is a **paused-run interaction layer**, not a
full orchestration layer — the developer starts runs via ``agent.run()``
in their own code.

Usage::

    from dendrite import Agent, bridge
    from dendrite.llm import AnthropicProvider

    agent = Agent(
        provider=AnthropicProvider(model="claude-sonnet-4-6"),
        database_url="sqlite+aiosqlite:///my.db",
        prompt="You are a spreadsheet analyst.",
        tools=[lookup_price, read_excel_range],
    )

    transport = bridge(agent)
    app.mount("/dendrite", transport)

Known gaps (G2):
  - No ``POST /runs`` — developer starts runs via ``agent.run()``.
    Initial-run SSE and cancellation are not bridge-managed.
  - No token-level streaming — SSE delivers orchestration events
    (step, tool call, completion), not LLM token deltas.
  - Crash-after-claim: if the process dies after ``submit_and_claim``
    succeeds but before the resume loop completes, the run is stuck
    in RUNNING. Recovery requires a stale-run sweep (post-G2).
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from dendrite.auth import extract_bearer_token, generate_run_token, verify_run_token
from dendrite.bridge.observer import ServerEvent, TransportObserver
from dendrite.bridge.tasks import RunTaskManager

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from dendrite.agent import Agent
    from dendrite.runtime.state import StateStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


class ToolResultItem(BaseModel):
    """A single tool result submitted by the client."""

    tool_call_id: str
    tool_name: str
    result: str  # JSON string payload
    success: bool = True
    error: str | None = None
    duration_ms: int = 0


class SubmitToolResultsRequest(BaseModel):
    tool_results: list[ToolResultItem]


class SubmitInputRequest(BaseModel):
    user_input: str


# ---------------------------------------------------------------------------
# Bridge factory
# ---------------------------------------------------------------------------


def bridge(
    agent: Agent,
    *,
    secret: str | None = None,
    allow_insecure_dev_mode: bool = False,
) -> FastAPI:
    """Create a mountable FastAPI sub-app for paused-run interaction.

    The bridge provides 5 endpoints — all scoped to existing run IDs:

    - ``POST /runs/{id}/tool-results`` — persist-first tool result handover
    - ``POST /runs/{id}/input`` — persist-first clarification handover
    - ``GET  /runs/{id}/events`` — SSE stream (snapshot + live resume events)
    - ``GET  /runs/{id}`` — poll current run status
    - ``DELETE /runs/{id}`` — cancel a run (kills task + CAS-finalizes)

    Args:
        agent: The agent instance. Must have persistence configured
            (``database_url`` or ``state_store``).
        secret: HMAC secret for run-scoped auth tokens. When provided,
            all endpoints require a valid ``Authorization: Bearer drn_...``
            header.
        allow_insecure_dev_mode: If True and no secret provided, auth is
            disabled. If False and no secret, raises ValueError (fail-closed).

    Returns:
        A FastAPI application to be mounted on your server.

    Raises:
        ValueError: If agent has no persistence configured, or if auth
            is required but no secret provided.
    """
    if not allow_insecure_dev_mode and secret is None:
        raise ValueError(
            "bridge() requires a secret for auth, or pass "
            "allow_insecure_dev_mode=True for local development."
        )

    if agent.provider is None:
        raise ValueError(
            "bridge() requires an agent with a provider configured. "
            "Pass provider= to the Agent constructor."
        )

    if agent._database_url is None and agent._state_store is None:
        raise ValueError(
            "bridge() requires persistence on the agent. "
            "Pass database_url or state_store to the Agent constructor."
        )

    # Capture provider reference (validated non-None above)
    provider = agent.provider

    app = FastAPI(title="Dendrite Bridge")
    task_manager = RunTaskManager()
    sse_queues: dict[str, asyncio.Queue[ServerEvent]] = {}

    # ------------------------------------------------------------------
    # Auth-header stripping middleware
    # ------------------------------------------------------------------
    # Unconditionally strip Authorization header from the ASGI scope after
    # extracting it into request.state. Prevents bearer tokens from leaking
    # into downstream middleware, logging, or error reporters.

    @app.middleware("http")
    async def _strip_auth_header(request: Request, call_next: Any) -> Any:
        auth_value: str | None = None
        stripped: list[tuple[bytes, bytes]] = []
        for k, v in request.scope.get("headers", []):
            if k == b"authorization":
                auth_value = v.decode()
            else:
                stripped.append((k, v))
        request.scope["headers"] = stripped
        request.state.auth_header = auth_value
        return await call_next(request)

    # ------------------------------------------------------------------
    # Auth helper
    # ------------------------------------------------------------------

    def _require_auth(request: Request, run_id: str) -> None:
        """Verify run-scoped HMAC token if auth is enabled."""
        if secret is None:
            return
        token = extract_bearer_token(getattr(request.state, "auth_header", None))
        if token is None or not verify_run_token(run_id, token, secret):
            raise HTTPException(status_code=401, detail="Invalid or missing auth token.")

    # ------------------------------------------------------------------
    # Resolve state store (lazy, cached on first call)
    # ------------------------------------------------------------------

    _cached_store: dict[str, StateStore] = {}

    async def _get_store() -> StateStore:
        if "store" in _cached_store:
            return _cached_store["store"]
        store = await agent._resolve_state_store()
        if store is None:  # pragma: no cover — guarded by eager check above
            raise ValueError("bridge() requires persistence on the agent.")
        _cached_store["store"] = store
        return store

    # ------------------------------------------------------------------
    # Shared ack handshake helper
    # ------------------------------------------------------------------

    async def _spawn_resume_with_ack(
        run_id: str,
        store: StateStore,
        *,
        expected_status: str,
        submitted_data: dict[str, Any],
    ) -> bool:
        """Spawn a background resume task with ack handshake.

        1. Register SSE queue (ordering guarantee)
        2. Spawn task that calls submit_and_claim
        3. Task signals back via Future: True=won, False=lost
        4. If won, task continues with resume_claimed()

        Returns True if the claim was won, False otherwise.
        Raises on internal errors (propagated as 500).
        """
        ack: asyncio.Future[bool] = asyncio.get_running_loop().create_future()

        # Reuse existing SSE queue if a client is already connected,
        # otherwise create a new one. Must happen BEFORE task starts
        # (ordering guarantee for snapshot→live transition).
        if run_id not in sse_queues:
            sse_queues[run_id] = asyncio.Queue()
        queue = sse_queues[run_id]

        async def _resume_task() -> None:
            transport_obs = TransportObserver(queue)
            try:
                won = await store.submit_and_claim(
                    run_id,
                    expected_status=expected_status,
                    submitted_data=submitted_data,
                )
                if not won:
                    ack.set_result(False)
                    return

                ack.set_result(True)

                from dendrite.runtime.runner import resume_claimed
                from dendrite.types import RunStatus

                result = await resume_claimed(
                    run_id,
                    state_store=store,
                    agent=agent,
                    provider=provider,
                    redact=agent._redact,
                    extra_observer=transport_obs,
                )

                # Emit SSE event based on outcome.
                # CAS winner rule: only the code path that won finalize_run
                # emits the terminal event. If cancel won, we emit nothing.
                if result.status in (
                    RunStatus.WAITING_CLIENT_TOOL,
                    RunStatus.WAITING_HUMAN_INPUT,
                ):
                    # Re-paused — not terminal. Client stays connected.
                    pause_state = result.meta.get("pause_state")
                    pending = []
                    if pause_state:
                        pending = [
                            {"id": tc.id, "name": tc.name} for tc in pause_state.pending_tool_calls
                        ]
                    await queue.put(
                        ServerEvent(
                            event="run.paused",
                            data={
                                "status": result.status.value,
                                "run_id": run_id,
                                "pending_tool_calls": pending,
                            },
                        )
                    )
                elif result.meta.get("_finalize_won"):
                    # We won the finalize CAS — emit terminal event.
                    terminal = ServerEvent(
                        event="run.completed",
                        data={"status": result.status.value, "run_id": run_id},
                    )
                    await queue.put(terminal)
                    task_manager.buffer_terminal_event(
                        run_id, {"event": terminal.event, "data": terminal.data}
                    )
                # else: cancel won the CAS race — cancel endpoint owns
                # the terminal event. We emit nothing.

            except Exception as exc:
                if not ack.done():
                    ack.set_exception(exc)
                else:
                    # CAS succeeded, resume failed — error already persisted by runner
                    logger.exception("Resume failed for run %s after claim", run_id)
                    error_event = ServerEvent(
                        event="run.error",
                        data={"run_id": run_id, "error": str(exc)[:500]},
                    )
                    await queue.put(error_event)
                    task_manager.buffer_terminal_event(
                        run_id, {"event": error_event.event, "data": error_event.data}
                    )

        task_manager.spawn(run_id, _resume_task())

        try:
            return await ack
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)[:500]) from exc

    # ------------------------------------------------------------------
    # POST /runs/{run_id}/tool-results
    # ------------------------------------------------------------------

    @app.post("/runs/{run_id}/tool-results")
    async def submit_tool_results(
        run_id: str, req: SubmitToolResultsRequest, request: Request
    ) -> JSONResponse:
        """Persist-first tool result handover with ack handshake."""
        _require_auth(request, run_id)
        store = await _get_store()

        results_dicts = [
            {
                "name": tr.tool_name,
                "call_id": tr.tool_call_id,
                "payload": tr.result,
                "success": tr.success,
                "error": tr.error,
                "duration_ms": tr.duration_ms,
            }
            for tr in req.tool_results
        ]

        won = await _spawn_resume_with_ack(
            run_id,
            store,
            expected_status="waiting_client_tool",
            submitted_data={"submitted_tool_results": results_dicts},
        )

        if not won:
            raise HTTPException(
                status_code=409,
                detail=f"Run '{run_id}' tool results already submitted "
                "or run not in expected state.",
            )

        response_data: dict[str, Any] = {"run_id": run_id, "status": "accepted"}
        if secret:
            response_data["token"] = generate_run_token(run_id, secret)
        return JSONResponse(content=response_data)

    # ------------------------------------------------------------------
    # POST /runs/{run_id}/input
    # ------------------------------------------------------------------

    @app.post("/runs/{run_id}/input")
    async def submit_input(run_id: str, req: SubmitInputRequest, request: Request) -> JSONResponse:
        """Persist-first clarification handover with ack handshake."""
        _require_auth(request, run_id)
        store = await _get_store()

        won = await _spawn_resume_with_ack(
            run_id,
            store,
            expected_status="waiting_human_input",
            submitted_data={"submitted_user_input": req.user_input},
        )

        if not won:
            raise HTTPException(
                status_code=409,
                detail=f"Run '{run_id}' input already submitted or run not in expected state.",
            )

        response_data: dict[str, Any] = {"run_id": run_id, "status": "accepted"}
        if secret:
            response_data["token"] = generate_run_token(run_id, secret)
        return JSONResponse(content=response_data)

    # ------------------------------------------------------------------
    # GET /runs/{run_id}/events  (SSE)
    # ------------------------------------------------------------------

    @app.get("/runs/{run_id}/events")
    async def stream_events(run_id: str, request: Request) -> StreamingResponse:
        """SSE stream: snapshot on connect, then live resume events.

        Ordering guarantee: the SSE queue is registered in the bridge's
        queue map BEFORE the snapshot is built, so events emitted between
        snapshot and first live read are captured in the queue.
        """
        _require_auth(request, run_id)
        store = await _get_store()

        # Ensure queue exists (may already exist from tool-results/input)
        if run_id not in sse_queues:
            sse_queues[run_id] = asyncio.Queue()
        queue = sse_queues[run_id]

        def _sse(event: str, data: Any) -> str:
            """Format a Server-Sent Event."""
            payload = json.dumps(data) if not isinstance(data, str) else data
            return f"event: {event}\ndata: {payload}\n\n"

        async def _generate() -> AsyncGenerator[str, None]:
            try:
                # 1. Snapshot: current run state from DB
                run_record = await store.get_run(run_id)
                if run_record is None:
                    yield _sse("error", {"detail": "Run not found"})
                    return

                snapshot: dict[str, Any] = {
                    "run_id": run_id,
                    "status": run_record.status,
                    "iteration_count": run_record.iteration_count,
                }

                # If paused, include pending tool calls from pause_data
                if run_record.status in (
                    "waiting_client_tool",
                    "waiting_human_input",
                ):
                    pause_data = await store.get_pause_state(run_id)
                    if pause_data:
                        snapshot["pending_tool_calls"] = [
                            {"id": tc["id"], "name": tc["name"]}
                            for tc in pause_data.get("pending_tool_calls", [])
                        ]

                yield _sse("snapshot", snapshot)

                # 2. Check for buffered terminal event (late subscriber)
                terminal = task_manager.get_terminal_event(run_id)
                if terminal:
                    yield _sse(terminal["event"], terminal["data"])
                    return

                # 3. Stream live events from queue
                while True:
                    try:
                        event = await asyncio.wait_for(
                            queue.get(),
                            timeout=30.0,
                        )
                        yield _sse(event.event, event.data)

                        # Terminal events end the stream
                        if event.event in (
                            "run.completed",
                            "run.done",
                            "run.error",
                            "run.cancelled",
                        ):
                            return
                    except TimeoutError:
                        yield _sse("ping", {})

            finally:
                sse_queues.pop(run_id, None)

        return StreamingResponse(
            _generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    # ------------------------------------------------------------------
    # GET /runs/{run_id}  (poll)
    # ------------------------------------------------------------------

    @app.get("/runs/{run_id}")
    async def get_run_status(run_id: str, request: Request) -> JSONResponse:
        """Poll current run status from DB."""
        _require_auth(request, run_id)
        store = await _get_store()

        run_record = await store.get_run(run_id)
        if run_record is None:
            raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")

        data: dict[str, Any] = {
            "run_id": run_id,
            "status": run_record.status,
            "iteration_count": run_record.iteration_count,
            "answer": run_record.answer,
            "error": run_record.error,
        }

        # Include pending tool calls if paused
        if run_record.status in ("waiting_client_tool", "waiting_human_input"):
            pause_data = await store.get_pause_state(run_id)
            if pause_data:
                data["pending_tool_calls"] = [
                    {
                        "id": tc["id"],
                        "name": tc["name"],
                        "params": tc.get("params", {}),
                    }
                    for tc in pause_data.get("pending_tool_calls", [])
                ]

        return JSONResponse(content=data)

    # ------------------------------------------------------------------
    # DELETE /runs/{run_id}  (cancel)
    # ------------------------------------------------------------------

    @app.delete("/runs/{run_id}")
    async def cancel_run(run_id: str, request: Request) -> JSONResponse:
        """Cancel a run: kill the background task + CAS-finalize in DB.

        Cancel/resume race: if a resume task has already started, the CAS
        winner (cancel or resume) determines who emits the terminal event.
        The loser's finalize_run returns False, so no duplicate events.
        """
        _require_auth(request, run_id)
        store = await _get_store()

        # 1. Kill the background task (best-effort)
        task_manager.cancel(run_id)

        # 2. CAS-finalize: only succeeds if status is still running/waiting
        #    Try multiple expected statuses — run may be in various states.
        won = False
        for expected in ("running", "waiting_client_tool", "waiting_human_input"):
            won = await store.finalize_run(
                run_id,
                status="cancelled",
                expected_current_status=expected,
            )
            if won:
                break

        if not won:
            # Run was already finalized (completed, error, or already cancelled)
            run_record = await store.get_run(run_id)
            if run_record is None:
                raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")
            return JSONResponse(
                content={"run_id": run_id, "status": run_record.status, "cancelled": False}
            )

        # 3. Emit terminal event for SSE subscribers
        cancel_event = ServerEvent(
            event="run.cancelled", data={"run_id": run_id, "status": "cancelled"}
        )
        queue = sse_queues.get(run_id)
        if queue:
            await queue.put(cancel_event)
        task_manager.buffer_terminal_event(
            run_id, {"event": cancel_event.event, "data": cancel_event.data}
        )

        return JSONResponse(content={"run_id": run_id, "status": "cancelled", "cancelled": True})

    return app
