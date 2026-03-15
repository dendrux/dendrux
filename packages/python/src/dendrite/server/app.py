"""Mountable FastAPI application for Dendrite.

Usage:
    from fastapi import FastAPI
    from dendrite.server import create_app, HostedAgentConfig

    app = FastAPI()
    dendrite_app = create_app(state_store=store, registry=registry)
    app.mount("/dendrite", dendrite_app)
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from dendrite.server.observer import CompositeObserver, ServerEvent, TransportObserver
from dendrite.server.schemas import (
    CreateRunRequest,
    CreateRunResponse,
    ResumeResponse,
    RunStatusResponse,
    SubmitInputRequest,
    SubmitToolResultsRequest,
)
from dendrite.server.tasks import RunTaskManager

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from dendrite.runtime.state import StateStore
    from dendrite.server.registry import AgentRegistry

logger = logging.getLogger(__name__)


def create_app(
    state_store: StateStore,
    registry: AgentRegistry,
    *,
    hmac_secret: str | None = None,
) -> FastAPI:
    """Create a mountable Dendrite FastAPI application.

    Args:
        state_store: Persistence backend for runs.
        registry: Agent configurations (HostedAgentConfig with factories).
        hmac_secret: HMAC secret for run-scoped tokens. None = auth disabled.
    """
    app = FastAPI(title="Dendrite", version="0.1.0a1")
    task_manager = RunTaskManager()
    # Per-run SSE queues. Cleaned up by:
    # - _run_agent finally block (non-paused terminal runs)
    # - SSE generator finally block (after client consumes terminal event)
    # - DELETE /runs endpoint (cancelled runs)
    # Edge case: paused runs abandoned without resume or cancel will leak.
    # A TTL-based eviction (Sprint 4) would address this.
    sse_queues: dict[str, asyncio.Queue[ServerEvent]] = {}

    @app.post("/runs", response_model=CreateRunResponse)
    async def create_run(req: CreateRunRequest) -> CreateRunResponse:
        """Start a new agent run in a background task."""
        try:
            config = registry.get(req.agent_name)
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e)) from None

        from dendrite.types import generate_ulid

        run_id = generate_ulid()

        # Create SSE queue for this run
        queue: asyncio.Queue[ServerEvent] = asyncio.Queue()
        sse_queues[run_id] = queue

        async def _run_agent() -> Any:
            """Background task — runs the agent with persistence + transport."""
            from dendrite.runtime.observer import PersistenceObserver
            from dendrite.tool import get_tool_def
            from dendrite.types import PauseState, RunStatus

            agent = config.agent
            provider = config.provider_factory()
            strategy = config.strategy_factory() if config.strategy_factory else None
            loop = config.loop_factory() if config.loop_factory else None
            redact = config.redact

            # Create persistence observer
            target_lookup = {}
            for fn in agent.tools:
                td = get_tool_def(fn)
                target_lookup[td.name] = td.target

            persistence_obs = PersistenceObserver(
                state_store,
                run_id,
                model=agent.model,
                provider_name=type(provider).__name__,
                target_lookup=target_lookup,
                redact=redact,
            )
            transport_obs = TransportObserver(queue)
            composite = CompositeObserver([persistence_obs, transport_obs])

            # Create run record
            redacted_input = redact(req.input) if redact else req.input
            await state_store.create_run(
                run_id,
                agent.name,
                input_data={"input": redacted_input},
                model=agent.model,
                strategy=type(strategy).__name__ if strategy else "NativeToolCalling",
                tenant_id=req.tenant_id,
            )

            await queue.put(ServerEvent(event="run.started", data={"run_id": run_id}))

            is_paused = False
            is_cancelled = False
            try:
                from dendrite.loops.react import ReActLoop
                from dendrite.strategies.native import NativeToolCalling

                resolved_strategy = strategy or NativeToolCalling()
                resolved_loop = loop or ReActLoop()

                result = await resolved_loop.run(
                    agent=agent,
                    provider=provider,
                    strategy=resolved_strategy,
                    user_input=req.input,
                    run_id=run_id,
                    observer=composite,
                )

                # Handle result
                if result.status in (
                    RunStatus.WAITING_CLIENT_TOOL,
                    RunStatus.WAITING_HUMAN_INPUT,
                ):
                    is_paused = True
                    pause_state: PauseState = result.meta["pause_state"]
                    await state_store.pause_run(
                        run_id,
                        status=result.status.value,
                        pause_data=pause_state.to_dict(),
                        iteration_count=result.iteration_count,
                    )
                    # Emit distinct pause events (D3)
                    if result.status == RunStatus.WAITING_CLIENT_TOOL:
                        pending = [
                            {
                                "tool_call_id": tc.id,
                                "tool_name": tc.name,
                                "params": tc.params,
                                "target": pause_state.pending_targets.get(tc.id, "client"),
                            }
                            for tc in pause_state.pending_tool_calls
                        ]
                        await queue.put(
                            ServerEvent(
                                event="run.paused.tool_needed",
                                data={"run_id": run_id, "pending_tool_calls": pending},
                            )
                        )
                    else:
                        await queue.put(
                            ServerEvent(
                                event="run.paused.input_needed",
                                data={"run_id": run_id, "question": result.answer},
                            )
                        )
                else:
                    redacted_answer = (
                        redact(result.answer) if redact and result.answer else result.answer
                    )
                    # Atomic conditional finalize: only if still RUNNING
                    # (prevents overwriting CANCELLED set by DELETE)
                    finalized = await state_store.finalize_run(
                        run_id,
                        status=result.status.value,
                        answer=redacted_answer,
                        iteration_count=result.iteration_count,
                        total_usage=result.usage,
                        expected_current_status="running",
                    )
                    if finalized:
                        # CAS winner — buffer + emit terminal event
                        task_manager.buffer_terminal_event(
                            run_id,
                            {
                                "event": "run.completed",
                                "data": {"run_id": run_id, "status": result.status.value},
                            },
                        )
                        await queue.put(
                            ServerEvent(
                                event="run.completed",
                                data={"run_id": run_id, "status": result.status.value},
                            )
                        )
                    else:
                        logger.info("Run %s already finalized (likely cancelled), skipping", run_id)

                return result

            except asyncio.CancelledError:
                # DELETE /runs already won the CAS to CANCELLED and is
                # responsible for emitting the terminal event + buffering.
                # We do nothing here — only the CAS winner emits.
                is_cancelled = True
                raise

            except Exception as exc:
                # Only the CAS winner emits the terminal event.
                error_won = False
                try:
                    redacted_err = redact(str(exc)) if redact else str(exc)
                    error_won = await state_store.finalize_run(
                        run_id,
                        status="error",
                        error=redacted_err,
                        total_usage=None,
                        expected_current_status="running",
                    )
                except Exception:
                    logger.warning("Failed to persist ERROR for run %s", run_id, exc_info=True)
                if error_won:
                    # CAS winner — buffer + emit terminal event
                    task_manager.buffer_terminal_event(
                        run_id,
                        {
                            "event": "run.error",
                            "data": {"run_id": run_id, "error": str(exc)[:200]},
                        },
                    )
                    await queue.put(
                        ServerEvent(
                            event="run.error",
                            data={"run_id": run_id, "error": str(exc)[:200]},
                        )
                    )
                raise

            finally:
                # Clean up SSE queue for terminal runs (not paused).
                # Paused runs keep the queue for resume SSE.
                # On cancellation, DELETE owns cleanup — don't touch the queue.
                if not is_paused and not is_cancelled:
                    sse_queues.pop(run_id, None)

        task_manager.spawn(run_id, _run_agent())

        return CreateRunResponse(run_id=run_id, status="running")

    @app.get("/runs/{run_id}", response_model=RunStatusResponse)
    async def get_run_status(run_id: str) -> RunStatusResponse:
        """Poll the current status of a run."""
        record = await state_store.get_run(run_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")

        pending = None
        if record.status in ("waiting_client_tool", "waiting_human_input"):
            raw = await state_store.get_pause_state(run_id)
            if raw and "pending_tool_calls" in raw:
                from dendrite.server.schemas import PendingToolCall

                targets_map = raw.get("pending_targets", {})
                pending = [
                    PendingToolCall(
                        tool_call_id=tc["id"],
                        tool_name=tc["name"],
                        params=tc.get("params"),
                        target=targets_map.get(tc["id"], "client"),
                    )
                    for tc in raw["pending_tool_calls"]
                ]

        return RunStatusResponse(
            run_id=record.id,
            status=record.status,
            answer=record.answer,
            error=record.error,
            iteration_count=record.iteration_count,
            pending_tool_calls=pending,
        )

    @app.get("/runs/{run_id}/events")
    async def stream_events(run_id: str) -> StreamingResponse:
        """SSE event stream for a run."""
        queue = sse_queues.get(run_id)

        async def _generate() -> AsyncGenerator[str, None]:
            is_final = False
            try:
                # If run already finished, send terminal event
                terminal = task_manager.get_terminal_event(run_id)
                if terminal is not None:
                    yield f"event: {terminal['event']}\ndata: {json.dumps(terminal['data'])}\n\n"
                    is_final = True
                    return

                if queue is None:
                    yield f"event: error\ndata: {json.dumps({'error': 'Run not found'})}\n\n"
                    return

                while True:
                    try:
                        event = await asyncio.wait_for(queue.get(), timeout=30.0)
                        yield f"event: {event.event}\ndata: {json.dumps(event.data)}\n\n"
                        # Terminal events end the stream
                        if event.event in (
                            "run.completed",
                            "run.error",
                            "run.cancelled",
                            "run.paused.tool_needed",
                            "run.paused.input_needed",
                        ):
                            is_final = event.event in (
                                "run.completed",
                                "run.error",
                                "run.cancelled",
                            )
                            return
                    except TimeoutError:
                        # Send keep-alive ping
                        yield ": keepalive\n\n"
            finally:
                # Clean up resources for final events (completed/error/cancelled)
                # Pause events keep the queue alive for resume SSE
                if is_final:
                    sse_queues.pop(run_id, None)
                    task_manager.cleanup(run_id)

        return StreamingResponse(
            _generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    @app.post("/runs/{run_id}/tool-results", response_model=ResumeResponse)
    async def submit_tool_results(run_id: str, req: SubmitToolResultsRequest) -> ResumeResponse:
        """Submit client tool results to resume a WAITING_CLIENT_TOOL run."""
        from dendrite.runtime.runner import resume as dendrite_resume
        from dendrite.types import ToolResult

        # Load config from pause state
        raw = await state_store.get_pause_state(run_id)
        if raw is None:
            raise HTTPException(status_code=404, detail=f"Run '{run_id}' has no pause state.")

        agent_name = raw.get("agent_name", "")
        try:
            config = registry.get(agent_name)
        except KeyError:
            raise HTTPException(
                status_code=404, detail=f"Agent '{agent_name}' not found in registry."
            ) from None

        tool_results = [
            ToolResult(
                name=tr.tool_name,
                call_id=tr.tool_call_id,
                payload=tr.result,
                success=tr.success,
                error=tr.error,
            )
            for tr in req.tool_results
        ]

        try:
            # Create fresh instances from factories
            provider = config.provider_factory()
            strategy = config.strategy_factory() if config.strategy_factory else None
            loop = config.loop_factory() if config.loop_factory else None

            # Create transport observer for SSE streaming during resume
            resume_queue = sse_queues.get(run_id)
            transport_obs = TransportObserver(resume_queue) if resume_queue else None

            result = await dendrite_resume(
                run_id,
                tool_results,
                state_store=state_store,
                agent=config.agent,
                provider=provider,
                strategy=strategy,
                loop=loop,
                redact=config.redact,
                extra_observer=transport_obs,
            )

            # Emit SSE based on CAS-winner invariant
            await _emit_resume_terminal(run_id, result)
            return ResumeResponse(run_id=run_id, status=result.status.value)

        except ValueError as e:
            raise HTTPException(status_code=409, detail=str(e)) from None
        except Exception as exc:
            # Non-ValueError exception — emit error event + buffer + clean
            await _handle_resume_error(run_id, exc)
            raise HTTPException(status_code=500, detail="Internal server error") from exc

    @app.post("/runs/{run_id}/input", response_model=ResumeResponse)
    async def submit_input(run_id: str, req: SubmitInputRequest) -> ResumeResponse:
        """Submit clarification input to resume a WAITING_HUMAN_INPUT run."""
        from dendrite.runtime.runner import resume_with_input as dendrite_resume_input

        raw = await state_store.get_pause_state(run_id)
        if raw is None:
            raise HTTPException(status_code=404, detail=f"Run '{run_id}' has no pause state.")

        agent_name = raw.get("agent_name", "")
        try:
            config = registry.get(agent_name)
        except KeyError:
            raise HTTPException(
                status_code=404, detail=f"Agent '{agent_name}' not found in registry."
            ) from None

        try:
            provider = config.provider_factory()
            strategy = config.strategy_factory() if config.strategy_factory else None
            loop = config.loop_factory() if config.loop_factory else None

            resume_queue = sse_queues.get(run_id)
            transport_obs = TransportObserver(resume_queue) if resume_queue else None

            result = await dendrite_resume_input(
                run_id,
                req.user_input,
                state_store=state_store,
                agent=config.agent,
                provider=provider,
                strategy=strategy,
                loop=loop,
                redact=config.redact,
                extra_observer=transport_obs,
            )

            # Emit SSE based on CAS-winner invariant
            await _emit_resume_terminal(run_id, result)
            return ResumeResponse(run_id=run_id, status=result.status.value)

        except ValueError as e:
            raise HTTPException(status_code=409, detail=str(e)) from None
        except Exception as exc:
            await _handle_resume_error(run_id, exc)
            raise HTTPException(status_code=500, detail="Internal server error") from exc

    # ------------------------------------------------------------------
    # Shared helpers for resume terminal-state handling
    # ------------------------------------------------------------------

    async def _emit_resume_terminal(run_id: str, result: Any) -> None:
        """Emit SSE event after a resume, respecting CAS-winner invariant."""
        from dendrite.types import PauseState, RunStatus

        queue = sse_queues.get(run_id)
        if result.status in (RunStatus.WAITING_CLIENT_TOOL, RunStatus.WAITING_HUMAN_INPUT):
            # Re-paused — keep queue alive, emit pause event
            if queue:
                ps: PauseState = result.meta["pause_state"]
                if result.status == RunStatus.WAITING_CLIENT_TOOL:
                    pending = [
                        {
                            "tool_call_id": tc.id,
                            "tool_name": tc.name,
                            "params": tc.params,
                            "target": ps.pending_targets.get(tc.id, "client"),
                        }
                        for tc in ps.pending_tool_calls
                    ]
                    await queue.put(
                        ServerEvent(
                            event="run.paused.tool_needed",
                            data={"run_id": run_id, "pending_tool_calls": pending},
                        )
                    )
                else:
                    await queue.put(
                        ServerEvent(
                            event="run.paused.input_needed",
                            data={"run_id": run_id, "question": result.answer},
                        )
                    )
        else:
            # Terminal — only emit if we won the CAS
            finalize_won = result.meta.get("_finalize_won", True)
            if finalize_won:
                terminal_data = {"run_id": run_id, "status": result.status.value}
                task_manager.buffer_terminal_event(
                    run_id,
                    {
                        "event": "run.completed",
                        "data": terminal_data,
                    },
                )
                if queue:
                    await queue.put(ServerEvent(event="run.completed", data=terminal_data))
                sse_queues.pop(run_id, None)
            else:
                logger.info(
                    "Run %s: resume lost terminal CAS (likely cancelled), skipping emit", run_id
                )

    async def _handle_resume_error(run_id: str, exc: Exception) -> None:
        """Handle non-ValueError exceptions from resume endpoints.

        Buffers run.error for late subscribers and cleans up the queue,
        but only if the error CAS was already won by _resume_core.
        """
        # _resume_core already attempted conditional finalize_run(status="error").
        # Check if error was persisted (run status is now "error")
        record = await state_store.get_run(run_id)
        if record and record.status == "error":
            # We own the error — buffer + emit
            error_data = {"run_id": run_id, "error": str(exc)[:200]}
            task_manager.buffer_terminal_event(run_id, {"event": "run.error", "data": error_data})
            queue = sse_queues.get(run_id)
            if queue:
                await queue.put(ServerEvent(event="run.error", data=error_data))
            sse_queues.pop(run_id, None)
        else:
            logger.info("Run %s error not persisted (likely cancelled), skipping emit", run_id)

    @app.delete("/runs/{run_id}")
    async def cancel_run(run_id: str) -> dict[str, str]:
        """Cancel a run. Works on paused/pending runs. Best-effort on running (D5)."""
        record = await state_store.get_run(run_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")

        from dendrite.types import RunStatus

        cancelable = {
            RunStatus.WAITING_CLIENT_TOOL.value,
            RunStatus.WAITING_HUMAN_INPUT.value,
            RunStatus.WAITING_APPROVAL.value,
            RunStatus.PENDING.value,
        }

        if record.status in cancelable:
            # Atomic CAS: only cancel if still in expected state
            did_cancel = await state_store.finalize_run(
                run_id,
                status=RunStatus.CANCELLED.value,
                expected_current_status=record.status,
            )
            if not did_cancel:
                raise HTTPException(status_code=409, detail="Run status changed concurrently.")
            # 1. Deliver cancel event BEFORE cleanup
            queue = sse_queues.get(run_id)
            if queue:
                await queue.put(ServerEvent(event="run.cancelled", data={"run_id": run_id}))
            # 2. Buffer terminal event for late subscribers
            task_manager.buffer_terminal_event(
                run_id,
                {
                    "event": "run.cancelled",
                    "data": {"run_id": run_id},
                },
            )
            # 3. Clean up queue (SSE generator's finally will handle its own ref)
            #    If no SSE client is connected, this prevents queue leak.
            #    If a client IS connected, it already holds a ref to the queue object.
            sse_queues.pop(run_id, None)
            return {"run_id": run_id, "status": "cancelled"}

        if record.status == RunStatus.RUNNING.value:
            # 1. Atomic CAS: transition running → cancelled (we are the winner)
            did_cancel = await state_store.finalize_run(
                run_id,
                status=RunStatus.CANCELLED.value,
                expected_current_status="running",
            )
            if not did_cancel:
                raise HTTPException(
                    status_code=409,
                    detail="Run already finalized before cancel took effect.",
                )
            # 2. Buffer terminal event for late subscribers (before any cleanup)
            task_manager.buffer_terminal_event(
                run_id,
                {
                    "event": "run.cancelled",
                    "data": {"run_id": run_id},
                },
            )
            # 3. Deliver to live SSE queue
            queue = sse_queues.get(run_id)
            if queue:
                await queue.put(ServerEvent(event="run.cancelled", data={"run_id": run_id}))
            # 4. Cancel the asyncio.Task (best-effort, D5/D8)
            #    _run_agent's CancelledError handler does nothing (we own terminal).
            #    _run_agent's finally skips queue cleanup (is_cancelled=True).
            task_manager.cancel(run_id)
            return {"run_id": run_id, "status": "cancelled"}
            raise HTTPException(
                status_code=409,
                detail="Run is RUNNING but no active task found. "
                "Full cooperative cancellation is planned for Sprint 4.",
            )

        raise HTTPException(
            status_code=409,
            detail=f"Run is in status '{record.status}' and cannot be cancelled.",
        )

    return app
