"""Characterize discovery, image lifetime and lifecycle seams before PR2 moves."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from dendrux.loops import _helpers
from dendrux.runtime import _run_events, _run_images, _run_tools, runner
from dendrux.types import RunResult, RunStatus, generate_ulid
from tests.unit.test_runner_move_contract import invoke
from tests.unit.test_runner_move_contract import store as store


@pytest.mark.parametrize("entry", ["run", "stream"])
async def test_discovery_fault_fires_before_provider_and_releases_edges(monkeypatch, entry):
    failure = RuntimeError("discovery refusal")
    fault = AsyncMock(side_effect=failure)
    release = Mock()
    monkeypatch.setattr(_run_tools, "_discover_runtime_tools", fault)
    monkeypatch.setattr(_run_images, "_release_memory_edges", release)
    if entry == "run":
        with pytest.raises(RuntimeError) as caught:
            await invoke(entry, None)
        assert caught.value is failure
    else:
        events = await invoke(entry, None)
        assert events[-1].error == "discovery refusal"
    fault.assert_awaited_once()
    release.assert_called_once()
    assert release.call_args.args[1] is None


async def test_discovery_diagnostic_failure_preserves_original_cause(monkeypatch):
    cause = RuntimeError("discovery refusal")
    wrapped = runner._MCPDiscoveryError("discovery refusal")
    wrapped.__cause__ = cause
    discover = AsyncMock(side_effect=wrapped)
    diagnostic = AsyncMock(side_effect=RuntimeError("diagnostic refusal"))
    monkeypatch.setattr(_run_tools, "_emit_init_events", discover)
    monkeypatch.setattr(_run_tools, "_emit_init_governance_event", diagnostic)
    with pytest.raises(RuntimeError) as caught:
        await runner._discover_runtime_tools(Mock(), None, None, generate_ulid(), Mock(), {})
    assert caught.value is cause
    discover.assert_awaited_once()
    diagnostic.assert_awaited_once()


@pytest.mark.parametrize("atomic", [False, True])
@pytest.mark.parametrize("won", [False, True])
async def test_finalize_cas_loser_never_requests_separate_event(atomic, won):
    store = SimpleNamespace(finalize_run=AsyncMock(return_value=won))
    if atomic:
        store.finalize_with_event = AsyncMock(return_value=7 if won else None)
    result = await runner._finalize_running_event(
        store, generate_ulid(), "success", "run.completed", {}
    )
    assert result == (won, won and not atomic)
    if atomic:
        store.finalize_with_event.assert_awaited_once()
        store.finalize_run.assert_not_awaited()
    else:
        store.finalize_run.assert_awaited_once()
        assert store.finalize_run.call_args.kwargs["expected_current_status"] == "running"


@pytest.mark.parametrize("cancelled", [False, True])
async def test_legacy_pause_cancel_cut_never_writes_after_cancel(monkeypatch, cancelled):
    store = SimpleNamespace(
        is_cancel_requested=AsyncMock(return_value=cancelled), pause_run=AsyncMock()
    )
    emit = AsyncMock()
    monkeypatch.setattr(_run_events, "_emit_event", emit)
    assert (
        await runner._pause_with_event(
            store,
            generate_ulid(),
            status="paused",
            pause_data={},
            event_data={},
            iteration_count=0,
            pii_mapping=None,
        )
        is not cancelled
    )
    assert store.pause_run.await_count == int(not cancelled)
    assert emit.await_count == int(not cancelled)


@pytest.mark.parametrize("stored", [False, True])
@pytest.mark.parametrize("fails", [False, True])
def test_memory_edge_release_is_owned_by_memory_run(stored, fails):
    release = Mock(side_effect=RuntimeError("release refusal") if fails else None)
    runtime = SimpleNamespace(catalog=SimpleNamespace(release_run=release))
    run_id = generate_ulid()
    runner._release_memory_edges(runtime, object() if stored else None, run_id)
    assert release.call_args_list == ([] if stored else [((run_id,), {})])


@pytest.mark.parametrize("driver", ["_record_finished", "_finish_fenced_lifecycle"])
async def test_finish_hooks_keep_order_and_fault_interception(monkeypatch, driver):
    calls = []

    async def record(*args):
        calls.append("record")

    async def notify(*args):
        calls.append("notify")

    record_fault = AsyncMock(side_effect=record)
    notify_fault = AsyncMock(side_effect=notify)
    monkeypatch.setattr(_helpers, "record_run_finished", record_fault)
    monkeypatch.setattr(_helpers, "notify_run_finished", notify_fault)
    result = RunResult(run_id=generate_ulid(), status=RunStatus.SUCCESS, answer="done")
    if driver == "_record_finished":
        await runner._record_finished(None, None, result.run_id, result)
    else:
        await runner._finish_fenced_lifecycle(result.run_id, result, None, None)
    assert calls == ["record", "notify"]
    record_fault.assert_awaited_once()
    notify_fault.assert_awaited_once()


@pytest.mark.parametrize("entry", ["run", "stream"])
async def test_caller_cancellation_keeps_cleanup_in_same_task(monkeypatch, entry):
    task = asyncio.current_task()
    fault = AsyncMock(side_effect=asyncio.CancelledError())
    released = []

    def release(*args):
        released.append(asyncio.current_task())

    monkeypatch.setattr(_run_tools, "_discover_runtime_tools", fault)
    monkeypatch.setattr(_run_images, "_release_memory_edges", release)
    with pytest.raises(asyncio.CancelledError):
        await invoke(entry, None)
    fault.assert_awaited_once()
    assert released == [task]


@pytest.mark.parametrize("path", ["resume", "resume_stream"])
async def test_image_restore_fault_fires_in_both_resume_modes(store, monkeypatch, path):
    from dendrux.runtime import _run_images
    from dendrux.types import LLMResponse
    from tests._helpers.runtime_paths import path_case
    from tests._helpers.scripted_provider import ScriptedLLM

    case = await path_case(store, None, path, ScriptedLLM([LLMResponse(text="done")]))
    failure = RuntimeError("image restore refusal")
    fault = Mock(side_effect=failure)
    with monkeypatch.context() as patcher:
        patcher.setattr(_run_images, "_resume_image_view", fault)
        if path == "resume":
            with pytest.raises(RuntimeError) as caught:
                await case.invoke()
            assert caught.value is failure
        else:
            result = await case.invoke()
            assert result.status == RunStatus.ERROR
            assert case.events[-1].error == "image restore refusal"
    fault.assert_called_once()
    assert (await store.get_run(case.run_id)).status == "error"
    assert case.run_id not in case.agent._task_manager._owned


def test_text_only_parts_do_not_register_blob_edges():
    from dendrux.types import TextPart

    runtime = SimpleNamespace(tenant_scope="tenant")
    assert _run_images._input_edge_kwargs(runtime, (TextPart(text="hello"),)) == {}


@pytest.mark.parametrize("runtime", [None, SimpleNamespace(catalog=None)])
async def test_live_edges_without_catalog_preserve_the_input(runtime):
    edges = [object()]
    assert await _run_images._live_edges(runtime, edges) is edges


async def test_seeded_images_refuse_a_loop_without_image_runtime():
    from dendrux.types import ImagePart, ImageRef, Message, Role

    ref = ImageRef(sha256="a" * 64, media_type="image/png", byte_length=3, width=1, height=1)
    seeded = [Message(role=Role.USER, content="", parts=(ImagePart(ref=ref),))]
    with pytest.raises(TypeError, match="cannot run a seeded prefix with images"):
        await _run_images._start_image_view(
            None,
            None,
            generate_ulid(),
            seeded=seeded,
            sources=[],
            single_full=False,
            input_images=False,
        )


def test_runtime_single_call_override_refuses_local_tools():
    from dendrux.loops.single import SingleCall

    agent = SimpleNamespace(name="move", tools=[lambda: None], _tool_sources=[])
    with pytest.raises(ValueError, match="SingleCall agents cannot have tools"):
        _run_tools._validate_loop_skill_compat(agent, SingleCall())


@pytest.mark.parametrize("method", ["run", "run_stream"])
@pytest.mark.parametrize("cut", ["missing_run", "missing_pause", "mismatched_calls"])
async def test_image_resume_preflight_refuses_before_ingestion(monkeypatch, method, cut):
    from dendrux.errors import RunNotFoundError
    from dendrux.loops.react import ReActLoop

    record = (
        None
        if cut == "missing_run"
        else SimpleNamespace(status=RunStatus.WAITING_CLIENT_TOOL.value, tenant_id=None)
    )
    pause = None if cut == "missing_pause" else {"pending_tool_calls": [{"id": "pending"}]}
    store = SimpleNamespace(
        get_run=AsyncMock(return_value=record), get_pause_state=AsyncMock(return_value=pause)
    )
    ingest = Mock(side_effect=AssertionError("ingestion must not start"))
    monkeypatch.setattr(_run_images, "_image_runtime", ingest)
    with pytest.raises(RunNotFoundError if cut == "missing_run" else ValueError):
        await _run_images._ingest_resume_images(
            SimpleNamespace(loop=None),
            Mock(),
            store,
            generate_ulid(),
            loop=ReActLoop(),
            tool_results=[],
            input_items=(),
            loop_method=method,
        )
    store.get_run.assert_awaited_once()
    assert store.get_pause_state.await_count == int(cut != "missing_run")
    ingest.assert_not_called()
