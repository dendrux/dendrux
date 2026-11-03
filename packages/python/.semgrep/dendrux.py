# Fixture for .semgrep/dendrux.yml: lines marked as matches must match, lines marked ok must not.
# Run: semgrep --test .semgrep   (see tools/QUALITY.md)
import logging

logger = logging.getLogger(__name__)


class SQLStateStore(Base):
    async def _write_tx(self, session):
        # The rule matches this commit too: the ratchet exempts exactly SQLStateStore._write_tx
        # (and helpers nested in it) through allow_in, which tests/tools covers.
        # ruleid: session-commit-outside-write-tx
        await session.commit()

    async def append(self, session):
        # ruleid: session-commit-outside-write-tx
        await session.commit()

    def sync_path(self, conn):
        # ruleid: session-commit-outside-write-tx
        conn.commit()

    async def good(self, session):
        # ok: session-commit-outside-write-tx
        await self._write_tx(session)

    async def other_method(self, session):
        async def _write_tx(session):
            # ruleid: session-commit-outside-write-tx
            await session.commit()

        await _write_tx(session)


class OtherStore:
    async def _write_tx(self, session):
        # ruleid: session-commit-outside-write-tx
        await session.commit()


async def _write_tx(session):
    # ruleid: session-commit-outside-write-tx
    await session.commit()


async def callers(extra, image_kwargs):
    # ruleid: process-tool-calls-without-tenant
    await _process_tool_calls(step=1, run_id="r")
    # ruleid: process-tool-calls-without-tenant
    await _process_tool_calls(step=1, **{})
    # ruleid: process-tool-calls-without-tenant
    await _process_tool_calls(step=1, **extra)
    # ruleid: process-tool-calls-without-tenant
    await _process_tool_calls(step=1, **image_kwargs(extra))
    # ok: process-tool-calls-without-tenant
    await _process_tool_calls(step=1, run_id="r", tenant_id=None)
    # ok: process-tool-calls-without-tenant
    await _process_tool_calls(step=1, tenant_id=tenant, run_id="r")
    # ok: process-tool-calls-without-tenant
    await _process_tool_calls(step=1, tenant_id=tenant, **image_kwargs(extra))


async def _process_tool_calls(*, step, run_id, tenant_id=None):
    return step


# ruleid: hardcoded-terminal-status-set
DONE = {"success", "error", "cancelled"}
# ruleid: hardcoded-terminal-status-set
DONE_TUPLE = ("success", "cancelled")
# ruleid: hardcoded-terminal-status-set
DONE_LIST = [RunStatus.SUCCESS, RunStatus.ERROR]
# ruleid: hardcoded-terminal-status-set
MIXED = frozenset({"error", RunStatus.CANCELLED, "other"})
# ok: hardcoded-terminal-status-set
ONE = {"success"}
# ok: hardcoded-terminal-status-set
UNRELATED = ["alpha", "beta", "success"]
# ok: hardcoded-terminal-status-set
SHARED = TERMINAL_STATUSES


def handler(run_id, record):
    try:
        work()
    except Exception as exc:
        # ruleid: str-exc-in-log
        logger.warning("failed: %s", str(exc))
        # ruleid: str-exc-in-log
        logger.error(f"failed: {exc}")
        # ruleid: str-exc-in-log
        logger.info(f"failed: {str(exc)}")
        # ruleid: str-exc-in-log
        logger.error("failed", extra={"error_info": str(exc)})
        # ruleid: str-exc-in-log
        logger.error("failed", extra={"detail": [f"{exc}"]})
        # ruleid: str-exc-in-log
        record(error_info=str(exc))
        # ruleid: str-exc-in-log
        record(error_info={"text": str(exc)})
        # ruleid: str-exc-in-log
        error_info = str(exc)
        # ruleid: str-exc-in-log
        payload = {"error_info": f"bad {exc}", "run_id": run_id}
        # ok: str-exc-in-log
        logger.warning("failed", extra={"run_id": run_id, "kind": type(exc).__name__})
        # ok: str-exc-in-log
        logger.error("failed", extra={"error_info": "provider_error", "code": "E_X"})
        # ok: str-exc-in-log
        record(error_info=type(exc).__name__)
        # ok: str-exc-in-log
        error_info = type(exc).__name__
        # ok: str-exc-in-log
        payload = {"error_info": "provider_error", "run_id": run_id}
        # ok: str-exc-in-log
        logger.warning("failed", exc_info=safe_exception_info())
        # ok: str-exc-in-log
        text = str(exc)
    # ok: str-exc-in-log
    logger.info("fine %s", str(run_id))
    # ok: str-exc-in-log
    logger.info("fine", extra={"error_info": str(run_id)})


# ruleid: dotenv-in-src
import dotenv

# ruleid: dotenv-in-src
from dotenv import load_dotenv

# ruleid: dotenv-in-src
PATH = ".env"
# ruleid: dotenv-in-src
OTHER = "../config/.env.local"
# ok: dotenv-in-src
NAME = "environment"
# ok: dotenv-in-src
DOC = "Reads settings from the environment, not from a .env file."


async def timeouts(client, url):
    # ruleid: bare-number-timeout
    await client.get(url, timeout=30)
    # ruleid: bare-number-timeout
    await client.get(url, timeout=2.5)
    # ruleid: bare-number-timeout
    call(connect_timeout_s=10)
    # ruleid: bare-number-timeout
    call(timeout_s=5)
    # ok: bare-number-timeout
    await client.get(url, timeout=TIMEOUT)
    # ok: bare-number-timeout
    await client.get(url, timeout=None)
    # ok: bare-number-timeout
    await client.get(url, retries=3)


# --- shared vocabulary: copies of a definition are matched; the ratchet then applies each
# rule's path scope and allow_in (module:function) list, which tests/tools covers ---


async def stored_events(store, ev):
    # ruleid: stored-event-name-literal
    await store.save_run_event("r", event_type="run.started")
    # ruleid: stored-event-name-literal
    KIND = "tool.completed"
    # ruleid: stored-event-name-literal
    flagged = ev.event_type == "llm.failed"
    # ruleid: stored-event-name-literal
    DONE_EVENTS = frozenset({"run.completed", "tools.declared"})
    # ok: stored-event-name-literal
    await store.save_run_event("r", event_type=StoredEventType.RUN_COMPLETED.value)
    # ok: stored-event-name-literal
    OTHER = "run.something_else"
    # ok: stored-event-name-literal
    STREAMED = "run_started"
    # ok: stored-event-name-literal
    TEXT = "the run.started event is written once"


async def reasons(store, meta, reason, result):
    # ruleid: lifecycle-reason-literal
    await store.finalize_run("r", failure_reason="stale_running")
    # ruleid: lifecycle-reason-literal
    await emit("r", {"reason": "cancel_requested"})
    # ruleid: lifecycle-reason-literal
    data = {"failure_reason": "reclaim_limit"}
    # ruleid: lifecycle-reason-literal
    stopped = reason == "task_cancelled"
    # ruleid: lifecycle-reason-literal
    over = meta.get("failure_reason") != "budget_exceeded"
    # ruleid: lifecycle-reason-literal
    failure_reason = "lease_lost"
    # ruleid: lifecycle-reason-literal
    meta["failure_reason"] = "never_started"
    # ruleid: lifecycle-reason-literal
    await emit("r", {"kind": "history_too_large"})
    # ruleid: lifecycle-reason-literal
    paused = reason in ("requested", "budget")
    # ruleid: lifecycle-reason-literal
    approve(reason="requires_approval")
    # ruleid: lifecycle-reason-literal
    pause_reason = "requested"
    # ruleid: lifecycle-reason-literal
    deny(reason="User declined to run this tool.")
    # ruleid: lifecycle-reason-literal
    lost = reason in {"lease_lost", "reclaim_limit"}
    # ruleid: lifecycle-reason-literal
    lost = reason in ["lease_lost"]
    # ruleid: lifecycle-reason-literal
    await emit("r", {"failure_reason": "lease_lost" if lost else None})
    # ruleid: lifecycle-reason-literal
    await emit("r", {"failure_reason": None if lost else "lease_lost"})
    # ruleid: lifecycle-reason-literal
    await store.finalize_run("r", failure_reason="lease_lost" if lost else None)
    # ruleid: lifecycle-reason-literal
    paused = reason in {"requested", "budget"}
    # ok: lifecycle-reason-literal
    other = kind in {"requested", "budget"}
    # ok: lifecycle-reason-literal
    await emit("r", {"kind": "requested" if lost else None})
    # ok: lifecycle-reason-literal
    await emit("r", {"failure_reason": FailureReason.LEASE_LOST.value if lost else None})
    # ok: lifecycle-reason-literal
    await store.finalize_run("r", failure_reason=FailureReason.STALE_RUNNING.value)
    # ok: lifecycle-reason-literal
    await emit("r", {"reason": CancelReason.CANCEL_REQUESTED.value})
    # ok: lifecycle-reason-literal
    first = result.status == "requested"
    # ok: lifecycle-reason-literal
    asked = {"kind": "budget"}
    # ok: lifecycle-reason-literal
    flag = meta["cancel_requested"]
    # ok: lifecycle-reason-literal
    column = {"cancel_requested": False}
    # ok: lifecycle-reason-literal
    skipped = {"reason": "not_found"}
    # ok: lifecycle-reason-literal
    seen = getattr(result, "cancel_requested", False)


def formats(record, row):
    # ruleid: storage-format-literal
    checkpointed = record.storage_format == 2
    # ruleid: storage-format-literal
    other = storage_format != 2
    # ruleid: storage-format-literal
    storage_format: int = 2
    # ruleid: storage-format-literal
    upgraded = {"storage_format": 2}
    # ruleid: storage-format-literal
    AgentRun(storage_format=2)
    # ruleid: storage-format-literal
    current = record.storage_format >= 2
    # ruleid: storage-format-literal
    reversed_check = 2 == record.storage_format
    # ruleid: storage-format-literal
    reversed_other = 2 != storage_format
    # ruleid: storage-format-literal
    known = record.storage_format in (1, 2)
    # ruleid: storage-format-literal
    known_set = record.storage_format in {1, 2}
    # ok: storage-format-literal
    legacy_only = record.storage_format in (1,)
    # ok: storage-format-literal
    reversed_legacy = 1 == record.storage_format
    # ok: storage-format-literal
    other_member = record.iteration_count in (1, 2)
    # ok: storage-format-literal
    checkpointed = record.storage_format == STORAGE_FORMAT_CHECKPOINTED
    # ok: storage-format-literal
    legacy: int = 1
    # ok: storage-format-literal
    storage_format: int = 1
    # ok: storage-format-literal
    is_legacy = record.storage_format == 1
    # ok: storage-format-literal
    row = {"storage_format": 1}
    # ok: storage-format-literal
    total = record.iteration_count == 2


def documented_key():
    # ok: meta-key-literal
    """dendrux.parts"""


def meta_keys(meta, session, run):
    # ruleid: meta-key-literal
    parts = meta["dendrux.parts"]
    # ruleid: meta-key-literal
    error = meta.get("dendrux.error_info")
    # ruleid: meta-key-literal
    fresh = {"dendrux.loop": "react"}
    # ruleid: meta-key-literal
    session.info["dendrux.store"] = True
    # ruleid: meta-key-literal
    MY_KEY = "dendrux.stop_reason"
    # ruleid: meta-key-literal
    seen = "dendrux.images.seeded_slots" in meta
    # ruleid: meta-key-literal
    value = meta.pop("dendrux.step", None)
    # ruleid: meta-key-literal
    value = meta.setdefault("dendrux.message", {})
    # ok: meta-key-literal
    module = importlib.import_module("dendrux.store")
    # ok: meta-key-literal
    named = logging.getLogger("dendrux.store")
    # ok: meta-key-literal
    return_value = f("dendrux.parts")
    # ok: meta-key-literal
    parts = meta[PARTS_META_KEY]
    # ok: meta-key-literal
    error = meta.get(ERROR_INFO_META_KEY)
    # ok: meta-key-literal
    PendingPause.__module__ = "dendrux.store"
    # ok: meta-key-literal
    log = logging.getLogger("dendrux.runtime")
    # ok: meta-key-literal
    doc = 'agent_runs.meta["dendrux.parts"] holds the wire parts'
    # ok: meta-key-literal
    other = meta["dendrux.unrelated"]


# ruleid: default-sqlite-url-literal
DEFAULT_URL = "sqlite+aiosqlite:///./dendrux.db"
# ruleid: default-sqlite-url-literal
DEFAULT_PATH = "./dendrux.db"
# ruleid: default-sqlite-url-literal
MESSAGE = "Start with ./dendrux.db or set DENDRUX_DATABASE_URL."
# ok: default-sqlite-url-literal
RESOLVED = get_database_url()
# ok: default-sqlite-url-literal
HOME_DB = "~/.dendrux/dendrux.db"
# ok: default-sqlite-url-literal
BARE_DB = "sqlite+aiosqlite:///dendrux.db"
# ok: default-sqlite-url-literal
OTHER_DB = "./other.db"


def connection_loss(code, classified):
    # ruleid: connection-loss-code-copy
    lost = code.startswith("08")
    # ruleid: connection-loss-code-copy
    shutdown = code in ("57P01", "57P02")
    # ruleid: connection-loss-code-copy
    starting = code == "57P03"
    # ruleid: connection-loss-code-copy
    class_08 = code[:2] == "08"
    # ruleid: connection-loss-code-copy
    lost_tuple = code.startswith(("08",))
    # ruleid: connection-loss-code-copy
    lost_zero_slice = code[0:2] == "08"
    # ruleid: connection-loss-code-copy
    lost_reversed = "08" == code[:2]
    # ruleid: connection-loss-code-copy
    lost_member = code[:2] in ("08", "57")
    # ok: connection-loss-code-copy
    integrity_tuple = code.startswith(("23", "40"))
    # ok: connection-loss-code-copy
    integrity_slice = code[0:2] == "23"
    # ok: connection-loss-code-copy
    too_many = code == "53300"
    # ok: connection-loss-code-copy
    integrity = code.startswith("23")
    # ok: connection-loss-code-copy
    lost = _is_connection_loss_code(code)


def fingerprints(agent_name, user_input, output_type):
    # ruleid: new-run-fingerprint-outside-helper
    fp = compute_idempotency_fingerprint(agent_name, user_input)
    # ruleid: new-run-fingerprint-outside-helper
    name = f"{output_type.__module__}.{output_type.__qualname__}"
    # ok: new-run-fingerprint-outside-helper
    meta, fp = _new_run_metadata(agent, loop, session, None, None, user_input="x")
    # ok: new-run-fingerprint-outside-helper
    short = f"{output_type.__name__}"
    # ok: new-run-fingerprint-outside-helper
    other = f"{output_type.__module__}"


async def startup(store, run_id, data, event):
    # ruleid: run-started-outside-startup
    await _emit_event(store, run_id, StoredEventType.RUN_STARTED.value, data)
    # ruleid: run-started-outside-startup
    await store.save_run_event(run_id, event_type=StoredEventType.RUN_STARTED)
    # ruleid: run-started-outside-startup
    await store.save_run_event(run_id, event_type=StoredEventType.RUN_STARTED.value, data=data)
    # ok: run-started-outside-startup
    await _start_new_run(store, run_id, data)
    # ok: run-started-outside-startup
    first = event.event_type == StoredEventType.RUN_STARTED.value
    # ok: run-started-outside-startup
    clause = RunEvent.event_type == literal_column(f"'{StoredEventType.RUN_STARTED.value}'")
    # ok: run-started-outside-startup
    await _emit_event(store, run_id, StoredEventType.RUN_RESUMED.value, data)


async def failures(store, run_id, exc, payload):
    # ruleid: error-finalize-outside-helper
    await store.finalize_run(run_id, status=RunStatus.ERROR.value, error="x")
    # ruleid: error-finalize-outside-helper
    await store.finalize_with_event(run_id, status=RunStatus.ERROR, event_type=EVENT)
    # ruleid: error-finalize-outside-helper
    await store.finalize_run_if_status_in(run_id, status="error", allowed=ALLOWED)
    # ruleid: error-finalize-outside-helper
    await _transition_with_event(store, run_id, status=RunStatus.ERROR.value, event_type=EVENT)
    # ruleid: error-finalize-outside-helper
    await _emit_event(store, run_id, StoredEventType.RUN_ERROR.value, payload)
    # ruleid: error-finalize-outside-helper
    await _emit_event_safe(store, run_id, StoredEventType.RUN_ERROR, payload)
    # ruleid: error-finalize-outside-helper
    await store.finalize_with_event(run_id, status=state, event_type=StoredEventType.RUN_ERROR)
    # ruleid: error-finalize-outside-helper
    await store.save_run_event(run_id, event_type=StoredEventType.RUN_ERROR.value, data=payload)
    # ok: error-finalize-outside-helper
    await store.finalize_run(run_id, status=RunStatus.SUCCESS.value)
    # ok: error-finalize-outside-helper
    await _persist_run_failure(store, run_id, exc)
    # ok: error-finalize-outside-helper
    result = RunResult(status=RunStatus.ERROR, error="x")
    # ok: error-finalize-outside-helper
    failed = payload.event_type == StoredEventType.RUN_ERROR.value
    # ok: error-finalize-outside-helper
    await store.finalize_run(run_id, status=result.status.value)


async def local_emitter_aliases(store, run_id, safe_event, payload):
    emit = _emit_event_safe if safe_event else _emit_event
    write = store.save_run_event
    if payload:
        # ruleid: error-finalize-outside-helper
        await emit(store, run_id, StoredEventType.RUN_ERROR.value, payload)
        # ruleid: run-started-outside-startup
        await emit(store, run_id, StoredEventType.RUN_STARTED.value, payload)
        # ruleid: error-finalize-outside-helper
        await write(run_id, event_type=StoredEventType.RUN_ERROR.value, data=payload)
        # ok: error-finalize-outside-helper
        await emit(store, run_id, StoredEventType.RUN_PAUSED.value, payload)


async def not_an_emitter_alias(store, run_id, payload, other):
    emit = other.emit
    # ok: error-finalize-outside-helper
    await emit(store, run_id, StoredEventType.RUN_ERROR.value, payload)
    # ok: error-finalize-outside-helper
    await other(store, run_id, StoredEventType.RUN_ERROR.value, payload)


async def cancellations(store, run_id, payload, allowed):
    # ruleid: cancel-finalize-outside-helper
    await store.finalize_run(run_id, status=RunStatus.CANCELLED.value, answer=None)
    # ruleid: cancel-finalize-outside-helper
    await store.finalize_run(run_id, status=RunStatus.CANCELLED)
    # ruleid: cancel-finalize-outside-helper
    await store.finalize_run(run_id, status="cancelled")
    # ruleid: cancel-finalize-outside-helper
    await store.finalize_run_if_status_in(run_id, status=RunStatus.CANCELLED.value, allowed=allowed)
    # ruleid: cancel-finalize-outside-helper
    await _transition_with_event(store, run_id, status=RunStatus.CANCELLED.value, event_type=EV)
    # ruleid: cancel-finalize-outside-helper
    await store.finalize_with_event(run_id, status=RunStatus.CANCELLED.value, event_type=EV)
    # ok: cancel-finalize-outside-helper
    await _persist_run_cancellation(store, run_id, payload, on_won=mark)
    # ok: cancel-finalize-outside-helper
    await store.finalize_run(run_id, status=RunStatus.SUCCESS.value)
    # ok: cancel-finalize-outside-helper
    await _transition_with_event(store, run_id, status=status, event_type=EV)
    # ok: cancel-finalize-outside-helper
    cancelled = payload.status == RunStatus.CANCELLED.value


# ruleid: hardcoded-terminal-status-set
DONE_VALUES = (RunStatus.SUCCESS.value, RunStatus.ERROR.value)
# ruleid: hardcoded-terminal-status-set
VALUE_LIST = [RunStatus.ERROR.value, "cancelled"]
# ruleid: hardcoded-terminal-status-set
VALUE_SET = frozenset({RunStatus.CANCELLED.value, RunStatus.MAX_ITERATIONS})
# ok: hardcoded-terminal-status-set
ONE_VALUE = (RunStatus.SUCCESS.value,)
# ok: hardcoded-terminal-status-set
SHARED_VALUES = TERMINAL_STATUS_VALUES
