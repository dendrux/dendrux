# Fixture for .semgrep/dendrux.yml: lines marked as matches must match, lines marked ok must not.
# Run: semgrep --test .semgrep   (see tools/QUALITY.md)
import logging

logger = logging.getLogger(__name__)


class Store:
    async def _write_tx(self, session):
        # ok: session-commit-outside-write-tx
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


async def callers(extra):
    # ruleid: process-tool-calls-without-tenant
    await _process_tool_calls(step=1, run_id="r")
    # ok: process-tool-calls-without-tenant
    await _process_tool_calls(step=1, run_id="r", tenant_id=None)
    # ok: process-tool-calls-without-tenant
    await _process_tool_calls(step=1, tenant_id=tenant, run_id="r")
    # ok: process-tool-calls-without-tenant
    await _process_tool_calls(step=1, **extra)


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
        record(error_info=str(exc))
        # ok: str-exc-in-log
        logger.warning("failed", extra={"run_id": run_id, "kind": type(exc).__name__})
        # ok: str-exc-in-log
        record(error_info=type(exc).__name__)
    # ok: str-exc-in-log
    logger.info("fine %s", str(run_id))


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
