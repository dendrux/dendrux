"""Shared homes keep the existing defaults and their consumer signatures aligned."""

import importlib
import inspect
from pathlib import Path

import pytest

HOMES = {
    "runtime._bounds": {
        "MAX_PAGE": 1000,
        "MAX_SMALL_PAGE": 500,
        "DEFAULT_LIST_LIMIT": 50,
        "DEFAULT_READ_LIMIT": 500,
        "DEFAULT_EVENT_LIMIT": 100,
        "MAX_LIST_OFFSET": 10_000,
        "IN_CHUNK": 500,
        "SWEEP_BATCH_DEFAULT": 100,
        "SWEEP_BATCH_MAX": 1000,
        "SWEEP_MAX_RUNS_DEFAULT": 10_000,
        "SWEEP_MAX_RUNS_MAX": 100_000,
        "SWEEP_MAX_RETURNED_DEFAULT": 1000,
        "SWEEP_MAX_RETURNED_MAX": 100_000,
        "SWEEP_MAX_RECLAIMS_DEFAULT": 3,
        "SWEEP_MAX_RECLAIMS_MAX": 32_767,
        "RECOVERY_SCAN_MAX": 100_000,
    },
    "runtime.context": {
        "DEFAULT_LEASE_INTERVAL_S": 30.0,
        "MIN_LEASE_INTERVAL_S": 5.0,
        "LEASE_FAILURE_LIMIT": 2,
        "RENEW_DEADLINE_CAP_S": 10.0,
    },
    "llm._resilience": {
        "DEFAULT_MAX_TOKENS": 16_000,
        "DEFAULT_MAX_RETRIES": 3,
        "DEFAULT_MAX_RESPONSE_BYTES": 16 * 1024 * 1024,
        "DEFAULT_MAX_STREAM_BYTES": 16 * 1024 * 1024,
        "DEFAULT_MAX_STREAM_TOOL_CALLS": 1000,
    },
    "mcp._source": {
        "DEFAULT_CONNECT_TIMEOUT": 30.0,
        "DEFAULT_CALL_TIMEOUT": 120.0,
        "DEFAULT_MAX_RESULT_BYTES": 1_000_000,
        "DEFAULT_MAX_REDIRECTS": 20,
        "DEFAULT_MAX_DISCOVERY_PAGES": 100,
        "DEFAULT_DISCOVERY_TIMEOUT": 60.0,
        "DEFAULT_MAX_DISCOVERY_TOOLS": 2000,
    },
    "types": {"JSON_MAX_DEPTH": 32, "JSON_MAX_NODES": 10_000, "JSON_MAX_TEXT": 1024 * 1024},
    "runtime.durability": {"MAX_RETRIES": 3, "BASE_BACKOFF_S": 0.1},
    "db._engine_options": {"POOL_RECYCLE_S": 1800, "POOL_PRE_PING": True},
}


def constant(module, name):
    spec = importlib.util.find_spec(f"dendrux.{module}")
    assert spec is not None, f"missing constants home: {module}"
    home = importlib.import_module(f"dendrux.{module}")
    assert hasattr(home, name), f"missing named default: {module}.{name}"
    return getattr(home, name)


@pytest.mark.parametrize("module", HOMES)
def test_named_defaults_preserve_existing_values(module):
    for name, value in HOMES[module].items():
        actual = constant(module, name)
        assert actual == value
        assert type(actual) is type(value)


@pytest.mark.parametrize(
    "provider",
    [
        "anthropic.AnthropicProvider",
        "openai.OpenAIProvider",
        "openai_responses.OpenAIResponsesProvider",
        "openrouter.OpenRouterProvider",
    ],
)
def test_provider_default_signatures(provider):
    module, name = provider.split(".")
    cls = getattr(importlib.import_module(f"dendrux.llm.{module}"), name)
    params = inspect.signature(cls).parameters
    for arg, bound in {
        "max_tokens": "DEFAULT_MAX_TOKENS",
        "max_output_tokens": "DEFAULT_MAX_TOKENS",
        "max_retries": "DEFAULT_MAX_RETRIES",
        "max_response_bytes": "DEFAULT_MAX_RESPONSE_BYTES",
        "max_stream_bytes": "DEFAULT_MAX_STREAM_BYTES",
        "max_stream_tool_calls": "DEFAULT_MAX_STREAM_TOOL_CALLS",
    }.items():
        if arg in params:
            assert params[arg].default == constant("llm._resilience", bound)


def test_sweep_default_signatures():
    from dendrux.runtime.state import StateStore
    from dendrux.runtime.stores.sql.base import SQLStateStore
    from dendrux.runtime.sweep import sweep

    for operation in (
        sweep,
        StateStore.sweep_stale_runs,
        StateStore.sweep_abandoned_runs,
        SQLStateStore.sweep_stale_runs,
        SQLStateStore.sweep_abandoned_runs,
    ):
        params = inspect.signature(operation).parameters
        for arg, bound in {
            "batch_size": "SWEEP_BATCH_DEFAULT",
            "max_runs": "SWEEP_MAX_RUNS_DEFAULT",
        }.items():
            assert params[arg].default == constant("runtime._bounds", bound)


def test_transport_and_tool_result_caps_remain_distinct():
    from dendrux.types import Limits

    assert constant("mcp._source", "DEFAULT_MAX_RESULT_BYTES") == 1_000_000
    assert Limits().max_tool_result_bytes == 1_048_576


def test_scaling_table_matches_named_defaults():
    rows = {}
    document = Path(__file__).parents[4] / "docs/guides/scaling.mdx"
    section = (
        document.read_text()
        .split("## Live readers and bounded work\n", 1)[1]
        .split("## Read replicas", 1)[0]
    )
    for line in section.splitlines():
        if line.startswith("| "):
            cells = [cell.strip() for cell in line.split("|")[1:-1]]
            rows[cells[0]] = cells[1]

    def c(module, name):
        return constant(module, name)

    expected = {
        "Live `follow` readers per store (process)": str(
            c("store._follow", "DEFAULT_MAX_FOLLOWERS")
        ),
        "Live `follow` readers per tenant scope": (
            f"{c('store._follow', 'DEFAULT_MAX_FOLLOWERS') // 5} (a fifth of the store cap)"
        ),
        "One `follow` reader": f"{c('store._follow', 'DEFAULT_MAX_DURATION_S'):g} s",
        "Sync tool threads per process": str(c("tools._pool", "DEFAULT_TOOL_THREADS")),
        "Sync tool threads per Agent": str(c("agent", "DEFAULT_TOOL_THREADS_PER_AGENT")),
        "Tool calls waiting per Agent": str(c("tools._pool", "DEFAULT_MAX_QUEUED_PER_AGENT")),
        "Tool calls queued per process": f"{c('tools._pool', 'MAX_QUEUED_CALLS'):,}",
        "MCP discovery": (
            f"{c('mcp._source', 'DEFAULT_MAX_DISCOVERY_PAGES')} pages, "
            f"{c('mcp._source', 'DEFAULT_MAX_DISCOVERY_TOOLS'):,} tools, "
            f"{c('mcp._source', 'DEFAULT_DISCOVERY_TIMEOUT'):g} s, "
            f"{c('mcp._source', 'DEFAULT_MAX_DISCOVERY_BYTES') // (1024 * 1024)} MiB (HTTP)"
        ),
        "`sweep` candidates per category": (
            f"{c('runtime._bounds', 'SWEEP_MAX_RUNS_DEFAULT'):,} (ceiling "
            f"{c('runtime._bounds', 'SWEEP_MAX_RUNS_MAX'):,})"
        ),
        "`sweep` page size": (
            f"{c('runtime._bounds', 'SWEEP_BATCH_DEFAULT')} (ceiling "
            f"{c('runtime._bounds', 'SWEEP_BATCH_MAX'):,})"
        ),
        "`sweep` returned reclaimable runs": (
            f"{c('runtime._bounds', 'SWEEP_MAX_RETURNED_DEFAULT'):,} (ceiling "
            f"{c('runtime._bounds', 'SWEEP_MAX_RETURNED_MAX'):,})"
        ),
        "`sweep` reclaim attempts": (
            f"{c('runtime._bounds', 'SWEEP_MAX_RECLAIMS_DEFAULT')} (ceiling "
            f"{c('runtime._bounds', 'SWEEP_MAX_RECLAIMS_MAX'):,})"
        ),
        "`list_runs` offset": f"{c('runtime._bounds', 'MAX_LIST_OFFSET'):,}",
        "Lease renewal interval": (
            f"{c('runtime.context', 'DEFAULT_LEASE_INTERVAL_S'):g} s (minimum "
            f"{c('runtime.context', 'MIN_LEASE_INTERVAL_S'):g} s)"
        ),
        "Lease renewal deadline": (
            f"half the interval, at most {c('runtime.context', 'RENEW_DEADLINE_CAP_S'):g} s"
        ),
        "Consecutive lease failures": str(c("runtime.context", "LEASE_FAILURE_LIMIT")),
        "Agent iterations": (
            f"{c('agent', 'DEFAULT_MAX_ITERATIONS')} (ceiling "
            f"{c('agent', 'MAX_ITERATIONS_CEILING')})"
        ),
        "Provider response / stream bytes": (
            f"{c('llm._resilience', 'DEFAULT_MAX_RESPONSE_BYTES') // (1024 * 1024)} MiB / "
            f"{c('llm._resilience', 'DEFAULT_MAX_STREAM_BYTES') // (1024 * 1024)} MiB"
        ),
        "Provider stream tool calls": f"{c('llm._resilience', 'DEFAULT_MAX_STREAM_TOOL_CALLS'):,}",
        "Provider output tokens / SDK retries": (
            f"{c('llm._resilience', 'DEFAULT_MAX_TOKENS'):,} / "
            f"{c('llm._resilience', 'DEFAULT_MAX_RETRIES')}"
        ),
    }
    assert rows.keys() == {"Resource", "---", *expected.keys()}
    for resource, default in expected.items():
        assert rows[resource] == default, resource


def test_mcp_factories_share_transport_defaults():
    from dendrux.mcp._result import normalize_mcp_result
    from dendrux.mcp._server import MCPServer, _normalize_mcp_result
    from dendrux.mcp._source import MCPSource

    for operation in (MCPSource, MCPSource.http, MCPSource.stdio, MCPServer):
        params = inspect.signature(operation).parameters
        for arg, bound in {
            "connect_timeout": "DEFAULT_CONNECT_TIMEOUT",
            "call_timeout": "DEFAULT_CALL_TIMEOUT",
            "max_result_bytes": "DEFAULT_MAX_RESULT_BYTES",
            "max_redirects": "DEFAULT_MAX_REDIRECTS",
            "max_discovery_pages": "DEFAULT_MAX_DISCOVERY_PAGES",
            "discovery_timeout": "DEFAULT_DISCOVERY_TIMEOUT",
            "max_discovery_tools": "DEFAULT_MAX_DISCOVERY_TOOLS",
            "max_discovery_bytes": "DEFAULT_MAX_DISCOVERY_BYTES",
        }.items():
            if arg in params:
                assert params[arg].default == constant("mcp._source", bound)
    for operation in (normalize_mcp_result, _normalize_mcp_result):
        assert inspect.signature(operation).parameters["max_result_bytes"].default == constant(
            "mcp._source", "DEFAULT_MAX_RESULT_BYTES"
        )


def test_database_error_kinds_match_log_projection():
    from typing import get_args

    from dendrux import _codes
    from dendrux.runtime.stores._errors import DbErrorKind

    assert frozenset(get_args(DbErrorKind)) == _codes._ENUMS["db_kind"]


def test_cost_precision_and_reclaim_storage_width_match_bounds():
    from dendrux.db._storage_schema import ADDITIONS
    from dendrux.db.models import AgentRun, LLMInteraction, TokenUsage
    from dendrux.types import COST_DECIMALS

    for column in (
        AgentRun.__table__.c.total_cost_usd,
        LLMInteraction.__table__.c.cost_usd,
        TokenUsage.__table__.c.cost_usd,
    ):
        assert column.type.scale == COST_DECIMALS == 6
    assert ADDITIONS["agent_runs"]["reclaim_limit"] == "SMALLINT NULL"
    assert constant("runtime._bounds", "SWEEP_MAX_RECLAIMS_MAX") == 2**15 - 1
