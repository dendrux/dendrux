"""Agent — the primary interface for Dendrux.

Describes the agent's identity, capabilities, and limits, and provides
runtime methods (run, resume) that delegate to the runner.

Two creation styles:
    # Constructor — primary API
    agent = Agent(
        provider=AnthropicProvider(model="claude-sonnet-4-6"),
        tools=[add, multiply],
        prompt="You are a math assistant.",
    )

    # Subclass — for reusable agent types
    class AuditAgent(Agent):
        tools = [readRange, finishAudit]
        prompt = "You are a workbook auditor."
        max_iterations = 15

    agent = AuditAgent(provider=AnthropicProvider(model="..."))
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import warnings
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, overload

from dendrux._sentinel import _OMITTED, _OmittedType, _UnsetType
from dendrux.runtime._entry_checks import entry_checks
from dendrux.runtime.state import UNSCOPED, TenantScope, require_store_method
from dendrux.skills._internal import is_script, validate_executor
from dendrux.skills._session import NAMES, SkillSession, continuation, current
from dendrux.tool import get_tool_def, is_tool

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping, Sequence

    from pydantic import BaseModel
    from sqlalchemy.ext.asyncio import AsyncEngine

    from dendrux.chat import ChatMessage
    from dendrux.context_blocks import ContextBlock
    from dendrux.guardrails._protocol import Guardrail
    from dendrux.llm.base import LLMProvider
    from dendrux.loops.base import Loop, LoopNotifier
    from dendrux.mcp._host import MCPHost
    from dendrux.mcp._runtime import MCPToolView
    from dendrux.mcp._server import MCPServer
    from dendrux.mcp._source import MCPSource
    from dendrux.media._catalog import BlobCatalog, BlobResolver
    from dendrux.media._image import UserContent
    from dendrux.media._policy import ImagePolicy
    from dendrux.media._store import BlobAccessor, BlobInfo, BlobReference, BlobStore, GcResult
    from dendrux.pricing import PriceTable
    from dendrux.runtime.state import StateStore
    from dendrux.skills import Skill, SkillExecutor
    from dendrux.tools import ToolLookups
    from dendrux.types import (
        ApprovalDecision,
        BeforeToolHook,
        Budget,
        ImageRef,
        Limits,
        PauseOutcome,
        RunEvent,
        RunResult,
        RunStream,
        ToolDef,
        ToolResult,
        ToolResultTransform,
    )

_agent_logger = logging.getLogger(__name__)

# Safety limit to prevent runaway LLM costs. Can be overridden per-agent
# once the worker/config layer ships (Sprint 6).
MAX_ITERATIONS_CEILING = 200

_UNSET = _UnsetType()

# Inbox note limits (E-P2-3): text is UTF-8 bytes; metadata is encoded JSON.
MAX_NOTE_TEXT_BYTES = 32 * 1024
MAX_NOTE_METADATA_BYTES = 8 * 1024
MAX_PAUSE_REASON_CHARS = 1000

_TERMINAL_STATUS_VALUES = frozenset({"success", "error", "cancelled", "max_iterations"})
_WAITING_STATUS_VALUES = frozenset(
    {"waiting_client_tool", "waiting_human_input", "waiting_approval"}
)


def _in_tenant_scope(record: Any, tenant_id: TenantScope) -> bool:
    """Omitted scope matches everything; otherwise the tenant must match exactly."""
    return tenant_id is UNSCOPED or record.tenant_id == tenant_id


async def _require_run_in_scope(store: StateStore, run_id: str, tenant_id: TenantScope) -> None:
    """Raise ``RunNotFoundError`` unless the run exists within ``tenant_id``.

    Run before any write so a cross-tenant call changes nothing. A run's
    tenant never changes, so this read cannot race. Missing runs use the same error.
    """
    from dendrux.errors import RunNotFoundError

    record = await store.get_run(run_id)
    if record is None or not _in_tenant_scope(record, tenant_id):
        raise RunNotFoundError(run_id)


def _cascade_order(subtree: Any) -> list[str]:
    """Run ids of a subtree snapshot: deepest descendants first, root last."""
    descendants = sorted((node for node in subtree.nodes if node.depth > 0), key=lambda n: -n.depth)
    return [node.run.id for node in descendants] + [subtree.nodes[0].run.id]


def _validate_max_delegation_depth(value: int | None | _UnsetType) -> None:
    """Validate max_delegation_depth — shared between constructor and _validate()."""
    if isinstance(value, _UnsetType):
        return
    if value is not None and (not isinstance(value, int) or value < 0):
        raise ValueError(
            f"max_delegation_depth must be a non-negative integer or None, got {value!r}"
        )


def _skipped_source_entry(source: Any, exc: Exception) -> dict[str, Any]:
    """Describe a best_effort MCP source skipped at discovery."""
    # Deferred: dendrux.mcp needs the optional mcp extra, and an Agent only
    # reaches here when it has MCP tool sources.
    from dendrux.mcp._errors import MCPAuthorizationRequired, error_code_for

    configured = getattr(getattr(source, "source", None), "name", None)
    entry: dict[str, Any] = {
        "source_name": configured if isinstance(configured, str) else source.name,
        "namespace": source.name,
        "error_type": type(exc).__name__,
        "error": str(exc),
        "code": error_code_for(exc),
    }
    if isinstance(exc, MCPAuthorizationRequired):
        entry["reason"] = exc.reason
    return entry


def _record_to_run_result(record: Any) -> RunResult:
    """Project a RunRecord onto a RunResult for cancel/no-op return paths."""
    from dendrux.types import RunResult, RunStatus, UsageStats

    usage = UsageStats(
        input_tokens=record.total_input_tokens,
        output_tokens=record.total_output_tokens,
        total_tokens=record.total_input_tokens + record.total_output_tokens,
        cost_usd=record.total_cost_usd,
        cache_read_input_tokens=record.total_cache_read_tokens,
        cache_creation_input_tokens=record.total_cache_creation_tokens,
    )
    from dendrux.runtime.runner import _stop_reason_from_meta

    status = RunStatus(record.status)
    return RunResult(
        run_id=record.id,
        status=status,
        answer=record.answer,
        error=record.error,
        iteration_count=record.iteration_count,
        usage=usage,
        meta={"cancel_requested": bool(record.cancel_requested)},
        stop_reason=(
            _stop_reason_from_meta(getattr(record, "meta", None))
            if status == RunStatus.SUCCESS
            else None
        ),
    )


def _warn_env_store_deprecated() -> None:
    """B6 ``DeprecationWarning`` for the implicit DENDRUX_DATABASE_URL store.

    Deduplicated by ``warnings`` per caller location; ``stacklevel`` points at
    the code that called the Agent method (this function,
    ``_resolve_state_store``, the Agent method).
    """
    import warnings

    warnings.warn(
        "Agent found no state_store or database_url and fell back to DENDRUX_DATABASE_URL. "
        "This implicit store is deprecated and is removed in dendrux 0.6. Create one state "
        "store per process (for example at app startup: "
        "SQLAlchemyStateStore(create_async_engine(url))) and pass state_store= to every "
        "Agent. Do not pass database_url= per request: each Agent would build its own pool.",
        DeprecationWarning,
        stacklevel=4,
    )


async def _verify_schema_initialized(engine: AsyncEngine, url: str) -> None:
    """Fail fast with setup guidance if a non-SQLite database has no tables.

    SQLite auto-creates its schema in ``_create_private_engine`` / ``get_engine``,
    so the check is skipped there. For Postgres and other backends the schema
    must be migrated first; a missing ``agent_runs`` table means the dev never
    ran ``dendrux db migrate``, so raise a clear, actionable error instead of
    letting a raw "relation does not exist" surface mid-run.
    """
    if url.startswith("sqlite"):
        return

    from sqlalchemy import inspect as _sa_inspect

    from dendrux.errors import SchemaNotInitializedError

    async with engine.connect() as conn:
        present = await conn.run_sync(
            lambda sync_conn: _sa_inspect(sync_conn).has_table("agent_runs")
        )
    if not present:
        raise SchemaNotInitializedError()


def _build_provider_from_spec(spec: str) -> LLMProvider:
    """Build a provider from a ``"vendor:model"`` recipe string.

    A convenience that mirrors ``database_url``: hand the agent a recipe
    string and it constructs the provider (reading the API key from the
    environment). Supported vendors: ``anthropic``, ``openai``,
    ``openai-responses``, ``openrouter``. Pass a provider instance for full
    configuration.
    """
    vendor, sep, model = spec.partition(":")
    vendor = vendor.strip().lower()
    model = model.strip()
    if not sep or not model:
        raise ValueError(
            f"Invalid provider spec {spec!r}. Use 'vendor:model', "
            "e.g. 'anthropic:claude-haiku-4-5'."
        )
    if vendor == "anthropic":
        from dendrux.llm.anthropic import AnthropicProvider

        return AnthropicProvider(model=model)
    if vendor == "openai":
        from dendrux.llm.openai import OpenAIProvider

        return OpenAIProvider(model=model)
    if vendor == "openai-responses":
        from dendrux.llm.openai_responses import OpenAIResponsesProvider

        return OpenAIResponsesProvider(model=model)
    if vendor == "openrouter":
        from dendrux.llm.openrouter import OpenRouterProvider

        return OpenRouterProvider(model=model)
    raise ValueError(
        f"Unknown provider vendor {vendor!r}. Supported: anthropic, openai, "
        "openai-responses, openrouter. Or pass a provider instance you "
        "construct yourself."
    )


def _blob_services_factory() -> Any:
    """``BlobServices`` from the persistence layer (the single wiring point).

    ``BlobServices.for_state_store(store, blob_store=)`` and
    ``BlobServices.in_memory(blob_store=)``; tests substitute fakes here.
    """
    from dendrux.runtime._blob_catalog import BlobServices

    return BlobServices


class _AgentBlobs:
    """``agent.blobs``: resolves the agent's store lazily, then delegates.

    Delegates to the blob services' accessor for this agent's database (or
    the process-local in-memory services without one).
    """

    def __init__(self, agent: Agent) -> None:
        self._agent = agent

    async def _target(self) -> BlobAccessor:
        from dendrux.errors import ImagePersistenceUnsupportedError

        store = await self._agent._resolve_state_store()
        services = self._agent._blob_services(store)
        if services is None:
            raise ImagePersistenceUnsupportedError(
                "state_store", store_type=type(store).__name__ if store is not None else None
            )
        accessor: BlobAccessor = services.accessor
        return accessor

    async def read(self, image: ImageRef | str, *, tenant_id: str | None = None) -> bytes:
        return await (await self._target()).read(image, tenant_id=tenant_id)

    async def info(self, sha256: str, *, tenant_id: str | None = None) -> BlobInfo | None:
        return await (await self._target()).info(sha256, tenant_id=tenant_id)

    async def references(self, sha256: str, *, tenant_id: str | None = None) -> list[BlobReference]:
        return await (await self._target()).references(sha256, tenant_id=tenant_id)

    async def purge(
        self, sha256: str, *, tenant_id: str | None = None, reason: str | None = None
    ) -> bool:
        return await (await self._target()).purge(sha256, tenant_id=tenant_id, reason=reason)

    async def collect_garbage(
        self, *, older_than: timedelta = timedelta(hours=24), limit: int = 1000
    ) -> GcResult:
        return await (await self._target()).collect_garbage(older_than=older_than, limit=limit)


class Agent:
    """Dendrux agent — definition and runtime facade.

    Holds the agent's identity, capabilities, limits, and runtime config
    (provider, persistence, guardrails). Provides run() and resume() as
    instance methods that delegate to the runner.

    Attributes:
        name: Agent identifier. Auto-derived from class name if subclassed.
        prompt: System prompt for the agent.
        tools: List of @tool-decorated functions this agent can use.
        max_iterations: Maximum ReAct loop iterations before stopping.
    """

    name: str = ""
    prompt: str = ""
    tools: list[Callable[..., Any]] = []
    max_iterations: int = 10
    max_delegation_depth: int | None | _UnsetType = _UNSET

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if "model" in cls.__dict__:
            warnings.warn(
                f"Class-level 'model' on {cls.__name__} is deprecated. "
                "Pass provider= to __init__() instead. The model is set on the provider.",
                DeprecationWarning,
                stacklevel=2,
            )
            # Remove so the base class property takes over
            delattr(cls, "model")

    @overload
    def __init__(
        self,
        *,
        provider: LLMProvider | str,
        prompt: str,
        name: str = ...,
        tools: list[Callable[..., Any]] = ...,
        tool_sources: list[MCPServer | MCPSource | MCPHost | MCPToolView] | None = ...,
        max_iterations: int = ...,
        max_delegation_depth: int | None = ...,
        loop: Loop | None = ...,
        output_type: type[BaseModel] | None = ...,
        database_url: str | None = ...,
        database_options: dict[str, Any] | None = ...,
        state_store: StateStore | None = ...,
        deny: list[str] | None = ...,
        require_approval: list[str] | None = ...,
        budget: Budget | None = ...,
        pricing: PriceTable | None = ...,
        guardrails: list[Guardrail] | None = ...,
        skills_dir: str | Path | None = ...,
        skills: list[Skill] | None = ...,
        deny_skills: list[str] | None = ...,
        skill_executor: SkillExecutor | None = ...,
        tool_result_transform: ToolResultTransform | None = ...,
        before_tool: BeforeToolHook | None = ...,
        images: ImagePolicy | None = ...,
        blob_store: BlobStore | None = ...,
        strict_tenancy: bool = ...,
        limits: Limits | None = ...,
        tool_threads: int = ...,
        on_governance_change: Literal["warn", "raise"] = ...,
    ) -> None: ...

    @overload
    def __init__(
        self,
        *,
        provider: LLMProvider | str | None = ...,
        name: str = ...,
        prompt: str = ...,
        tools: list[Callable[..., Any]] = ...,
        tool_sources: list[MCPServer | MCPSource | MCPHost | MCPToolView] | None = ...,
        max_iterations: int = ...,
        max_delegation_depth: int | None = ...,
        loop: Loop | None = ...,
        output_type: type[BaseModel] | None = ...,
        database_url: str | None = ...,
        database_options: dict[str, Any] | None = ...,
        state_store: StateStore | None = ...,
        deny: list[str] | None = ...,
        require_approval: list[str] | None = ...,
        budget: Budget | None = ...,
        pricing: PriceTable | None = ...,
        guardrails: list[Guardrail] | None = ...,
        skills_dir: str | Path | None = ...,
        skills: list[Skill] | None = ...,
        deny_skills: list[str] | None = ...,
        skill_executor: SkillExecutor | None = ...,
        tool_result_transform: ToolResultTransform | None = ...,
        before_tool: BeforeToolHook | None = ...,
        images: ImagePolicy | None = ...,
        blob_store: BlobStore | None = ...,
        strict_tenancy: bool = ...,
        limits: Limits | None = ...,
        tool_threads: int = ...,
        on_governance_change: Literal["warn", "raise"] = ...,
    ) -> None: ...

    def __init__(
        self,
        *,
        name: str | _UnsetType = _UNSET,
        prompt: str | _UnsetType = _UNSET,
        tools: list[Callable[..., Any]] | _UnsetType = _UNSET,
        tool_sources: list[MCPServer | MCPSource | MCPHost | MCPToolView] | None = None,
        max_iterations: int | _UnsetType = _UNSET,
        max_delegation_depth: int | None | _UnsetType = _UNSET,
        loop: Loop | None = None,
        output_type: type[BaseModel] | None = None,
        provider: LLMProvider | str | None | _UnsetType = _UNSET,
        database_url: str | None = None,
        database_options: dict[str, Any] | None = None,
        state_store: StateStore | None = None,
        deny: list[str] | None = None,
        require_approval: list[str] | None = None,
        budget: Budget | None = None,
        pricing: PriceTable | None = None,
        guardrails: list[Guardrail] | None = None,
        skills_dir: str | Path | None = None,
        skills: list[Skill] | None = None,
        deny_skills: list[str] | None = None,
        skill_executor: SkillExecutor | None = None,
        tool_result_transform: ToolResultTransform | None = None,
        before_tool: BeforeToolHook | None = None,
        images: ImagePolicy | None = None,
        blob_store: BlobStore | None = None,
        strict_tenancy: bool | _UnsetType = _UNSET,
        limits: Limits | None = None,
        tool_threads: int = 8,
        on_governance_change: Literal["warn", "raise"] | _UnsetType = _UNSET,
    ) -> None:
        # --- Subclass guard: block class-level provider ---
        from dendrux.llm.base import LLMProvider as _LLMBase

        cls_provider = self.__class__.__dict__.get("provider")
        if isinstance(cls_provider, _LLMBase):
            raise ValueError(
                "provider must be passed to __init__(), not set as a class attribute. "
                "Class-level provider creates a live HTTP client at class definition time."
            )

        # --- Identity ---
        if not isinstance(name, _UnsetType):
            self.name = name
        elif not self.name:
            self.name = type(self).__name__

        if not isinstance(prompt, _UnsetType):
            self.prompt = prompt
        if not isinstance(tools, _UnsetType):
            self.tools = list(tools)
        else:
            self.tools = list(self.__class__.tools)
        if not isinstance(max_iterations, _UnsetType):
            self.max_iterations = max_iterations
        if not isinstance(max_delegation_depth, _UnsetType):
            _validate_max_delegation_depth(max_delegation_depth)
            self.max_delegation_depth = max_delegation_depth

        # --- Loop ---
        self._loop: Loop | None = loop
        self._output_type: type[BaseModel] | None = output_type

        # --- Provider ---
        # A "vendor:model" recipe string builds the provider for you (mirrors
        # how database_url builds an engine). A provider instance is used as-is.
        _resolved_provider = None if isinstance(provider, _UnsetType) else provider
        self._provider: LLMProvider | None = (
            _build_provider_from_spec(_resolved_provider)
            if isinstance(_resolved_provider, str)
            else _resolved_provider
        )

        # --- Persistence ---
        if database_url is not None and state_store is not None:
            raise ValueError(
                "database_url and state_store are mutually exclusive. Pass one or the other."
            )
        if database_options is not None and database_url is None:
            raise ValueError("database_options requires database_url.")

        self._database_url = database_url
        self._database_options = database_options or {}
        self._state_store = state_store

        from dendrux.runtime.tasks import RunTaskManager

        self._task_manager: RunTaskManager = RunTaskManager()
        self._deny: frozenset[str] = frozenset(deny) if deny else frozenset()
        self._require_approval: frozenset[str] = (
            frozenset(require_approval) if require_approval else frozenset()
        )
        self._budget: Budget | None = budget
        self._pricing: PriceTable | None = pricing
        self._guardrails: list[Guardrail] | None = guardrails if guardrails else None
        if tool_result_transform is not None and not callable(tool_result_transform):
            raise ValueError(
                f"Agent '{self.name}' tool_result_transform must be callable, "
                f"got {type(tool_result_transform).__name__}."
            )
        self._tool_result_transform: ToolResultTransform | None = tool_result_transform
        if before_tool is not None and not callable(before_tool):
            raise ValueError(
                f"Agent '{self.name}' before_tool must be callable, "
                f"got {type(before_tool).__name__}."
            )
        self._before_tool: BeforeToolHook | None = before_tool

        # --- Trust (0.5.0): stored here, enforced by the runtime checks ---
        from dendrux.tools._lookups import make_call_limiter
        from dendrux.types import Limits as _Limits

        # "Not chosen" is kept apart from an explicit False / "warn": the 0.6
        # default flips warn only developers who did not choose.
        self._strict_tenancy_chosen: bool = not isinstance(strict_tenancy, _UnsetType)
        self._on_governance_change_chosen: bool = not isinstance(on_governance_change, _UnsetType)
        if isinstance(strict_tenancy, _UnsetType):
            strict_tenancy = False
        if isinstance(on_governance_change, _UnsetType):
            on_governance_change = "warn"
        if not isinstance(strict_tenancy, bool):
            raise ValueError(f"Agent '{self.name}' strict_tenancy must be a bool.")
        if limits is not None and not isinstance(limits, _Limits):
            raise ValueError(
                f"Agent '{self.name}' limits must be a Limits instance, "
                f"got {type(limits).__name__}."
            )
        if isinstance(tool_threads, bool) or not isinstance(tool_threads, int) or tool_threads < 1:
            raise ValueError(
                f"Agent '{self.name}' tool_threads must be a positive int, got {tool_threads!r}."
            )
        if on_governance_change not in ("warn", "raise"):
            raise ValueError(
                f"Agent '{self.name}' on_governance_change must be 'warn' or 'raise', "
                f"got {on_governance_change!r}."
            )
        self._strict_tenancy: bool = strict_tenancy
        # None means the default caps: B12 applies them unless the developer overrides.
        self._limits: Limits = limits if limits is not None else _Limits()
        self._tool_threads: int = tool_threads
        self._on_governance_change: Literal["warn", "raise"] = on_governance_change
        self._tool_call_limiter: Any = make_call_limiter(tool_threads)

        # --- Images ---
        from dendrux.media._policy import ImagePolicy as _ImagePolicy
        from dendrux.media._store import BlobStore as _BlobStore

        if images is not None and not isinstance(images, _ImagePolicy):
            raise ValueError(
                f"Agent '{self.name}' images must be an ImagePolicy, got {type(images).__name__}."
            )
        if blob_store is not None and not isinstance(blob_store, _BlobStore):
            raise ValueError(
                f"Agent '{self.name}' blob_store must implement BlobStore "
                f"(put/get/delete), got {type(blob_store).__name__}."
            )
        self._image_policy: ImagePolicy = images if images is not None else _ImagePolicy()
        self._blob_store: BlobStore | None = blob_store
        # Blob services per state store (catalog, resolver cache, accessor);
        # key None = process-local in-memory services when there is no database.
        self._services: dict[int | None, tuple[Any, Any]] = {}
        self._lazy_store: StateStore | None = None
        self._private_engine: AsyncEngine | None = None
        self._store_lock = threading.Lock()
        self._store_generation = 0

        # --- MCP tool sources ---
        self._tool_sources: list[MCPServer] = []
        if tool_sources:
            from dendrux.mcp._host import MCPHost as _MCPHost
            from dendrux.mcp._runtime import (
                MCPConnection as _MCPConnection,
            )
            from dendrux.mcp._runtime import (
                MCPToolView as _MCPToolView,
            )
            from dendrux.mcp._runtime import (
                _ViewToolSource,
            )
            from dendrux.mcp._server import MCPServer as _MCPServer
            from dendrux.mcp._source import MCPSource as _MCPSource

            seen_names: set[str] = set()
            expanded_sources: list[MCPServer | MCPSource | MCPToolView] = []
            for configured_source in tool_sources:
                if isinstance(configured_source, _MCPHost):
                    expanded_sources.extend(configured_source.tool_sources)
                else:
                    expanded_sources.append(configured_source)

            for i, configured_source in enumerate(expanded_sources):
                if isinstance(configured_source, _MCPSource):
                    src = _MCPServer.from_source(configured_source)
                elif isinstance(configured_source, _MCPToolView):
                    src = _ViewToolSource(configured_source)
                elif isinstance(configured_source, _MCPServer):
                    src = configured_source
                elif isinstance(configured_source, _MCPConnection):
                    raise ValueError(
                        f"Agent '{self.name}' tool_sources[{i}] is an MCPConnection, "
                        "which exposes no tools by itself. Call connection.tools(...) "
                        "and pass the resulting view instead."
                    )
                else:
                    raise ValueError(
                        f"Agent '{self.name}' tool_sources[{i}] is "
                        f"{type(configured_source).__name__}, not an MCPServer, MCPSource, "
                        "MCPHost, or MCPToolView instance. Use MCPSource.http(...) or "
                        "MCPSource.stdio(...)."
                    )
                if src.name in seen_names:
                    raise ValueError(
                        f"Agent '{self.name}' has duplicate tool_sources name '{src.name}'. "
                        f"Each MCP source must have a unique name."
                    )
                seen_names.add(src.name)
                self._tool_sources.append(src)
        self._discovered_tool_defs: list[ToolDef] | None = None
        self._mcp_executors: dict[str, Callable[..., Any]] | None = None
        # Optional (best_effort) sources skipped at the last discovery, each
        # as {source_name, namespace, error_type, error, code[, reason]}.
        # Surfaced on the mcp.error governance event and
        # RunResult.meta["mcp_skipped_sources"].
        self._mcp_skipped_sources: list[dict[str, Any]] = []
        # Individual tools views left out at the last discovery, each as
        # {source_name, namespace, tool, reason}. Surfaced on the
        # mcp.tools_skipped event and RunResult.meta["mcp_skipped_tools"].
        self._mcp_skipped_tools: list[dict[str, Any]] = []
        self._discovery_lock = asyncio.Lock()

        # --- Skills ---
        if skills_dir is not None and not isinstance(skills_dir, (str, Path)):
            raise ValueError(
                f"Agent '{self.name}' skills_dir must be a string or Path, "
                f"got {type(skills_dir).__name__}."
            )
        self._skills_dir: Path | None = Path(skills_dir) if skills_dir is not None else None

        self._explicit_skills: list[Skill] | None = None
        if skills is not None:
            from dendrux.skills import Skill as _Skill

            for i, s in enumerate(skills):
                if not isinstance(s, _Skill):
                    raise ValueError(
                        f"Agent '{self.name}' skills[{i}] is {type(s).__name__}, "
                        f"not a Skill instance. Use Skill.from_dir(path)."
                    )
            self._explicit_skills = list(skills)

        self._deny_skills: frozenset[str] = frozenset(deny_skills) if deny_skills else frozenset()
        self._loaded_skills: list[Skill] | None = None
        self._denied_skill_names: list[str] | None = None
        self._inline_skill_warning_emitted: bool = False
        validate_executor(skill_executor)
        self._skill_executor = skill_executor
        self._script_warning_names: set[str] = set()
        self._validate_skill_contents(self._explicit_skills or [])

        self._validate()

        from dendrux import pricing as _pricing

        _pricing.check_budget_config(self)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def model(self) -> str:
        """Model identifier — reads from provider."""
        if self._provider is None:
            return ""
        return self._provider.model

    @property
    def loop(self) -> Loop | None:
        """The execution loop, or None (runner defaults to ReActLoop)."""
        return self._loop

    @property
    def output_type(self) -> type[BaseModel] | None:
        """The default output type for structured output, or None."""
        return self._output_type

    @property
    def deny(self) -> frozenset[str]:
        """Tool names denied by policy. Empty frozenset if none."""
        return self._deny

    @property
    def require_approval(self) -> frozenset[str]:
        """Tool names requiring human approval. Empty frozenset if none."""
        return self._require_approval

    @property
    def budget(self) -> Budget | None:
        """Token and/or dollar budget (see :class:`~dendrux.types.Budget`), or None."""
        return self._budget

    @property
    def pricing(self) -> PriceTable | None:
        """Developer-supplied price table used to fill ``cost_usd``, or None."""
        return self._pricing

    @property
    def guardrails(self) -> list[Guardrail] | None:
        """Content guardrails, or None."""
        return self._guardrails

    @property
    def tool_result_transform(self) -> ToolResultTransform | None:
        """Hook that rewrites server tool results, or None.

        Called with a :class:`~dendrux.types.ToolResultContext` after every
        successful server-executed tool call (local ``@tool`` and MCP alike,
        including tools run on resume after ``submit_approval``) and before
        the result is JSON-serialized. Its return value always replaces the
        result — return ``ctx.result`` to keep it. The transformed value is
        what gets persisted, notified and sent to the model.

        - Not applied to failed tool calls or to client-submitted results
          (``submit_tool_results``). The built-in ``use_skill`` tool is a
          server tool too; check ``ctx.tool_name`` to skip it.
        - If the transform raises, the tool call fails (``success=False``)
          with an error naming the transform.
        - Runs on the raw result; guardrail scans of tool messages happen
          afterwards, on the serialized content.
        - Sync transforms run inline on the event loop — keep them cheap.
        """
        return self._tool_result_transform

    @property
    def image_policy(self) -> ImagePolicy:
        """Image input policy (``Agent(images=...)``; defaults to ``ImagePolicy()``)."""
        return self._image_policy

    @property
    def blobs(self) -> BlobAccessor:
        """Blob reads and lifecycle (read, info, references, purge, collect_garbage).

        Uses this agent's database and ``blob_store``; without a database,
        the process-local in-memory store.
        """
        return _AgentBlobs(self)

    def _blob_services(self, state_store: StateStore | None) -> Any:
        """Blob services for ``state_store`` (cached per store), or None when unavailable."""
        key = None if state_store is None else id(state_store)
        cached = self._services.get(key)
        if cached is not None and cached[0] is state_store:
            return cached[1]
        factory = _blob_services_factory()
        if factory is None:
            return None
        if state_store is None:
            services = factory.in_memory(blob_store=self._blob_store)
        else:
            services = factory.for_state_store(state_store, blob_store=self._blob_store)
        self._services[key] = (state_store, services)
        return services

    def _image_backend(
        self, state_store: StateStore | None
    ) -> tuple[BlobCatalog | None, BlobStore | None, BlobResolver | None]:
        """Catalog, blob store and resolver for one run's ``ImageRuntime``.

        With a database the state store is the catalog when it implements
        ``BlobCatalog`` (else ``None``: ingest then raises
        ``ImagePersistenceUnsupportedError``). Without one, process-local
        in-memory services (non-durable).
        """
        services = self._blob_services(state_store)
        if services is None:
            return None, self._blob_store, None
        return services.catalog, services.blob_store, services.resolver

    def _image_gc(self, state_store: StateStore | None) -> Callable[[], Awaitable[object]] | None:
        """One GC pass (``older_than=1h``) that ingest runs when the blob store is full (§5.12).

        Database-backed agents only: the in-memory services (no database)
        wrap their store so a full ``put`` already runs that single pass.
        """
        if state_store is None:
            return None
        services = self._blob_services(state_store)
        accessor = getattr(services, "accessor", None)
        if accessor is None:
            return None

        async def _collect() -> object:
            return await accessor.collect_garbage(older_than=timedelta(hours=1))

        return _collect

    @property
    def before_tool(self) -> BeforeToolHook | None:
        """Policy hook evaluated before every tool call, or None.

        Called with a :class:`~dendrux.types.ToolCallContext` (deep-copied
        ``params``/``meta`` — mutation never affects execution) for local,
        MCP and skill tools in ``run`` and ``stream``, after static
        ``deny=`` and skill preflight and before approval. Return a
        :class:`~dendrux.types.ToolDecision` or ``None`` (allow); sync or
        async.

        - ``deny(reason)``: not executed; the model gets ``{"denied": reason}``
          and ``policy.denied`` fires with ``source="hook"``.
        - ``require_approval(reason)``: joins the approval pause
          (``approval.requested`` with ``source="hook"``). Server-target tools
          only — on a client-target call it is treated as a deny with a
          ``policy.hook_warning`` event.
        - ``substitute(result)``: not executed; ``result`` (JSON-serialized
          first) is the successful tool output, the tool_calls row carries
          ``meta.substituted=true`` and ``policy.substituted`` fires.
        - Static ``deny=`` applies first and cannot be undone. A tool in
          ``require_approval=`` stays gated unless the hook denies it (allow,
          substitute and require_approval leave the static gate in place).
        - Fails closed: a hook that raises, returns anything other than a
          ``ToolDecision``/``None``, or substitutes a non-JSON value denies
          the call with ``"policy hook error"`` and emits
          ``policy.hook_error`` (exception type only). Cancellation is never
          swallowed.
        - Not re-run on resume: approved calls (the human decision wins) and
          held non-gated calls were evaluated before the pause. The hook is
          not persisted — pass it to every Agent that runs or resumes.
        - Sync hooks run inline on the event loop — keep them cheap.
        """
        return self._before_tool

    @property
    def provider(self) -> LLMProvider | None:
        """The LLM provider, or None if not configured."""
        return self._provider

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_provider(self) -> LLMProvider:
        """Return the provider or raise if not configured."""
        if self._provider is None:
            raise ValueError("Agent requires a provider. Pass provider= to the constructor.")
        return self._provider

    def _resolve_run_max_delegation_depth(
        self, requested: int | None | _UnsetType
    ) -> int | None | _UnsetType:
        """Resolve max_delegation_depth for a run/stream call.

        Returns the value to pass to the runner, or _UNSET if the runner
        should handle parent/default logic itself.

        Precedence: explicit run kwarg → agent default → _UNSET (runner decides).
        """
        if not isinstance(requested, _UnsetType):
            return requested
        if not isinstance(self.max_delegation_depth, _UnsetType):
            return self.max_delegation_depth
        return _UNSET

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate(self) -> None:
        """Validate agent configuration at creation time.

        Governance name validation (deny/require_approval names exist in
        tools) is deferred to discovery time when tool_sources are present,
        because MCP tool names aren't known until the server is connected.
        """
        if not self.prompt:
            raise ValueError(
                f"Agent '{self.name}' requires a prompt. "
                f"Set prompt as a class attribute or pass it to the constructor."
            )

        # Validate max_delegation_depth (catches subclass defaults like -1)
        _validate_max_delegation_depth(self.max_delegation_depth)

        # SingleCall cannot have tools or tool_sources
        from dendrux.loops.single import SingleCall

        if isinstance(self._loop, SingleCall) and self.tools:
            tool_names = [getattr(fn, "__name__", str(fn)) for fn in self.tools]
            raise ValueError(
                f"Agent '{self.name}' uses SingleCall loop but has {len(self.tools)} "
                f"tool(s): {tool_names}. SingleCall agents cannot have tools — tool "
                f"execution requires multi-turn iteration. Skills are still supported "
                f"on SingleCall (delivered inline into the system prompt). Either "
                f"remove the tools or switch to ReActLoop."
            )

        if isinstance(self._loop, SingleCall) and self._tool_sources:
            raise ValueError(
                f"Agent '{self.name}' uses SingleCall loop but has "
                f"{len(self._tool_sources)} tool_source(s). SingleCall agents cannot "
                f"have tool_sources for the same reason as tools — multi-turn execution "
                f"is required. Skills (inline-delivered) are still supported."
            )

        if isinstance(self._loop, SingleCall) and self._before_tool is not None:
            raise ValueError(
                f"Agent '{self.name}' has before_tool but uses SingleCall loop. "
                f"before_tool is not supported with SingleCall (no tool calls)."
            )

        # output_type requires SingleCall in v1
        if self._output_type is not None and not isinstance(self._loop, SingleCall):
            raise ValueError(
                f"Agent '{self.name}' has output_type={self._output_type.__name__} "
                f"but does not use SingleCall loop. Structured output is only "
                f"supported with SingleCall in this version. Either set "
                f"loop=SingleCall() or remove output_type."
            )

        for fn in self.tools:
            if not is_tool(fn):
                raise ValueError(
                    f"Agent '{self.name}' has a non-tool in its tools list: "
                    f"'{getattr(fn, '__name__', fn)}'. Decorate it with @tool()."
                )

        # Reserve 'use_skill' name when skills are configured
        has_skills = bool(self._skills_dir or self._explicit_skills)
        if has_skills:
            for fn in self.tools:
                if getattr(fn, "__name__", None) in NAMES:
                    raise ValueError(
                        f"Agent '{self.name}' has a tool named '{fn.__name__}' but also "
                        f"has skills configured. '{fn.__name__}' is reserved "
                        "for skill runtime tools when skills are present."
                    )

        if self.max_iterations < 1:
            raise ValueError(
                f"Agent '{self.name}' max_iterations must be >= 1, got {self.max_iterations}."
            )
        if self.max_iterations > MAX_ITERATIONS_CEILING:
            raise ValueError(
                f"Agent '{self.name}' max_iterations cannot exceed "
                f"{MAX_ITERATIONS_CEILING}, got {self.max_iterations}."
            )

        has_mcp = bool(self._tool_sources)

        # --- Shared tool name set for deny/require_approval validation ---
        known_tools: frozenset[str] = frozenset()
        if self.tools and (self._deny or self._require_approval):
            known_tools = frozenset(str(getattr(fn, "__name__", fn)) for fn in self.tools)

        if has_skills:
            known_tools |= NAMES

        # --- deny= validation ---
        if self._deny:
            # deny requires tools or tool_sources
            if not self.tools and not has_mcp and not has_skills:
                raise ValueError(f"Agent '{self.name}' has deny={sorted(self._deny)} but no tools.")

            # deny + SingleCall is not supported
            if isinstance(self._loop, SingleCall):
                raise ValueError(
                    f"Agent '{self.name}' has deny={sorted(self._deny)} but uses "
                    f"SingleCall loop. deny is not supported with SingleCall."
                )

            # Name existence check — skip if tool_sources present (names
            # may refer to MCP tools not yet discovered). Validated at
            # discovery time in _validate_governance_names().
            if not has_mcp:
                unknown = self._deny - known_tools
                if unknown:
                    raise ValueError(
                        f"Agent '{self.name}' deny contains unknown tool(s): "
                        f"{sorted(unknown)}. Available tools: {sorted(known_tools)}."
                    )

        # --- require_approval= validation ---
        if self._require_approval:
            # require_approval requires tools or tool_sources
            if not self.tools and not has_mcp and not has_skills:
                raise ValueError(
                    f"Agent '{self.name}' has require_approval="
                    f"{sorted(self._require_approval)} but no tools."
                )

            # require_approval + SingleCall is not supported
            if isinstance(self._loop, SingleCall):
                raise ValueError(
                    f"Agent '{self.name}' has require_approval="
                    f"{sorted(self._require_approval)} but uses SingleCall loop. "
                    f"require_approval is not supported with SingleCall."
                )

            # Name existence check — skip if tool_sources present (names
            # may refer to MCP tools not yet discovered). Full check at
            # discovery time in _validate_governance_names().
            if not has_mcp:
                unknown_approval = self._require_approval - known_tools
                if unknown_approval:
                    raise ValueError(
                        f"Agent '{self.name}' require_approval contains unknown tool(s): "
                        f"{sorted(unknown_approval)}. Available tools: {sorted(known_tools)}."
                    )

            # Target check on LOCAL tools — always runs, even with
            # tool_sources. Catches client-tool approval errors early
            # without waiting for MCP discovery.
            if self.tools:
                from dendrux.tool import get_tool_def as _get_tool_def
                from dendrux.types import ToolTarget as _ToolTarget

                for fn in self.tools:
                    td = _get_tool_def(fn)
                    if td.name in self._require_approval and td.target != _ToolTarget.SERVER:
                        raise ValueError(
                            f"Agent '{self.name}' require_approval contains "
                            f"'{td.name}' which has target={td.target.value}. "
                            f"require_approval only supports server tools."
                        )

            # deny and require_approval must not overlap
            overlap = self._deny & self._require_approval
            if overlap:
                raise ValueError(
                    f"Agent '{self.name}' has tools in both deny and "
                    f"require_approval: {sorted(overlap)}. "
                    f"A tool cannot be both denied and require approval."
                )

    def _validate_governance_names(self, all_tool_defs: dict[str, ToolDef]) -> None:
        """Validate deny/require_approval names against full tool set.

        Called at discovery time after MCP tools are discovered. This is
        the deferred half of governance validation — catches typos in
        deny/require_approval that refer to MCP tool names AND enforces
        the require_approval server-tools-only invariant on the merged set.
        """
        from dendrux.types import ToolTarget as _ToolTarget

        all_names = frozenset(all_tool_defs.keys())
        if self._skills_dir or self._explicit_skills:
            all_names |= NAMES

        if self._deny:
            unknown = self._deny - all_names
            if unknown:
                raise ValueError(
                    f"Agent '{self.name}' deny contains unknown tool(s): "
                    f"{sorted(unknown)}. Available tools: {sorted(all_names)}."
                )

        if self._require_approval:
            unknown_approval = self._require_approval - all_names
            if unknown_approval:
                raise ValueError(
                    f"Agent '{self.name}' require_approval contains unknown tool(s): "
                    f"{sorted(unknown_approval)}. Available tools: {sorted(all_names)}."
                )

            # require_approval tools must be server tools (including MCP + local)
            for name in self._require_approval:
                td = all_tool_defs.get(name)
                if td is not None and td.target != _ToolTarget.SERVER:
                    raise ValueError(
                        f"Agent '{self.name}' require_approval contains "
                        f"'{name}' which has target={td.target.value}. "
                        f"require_approval only supports server tools."
                    )

    # ------------------------------------------------------------------
    # Persistence (lazy init)
    # ------------------------------------------------------------------

    async def _resolve_state_store(self) -> StateStore | None:
        """Lazily resolve the state store from config.

        Resolution order:
          1. Explicit state_store from constructor → returned as-is
          2. database_url from constructor → private engine (agent-owned)
          3. DENDRUX_DATABASE_URL env var → shared global engine
          4. None (ephemeral mode)

        Ownership rule:
          - Explicit database_url creates a private engine disposed by close().
          - Env-var path uses the shared singleton from get_engine() — NOT
            owned by this agent, NOT disposed by close().
        """
        if self._state_store is not None:
            return self._state_store

        if self._lazy_store is not None:
            return self._lazy_store

        from dendrux.db._schema_check import was_verified
        from dendrux.runtime.state import SQLAlchemyStateStore

        if self._database_url is not None:
            with self._store_lock:
                generation = self._store_generation
            engine = await self._create_private_engine(self._database_url)
            try:
                if not was_verified(engine):
                    await _verify_schema_initialized(engine, self._database_url)
            except BaseException:
                await engine.dispose()
                raise
            with self._store_lock:
                if self._store_generation != generation:
                    closed = True
                elif self._lazy_store is not None:
                    closed = False  # a concurrent first use published first
                else:
                    self._private_engine = engine
                    self._lazy_store = SQLAlchemyStateStore(engine)
                    return self._lazy_store
            await engine.dispose()
            if closed:
                raise RuntimeError("Agent was closed while its database was initializing.")
            assert self._lazy_store is not None
            return self._lazy_store

        env_url = os.environ.get("DENDRUX_DATABASE_URL")
        if env_url is not None:
            from dendrux.db.session import get_engine

            _warn_env_store_deprecated()
            engine = await get_engine(env_url)
            if not was_verified(engine):
                await _verify_schema_initialized(engine, env_url)
            self._lazy_store = SQLAlchemyStateStore(engine)
            return self._lazy_store

        return None

    async def _create_private_engine(self, url: str) -> AsyncEngine:
        """Create an agent-owned engine from an explicit database_url.

        Unlike the global get_engine() singleton, this creates a fresh engine
        per agent so multiple agents with different URLs can coexist.
        database_options are applied here.
        """
        from pathlib import Path

        from sqlalchemy.ext.asyncio import create_async_engine

        from dendrux.db._schema_check import ensure_schema_current
        from dendrux.db._sqlite_sync import create_sqlite_schema

        connect_args: dict[str, Any] = {}
        if url.startswith("sqlite"):
            connect_args["check_same_thread"] = False

        engine = create_async_engine(
            url,
            echo=False,
            connect_args=connect_args,
            **self._database_options,
        )

        try:
            if url.startswith("sqlite"):
                # Auto-create tables for SQLite (zero-config promise)
                db_path = url.split("///", 1)[-1] if "///" in url else None
                if db_path:
                    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
                await create_sqlite_schema(engine)
            else:
                await ensure_schema_current(engine)
        except BaseException:
            await engine.dispose()
            raise

        return engine

    # ------------------------------------------------------------------
    # Runtime methods
    # ------------------------------------------------------------------

    @overload
    async def run(
        self,
        user_input: UserContent,
        *,
        history: list[ChatMessage] | None = ...,
        context: list[ContextBlock] | None = ...,
        activate_skills: Sequence[str] | None = ...,
        runs: list[str] | None = ...,
        replay_last: int | Literal["all"] = ...,
        replay_reasoning: bool = ...,
        tenant_id: str | None = ...,
        metadata: dict[str, Any] | None = ...,
        notifier: LoopNotifier | None = ...,
        max_delegation_depth: int | None,
        idempotency_key: str | None = ...,
        output_type: type[BaseModel] | None = ...,
        **kwargs: Any,
    ) -> RunResult: ...

    @overload
    async def run(
        self,
        user_input: UserContent,
        *,
        history: list[ChatMessage] | None = ...,
        context: list[ContextBlock] | None = ...,
        activate_skills: Sequence[str] | None = ...,
        runs: list[str] | None = ...,
        replay_last: int | Literal["all"] = ...,
        replay_reasoning: bool = ...,
        tenant_id: str | None = ...,
        metadata: dict[str, Any] | None = ...,
        notifier: LoopNotifier | None = ...,
        idempotency_key: str | None = ...,
        output_type: type[BaseModel] | None = ...,
        **kwargs: Any,
    ) -> RunResult: ...

    async def run(
        self,
        user_input: UserContent,
        *,
        history: list[ChatMessage] | None = None,
        context: list[ContextBlock] | None = None,
        activate_skills: Sequence[str] | None = None,
        runs: list[str] | None = None,
        replay_last: int | Literal["all"] = 0,
        replay_reasoning: bool = True,
        tenant_id: str | None | _OmittedType = _OMITTED,
        metadata: dict[str, Any] | None = None,
        notifier: LoopNotifier | None = None,
        max_delegation_depth: int | None | _UnsetType = _UNSET,
        idempotency_key: str | None = None,
        output_type: type[BaseModel] | None | _UnsetType = _UNSET,
        **kwargs: Any,
    ) -> RunResult:
        """Start a new agent run.

        Delegates to runner.run() with this agent's provider and persistence
        config.

        Args:
            user_input: The user's input: a string, or a sequence of strings
                and images (``Image``, ``ImageRef``, ``ImagePart``) in order.
                Images are validated and stored before the run is created;
                an image error (``ImageError``, ``UnsupportedModalityError``
                for a model that cannot take images, ...) raises before any
                run row exists.
            history: Optional prior conversation turns to seed the run with —
                for building chatbots on top of Dendrux. Each ``ChatMessage``
                is a ``role`` (USER or ASSISTANT) plus ``content``; user turns
                may carry images (``ChatMessage.user([...])``), which count as
                already sent. The dev's
                app owns conversation storage; Dendrux only reads ``history``
                as input for this turn and does NOT persist it to
                ``react_traces``. Validation rules: must start with USER, end
                with ASSISTANT, no consecutive same-role messages, no empty
                content. Invalid history raises ``ValueError`` before any
                run is created. ``None`` and ``[]`` mean "no prior context."
            context: Optional per-run context blocks (``ContextBlock``) folded
                into the message content — retrieved docs, project
                instructions/memory, referenced files. ``placement="stable"``
                blocks fold into the first user message (cached frozen prefix);
                ``"dynamic"`` (default) fold into the current user message
                (volatile tail). Read-only, never persisted as forward state;
                the idempotency hash includes it. See ``dendrux.context_blocks``.
            runs: Optional ordered run ids of this conversation branch (the
                app owns the thread; Dendrux has no thread concept). Each run
                is rebuilt from its own persisted traces and replayed ahead of
                ``history`` and the new input, so the model remembers its
                earlier tool work. Requires persistence. Runs must exist,
                belong to ``tenant_id``, and be terminal, else
                ``ReplayRunError`` is raised before anything is created.
                Cannot be combined with ``placement="stable"`` context blocks
                (the first replayed run already carries its stable band).
            replay_last: How many of the most recent ``runs`` replay in full
                (text + tool calls + results); older ones replay as question +
                answer only. ``0`` (default) = all text-only; ``"all"`` = all
                full. SingleCall agents accept only ``0``.
            replay_reasoning: Also replay stored reasoning blocks for runs in
                the full window (default True). Kept only while a run's
                recorded prefix (provider, model, rendered system prompt,
                tools, and what preceded it) matches the current request;
                from the first mismatch onward reasoning is dropped with a
                warning. Forced off when guardrails are configured. What was
                replayed and dropped is reported on ``RunResult.replay``.
            tenant_id: Optional tenant ID for multi-tenant isolation.
            metadata: Optional developer linking data (thread_id, user_id, etc.).
            notifier: Optional notifier for lifecycle events (e.g. ConsoleNotifier
                for terminal output, custom notifiers for Slack/Telegram/etc.).
                Composed with PersistenceRecorder internally if persistence is enabled.
            max_delegation_depth: Maximum allowed delegation depth for the run
                tree. Default 10. None means unbounded. Child runs inherit
                this limit automatically.
            idempotency_key: Optional key for duplicate run prevention. If a run
                with this key already exists and completed, the cached result is
                returned. If still active, RunAlreadyActiveError is raised. If the
                same key is reused with different input, IdempotencyConflictError
                is raised. Requires persistence (database_url, state_store, or
                DENDRUX_DATABASE_URL). The hash includes ``history``, ``context``
                and the ``runs`` / ``replay_last`` / ``replay_reasoning`` knobs —
                same input with a different prefix is a different request.
            **kwargs: Forwarded to the LLM provider. Pass ``model=`` to override
                the provider's configured model for this call only (e.g.
                ``model="claude-opus-4-1"``) — useful for per-turn model selection
                in a chatbot. Also accepts ``temperature``, ``max_tokens``, and
                other provider sampling params. The model actually used is recorded
                per LLM call (see ``RunStore.get_llm_calls``).

        Returns:
            RunResult with status, answer, steps, and usage stats. On success,
            ``answer`` is the final answer turn, not the concatenation of text
            across tool calls. Collect stream deltas or use RunStore.get_traces()
            for earlier assistant text. Stream cancellation preserves only the
            current text buffer (reset at each TOOL_RESULT), possibly None.
            Cancelling this coroutine finalizes a still-running persisted run
            as cancelled and re-raises CancelledError; no RunResult or in-flight
            model text is returned. :meth:`cancel_run` from the same process
            (same Agent instance) instead interrupts the run wherever it is
            awaiting — provider call or tool — and this call returns
            ``RunResult(status=CANCELLED)`` with ``meta["interrupted"]=True``
            and ``meta["partial_output"]=False``. A run paused via
            :meth:`request_pause` returns with ``status=PAUSED``.

        Raises:
            ValueError: If no provider is configured, if idempotency_key
                is provided without persistence, or if the replay knobs are
                invalid (``ReplayRunError`` for a bad run id).
            RunAlreadyActiveError: If idempotency_key matches an active run.
            IdempotencyConflictError: If idempotency_key is reused with
                different request parameters.
            DelegationDepthExceededError: If this run would exceed the inherited
                max_delegation_depth from a parent run.
        """
        tenant_id = entry_checks(
            self,
            "run",
            tenant_id,
            default_tenant=None,
            metadata=metadata,
            user_input=user_input,
            history=history,
            context=context,
            provider_kwargs=kwargs,
        )
        _validate_max_delegation_depth(max_delegation_depth)
        provider = self._require_provider()

        # Resolve output_type: run-level overrides agent default
        resolved_output_type = (
            self._output_type if isinstance(output_type, _UnsetType) else output_type
        )

        # Validate output_type + loop compatibility at call time
        if resolved_output_type is not None:
            from dendrux.loops.single import SingleCall

            effective_loop = self._loop
            if effective_loop is None:
                # Runner will default to ReActLoop — that's not SingleCall
                raise ValueError(
                    f"output_type={resolved_output_type.__name__} requires SingleCall loop. "
                    f"Set loop=SingleCall() on the Agent or remove output_type."
                )
            if not isinstance(effective_loop, SingleCall):
                raise ValueError(
                    f"output_type={resolved_output_type.__name__} is only supported with "
                    f"SingleCall loop, not {type(effective_loop).__name__}."
                )

        store = await self._resolve_state_store()

        if idempotency_key is not None and store is None:
            raise ValueError(
                "idempotency_key requires persistence (database_url, state_store, "
                "or DENDRUX_DATABASE_URL). Idempotency cannot work without durable state."
            )

        from dendrux.runtime.runner import run as runner_run

        run_kwargs: dict[str, Any] = {}
        resolved_depth = self._resolve_run_max_delegation_depth(max_delegation_depth)
        if not isinstance(resolved_depth, _UnsetType):
            run_kwargs["max_delegation_depth"] = resolved_depth

        return await runner_run(
            self,
            provider=provider,
            user_input=user_input,
            history=history,
            context=context,
            activate_skills=activate_skills,
            runs=runs,
            replay_last=replay_last,
            replay_reasoning=replay_reasoning,
            state_store=store,
            tenant_id=tenant_id,
            metadata=metadata,
            extra_notifier=notifier,
            idempotency_key=idempotency_key,
            output_type=resolved_output_type,
            **run_kwargs,
            **kwargs,
        )

    async def retry(
        self,
        run_id: str,
        *,
        tenant_id: str | None | _OmittedType = _OMITTED,
        metadata: dict[str, Any] | None = None,
        notifier: LoopNotifier | None = None,
        **kwargs: Any,
    ) -> RunResult:
        """Retry a terminal run with approximate prior context.

        Creates a fresh run seeded with the original run's conversation
        history from persisted traces. The LLM sees the prior conversation
        and can continue from context. This is NOT resume — it's a new
        run with approximate context, fresh counters, fresh usage.

        Works for any terminal run (error, cancelled, max_iterations,
        success). Source run must have used a retry-capable loop (not
        SingleCall). The retry agent can differ from the original.

        Args:
            run_id: The terminal run to retry from.
            tenant_id: Optional tenant ID for the retry run.
            metadata: Optional developer metadata for the retry run.
            notifier: Optional notifier for lifecycle events.
            **kwargs: Forwarded to the LLM provider.

        Returns:
            RunResult from the retry run.

        Raises:
            ValueError: If no provider/persistence configured, if the
                source run is not terminal, or if the source run was
                SingleCall.
        """
        tenant_id = entry_checks(
            self,
            "retry",
            tenant_id,
            default_tenant=None,
            run_id=run_id,
            metadata=metadata,
            provider_kwargs=kwargs,
        )
        provider = self._require_provider()
        store = await self._resolve_state_store()

        if store is None:
            raise ValueError(
                "retry() requires persistence (database_url, state_store, "
                "or DENDRUX_DATABASE_URL). Retry reads traces from the original run."
            )

        from dendrux.runtime.runner import retry as runner_retry

        return await runner_retry(
            run_id,
            agent=self,
            provider=provider,
            state_store=store,
            tenant_id=tenant_id,
            metadata=metadata,
            extra_notifier=notifier,
            **kwargs,
        )

    async def resume(
        self,
        run_id: str,
        *,
        tool_results: list[ToolResult] | None = None,
        user_input: UserContent | None = None,
        notifier: LoopNotifier | None = None,
        tenant_id: TenantScope | _OmittedType = _OMITTED,
        expected_pause_id: str | None = None,
    ) -> RunResult:
        """Resume a paused run.

        Three modes:
          - tool_results provided -> store + resume with tool results
          - user_input provided -> resume with clarification answer
          - neither -> approve a WAITING_APPROVAL batch, or continue a
            PAUSED run (see :meth:`request_pause`) with its full history

        Args:
            run_id: The paused run's ID.
            tool_results: Results for pending tool calls (for client tool runs).
                ``ToolResult.images`` are ingested before the run is claimed.
            user_input: Clarification answer (for human-in-the-loop runs): a
                string, or strings and images.
            expected_pause_id: Current pause generation; stale submissions are rejected.
            notifier: Optional additional loop notifier. Composed with
                PersistenceRecorder internally.
            tenant_id: Omitted = no tenant check. Otherwise the run must belong
                to this tenant (``None`` = untenanted) or ``RunNotFoundError``
                is raised before anything is written.

        Returns:
            RunResult with updated status.

        Raises:
            ValueError: If provider not set, persistence not configured,
                both args provided, or neither provided.
            RunNotFoundError: ``run_id`` is outside ``tenant_id``.
        """
        tenant_id = entry_checks(
            self,
            "resume",
            tenant_id,
            default_tenant=UNSCOPED,
            run_id=run_id,
            tool_results=tool_results,
            user_input=user_input,
        )
        # Validate args before any I/O
        provider = self._require_provider()
        if tool_results is not None and user_input is not None:
            raise ValueError("Cannot provide both tool_results and user_input to resume().")

        store = await self._resolve_state_store()
        if store is None:
            raise ValueError(
                "resume() requires persistence. "
                "Pass database_url or state_store to the constructor."
            )

        from dendrux.runtime.runner import resume as runner_resume
        from dendrux.runtime.runner import resume_with_input

        await _require_run_in_scope(store, run_id, tenant_id)

        if tool_results is not None:
            return await runner_resume(
                run_id,
                tool_results,
                state_store=store,
                agent=self,
                provider=provider,
                extra_notifier=notifier,
                expected_pause_id=expected_pause_id,
            )
        if user_input is not None:
            return await resume_with_input(
                run_id,
                user_input,
                state_store=store,
                agent=self,
                provider=provider,
                extra_notifier=notifier,
                expected_pause_id=expected_pause_id,
            )
        # No-arg resume — approve path. Runner validates status is
        # WAITING_APPROVAL and executes pending tools.
        return await runner_resume(
            run_id,
            None,
            state_store=store,
            agent=self,
            provider=provider,
            extra_notifier=notifier,
            expected_pause_id=expected_pause_id,
        )

    async def submit_tool_results(
        self,
        run_id: str,
        results: list[ToolResult],
        *,
        notifier: LoopNotifier | None = None,
        tenant_id: TenantScope | _OmittedType = _OMITTED,
        expected_pause_id: str | None = None,
    ) -> RunResult:
        """Submit client-side tool results and resume the paused run.

        Race-safe: persist-first handoff + CAS claim in one atomic step,
        then blocks until the run reaches the next pause or terminal
        state.

        Args:
            run_id: A run in status ``WAITING_CLIENT_TOOL``.
            results: One :class:`ToolResult` per pending tool call. Must
                cover every pending call; unknown or missing ids raise
                ``InvalidToolResultError``. Text and JSON payloads keep their original
                bytes unless capped or changed by ``tool_result_transform``.
                ``ToolResult.images`` are validated and stored before the claim; an image error
                (``ImageError``, ``UnsupportedModalityError``) raises with the
                run still waiting.
            expected_pause_id: Current pause generation; stale submissions are rejected.
            notifier: Optional additional notifier composed during resume.
            tenant_id: Omitted = no tenant check. Otherwise the run must belong
                to this tenant (``None`` = untenanted) or ``RunNotFoundError``
                is raised before anything is written.

        Returns:
            :class:`RunResult` from the resumed loop.

        Raises:
            PersistenceNotConfiguredError: Agent has no DB.
            RunNotFoundError: ``run_id`` does not exist (or is outside
                ``tenant_id``).
            RunNotPausedError: run is not in a paused state.
            PauseStatusMismatchError: run is paused, but not for client tools.
            RunAlreadyClaimedError: a concurrent submit won the CAS.
            RunAlreadyTerminalError: run already reached a terminal state.
            InvalidToolResultError: results don't match pending calls.
        """
        tenant_id = entry_checks(
            self,
            "submit_tool_results",
            tenant_id,
            default_tenant=UNSCOPED,
            run_id=run_id,
            tool_results=results,
        )
        from dendrux.errors import PersistenceNotConfiguredError
        from dendrux.runtime.runner import resume_claimed
        from dendrux.runtime.submit import (
            claim_or_raise,
            prepare_client_results,
            serialize_tool_results,
            validate_tool_results,
        )
        from dendrux.types import RunStatus

        provider = self._require_provider()
        store = await self._resolve_state_store()
        if store is None:
            raise PersistenceNotConfiguredError()

        await _require_run_in_scope(store, run_id, tenant_id)
        skill_session = await continuation(self, store, run_id)
        with skill_session.bind():
            pause = await validate_tool_results(store, run_id, results)
            from dendrux.runtime.submit import warn_missing_pause_id

            warn_missing_pause_id(self, expected_pause_id, pause_id=pause.pause_id)
            results = await prepare_client_results(self, run_id, pause, results)
            claim_kwargs: dict[str, Any] = {}
            if any(r.images for r in results):
                from dendrux.runtime.runner import ingest_submitted_tool_images

                results, claim_kwargs = await ingest_submitted_tool_images(
                    self, provider, store, run_id, pause, results
                )
            epoch = await claim_or_raise(
                store,
                run_id,
                expected_status=RunStatus.WAITING_CLIENT_TOOL,
                submitted_data={"submitted_tool_results": serialize_tool_results(results)},
                agent=self,
                expected_pause_id=(
                    expected_pause_id if expected_pause_id is not None else pause.pause_id
                ),
                call_results=results,
                **claim_kwargs,
            )
            from dendrux.runtime.state import claimed_store

            store = claimed_store(store, run_id, epoch)

            task = self._task_manager.spawn(
                run_id,
                resume_claimed(
                    run_id,
                    state_store=store,
                    agent=self,
                    provider=provider,
                    extra_notifier=notifier,
                    pre_claim_status=RunStatus.WAITING_CLIENT_TOOL.value,
                ),
            )
            return await task

    async def submit_input(
        self,
        run_id: str,
        text: UserContent,
        *,
        notifier: LoopNotifier | None = None,
        tenant_id: TenantScope | _OmittedType = _OMITTED,
        expected_pause_id: str | None = None,
    ) -> RunResult:
        """Submit clarification input and resume the paused run.

        Race-safe: persist-first + CAS claim, blocks until next pause or
        terminal state.

        Args:
            run_id: A run in status ``WAITING_HUMAN_INPUT``.
            text: The user's clarification answer: a string, or strings and
                images (ingested before the claim).
            expected_pause_id: Current pause generation; stale submissions are rejected.
            notifier: Optional additional notifier composed during resume.
            tenant_id: Omitted = no tenant check. Otherwise the run must belong
                to this tenant (``None`` = untenanted) or ``RunNotFoundError``
                is raised before anything is written.

        Raises:
            PersistenceNotConfiguredError, RunNotFoundError (also when
            ``run_id`` is outside ``tenant_id``), RunNotPausedError,
            PauseStatusMismatchError, RunAlreadyClaimedError,
            RunAlreadyTerminalError.
        """
        tenant_id = entry_checks(
            self,
            "submit_input",
            tenant_id,
            default_tenant=UNSCOPED,
            run_id=run_id,
            user_input=text,
        )
        from dendrux.errors import PersistenceNotConfiguredError
        from dendrux.runtime.runner import resume_claimed
        from dendrux.runtime.submit import claim_or_raise
        from dendrux.types import RunStatus

        provider = self._require_provider()
        store = await self._resolve_state_store()
        if store is None:
            raise PersistenceNotConfiguredError()

        from dendrux.chat import split_user_content

        projection, items = split_user_content(text, owner="text")
        await _require_run_in_scope(store, run_id, tenant_id)
        skill_session = await continuation(self, store, run_id)
        with skill_session.bind():
            submitted: dict[str, Any] = {"submitted_user_input": projection}
            claim_kwargs: dict[str, Any] = {}
            if items:
                from dendrux.runtime.runner import ingest_submitted_user_parts

                submitted, claim_kwargs = await ingest_submitted_user_parts(
                    self, provider, store, run_id, projection, items
                )
            epoch = await claim_or_raise(
                store,
                run_id,
                expected_status=RunStatus.WAITING_HUMAN_INPUT,
                submitted_data=submitted,
                agent=self,
                expected_pause_id=expected_pause_id,
                **claim_kwargs,
            )
            from dendrux.runtime.state import claimed_store

            store = claimed_store(store, run_id, epoch)

            task = self._task_manager.spawn(
                run_id,
                resume_claimed(
                    run_id,
                    state_store=store,
                    agent=self,
                    provider=provider,
                    extra_notifier=notifier,
                    pre_claim_status=RunStatus.WAITING_HUMAN_INPUT.value,
                ),
            )
            return await task

    async def submit_approval(
        self,
        run_id: str,
        *,
        approved: bool | None = None,
        decisions: Mapping[str, ApprovalDecision] | None = None,
        rejection_reason: str | None = None,
        notifier: LoopNotifier | None = None,
        tenant_id: TenantScope | _OmittedType = _OMITTED,
        expected_pause_id: str | None = None,
    ) -> RunResult:
        """Submit an approval decision for a run awaiting human approval.

        Pass exactly one of ``approved=`` (whole batch) or ``decisions=``
        (per call).

        - ``approved=True``: claims and resumes. Approved and held server
          calls execute server-side and their output is fed to the LLM.
        - ``approved=False``: every pending call receives a synthetic failed
          :class:`ToolResult` whose payload is ``rejection_reason`` as a JSON
          string (default: ``"User declined to run this tool."``). The LLM
          decides what to do next. ``rejection_reason`` is ignored when
          ``approved=True``.
        - ``decisions={call_id: ApprovalDecision.approve() | .deny(reason)}``:
          keys must be exactly the approval-gated call ids (those that fired
          ``approval.requested``); missing, unknown or duplicate ids raise
          ``ValueError`` before anything is claimed. Validation uses the gate
          snapshot persisted at pause time, so a different Agent instance
          (with or without the ``before_tool`` hook) can decide. Denied calls
          get a failed result ``{"denied": reason}``; held non-gated server
          calls execute automatically.

        In every form, client-target calls held in the batch never run on the
        server: after the server calls resolve, the run pauses again as
        ``WAITING_CLIENT_TOOL`` for those calls.

        ``approval.decided`` carries a scalar ``decision``
        (``approved|rejected|mixed``) and ``decisions``: one entry per pending
        call with ``approved|denied|not_gated_executed|client_pending``.

        Args:
            run_id: A run in status ``WAITING_APPROVAL``.
            approved: Whole-batch decision.
            decisions: Per-call decisions keyed by tool call id.
            rejection_reason: Reason used with ``approved=False``.
            expected_pause_id: Current pause generation; stale submissions are rejected.
            notifier: Optional additional notifier composed during resume.
            tenant_id: Omitted = no tenant check. Otherwise the run must belong
                to this tenant (``None`` = untenanted) or ``RunNotFoundError``
                is raised before anything is written.

        Raises:
            ValueError: invalid argument combination or decision ids.
            PersistenceNotConfiguredError, RunNotFoundError (also when
            ``run_id`` is outside ``tenant_id``), RunNotPausedError,
            PauseStatusMismatchError, RunAlreadyClaimedError,
            RunAlreadyTerminalError.
        """
        tenant_id = entry_checks(
            self,
            "submit_approval",
            tenant_id,
            default_tenant=UNSCOPED,
            run_id=run_id,
            rejection_reason=rejection_reason,
            decisions=decisions,
        )
        from dendrux.errors import PersistenceNotConfiguredError
        from dendrux.runtime.runner import resume_approval_approved_claim, resume_claimed
        from dendrux.runtime.submit import (
            build_rejection_results,
            claim_or_raise,
            normalize_decisions,
            preflight_status,
            raise_for_non_paused_status,
            serialize_decisions,
            serialize_tool_results,
            validate_approval_decisions,
            warn_missing_pause_id,
        )
        from dendrux.types import PauseState, RunStatus

        if (approved is None) == (decisions is None):
            raise ValueError("submit_approval() takes exactly one of approved= or decisions=.")
        if decisions is not None and rejection_reason is not None:
            raise ValueError(
                "rejection_reason is only valid with approved=False; "
                "give per-call reasons with ApprovalDecision.deny(reason)."
            )
        pairs = normalize_decisions(decisions) if decisions is not None else None

        provider = self._require_provider()
        store = await self._resolve_state_store()
        if store is None:
            raise PersistenceNotConfiguredError()

        await _require_run_in_scope(store, run_id, tenant_id)
        skill_session = await continuation(self, store, run_id)
        with skill_session.bind():
            if pairs is not None:
                await preflight_status(store, run_id, RunStatus.WAITING_APPROVAL)
                raw_pause = await store.get_pause_state(run_id)
                if raw_pause is None:
                    await raise_for_non_paused_status(store, run_id)
                pause_state = PauseState.from_dict(raw_pause)
                target_lookup = {
                    get_tool_def(fn).name: get_tool_def(fn).target for fn in self.tools
                }
                decision_map = validate_approval_decisions(
                    run_id, pause_state, pairs, target_lookup
                )

                warn_missing_pause_id(
                    self,
                    expected_pause_id,
                    pause_id=pause_state.pause_id,
                    approval=any(decision.approved for decision in decision_map.values()),
                )

                def same_pause_generation(current: dict[str, Any]) -> None:
                    # Re-validate against the exact pause being claimed: a
                    # newer approval pause (ABA) must reject stale decisions
                    # and stay WAITING_APPROVAL.
                    validate_approval_decisions(
                        run_id, PauseState.from_dict(current), pairs, target_lookup
                    )

                epoch = await claim_or_raise(
                    store,
                    run_id,
                    expected_status=RunStatus.WAITING_APPROVAL,
                    submitted_data={
                        "submitted_decisions": serialize_decisions(pause_state, decision_map)
                    },
                    validate=same_pause_generation,
                    agent=self,
                    expected_pause_id=(
                        expected_pause_id if expected_pause_id is not None else pause_state.pause_id
                    ),
                )
                from dendrux.runtime.state import claimed_store

                store = claimed_store(store, run_id, epoch)
                task = self._task_manager.spawn(
                    run_id,
                    resume_claimed(
                        run_id,
                        state_store=store,
                        agent=self,
                        provider=provider,
                        extra_notifier=notifier,
                        pre_claim_status=RunStatus.WAITING_APPROVAL.value,
                    ),
                )
                return await task

            if approved:
                epoch = await claim_or_raise(
                    store,
                    run_id,
                    expected_status=RunStatus.WAITING_APPROVAL,
                    agent=self,
                    expected_pause_id=expected_pause_id,
                    approval=True,
                )
                from dendrux.runtime.state import claimed_store

                store = claimed_store(store, run_id, epoch)
                task = self._task_manager.spawn(
                    run_id,
                    resume_approval_approved_claim(
                        run_id,
                        state_store=store,
                        agent=self,
                        provider=provider,
                        extra_notifier=notifier,
                    ),
                )
                return await task

            # Rejection path — fabricate one failed result per pending tool call,
            # then submit-and-claim so the runner's resume_claimed sees them as
            # ordinary tool results.
            raw_pause = await store.get_pause_state(run_id)
            if raw_pause is None:
                await raise_for_non_paused_status(store, run_id)
            pause_state = PauseState.from_dict(raw_pause)
            warn_missing_pause_id(self, expected_pause_id, pause_id=pause_state.pause_id)
            synthetic = build_rejection_results(pause_state, rejection_reason)
            epoch = await claim_or_raise(
                store,
                run_id,
                expected_status=RunStatus.WAITING_APPROVAL,
                submitted_data={"submitted_tool_results": serialize_tool_results(synthetic)},
                agent=self,
                call_results=synthetic,
                expected_pause_id=(
                    expected_pause_id if expected_pause_id is not None else pause_state.pause_id
                ),
            )
            from dendrux.runtime.state import claimed_store

            store = claimed_store(store, run_id, epoch)
            task = self._task_manager.spawn(
                run_id,
                resume_claimed(
                    run_id,
                    state_store=store,
                    agent=self,
                    provider=provider,
                    extra_notifier=notifier,
                    pre_claim_status=RunStatus.WAITING_APPROVAL.value,
                ),
            )
            return await task

    async def cancel_run(
        self,
        run_id: str,
        *,
        cascade: bool = False,
        tenant_id: TenantScope | _OmittedType = _OMITTED,
    ) -> RunResult:
        """Request cancellation for ``run_id``.

        Behavior by run state:

        - **Paused run (any process; WAITING_* or PAUSED):** atomic CAS
          finalize to ``CANCELLED`` in one round-trip. Deterministic.
        - **Running run (any process, persisted):** sets
          ``cancel_requested=True`` on the DB row. The runner observes
          this flag at the top of the next iteration (and at the
          pre-pause checkpoint after the current iteration's loop body
          returns) and exits cleanly as ``CANCELLED``. Cancellation wins
          over a pending pause request. Cancellation is cooperative: a run
          that finishes before observing the request (its last LLM call
          returned a final answer) completes normally — its work and side
          effects happened — and the request is cleared. A stale RUNNING
          row swept with the request pending is finalized ``CANCELLED``.
        - **Batch** :meth:`run` **executing in this process on this Agent
          instance:** additionally cancels the task awaiting it, so an
          in-flight provider call or tool is interrupted. That ``run()``
          call returns ``RunResult(status=CANCELLED)`` with
          ``meta["interrupted"] = True`` and ``meta["partial_output"] =
          False`` (a batch call has no partial text). A caller cancelling
          its own task still gets ``CancelledError``.
        - **Streamed run owned by this Agent instance** (``stream()`` /
          ``resume_stream()``): additionally sets an in-process interrupt
          signal. The loop races every provider pull against it, so an
          in-flight model stream is abandoned immediately — even while
          waiting for the next token. The stream then yields
          ``RUN_CANCELLED`` whose ``answer`` is the text streamed so far
          (``meta["interrupted"]`` / ``meta["partial_output"]``). Once the
          provider's final event has been received, completion wins and
          the run finishes as ``success``.
        - **In-process submit/resume task on this Agent instance:** the
          asyncio task spawned by :meth:`submit_tool_results`,
          :meth:`submit_input`, :meth:`submit_approval` or
          :meth:`submit_resume` is cancelled synchronously in addition to
          the DB-level cancel signal.
        - **Terminal run:** no-op; returns the current persisted state.
          Does not raise.
        - **Non-persisted runs:** unsupported. ``cancel_run`` requires a
          configured DB. Raises ``PersistenceNotConfiguredError``.

        ``cascade=True`` also cancels every non-terminal descendant run
        (delegated child runs), deepest first and ``run_id`` last, over a
        point-in-time snapshot of the subtree: a child created after the
        snapshot is not included. Per-run outcomes are listed in
        ``meta["cascade"]`` as ``{"run_id", "outcome"}`` with outcome
        ``cancelled`` (finalized now), ``cancel_requested`` (running;
        stops at its next checkpoint) or ``terminal``.

        ``tenant_id``: when given, ``run_id`` must belong to exactly that
        tenant (``None`` = untenanted) or ``RunNotFoundError`` is raised,
        and the cascade only reaches descendants of the same tenant.
        Omitted means unscoped (trusted single-tenant use).

        Returning is *not* confirmation that the run stopped. A result
        with a non-terminal ``status`` and ``meta["cancel_requested"]``
        set means cancellation was requested; the terminal event
        (``RUN_CANCELLED`` on the stream, ``run.cancelled`` in the event
        log, ``status == "cancelled"`` in the store) confirms it.

        Returns:
            :class:`RunResult` reflecting the persisted state of ``run_id``
            after the cancel attempt. When that state is terminal,
            ``meta["undelivered_notes"]`` lists inbox notes the run never
            delivered (only when there are any).

        Raises:
            PersistenceNotConfiguredError: Agent has no DB.
            RunNotFoundError: ``run_id`` does not exist (or is outside
                ``tenant_id``).
        """
        tenant_id = entry_checks(
            self, "cancel_run", tenant_id, default_tenant=UNSCOPED, run_id=run_id
        )
        from dendrux.errors import PersistenceNotConfiguredError, RunNotFoundError

        store = await self._resolve_state_store()
        if store is None:
            raise PersistenceNotConfiguredError()

        # Preflight: distinguish missing from terminal so callers get
        # specific errors / no-op behavior without ambiguity.
        record = await store.get_run(run_id)
        if record is None or not _in_tenant_scope(record, tenant_id):
            raise RunNotFoundError(run_id)

        outcomes: list[dict[str, str]] = []
        # Descendants are signalled together with the root, after every
        # durable request: a parent and child sharing one task must both
        # register their interrupt before the task sees the cancellation.
        to_signal: list[str] = []
        if cascade:
            get_subtree = require_store_method(store, "get_subtree")
            subtree = await get_subtree(run_id, tenant_id=tenant_id)
            if subtree is not None:
                for rid in _cascade_order(subtree)[:-1]:
                    outcome = await self._cancel_one(store, rid, signal=False)
                    outcomes.append({"run_id": rid, "outcome": outcome})
                    if outcome != "terminal":
                        to_signal.append(rid)

        if record.status in _TERMINAL_STATUS_VALUES:
            for rid in to_signal:
                self._task_manager.cancel(rid)
            result = _record_to_run_result(record)
            outcomes.append({"run_id": run_id, "outcome": "terminal"})
        else:
            outcome = await self._cancel_one(store, run_id, signal=False)
            outcomes.append({"run_id": run_id, "outcome": outcome})
            record = await store.get_run(run_id)
            if record is None:  # pragma: no cover — we just preflight-checked
                raise RunNotFoundError(run_id)
            # In-process signals last, after the durable request is recorded
            # and the returned snapshot is read: a locally owned run or task is
            # preempted from here on, and the returned RunResult
            # deterministically reflects "requested" (non-terminal status +
            # meta.cancel_requested) rather than racing the loop's own finalize.
            for rid in to_signal:
                self._task_manager.cancel(rid)
            self._task_manager.cancel(run_id)
            result = _record_to_run_result(record)
        if result.status.value in _TERMINAL_STATUS_VALUES:
            from dendrux.runtime.runner import _attach_undelivered_notes

            await _attach_undelivered_notes(store, run_id, result)
        if cascade:
            result.meta["cascade"] = outcomes
        return result

    async def _cancel_one(self, store: StateStore, run_id: str, *, signal: bool = True) -> str:
        """Cancel one run: set the flag, CAS-finalize it if paused, signal in-process."""
        from dendrux.types import RunStatus

        # Set cooperative cancel flag (observed by runner checkpoints) AND
        # attempt atomic CAS finalize over the paused statuses. If paused:
        # CAS wins instantly. If running: CAS misses, the runner sees the
        # flag at its next checkpoint and finalizes itself.
        if not await store.request_cancel(run_id):
            return "terminal"
        # Paused-run cancel: we own the terminal event, written in the CAS's
        # transaction (sequence allocated under the row lock) so it always
        # sorts after the runner's run.paused. Running-run cancel: the runner
        # emits run.cancelled at a checkpoint.
        from dendrux.runtime.runner import _transition_with_event

        cancel_won = await _transition_with_event(
            store,
            run_id,
            status=RunStatus.CANCELLED.value,
            allowed_current_statuses=[
                RunStatus.WAITING_CLIENT_TOOL.value,
                RunStatus.WAITING_HUMAN_INPUT.value,
                RunStatus.WAITING_APPROVAL.value,
                RunStatus.PAUSED.value,
            ],
            event_type="run.cancelled",
            event_data={"reason": "cancel_requested"},
        )
        if signal:
            self._task_manager.cancel(run_id)
        return "cancelled" if cancel_won else "cancel_requested"

    async def request_pause(
        self,
        run_id: str,
        *,
        cascade: bool = True,
        tenant_id: TenantScope | _OmittedType = _OMITTED,
        reason: str | None = None,
    ) -> PauseOutcome:
        """Ask a running run to pause durably at its next checkpoint.

        Cooperative, like :meth:`cancel_run`: a RUNNING run gets a
        ``pause_requested`` flag and the runner that owns it pauses at the
        top of its next iteration — or between the approved calls of a
        resumed approval batch — before any further side effect. The run
        becomes ``PAUSED`` with its full history persisted, emits
        ``run.paused`` with ``reason="requested"``, and the waiting
        ``run()`` / stream returns that paused result. In-flight LLM calls
        and tools are never preempted. Cancellation wins over pause.

        - WAITING_* run: no-op (already durable); nothing latches, so the
          next submit continues normally.
        - PAUSED or terminal run: no-op; returns the current status.
        - If the run reaches a WAITING_* pause first, that pause satisfies
          the request and clears it.

        Resume a PAUSED run with :meth:`submit_resume`, ``resume(run_id)``
        or ``resume_stream(run_id)`` (no payload); :meth:`cancel_run`
        finalizes it as cancelled. Notes submitted while paused are
        delivered after resume.

        ``cascade=True`` (default) also flags every non-terminal descendant
        over a point-in-time snapshot (deepest first, root last). This is
        flag propagation only, not a resumable call stack: a paused child
        returns its paused ``RunResult`` to the parent's delegating tool,
        which records whatever the tool returns; the parent pauses at its
        own next checkpoint. Resuming the child later never rewrites the
        parent's recorded tool result — the delegating tool or the app owns
        correlating and re-injecting the child's outcome. Resume is per run
        (``RunStore.get_subtree`` lists them).

        Args:
            run_id: The run to pause.
            cascade: Also flag descendants (default True).
            tenant_id: When given, ``run_id`` must belong to exactly this
                tenant (``None`` = untenanted) or ``RunNotFoundError`` is
                raised; the cascade reaches only that tenant's descendants.
                Omitted means unscoped (trusted single-tenant use).
            reason: Optional free text (≤ 1000 chars) stored in the pause
                and on the ``run.paused`` event as ``message``.

        Returns:
            :class:`PauseOutcome` with the root's status before/after and
            each affected run's outcome.

        Raises:
            PersistenceNotConfiguredError: Agent has no DB.
            RunNotFoundError: ``run_id`` does not exist (or is outside
                ``tenant_id``).
            ValueError: ``reason`` is not a string or is too long.
        """
        tenant_id = entry_checks(
            self, "request_pause", tenant_id, default_tenant=UNSCOPED, run_id=run_id
        )
        from dendrux.errors import PersistenceNotConfiguredError, RunNotFoundError
        from dendrux.types import PauseOutcome, RunStatus

        if reason is not None and (
            not isinstance(reason, str) or len(reason) > MAX_PAUSE_REASON_CHARS
        ):
            raise ValueError(
                f"reason must be a string of at most {MAX_PAUSE_REASON_CHARS} characters."
            )
        store = await self._resolve_state_store()
        if store is None:
            raise PersistenceNotConfiguredError()

        record = await store.get_run(run_id)
        if record is None or not _in_tenant_scope(record, tenant_id):
            raise RunNotFoundError(run_id)

        store_request_pause = require_store_method(store, "request_pause")
        order = [run_id]
        if cascade:
            subtree = await require_store_method(store, "get_subtree")(run_id, tenant_id=tenant_id)
            if subtree is not None:
                order = _cascade_order(subtree)

        affected: list[tuple[str, Any]] = []
        for rid in order:
            if await store_request_pause(rid, reason=reason):
                affected.append((rid, "requested"))
                continue
            current = await store.get_run(rid)
            status = current.status if current is not None else "cancelled"
            if status in _WAITING_STATUS_VALUES:
                affected.append((rid, "already_waiting"))
            elif status == RunStatus.PAUSED.value:
                affected.append((rid, "already_paused"))
            else:
                affected.append((rid, "terminal"))

        after = await store.get_run(run_id)
        return PauseOutcome(
            run_id=run_id,
            status_before=RunStatus(record.status),
            status_after=RunStatus(after.status if after is not None else record.status),
            affected=affected,
        )

    async def submit_resume(
        self,
        run_id: str,
        *,
        notifier: LoopNotifier | None = None,
        tenant_id: TenantScope | _OmittedType = _OMITTED,
        expected_pause_id: str | None = None,
    ) -> RunResult:
        """Resume a PAUSED run and block until its next pause or terminal state.

        Race-safe like the other ``submit_*`` methods: a status preflight,
        then an atomic claim (PAUSED → RUNNING) that also clears
        ``pause_requested``. The run continues with its full history; a
        pause that interrupted an approval batch first executes the rest of
        that batch with the decisions already made. Pending inbox notes are
        delivered at the first iteration.

        Args:
            run_id: A run in status ``PAUSED``.
            expected_pause_id: Current pause generation; stale submissions are rejected.
            notifier: Optional additional notifier composed during resume.
            tenant_id: Omitted = no tenant check. Otherwise the run must belong
                to this tenant (``None`` = untenanted) or ``RunNotFoundError``
                is raised before anything is written.

        Raises:
            PersistenceNotConfiguredError, RunNotFoundError (also when
            ``run_id`` is outside ``tenant_id``), RunNotPausedError,
            PauseStatusMismatchError (paused for another reason, e.g.
            WAITING_APPROVAL), RunAlreadyClaimedError,
            RunAlreadyTerminalError.
        """
        tenant_id = entry_checks(
            self, "submit_resume", tenant_id, default_tenant=UNSCOPED, run_id=run_id
        )
        from dendrux.errors import PersistenceNotConfiguredError
        from dendrux.runtime.runner import resume_paused_claim
        from dendrux.runtime.submit import claim_or_raise
        from dendrux.types import RunStatus

        provider = self._require_provider()
        store = await self._resolve_state_store()
        if store is None:
            raise PersistenceNotConfiguredError()

        await _require_run_in_scope(store, run_id, tenant_id)
        skill_session = await continuation(self, store, run_id)
        with skill_session.bind():
            epoch = await claim_or_raise(
                store,
                run_id,
                expected_status=RunStatus.PAUSED,
                agent=self,
                expected_pause_id=expected_pause_id,
            )
            from dendrux.runtime.state import claimed_store

            store = claimed_store(store, run_id, epoch)
            task = self._task_manager.spawn(
                run_id,
                resume_paused_claim(
                    run_id,
                    state_store=store,
                    agent=self,
                    provider=provider,
                    extra_notifier=notifier,
                ),
            )
            return await task

    async def submit_note(
        self,
        run_id: str,
        text: str,
        *,
        metadata: Any = None,
        tenant_id: TenantScope | _OmittedType = _OMITTED,
    ) -> str:
        """Leave a note for a non-terminal run; the model sees it at the next iteration.

        The note is stored in the run's inbox in one conditional insert
        (only while the run is non-terminal). The runner that owns the run
        delivers pending notes at the top of its next iteration, in
        submission order, as a USER message ``"NOTE FROM USER:\n{text}"``
        (``kind="inbox_note"``, ``source="run_inbox"``) and records an
        ``inbox.delivered`` event carrying only the note id. Incoming
        guardrails apply to the note text like any other message.

        Notes on a WAITING_* or PAUSED run wait until it resumes. Delivery
        is at-least-once (a crash between recording the message and marking
        the note delivered can redeliver it). A note accepted while the run
        makes its last LLM call may never be delivered: the run's result
        then lists it in ``meta["undelivered_notes"]`` and
        ``RunStore.get_notes`` shows it undelivered.

        Args:
            run_id: The run to notify.
            text: Note text, non-empty, at most 32 KiB as UTF-8.
            metadata: Optional JSON-serializable app data (at most 8 KiB
                encoded). Stored with the note, never shown to the model.
            tenant_id: When given, ``run_id`` must belong to exactly this
                tenant (``None`` = untenanted) or ``RunNotFoundError`` is
                raised. Omitted means unscoped.

        Returns:
            The note id.

        Raises:
            ValueError: empty/oversized text or invalid/oversized metadata.
            PersistenceNotConfiguredError: Agent has no DB.
            RunNotFoundError: ``run_id`` does not exist (or is outside
                ``tenant_id``).
            RunAlreadyTerminalError: the run already finished.
        """
        tenant_id = entry_checks(
            self,
            "submit_note",
            tenant_id,
            default_tenant=UNSCOPED,
            run_id=run_id,
            note_text=text,
            note_metadata=metadata,
        )
        import json

        from dendrux.errors import (
            PersistenceNotConfiguredError,
            RunAlreadyTerminalError,
            RunNotFoundError,
        )
        from dendrux.types import RunStatus, generate_ulid

        if not isinstance(text, str) or not text:
            raise ValueError("Note text must be a non-empty string.")
        if len(text.encode("utf-8")) > MAX_NOTE_TEXT_BYTES:
            raise ValueError(f"Note text exceeds {MAX_NOTE_TEXT_BYTES} bytes as UTF-8.")
        if metadata is not None:
            try:
                encoded = json.dumps(metadata, allow_nan=False)
            except (TypeError, ValueError) as exc:
                raise ValueError("Note metadata must be JSON-serializable.") from exc
            if len(encoded.encode("utf-8")) > MAX_NOTE_METADATA_BYTES:
                raise ValueError(
                    f"Note metadata exceeds {MAX_NOTE_METADATA_BYTES} bytes as encoded JSON."
                )

        store = await self._resolve_state_store()
        if store is None:
            raise PersistenceNotConfiguredError()

        add_note = require_store_method(store, "add_note")
        from dendrux.runtime.runner import _accepts_kwarg

        note_kwargs: dict[str, Any] = {}
        if _accepts_kwarg(add_note, "max_notes"):  # custom stores written before 0.5.0
            note_kwargs["max_notes"] = self._limits.max_notes_per_run
        note_id = generate_ulid()
        if await add_note(
            run_id,
            note_id=note_id,
            text=text,
            metadata=metadata,
            tenant_id=tenant_id,
            **note_kwargs,
        ):
            return note_id
        record = await store.get_run(run_id)
        if record is None or not _in_tenant_scope(record, tenant_id):
            raise RunNotFoundError(run_id)
        raise RunAlreadyTerminalError(run_id, RunStatus(record.status))

    @overload
    def stream(
        self,
        user_input: UserContent,
        *,
        history: list[ChatMessage] | None = ...,
        context: list[ContextBlock] | None = ...,
        activate_skills: Sequence[str] | None = ...,
        runs: list[str] | None = ...,
        replay_last: int | Literal["all"] = ...,
        replay_reasoning: bool = ...,
        tenant_id: str | None = ...,
        metadata: dict[str, Any] | None = ...,
        notifier: LoopNotifier | None = ...,
        idempotency_key: str | None = ...,
        max_delegation_depth: int | None,
        **kwargs: Any,
    ) -> RunStream: ...

    @overload
    def stream(
        self,
        user_input: UserContent,
        *,
        history: list[ChatMessage] | None = ...,
        context: list[ContextBlock] | None = ...,
        activate_skills: Sequence[str] | None = ...,
        runs: list[str] | None = ...,
        replay_last: int | Literal["all"] = ...,
        replay_reasoning: bool = ...,
        tenant_id: str | None = ...,
        metadata: dict[str, Any] | None = ...,
        notifier: LoopNotifier | None = ...,
        idempotency_key: str | None = ...,
        **kwargs: Any,
    ) -> RunStream: ...

    def stream(
        self,
        user_input: UserContent,
        *,
        history: list[ChatMessage] | None = None,
        context: list[ContextBlock] | None = None,
        activate_skills: Sequence[str] | None = None,
        runs: list[str] | None = None,
        replay_last: int | Literal["all"] = 0,
        replay_reasoning: bool = True,
        tenant_id: str | None | _OmittedType = _OMITTED,
        metadata: dict[str, Any] | None = None,
        notifier: LoopNotifier | None = None,
        idempotency_key: str | None = None,
        max_delegation_depth: int | None | _UnsetType = _UNSET,
        **kwargs: Any,
    ) -> RunStream:
        """Stream an agent run as RunEvents.

        Same parameters as run(). Returns a RunStream immediately —
        no ``await`` needed. The run_id is available before iteration; with
        idempotency_key it is provisional until lazy setup resolves the key.
        Async setup (DB row, notifiers) runs on first iteration.

        Usage:
            # Full event stream
            async for event in agent.stream("analyze revenue"):
                print(event.type, event.text or "")

            # Just the text
            async for chunk in agent.stream("analyze revenue").text():
                print(chunk, end="")

            # With explicit cleanup (recommended)
            async with agent.stream("analyze revenue") as stream:
                async for event in stream:
                    ...

        After any terminal event (RUN_COMPLETED, RUN_PAUSED, RUN_ERROR,
        RUN_CANCELLED), the stream ends. If the consumer breaks early,
        the run is cancelled via CAS-guarded cleanup. The persisted partial
        ``answer`` contains the current text buffer, reset at each TOOL_RESULT;
        it can be None even if earlier turns produced text. On success,
        ``answer`` contains the final answer turn only. Collect TEXT_DELTA
        events (or .text()) across turns, or use RunStore.get_traces(), for
        earlier assistant text. To interrupt from another coroutine, call
        :meth:`cancel_run` with ``stream.run_id`` after the first event;
        the stream then yields ``RUN_CANCELLED`` carrying the partial
        answer.

        Args:
            user_input: The user's input — a string, or strings and images
                (see :meth:`run`). Item types are checked synchronously
                (``TypeError``); image ingest runs on first iteration before
                the run row is created, and its errors are raised from the
                first ``__anext__`` (not as ``RUN_ERROR``), with no run row.
            history: Optional prior conversation turns to seed the run with —
                for building chatbots on top of Dendrux. Each ``ChatMessage``
                is a ``role`` (USER or ASSISTANT) plus ``content``. The dev's
                app owns conversation storage; Dendrux only reads ``history``
                as input for this turn and does NOT persist it to
                ``react_traces``. Validation rules: must start with USER, end
                with ASSISTANT, no consecutive same-role messages, no empty
                content. Invalid history raises ``ValueError`` synchronously
                (before the stream begins).
            context: Optional per-run context blocks — see :meth:`run`.
            runs: Earlier run ids to replay — see :meth:`run`. Invalid knobs
                raise synchronously; run-id problems (missing, other tenant,
                not terminal) arrive as ``RUN_ERROR`` with
                ``result.meta["error_type"] == "ReplayRunError"``.
            replay_last: Full-replay window — see :meth:`run`.
            replay_reasoning: Reasoning replay — see :meth:`run`.
            tenant_id: Optional tenant ID for multi-tenant isolation.
            metadata: Optional developer linking data (thread_id, user_id, etc.).
            notifier: Optional notifier for lifecycle events.
            idempotency_key: Duplicate prevention using run()'s fingerprint
                (agent name, input, history, context, replay knobs, and output
                type). Requires
                persistence. A terminal match yields one cached terminal event
                with the existing run_id and summary result; no text/tool events
                are replayed, so .text() yields nothing on a cache hit. An active
                match or input conflict yields RUN_ERROR with
                result.meta["error_type"] of RunAlreadyActiveError or
                IdempotencyConflictError. Missing persistence also yields
                RUN_ERROR during lazy setup. Existing runs are never modified.
            max_delegation_depth: Maximum allowed delegation depth for the run
                tree. Default 10. None means unbounded.
            **kwargs: Forwarded to the LLM provider. Pass ``model=`` to override
                the provider's configured model for this call only (e.g.
                ``model="claude-opus-4-1"``) — useful for per-turn model selection
                in a chatbot. Also accepts ``temperature``, ``max_tokens``, and
                other provider sampling params. The model actually used is recorded
                per LLM call (see ``RunStore.get_llm_calls``).

        Returns:
            RunStream — async iterable of RunEvent objects.

        Raises:
            ValueError: If no provider is configured, or if ``history`` is invalid.
        """
        tenant_id = entry_checks(
            self,
            "stream",
            tenant_id,
            default_tenant=None,
            metadata=metadata,
            user_input=user_input,
            history=history,
            context=context,
            provider_kwargs=kwargs,
        )
        _validate_max_delegation_depth(max_delegation_depth)

        if self._guardrails:
            raise ValueError(
                "guardrails are not supported with stream() in this version. "
                "Guardrails require the full response before scanning. "
                "Use agent.run() instead."
            )

        provider = self._require_provider()

        if self._output_type is not None:
            raise NotImplementedError(
                "Structured output streaming is not supported in this version. "
                "Use agent.run() for structured output, or create a separate "
                "Agent without output_type for streaming."
            )

        from dendrux.runtime.runner import run_stream as runner_run_stream

        stream_kwargs: dict[str, Any] = {}
        resolved_depth = self._resolve_run_max_delegation_depth(max_delegation_depth)
        if not isinstance(resolved_depth, _UnsetType):
            stream_kwargs["max_delegation_depth"] = resolved_depth

        return runner_run_stream(
            self,
            provider=provider,
            user_input=user_input,
            history=history,
            context=context,
            activate_skills=activate_skills,
            runs=runs,
            replay_last=replay_last,
            replay_reasoning=replay_reasoning,
            state_store_resolver=self._resolve_state_store,
            tenant_id=tenant_id,
            metadata=metadata,
            extra_notifier=notifier,
            idempotency_key=idempotency_key,
            **stream_kwargs,
            **kwargs,
        )

    def resume_stream(
        self,
        run_id: str,
        *,
        tool_results: list[ToolResult] | None = None,
        user_input: UserContent | None = None,
        notifier: LoopNotifier | None = None,
        tenant_id: TenantScope | _OmittedType = _OMITTED,
        expected_pause_id: str | None = None,
    ) -> RunStream:
        """Stream a resumed run as RunEvents.

        Same parameters as resume(). Returns a RunStream immediately —
        no ``await`` needed. Async setup (store resolution, pause state
        load, claim, history build) runs lazily on first iteration.

        First event is RUN_RESUMED (not RUN_STARTED), carrying the
        existing run_id for correlation. With neither ``tool_results`` nor
        ``user_input`` it approves a WAITING_APPROVAL batch or continues a
        PAUSED run.

        Usage:
            async with agent.resume_stream(run_id, tool_results=results) as stream:
                async for event in stream:
                    if event.type == RunEventType.TEXT_DELTA:
                        print(event.text, end="")

        Args:
            run_id: The paused run's ID.
            tool_results: Results for pending tool calls (client tool runs).
            user_input: Clarification answer (human-in-the-loop runs).
            expected_pause_id: Current pause generation; stale submissions are rejected.
            notifier: Optional additional loop notifier.
            tenant_id: Omitted = no tenant check. Otherwise the run must belong
                to this tenant (``None`` = untenanted) or ``RunNotFoundError``
                is raised before anything is written.

        Returns:
            RunStream — async iterable of RunEvent objects.

        Raises:
            ValueError: If provider not set, both args provided, or neither.
            RunNotFoundError: Raised from the first iteration, before any
                event, when ``run_id`` is outside ``tenant_id``. The run is
                left untouched (the stream's close-time cleanup is a no-op).
        """
        tenant_id = entry_checks(
            self,
            "resume_stream",
            tenant_id,
            default_tenant=UNSCOPED,
            run_id=run_id,
            tool_results=tool_results,
            user_input=user_input,
        )
        if self._guardrails:
            raise ValueError(
                "guardrails are not supported with resume_stream() in this version. "
                "Use agent.resume() instead."
            )
        provider = self._require_provider()
        if tool_results is not None and user_input is not None:
            raise ValueError("Cannot provide both tool_results and user_input to resume_stream().")

        from dendrux.runtime.runner import resume_stream as runner_resume_stream
        from dendrux.runtime.submit import capture_warning_location

        stream = runner_resume_stream(
            run_id,
            agent=self,
            provider=provider,
            state_store_resolver=self._resolve_state_store,
            tool_results=tool_results,
            user_input=user_input,
            extra_notifier=notifier,
            expected_pause_id=expected_pause_id,
            warning_location=capture_warning_location(),
        )
        return self._tenant_checked_stream(stream, run_id, tenant_id)

    def _tenant_checked_stream(
        self, inner: RunStream, run_id: str, tenant_id: TenantScope
    ) -> RunStream:
        """Gate ``inner`` behind a tenant preflight that raises, not yields RUN_ERROR.

        The check runs before ``inner`` starts, so its setup, claim and
        close-time cancel never touch a run outside the tenant.
        """
        from dendrux.types import RunStream

        async def _generate() -> AsyncGenerator[RunEvent, None]:
            store = await self._resolve_state_store()
            if store is not None:
                await _require_run_in_scope(store, run_id, tenant_id)
            events = inner.__aiter__()
            try:
                async for event in events:
                    yield event
            finally:
                await events.aclose()

        async def _cleanup() -> RunResult | None:
            await inner.aclose()
            return inner.result

        return RunStream(run_id=run_id, generator=_generate(), cleanup=_cleanup)

    # ------------------------------------------------------------------
    # MCP discovery
    # ------------------------------------------------------------------

    async def _ensure_discovered(self) -> None:
        """Lazy MCP tool discovery — called internally by get_tool_lookups().

        Uses asyncio.Lock with double-check pattern: fast path (already
        discovered) skips the lock. Lock protects one-time discovery
        against concurrent agent.run() calls.

        On any failure (connection, discovery, governance validation),
        all successfully opened sources are closed and caches stay unset
        so the next call retries.
        """
        if self._discovered_tool_defs is not None:
            return  # fast path

        if not self._tool_sources:
            # No MCP sources — set empty cache so fast path works
            self._discovered_tool_defs = []
            self._mcp_executors = {}
            self._mcp_skipped_sources = []
            self._mcp_skipped_tools = []
            return

        async with self._discovery_lock:
            if self._discovered_tool_defs is not None:
                return  # lost race, another caller finished

            discovered_defs: list[ToolDef] = []
            executors: dict[str, Callable[..., Any]] = {}
            opened_sources: list[MCPServer] = []
            skipped: list[dict[str, Any]] = []
            skipped_tools: list[dict[str, Any]] = []

            # Deferred: dendrux.mcp needs the optional mcp extra.
            from dendrux.mcp._errors import MCPToolNameCollisionError

            # Hoist local tool info — computed once, not per-tool
            local_defs = {get_tool_def(fn).name: get_tool_def(fn) for fn in self.tools}

            try:
                for source in self._tool_sources:
                    try:
                        tool_defs = await source._discover()
                    except Exception as exc:
                        if getattr(source, "failure_mode", "strict") != "best_effort":
                            raise
                        _agent_logger.warning(
                            "Optional MCP source '%s' is unavailable; continuing without it",
                            source.name,
                            exc_info=True,
                        )
                        skipped.append(_skipped_source_entry(source, exc))
                        continue
                    opened_sources.append(source)
                    source_skips = getattr(source, "_skipped_tools", None)
                    if isinstance(source_skips, list):
                        skipped_tools.extend(asdict(entry) for entry in source_skips)
                    for td in tool_defs:
                        # Collision check against reserved name
                        if td.name in NAMES and (self._skills_dir or self._explicit_skills):
                            raise MCPToolNameCollisionError(
                                f"MCP tool '{td.name}' from source "
                                f"'{td.meta.get('source_name')}' collides with "
                                f"the reserved '{td.name}' meta-tool."
                            )
                        # Collision check against local tools
                        if td.name in local_defs:
                            raise MCPToolNameCollisionError(
                                f"MCP tool '{td.name}' from source "
                                f"'{td.meta.get('source_name')}' collides with "
                                f"a local tool of the same name."
                            )
                        # Collision check against other MCP tools
                        if td.name in executors:
                            raise MCPToolNameCollisionError(
                                f"MCP tool '{td.name}' collides with a tool from "
                                f"another MCP source."
                            )
                        discovered_defs.append(td)
                        executors[td.name] = source._create_executor(
                            td.meta["mcp_tool_name"], exposed_name=td.name
                        )

                # Build merged name→ToolDef map for full governance validation
                all_tool_defs = dict(local_defs)
                for td in discovered_defs:
                    all_tool_defs[td.name] = td
                self._validate_governance_names(all_tool_defs)

            except BaseException:
                # Multi-source cleanup: close all opened sources
                for src in opened_sources:
                    try:
                        await src.close()
                    except Exception:
                        _agent_logger.warning(
                            "Failed to close MCP source '%s' during cleanup",
                            src.name,
                            exc_info=True,
                        )
                raise

            # Only commit cache after ALL sources discovered + validated.
            # Sort at storage so developer-visible order is alphabetical
            # everywhere, not just at the cache-facing getters.
            self._discovered_tool_defs = sorted(discovered_defs, key=lambda td: td.name)
            self._mcp_executors = executors
            self._mcp_skipped_sources = skipped
            self._mcp_skipped_tools = skipped_tools

    async def get_tool_lookups(self, loop: Loop | None = None) -> ToolLookups:
        """Build ToolLookups for local + MCP + use_skill tools.

        Calls _ensure_discovered() and _ensure_skills_loaded() internally.
        Safe to call multiple times (discovery/loading is cached).

        ``loop`` is the effective loop for the current run — pass the
        runner's ``resolved_loop`` so a runtime override
        (``agent.run(loop=...)``) is honored. Defaults to the agent's
        configured loop when omitted.
        """

        await self._ensure_discovered()
        self._ensure_skills_loaded()
        from dendrux.tools import build_tool_lookups
        from dendrux.types import ToolTarget as _ToolTarget

        effective_loop = self._effective_loop(loop)

        lookups = build_tool_lookups(
            self.tools,
            mcp_executors=self._mcp_executors,
            mcp_tool_defs=self._discovered_tool_defs,
            call_limiter=self._tool_call_limiter,
        )

        from dendrux.skills._session import SkillTool

        session = current(self) or SkillSession.prepare(self, loop=effective_loop)
        for td in session.tool_defs():
            lookups.fn[td.name] = SkillTool(session, td.name)
            lookups.target[td.name] = _ToolTarget.SERVER
            lookups.timeout[td.name] = td.timeout_seconds
            lookups.explicit_timeout[td.name] = td.has_explicit_timeout
            lookups.max_calls[td.name] = td.max_calls_per_run
            lookups.parallel[td.name] = td.parallel
            lookups.meta[td.name] = td.meta
            lookups.defs[td.name] = td

        return lookups

    def get_all_tool_defs(self, loop: Loop | None = None) -> list[ToolDef]:
        """Local + discovered MCP + use_skill tool defs, sorted by name.

        Sorted output gives cache-stable prefixes to provider tool arrays.
        ``use_skill`` is omitted on SingleCall (inlined skill mode).
        ``loop`` overrides the agent's configured loop for delivery-mode
        selection — see :meth:`get_tool_lookups`.
        """

        self._ensure_skills_loaded()
        effective_loop = self._effective_loop(loop)
        local = [get_tool_def(fn) for fn in self.tools]
        discovered = self._discovered_tool_defs or []
        all_defs = local + discovered
        all_defs.extend(
            (current(self) or SkillSession.prepare(self, loop=effective_loop)).tool_defs()
        )
        return sorted(all_defs, key=lambda td: td.name)

    def _effective_loop(self, loop: Loop | None) -> Loop | None:
        """Return ``loop`` if explicit, else the agent's configured loop.

        Used by the prompt/tool getters so a runtime ``loop=`` override
        controls skill delivery mode instead of silently diverging from
        what the runner is actually executing.
        """
        return loop if loop is not None else self._loop

    # ------------------------------------------------------------------
    # Skills
    # ------------------------------------------------------------------

    def _validate_skill_contents(self, skills: list[Skill], loop: Loop | None = None) -> None:
        from dendrux.loops.single import SingleCall

        if isinstance(self._effective_loop(loop), SingleCall) and any(s.files for s in skills):
            raise ValueError("SingleCall cannot use skills with files.")
        unavailable = {s.name for s in skills if any(is_script(p) for p in s.files)}
        if self._skill_executor is None and unavailable - self._script_warning_names:
            warnings.warn(
                f"Skills with scripts have no skill_executor: {', '.join(sorted(unavailable))}",
                UserWarning,
                stacklevel=3,
            )
            self._script_warning_names.update(unavailable)

    def _ensure_skills_loaded(self) -> None:
        """Lazy skill loading — scans skills_dir and merges explicit skills.

        Called at run start. Cached after first call. Cleared by refresh().
        Filters out deny_skills matches.
        """
        if self._loaded_skills is not None:
            return  # already loaded

        from dendrux.skills import Skill as _Skill

        all_skills: list[Skill] = []

        # Scan skills_dir
        if self._skills_dir is not None:
            all_skills.extend(_Skill.scan_dir(self._skills_dir))

        # Add explicit skills
        if self._explicit_skills:
            all_skills.extend(self._explicit_skills)

        # Check for duplicate names
        seen: set[str] = set()
        for s in all_skills:
            if s.name in seen:
                raise ValueError(
                    f"Agent '{self.name}' has duplicate skill name '{s.name}'. "
                    f"Each skill must have a unique name."
                )
            seen.add(s.name)

        # Apply deny_skills filter
        loaded: list[Skill] = []
        denied: list[str] = []
        for s in all_skills:
            if s.name in self._deny_skills:
                denied.append(s.name)
            else:
                loaded.append(s)

        self._loaded_skills = loaded
        self._denied_skill_names = denied

    def get_system_prompt(self, loop: Loop | None = None) -> str:
        """Compose system prompt with skill catalog.

        Two delivery modes, picked by the effective loop:

        - **ReAct (default):** progressive disclosure. Only skill names +
          descriptions are listed; the LLM pulls full bodies via the
          auto-injected ``use_skill`` tool.
        - **SingleCall (inlined skill mode):** full skill bodies are
          rendered into the prompt at construction time, alphabetically
          sorted. ``use_skill`` is not injected (SingleCall is one LLM
          call by definition; progressive disclosure requires ≥2 turns).

        The architectural rule: in SingleCall, skills are static prompt
        context; in ReAct, skills are dynamically discoverable
        capabilities.

        ``loop`` overrides the agent's configured loop for delivery-mode
        selection. The runner passes its ``resolved_loop`` so an
        ``agent.run(loop=...)`` override stays consistent end-to-end.

        Used by loops, runner events, and LLM semantic evidence.
        Single source of truth for the complete system prompt.
        """

        self._ensure_skills_loaded()
        session = current(self) or SkillSession.prepare(self, loop=loop)
        if not session.available or (
            session.delivery == "progressive" and "use_skill" in session.deny
        ):
            return self.prompt

        sorted_skills = sorted(session.available.values(), key=lambda s: s.name)

        if session.delivery == "inlined":
            # Inlined skill mode: render bodies verbatim into the prompt.
            parts = [self.prompt, "\n\n## Skills"]
            for skill in sorted_skills:
                parts.append(f"\n### Skill: {skill.name}\n\n{skill.body}")
            rendered = "\n".join(parts)
            self._maybe_warn_inlined_skill_size(rendered)
            return rendered

        parts = [
            self.prompt,
            "\n\n## Available Skills",
            "\nYou have access to the following skills. To use a skill, "
            "call the `use_skill` tool with the skill name. The skill's "
            "detailed instructions will be returned to you.\n",
        ]
        for skill in sorted_skills:
            parts.append(f"- **{skill.name}**: {skill.description}")

        return "\n".join(parts)

    def _maybe_warn_inlined_skill_size(self, rendered_prompt: str) -> None:
        """Emit a one-shot UserWarning naming the rendered prompt size.

        Fires the first time the SingleCall + skills system prompt is
        built. Devs see the character cost so they can decide whether
        to slim down skill bodies or switch to ReAct for progressive
        disclosure. Suppressible via the standard ``warnings`` module.
        """
        if self._inline_skill_warning_emitted:
            return
        import warnings

        skill_count = len(self._loaded_skills or [])
        warnings.warn(
            f"Agent '{self.name}': SingleCall + {skill_count} skill(s) → "
            f"inlined skill mode. Skill bodies are rendered into the system "
            f"prompt at every turn (no progressive disclosure with one LLM "
            f"call). Rendered system prompt is {len(rendered_prompt)} "
            f"characters. Switch to ReActLoop for tool-mediated disclosure.",
            UserWarning,
            stacklevel=3,
        )
        self._inline_skill_warning_emitted = True

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def refresh(self) -> None:
        """Re-discover all dynamic capabilities.

        Clears MCP discovery cache and skill cache. Next run()
        re-discovers MCP tools and re-scans skills.

        IMPORTANT: Call only when all runs are terminal (success, error,
        cancelled). Paused runs are still active. Calling refresh()
        with active or paused runs closes MCP sessions under active
        executors and invalidates the frozen skill context.

        For MCPHost-backed sources this clears only the agent snapshot;
        call ``host.refresh()`` first to refresh the shared server catalog.
        """
        # Close MCP sessions
        for source in self._tool_sources:
            try:
                await source.close()
            except Exception:
                _agent_logger.warning(
                    "Failed to close MCP source '%s' during refresh",
                    source.name,
                    exc_info=True,
                )

        # Clear all caches
        self._discovered_tool_defs = None
        self._mcp_executors = None
        self._mcp_skipped_sources = []
        self._mcp_skipped_tools = []
        self._loaded_skills = None
        self._denied_skill_names = None
        self._inline_skill_warning_emitted = False

    async def close(self) -> None:
        """Close MCP sources, provider, and agent-owned engine.

        Only disposes the private engine created from an explicit database_url.
        The shared global engine (from DENDRUX_DATABASE_URL env var) is NOT
        owned by this agent and is left untouched.

        Use ``MCPHost`` when sharing MCP sources across agents; its leases are
        safe to close per agent and the host remains application-owned. A
        directly shared ``MCPServer`` is not safe to close per request.
        """
        for source in self._tool_sources:
            try:
                await source.close()
            except Exception:
                _agent_logger.warning("Failed to close MCP source '%s'", source.name, exc_info=True)
        # Clear all caches so stale executors/skills are not reused.
        self._discovered_tool_defs = None
        self._mcp_executors = None
        self._mcp_skipped_sources = []
        self._mcp_skipped_tools = []
        self._loaded_skills = None
        self._denied_skill_names = None
        if self._provider is not None:
            await self._provider.close()
        with self._store_lock:
            self._store_generation += 1
            engine, self._private_engine = self._private_engine, None
            if engine is not None:
                self._lazy_store = None
        if engine is not None:
            await engine.dispose()

    async def __aenter__(self) -> Agent:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_tool_defs(self, loop: Loop | None = None) -> list[ToolDef]:
        """Get ToolDef for each LOCAL tool registered on this agent.

        Includes the auto-injected use_skill tool when skills are loaded
        AND the effective loop supports tool calls. SingleCall delivers
        skills inline, so use_skill is omitted there. ``loop`` overrides
        the agent's configured loop — see :meth:`get_tool_lookups`.

        Does NOT include MCP tools. Use get_all_tool_defs() for the
        full set (local + discovered + use_skill).

        Output is sorted by ToolDef.name for cache-stable prefixes across
        runs. self.tools storage preserves caller order.
        """

        self._ensure_skills_loaded()
        effective_loop = self._effective_loop(loop)
        defs = [get_tool_def(fn) for fn in self.tools]
        defs.extend((current(self) or SkillSession.prepare(self, loop=effective_loop)).tool_defs())
        return sorted(defs, key=lambda td: td.name)

    def __repr__(self) -> str:
        return (
            f"Agent(name={self.name!r}, model={self.model!r}, "
            f"tools={len(self.tools)}, max_iterations={self.max_iterations})"
        )
