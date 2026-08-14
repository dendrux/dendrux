"""Contract tests for the public ``dendrux.mcp`` API surface.

The exact ``__all__`` list below is the compatibility contract: adding a
name is a deliberate API decision and removing one is a breaking change.
These tests also pin the properties that make the surface production-safe:
internals stay private, every public annotation resolves at runtime, and
imports and construction perform no connection or credential work.
"""

from __future__ import annotations

import asyncio
import inspect
import subprocess
import sys
import textwrap
import types
from typing import TYPE_CHECKING, get_type_hints

import dendrux.mcp as mcp_pkg
from dendrux.mcp import (
    MCPConnectionStatus,
    MCPCredentialProvider,
    MCPRuntime,
    MCPRuntimeObserver,
    MCPRuntimeSnapshot,
    MCPRuntimeState,
    MCPSource,
)

if TYPE_CHECKING:
    import pytest

EXPECTED_ALL = (
    "MCPAuthenticationError",
    "MCPBindingConflictError",
    "MCPCallCapacityError",
    "MCPCallCapacityRejected",
    "MCPCapacityError",
    "MCPCircuitClosed",
    "MCPCircuitOpenError",
    "MCPCircuitOpened",
    "MCPCircuitProbing",
    "MCPConnection",
    "MCPConnectionAborted",
    "MCPConnectionCapacityError",
    "MCPConnectionCapacityRejected",
    "MCPConnectionClosed",
    "MCPConnectionError",
    "MCPConnectionEvent",
    "MCPConnectionEvictingError",
    "MCPConnectionFailed",
    "MCPConnectionLostError",
    "MCPConnectionOpened",
    "MCPConnectionOpening",
    "MCPConnectionSnapshot",
    "MCPConnectionStateError",
    "MCPConnectionStatus",
    "MCPCredentialError",
    "MCPCredentialProvider",
    "MCPError",
    "MCPEvictionCompleted",
    "MCPEvictionMode",
    "MCPEvictionStarted",
    "MCPFailureMode",
    "MCPHost",
    "MCPOpenCircuitSnapshot",
    "MCPOutcomeUnknownError",
    "MCPPhysicalConnectionEvent",
    "MCPResultTooLargeError",
    "MCPRuntime",
    "MCPRuntimeClosedError",
    "MCPRuntimeEvent",
    "MCPRuntimeObserver",
    "MCPRuntimeShutdownCompleted",
    "MCPRuntimeShutdownStarted",
    "MCPRuntimeSnapshot",
    "MCPRuntimeState",
    "MCPServer",
    "MCPSource",
    "MCPStaleConnectionError",
    "MCPToolCallCancelled",
    "MCPToolCallCompleted",
    "MCPToolCallError",
    "MCPToolCallFailed",
    "MCPToolCallOutcomeUnknown",
    "MCPToolCallStarted",
    "MCPToolPolicy",
    "MCPToolView",
)


class TestPublicSurfaceContract:
    def test_all_matches_the_contract_exactly(self) -> None:
        assert tuple(mcp_pkg.__all__) == EXPECTED_ALL

    def test_contract_is_sorted(self) -> None:
        assert list(EXPECTED_ALL) == sorted(EXPECTED_ALL)

    def test_every_exported_name_resolves_to_package_code(self) -> None:
        for name in EXPECTED_ALL:
            obj = getattr(mcp_pkg, name)
            assert obj is not None, name
            if isinstance(obj, type):
                assert obj.__module__.startswith("dendrux.mcp"), name

    def test_star_import_exposes_exactly_the_contract(self) -> None:
        namespace: dict[str, object] = {}
        exec("from dendrux.mcp import *", namespace)  # noqa: S102
        imported = {name for name in namespace if not name.startswith("__")}
        assert imported == set(EXPECTED_ALL)

    def test_no_unlisted_public_attributes(self) -> None:
        # Submodules and the SDK import are module objects; everything else
        # reachable without an underscore must be part of the contract.
        public = {
            name
            for name, value in vars(mcp_pkg).items()
            if not name.startswith("_") and not isinstance(value, types.ModuleType)
        }
        assert public == set(EXPECTED_ALL)

    def test_internal_machinery_stays_private(self) -> None:
        for name in (
            "MCPClientAdapter",
            "_ConnectionEntry",
            "_CallPermit",
            "_CallWaiter",
            "_Admission",
            "_CircuitState",
            "_Registration",
            "_ViewToolSource",
            "build_mcp_tool_defs",
            "create_mcp_executor",
            "redact_source_text",
            "redact_source_value",
        ):
            assert not hasattr(mcp_pkg, name), name


class TestPublicAnnotationsResolve:
    def test_every_exported_class_and_public_method_resolves_type_hints(self) -> None:
        for name in EXPECTED_ALL:
            obj = getattr(mcp_pkg, name)
            if isinstance(obj, type):
                get_type_hints(obj)
                for member_name, descriptor in vars(obj).items():
                    if member_name.startswith("_") and member_name != "__init__":
                        continue
                    if isinstance(descriptor, (classmethod, staticmethod)):
                        target = descriptor.__func__
                    elif isinstance(descriptor, property):
                        target = descriptor.fget
                    elif inspect.isfunction(descriptor):
                        target = descriptor
                    else:
                        target = None
                    if target is not None:
                        get_type_hints(target)

    def test_protocol_methods_resolve_to_public_types(self) -> None:
        observer_hints = get_type_hints(MCPRuntimeObserver.on_event)
        assert observer_hints["event"] is mcp_pkg.MCPRuntimeEvent
        provider_hints = get_type_hints(MCPCredentialProvider.get_auth)
        assert "return" in provider_hints

    def test_snapshot_hints_name_public_types(self) -> None:
        hints = get_type_hints(MCPRuntimeSnapshot)
        assert hints["state"] is MCPRuntimeState
        connection_hints = get_type_hints(mcp_pkg.MCPConnectionSnapshot)
        assert connection_hints["status"] is MCPConnectionStatus


class TestConstructionPurity:
    def test_import_and_inert_construction_perform_no_external_io(self) -> None:
        probe = textwrap.dedent(
            """
            import sys

            def reject_external_io(event, args):
                if event in {"socket.__new__", "socket.connect", "subprocess.Popen"}:
                    raise AssertionError(f"external I/O during MCP import: {event}")

            sys.addaudithook(reject_external_io)

            from dendrux.mcp import MCPRuntime, MCPSource

            runtime = MCPRuntime()
            connection = runtime.bind(
                connection_key="github-1",
                source=MCPSource.http("github", "https://mcp.example.com"),
            )
            view = connection.tools(allowed_tools=["read_file"])
            snapshot = runtime.snapshot()
            assert view.namespace == "github"
            assert snapshot.connections == ()
            """
        )

        completed = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            check=False,
            text=True,
            timeout=10,
        )

        assert completed.returncode == 0, completed.stderr

    def test_bind_and_views_perform_no_connection_or_credential_work(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class ExplodingAdapter:
            def __init__(self, source: MCPSource) -> None:
                raise AssertionError("adapter constructed without a lease")

        monkeypatch.setattr("dendrux.mcp._runtime.MCPClientAdapter", ExplodingAdapter)

        class Provider:
            calls = 0

            async def get_auth(self) -> dict[str, str]:
                type(self).calls += 1
                return {"Authorization": "Bearer never-used"}

        # Through snapshot, construction remains synchronous and loop-free.
        # asyncio.run below exists only to exercise inert-runtime cleanup.
        runtime = MCPRuntime()
        connection = runtime.bind(
            connection_key="github-1",
            tenant_key="tenant-a",
            source=MCPSource.http("github", "https://mcp.example.com"),
            credentials=Provider(),
        )
        view = connection.tools(allowed_tools=["read_file"])
        snapshot = runtime.snapshot()

        assert snapshot.state is MCPRuntimeState.OPEN
        assert snapshot.connections == ()
        assert snapshot.in_flight_calls == 0
        assert Provider.calls == 0
        assert view.namespace == "github"
        asyncio.run(runtime.close())
