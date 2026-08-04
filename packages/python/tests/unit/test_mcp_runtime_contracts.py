"""Contract tests for the inert managed MCP runtime API."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError

import pytest

from dendrux.mcp._errors import MCPBindingConflictError
from dendrux.mcp._runtime import (
    MCPBinding,
    MCPCredentialProvider,
    MCPRuntime,
    MCPRuntimeState,
    MCPToolPolicy,
)
from dendrux.mcp._source import MCPSource


class _Credentials:
    def __init__(self) -> None:
        self.calls = 0

    async def get_auth(self) -> object:
        self.calls += 1
        return object()

    def __repr__(self) -> str:
        return "_Credentials(token=must-not-leak)"


class TestMCPRuntimeConfiguration:
    def test_defaults_are_bounded_and_runtime_starts_open(self) -> None:
        runtime = MCPRuntime()

        assert runtime.max_connections == 100
        assert runtime.max_in_flight_calls == 100
        assert runtime.idle_timeout == 300.0
        assert runtime.shutdown_timeout == 30.0
        assert runtime.state is MCPRuntimeState.OPEN

    @pytest.mark.parametrize(
        ("field", "value", "message"),
        [
            ("max_connections", 0, "max_connections"),
            ("max_in_flight_calls", 0, "max_in_flight_calls"),
            ("idle_timeout", -1.0, "idle_timeout"),
            ("shutdown_timeout", 0.0, "shutdown_timeout"),
        ],
    )
    def test_invalid_limits_are_rejected(
        self,
        field: str,
        value: int | float,
        message: str,
    ) -> None:
        kwargs = {field: value}

        with pytest.raises(ValueError, match=message):
            MCPRuntime(**kwargs)  # type: ignore[arg-type]

    @pytest.mark.asyncio
    async def test_close_is_idempotent_and_rejects_new_bindings(self) -> None:
        runtime = MCPRuntime()
        source = MCPSource.http("github", "https://mcp.example.com")

        await runtime.close()
        await runtime.close()

        assert runtime.state is MCPRuntimeState.CLOSED
        with pytest.raises(RuntimeError, match="closed"):
            runtime.bind(connection_key="connection-1", source=source)

    @pytest.mark.asyncio
    async def test_async_context_owns_runtime_lifecycle(self) -> None:
        runtime = MCPRuntime()

        async with runtime as entered:
            assert entered is runtime
            assert runtime.state is MCPRuntimeState.OPEN

        assert runtime.state is MCPRuntimeState.CLOSED


class TestMCPBindingContracts:
    def test_bind_is_lazy_and_defaults_to_application_partition(self) -> None:
        runtime = MCPRuntime()
        source = MCPSource.http("github", "https://mcp.example.com")
        credentials = _Credentials()

        binding = runtime.bind(
            connection_key="connection-1",
            source=source,
            credentials=credentials,
        )

        assert binding.runtime is runtime
        assert binding.tenant_key is None
        assert binding.connection_key == "connection-1"
        assert binding.source is source
        assert binding.namespace == "github"
        assert binding.policy == MCPToolPolicy()
        assert binding.credentials is credentials
        assert credentials.calls == 0

    def test_tenant_and_connection_identity_are_independent(self) -> None:
        runtime = MCPRuntime()
        source = MCPSource.http("github", "https://mcp.example.com")

        first = runtime.bind(
            tenant_key="tenant-a",
            connection_key="github-work",
            source=source,
        )
        second = runtime.bind(
            tenant_key="tenant-b",
            connection_key="github-work",
            source=source,
        )

        assert first.identity == ("tenant-a", "github-work")
        assert second.identity == ("tenant-b", "github-work")
        assert first.identity != second.identity

    def test_same_identity_can_create_different_agent_views(self) -> None:
        runtime = MCPRuntime()

        first = runtime.bind(
            tenant_key="tenant-a",
            connection_key="github-work",
            source=MCPSource.http(
                "github",
                "https://mcp.example.com",
                headers={"Authorization": "Bearer old-token"},
                allowed_tools=["read_issue"],
                call_timeout=10.0,
            ),
        )
        second = runtime.bind(
            tenant_key="tenant-a",
            connection_key="github-work",
            source=MCPSource.http(
                "github_work",
                "https://mcp.example.com",
                headers={"Authorization": "Bearer new-token"},
                allowed_tools=["create_issue"],
                call_timeout=60.0,
            ),
        )

        assert first.identity == second.identity
        assert first.namespace == "github"
        assert second.namespace == "github_work"

    @pytest.mark.parametrize(
        "conflicting_source",
        [
            MCPSource.http("github", "https://other.example.com"),
            MCPSource.stdio("github", ["github-mcp", "serve"]),
        ],
    )
    def test_same_identity_rejects_a_different_physical_target(
        self,
        conflicting_source: MCPSource,
    ) -> None:
        runtime = MCPRuntime()
        runtime.bind(
            tenant_key="tenant-a",
            connection_key="github-work",
            source=MCPSource.http("github", "https://mcp.example.com"),
        )

        with pytest.raises(MCPBindingConflictError, match="different physical target") as caught:
            runtime.bind(
                tenant_key="tenant-a",
                connection_key="github-work",
                source=conflicting_source,
            )

        assert caught.value.identity == ("tenant-a", "github-work")

    def test_tenants_can_reuse_connection_keys_for_different_targets(self) -> None:
        runtime = MCPRuntime()

        first = runtime.bind(
            tenant_key="tenant-a",
            connection_key="github",
            source=MCPSource.http("github", "https://one.example.com"),
        )
        second = runtime.bind(
            tenant_key="tenant-b",
            connection_key="github",
            source=MCPSource.http("github", "https://two.example.com"),
        )

        assert first.identity != second.identity

    def test_concurrent_conflicting_binds_register_only_one_target(self) -> None:
        runtime = MCPRuntime()
        sources = (
            MCPSource.http("github", "https://one.example.com"),
            MCPSource.http("github", "https://two.example.com"),
        )

        def bind(source: MCPSource) -> object:
            try:
                return runtime.bind(connection_key="github", source=source)
            except MCPBindingConflictError as error:
                return error

        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(bind, sources))

        assert sum(isinstance(result, MCPBindingConflictError) for result in results) == 1
        winning_binding = next(
            result for result in results if not isinstance(result, MCPBindingConflictError)
        )
        assert isinstance(winning_binding, MCPBinding)
        assert (
            runtime.bind(
                connection_key="github",
                source=winning_binding.source,
            ).source.physical_identity
            == winning_binding.source.physical_identity
        )

    @pytest.mark.parametrize(
        ("tenant_key", "connection_key"),
        [
            ("", "connection"),
            ("   ", "connection"),
            ("tenant\nsecret", "connection"),
            (None, ""),
            (None, "  "),
            (None, "connection\rsecret"),
        ],
    )
    def test_invalid_identity_keys_are_rejected(
        self,
        tenant_key: str | None,
        connection_key: str,
    ) -> None:
        runtime = MCPRuntime()
        source = MCPSource.http("github", "https://mcp.example.com")

        with pytest.raises(ValueError, match="key"):
            runtime.bind(
                tenant_key=tenant_key,
                connection_key=connection_key,
                source=source,
            )

    def test_credentials_are_redacted_from_repr(self) -> None:
        runtime = MCPRuntime()
        credentials = _Credentials()
        binding = runtime.bind(
            connection_key="connection-1",
            source=MCPSource.http("github", "https://mcp.example.com"),
            credentials=credentials,
        )

        rendered = repr(binding)

        assert "must-not-leak" not in rendered
        assert "credentials" not in rendered

    def test_binding_is_immutable(self) -> None:
        binding = MCPRuntime().bind(
            connection_key="connection-1",
            source=MCPSource.http("github", "https://mcp.example.com"),
        )

        with pytest.raises(FrozenInstanceError):
            binding.namespace = "changed"  # type: ignore[misc]

    def test_credential_provider_protocol_is_structural(self) -> None:
        assert isinstance(_Credentials(), MCPCredentialProvider)


class TestMCPToolPolicyContracts:
    def test_policy_copies_mutable_inputs(self) -> None:
        allowed = ["read_file"]
        serial = ["write_file"]

        policy = MCPToolPolicy(
            allowed_tools=allowed,
            force_serial_tools=serial,
        )
        allowed.append("delete_file")
        serial.append("delete_file")

        assert policy.allowed_tools == frozenset({"read_file"})
        assert policy.force_serial_tools == frozenset({"write_file"})

    def test_empty_allowlist_means_expose_no_tools(self) -> None:
        policy = MCPToolPolicy(allowed_tools=[])

        assert policy.allowed_tools == frozenset()
        assert MCPToolPolicy().allowed_tools is None

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("allowed_tools", [""]),
            ("allowed_tools", ["tool\nname"]),
            ("force_serial_tools", [""]),
            ("force_serial_tools", ["tool\rname"]),
        ],
    )
    def test_invalid_remote_tool_names_are_rejected(
        self,
        field: str,
        value: list[str],
    ) -> None:
        with pytest.raises(ValueError, match=field):
            MCPToolPolicy(**{field: value})  # type: ignore[arg-type]

    def test_with_policy_returns_an_independent_agent_view(self) -> None:
        binding = MCPRuntime().bind(
            tenant_key="tenant-a",
            connection_key="github-work",
            source=MCPSource.http("github", "https://mcp.example.com"),
        )

        restricted = binding.with_policy(
            namespace="github_work",
            allowed_tools=["read_file"],
            force_serial_tools=["write_file"],
        )

        assert restricted is not binding
        assert restricted.identity == binding.identity
        assert restricted.source is binding.source
        assert restricted.namespace == "github_work"
        assert restricted.policy == MCPToolPolicy(
            allowed_tools=["read_file"],
            force_serial_tools=["write_file"],
        )
        assert binding.namespace == "github"
        assert binding.policy == MCPToolPolicy()

    @pytest.mark.parametrize(
        "namespace",
        ["", "github work", "github.personal", "github__work", "github\nwork"],
    )
    def test_invalid_namespaces_are_rejected(self, namespace: str) -> None:
        binding = MCPRuntime().bind(
            connection_key="github-work",
            source=MCPSource.http("github", "https://mcp.example.com"),
        )

        with pytest.raises(ValueError, match="namespace"):
            binding.with_policy(namespace=namespace)
