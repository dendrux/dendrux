"""MCP (Model Context Protocol) integration for Dendrux.

The managed runtime is the production entry point: one process-wide
:class:`MCPRuntime` shares physical connections across Agents, enforces
capacity, breaks circuits, and reports operational telemetry.

Usage:
    from dendrux.mcp import MCPRuntime, MCPSource

    runtime = MCPRuntime()
    connection = runtime.bind(
        connection_key="github-1",
        source=MCPSource.http("github", "https://mcp.example.com"),
    )

    agent = Agent(
        provider=provider,
        tool_sources=[connection.tools()],
    )

:class:`MCPHost` remains the simpler application-owned facade for sharing a
fixed set of connections across Agents. :class:`MCPServer` is the directly
Agent-owned, single-connection facade.

Requires the ``mcp`` package: ``pip install dendrux[mcp]``
"""

try:
    import mcp  # noqa: F401
except ModuleNotFoundError as err:
    if err.name == "mcp":
        raise ImportError(
            "MCP support requires the 'mcp' package. Install with: pip install dendrux[mcp]"
        ) from None
    raise  # Real import error from within the mcp package

from dendrux.mcp._credentials import (  # noqa: E402
    MCPAuthKey,
    MCPCredentialProvider,
    MCPRefreshingCredentialProvider,
)
from dendrux.mcp._destination import MCPDestination, MCPDestinationPolicy  # noqa: E402
from dendrux.mcp._discovery import MCPDiscovery, MCPToolInfo  # noqa: E402
from dendrux.mcp._errors import (  # noqa: E402
    MCPAuthChallenge,
    MCPAuthenticationError,
    MCPAuthorizationRequired,
    MCPBindingConflictError,
    MCPCallCapacityError,
    MCPCapacityError,
    MCPCircuitOpenError,
    MCPConnectionCapacityError,
    MCPConnectionError,
    MCPConnectionEvictingError,
    MCPConnectionLostError,
    MCPConnectionStateError,
    MCPCredentialError,
    MCPDestinationDeniedError,
    MCPError,
    MCPOAuthError,
    MCPOAuthFlowError,
    MCPOAuthRefreshError,
    MCPOrigin,
    MCPOutcomeUnknownError,
    MCPResultTooLargeError,
    MCPRuntimeClosedError,
    MCPStaleConnectionError,
    MCPToolCallError,
    MCPToolNameCollisionError,
    error_code_for,
)
from dendrux.mcp._host import MCPHost  # noqa: E402
from dendrux.mcp._oauth import (  # noqa: E402
    MCPOAuth,
    MCPOAuthClaim,
    MCPOAuthClient,
    MCPOAuthClientRecord,
    MCPOAuthCredential,
    MCPOAuthFlow,
    MCPOAuthMemoryStore,
    MCPOAuthRecord,
    MCPOAuthRevocation,
    MCPOAuthStatus,
    MCPOAuthStore,
    MCPOAuthVersion,
    check_oauth_store,
)
from dendrux.mcp._observability import (  # noqa: E402
    MCPCallCapacityRejected,
    MCPCircuitClosed,
    MCPCircuitOpened,
    MCPCircuitProbing,
    MCPConnectionAborted,
    MCPConnectionCapacityRejected,
    MCPConnectionClosed,
    MCPConnectionEvent,
    MCPConnectionFailed,
    MCPConnectionOpened,
    MCPConnectionOpening,
    MCPConnectionSnapshot,
    MCPConnectionStatus,
    MCPEvictionCompleted,
    MCPEvictionStarted,
    MCPOAuthRefreshed,
    MCPOAuthRefreshFailed,
    MCPOAuthRequired,
    MCPOAuthRevoked,
    MCPOpenCircuitSnapshot,
    MCPPhysicalConnectionEvent,
    MCPRuntimeEvent,
    MCPRuntimeObserver,
    MCPRuntimeShutdownCompleted,
    MCPRuntimeShutdownStarted,
    MCPRuntimeSnapshot,
    MCPRuntimeState,
    MCPToolCallCancelled,
    MCPToolCallCompleted,
    MCPToolCallFailed,
    MCPToolCallOutcomeUnknown,
    MCPToolCallStarted,
)
from dendrux.mcp._runtime import (  # noqa: E402
    MCPConnection,
    MCPEvictionMode,
    MCPRuntime,
    MCPToolPolicy,
    MCPToolView,
)
from dendrux.mcp._server import MCPServer, MCPSkippedTool  # noqa: E402
from dendrux.mcp._source import MCPFailureMode, MCPSource  # noqa: E402

__all__ = [
    "MCPAuthChallenge",
    "MCPAuthKey",
    "MCPAuthenticationError",
    "MCPAuthorizationRequired",
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
    "MCPDestination",
    "MCPDestinationDeniedError",
    "MCPDestinationPolicy",
    "MCPDiscovery",
    "MCPError",
    "MCPEvictionCompleted",
    "MCPEvictionMode",
    "MCPEvictionStarted",
    "MCPFailureMode",
    "MCPHost",
    "MCPOAuth",
    "MCPOAuthClaim",
    "MCPOAuthClient",
    "MCPOAuthClientRecord",
    "MCPOAuthCredential",
    "MCPOAuthError",
    "MCPOAuthFlow",
    "MCPOAuthFlowError",
    "MCPOAuthMemoryStore",
    "MCPOAuthRecord",
    "MCPOAuthRefreshError",
    "MCPOAuthRefreshFailed",
    "MCPOAuthRefreshed",
    "MCPOAuthRequired",
    "MCPOAuthRevocation",
    "MCPOAuthRevoked",
    "MCPOAuthStatus",
    "MCPOAuthStore",
    "MCPOAuthVersion",
    "MCPOpenCircuitSnapshot",
    "MCPOrigin",
    "MCPOutcomeUnknownError",
    "MCPPhysicalConnectionEvent",
    "MCPRefreshingCredentialProvider",
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
    "MCPSkippedTool",
    "MCPSource",
    "MCPStaleConnectionError",
    "MCPToolCallCancelled",
    "MCPToolCallCompleted",
    "MCPToolCallError",
    "MCPToolCallFailed",
    "MCPToolCallOutcomeUnknown",
    "MCPToolCallStarted",
    "MCPToolInfo",
    "MCPToolNameCollisionError",
    "MCPToolPolicy",
    "MCPToolView",
    "check_oauth_store",
    "error_code_for",
]
