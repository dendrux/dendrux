"""Value-free errors, events, and representations for MCP OAuth."""

from __future__ import annotations

import pytest

from dendrux.mcp._credentials import (
    MCPAuthKey,
    MCPCredentialProvider,
    MCPRefreshingCredentialProvider,
)
from dendrux.mcp._errors import (
    MCPAuthorizationRequired,
    MCPConnectionError,
    MCPCredentialError,
    MCPError,
    MCPOAuthError,
    MCPOAuthFlowError,
    MCPOAuthRefreshError,
)
from dendrux.mcp._observability import (
    MCPConnectionEvent,
    MCPOAuthRefreshed,
    MCPOAuthRefreshFailed,
    MCPOAuthRequired,
    MCPOAuthRevoked,
)

SENTINEL = "SENTINEL-redaction-4be1"


def test_runtime_module_keeps_reexporting_the_credential_protocols() -> None:
    from dendrux.mcp import _runtime

    assert _runtime.MCPCredentialProvider is MCPCredentialProvider
    assert _runtime.MCPRefreshingCredentialProvider is MCPRefreshingCredentialProvider


@pytest.mark.parametrize(
    ("connection_key", "tenant_key"),
    [("", None), ("  ", None), ("a\nb", None), ("ok", ""), ("ok", "t\x00"), (3, None)],
)
def test_auth_key_validation(connection_key: object, tenant_key: object) -> None:
    with pytest.raises(ValueError):
        MCPAuthKey(connection_key, tenant_key=tenant_key)  # type: ignore[arg-type]


def test_auth_key_equality_is_by_value() -> None:
    assert MCPAuthKey("c", "t") == MCPAuthKey("c", tenant_key="t")
    assert MCPAuthKey("c") != MCPAuthKey("c", "t")


def test_authorization_required_is_value_free() -> None:
    key = MCPAuthKey(SENTINEL, tenant_key=SENTINEL)
    error = MCPAuthorizationRequired(
        key,
        "insufficient_scope",
        scopes=(SENTINEL,),
        resource_metadata_url=f"https://rs.example/{SENTINEL}",
        status_code=403,
    )
    assert isinstance(error, MCPConnectionError)
    assert not isinstance(error, MCPCredentialError)
    assert SENTINEL not in str(error)
    assert SENTINEL not in repr(error)
    assert error.reason == "insufficient_scope"
    assert error.scopes == (SENTINEL,)
    assert error.key is key
    assert error.status_code == 403


def test_flow_error_carries_reason_and_status_only() -> None:
    error = MCPOAuthFlowError("token_exchange_failed", status_code=502)
    assert isinstance(error, MCPOAuthError) and isinstance(error, MCPError)
    assert error.reason == "token_exchange_failed"
    assert error.status_code == 502
    assert "token_exchange_failed" in str(error)


def test_refresh_error_is_a_transient_credential_error() -> None:
    error = MCPOAuthRefreshError("MCP OAuth refresh is held by another worker.")
    assert isinstance(error, MCPOAuthError)
    assert isinstance(error, MCPCredentialError)
    assert isinstance(error, MCPConnectionError)


def _identity() -> dict[str, str]:
    return {"tenant_key": SENTINEL, "connection_key": SENTINEL, "source_name": "demo"}


@pytest.mark.parametrize(
    ("event", "expected"),
    [
        (
            MCPOAuthRequired(**_identity(), reason="expired", scope_count=2),
            "MCPOAuthRequired(reason='expired', scope_count=2)",
        ),
        (MCPOAuthRefreshed(**_identity(), expires_in=60.0), "MCPOAuthRefreshed(expires_in=60.0)"),
        (
            MCPOAuthRefreshFailed(**_identity(), outcome="uncertain", cause=SENTINEL),
            "MCPOAuthRefreshFailed(outcome='uncertain')",
        ),
        (MCPOAuthRevoked(**_identity(), remote="confirmed"), "MCPOAuthRevoked(remote='confirmed')"),
    ],
)
def test_oauth_event_reprs_show_outcomes_and_counts_only(
    event: MCPConnectionEvent, expected: str
) -> None:
    assert isinstance(event, MCPConnectionEvent)
    assert repr(event) == expected
    assert event.identity == (SENTINEL, SENTINEL)
