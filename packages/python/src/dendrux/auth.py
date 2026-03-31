"""Run-scoped HMAC token generation and verification.

Each POST /runs response includes a token bound to the run_id.
All subsequent requests for that run must present the token via
Authorization: Bearer drn_<signature>.

Token format (v0):
    HMAC-SHA256("drn:v0:" + run_id, secret) → hex-encoded, prefixed with "drn_"

The "drn:v0:" prefix in the signed message provides clean versioning —
future token formats (v1 with expiry/scope) can coexist during migration.

When hmac_secret is not provided AND allow_insecure_dev_mode=True,
auth is disabled entirely (local dev). Without explicit opt-in,
missing secret causes startup failure (fail-closed).
"""

from __future__ import annotations

import hashlib
import hmac

_TOKEN_PREFIX = "drn_"
_SIGN_VERSION = "v0"


def _build_message(run_id: str) -> bytes:
    """Build the versioned message to sign."""
    return f"drn:{_SIGN_VERSION}:{run_id}".encode()


def generate_run_token(run_id: str, secret: str) -> str:
    """Generate an HMAC-SHA256 token bound to a specific run_id.

    Signs the versioned message "drn:v0:<run_id>" and prefixes the
    hex digest with "drn_" for easy identification.
    """
    sig = hmac.new(secret.encode(), _build_message(run_id), hashlib.sha256).hexdigest()
    return f"{_TOKEN_PREFIX}{sig}"


def verify_run_token(run_id: str, token: str, secret: str) -> bool:
    """Verify that a token is valid for the given run_id.

    Uses constant-time comparison to prevent timing attacks.
    Returns False for malformed tokens (missing prefix, wrong length).
    """
    if not token.startswith(_TOKEN_PREFIX):
        return False
    expected = generate_run_token(run_id, secret)
    return hmac.compare_digest(token, expected)


def extract_bearer_token(authorization: str | None) -> str | None:
    """Extract the token from an Authorization: Bearer header.

    Returns None if the header is missing or malformed.
    """
    if not authorization:
        return None
    parts = authorization.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    return parts[1]
