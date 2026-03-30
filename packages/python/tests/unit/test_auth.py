"""Tests for dendrite.auth — HMAC token generation and verification."""

from __future__ import annotations

from dendrite.auth import (
    _build_message,
    extract_bearer_token,
    generate_run_token,
    verify_run_token,
)

_TEST_SECRET = "test-secret-key-for-hmac"


class TestAuthTokenGeneration:
    def test_generate_and_verify_roundtrip(self) -> None:
        token = generate_run_token("run-123", _TEST_SECRET)
        assert token.startswith("drn_")
        assert verify_run_token("run-123", token, _TEST_SECRET)

    def test_wrong_run_id_fails(self) -> None:
        token = generate_run_token("run-123", _TEST_SECRET)
        assert not verify_run_token("run-456", token, _TEST_SECRET)

    def test_tampered_token_fails(self) -> None:
        token = generate_run_token("run-123", _TEST_SECRET)
        tampered = token[:-4] + "dead"
        assert not verify_run_token("run-123", tampered, _TEST_SECRET)

    def test_wrong_secret_fails(self) -> None:
        token = generate_run_token("run-123", _TEST_SECRET)
        assert not verify_run_token("run-123", token, "wrong-secret")

    def test_missing_prefix_fails(self) -> None:
        assert not verify_run_token("run-123", "not_a_drn_token", _TEST_SECRET)

    def test_versioned_message_format(self) -> None:
        """Token signs 'drn:v0:<run_id>', not raw run_id."""
        msg = _build_message("run-123")
        assert msg == b"drn:v0:run-123"


class TestExtractBearerToken:
    def test_valid_bearer(self) -> None:
        assert extract_bearer_token("Bearer drn_abc123") == "drn_abc123"

    def test_case_insensitive_scheme(self) -> None:
        assert extract_bearer_token("bearer drn_abc123") == "drn_abc123"

    def test_missing_header(self) -> None:
        assert extract_bearer_token(None) is None

    def test_malformed_header(self) -> None:
        assert extract_bearer_token("Basic abc123") is None
        assert extract_bearer_token("drn_abc123") is None  # no scheme
        assert extract_bearer_token("") is None
