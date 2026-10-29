"""Recovery error compatibility and provider-code handoff."""

import pickle

from dendrux import errors


def test_run_control_hierarchy_and_copy():
    base = getattr(errors, "RunControlError", None)
    assert base is not None
    for cls in [
        errors.RunNotPausedError,
        errors.PauseStatusMismatchError,
        errors.RunAlreadyClaimedError,
        errors.RunAlreadyTerminalError,
    ]:
        assert issubclass(cls, base) and issubclass(cls, RuntimeError)
    for original in [
        errors.RunNotReclaimableError("00000000000000000000000401", "lease_fresh"),
        errors.RunCancelledError("00000000000000000000000401", "00000000000000000000000402"),
    ]:
        restored = pickle.loads(pickle.dumps(original))
        assert type(restored) is type(original) and vars(restored) == vars(original)


def test_provider_code_raw_is_separate_and_mapper_controls_safe_properties(monkeypatch):
    error = errors.ProviderError(provider="anthropic", provider_code="vendor_code")
    assert getattr(error, "provider_code_raw", None) == "vendor_code", "raw code needs its own slot"
    monkeypatch.setattr(errors, "_provider_code_mapper", lambda provider, raw: "provider_unknown")
    assert error.provider_code == error.safe_provider_code == "provider_unknown"
    assert "vendor_code" not in error.safe_message
    assert pickle.loads(pickle.dumps(error)).provider_code_raw == "vendor_code"


def test_invalid_raw_provider_code_is_never_projected():
    error = errors.ProviderError(provider="p", provider_code="private text with spaces")
    assert error.provider_code is None
    assert getattr(error, "provider_code_raw", None) is None


def test_persistence_error_preserves_stable_code():
    error = errors.PersistenceError(code="write_conflict")
    assert error.safe_message == "Database operation failed (write_conflict)."
    assert str(error) == error.safe_message
