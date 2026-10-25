"""Secondary stacks cannot format typed failures nested in worker groups."""

import logging

from dendrux.errors import ProviderTimeoutError, safe_exception_info


def test_secondary_group_chain_keeps_its_own_stack_without_worker_messages(caplog):
    private = "PRIVATE_GROUP_VENDOR_SENTINEL"
    try:
        raise ExceptionGroup(private, [ProviderTimeoutError(message=private)])
    except ExceptionGroup:
        try:
            raise ValueError("secondary failure")
        except ValueError:
            with caplog.at_level(logging.ERROR):
                logging.getLogger("dendrux.fixture").error("failed", exc_info=safe_exception_info())
    assert private not in caplog.text
    assert "ValueError: secondary failure" in caplog.text
    assert "raise ValueError" in caplog.text


def test_persistence_failure_has_structured_safe_wire_information():
    from dendrux.errors import PersistenceError
    from dendrux.runtime.runner import _provider_error_info
    from dendrux.types import RunResult, RunStatus

    info = _provider_error_info(PersistenceError(retryable=True))
    assert info is not None, "database failures need a stable code in the allowlisted wire DTO"
    assert info.category == "persistence_error" and info.retryable is True
    wire = RunResult("r", RunStatus.ERROR, error_info=info).to_dict()
    assert wire["error_info"]["category"] == "persistence_error"
    assert wire["error_info"]["safe_message"] == PersistenceError.safe_message
