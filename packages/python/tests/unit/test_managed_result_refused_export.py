"""ManagedResultRefusedError is public API: importable from dendrux.errors."""

import pytest

import dendrux.errors as public_errors
from dendrux.errors import InvalidToolResultError, ManagedResultRefusedError

RUN = "01JABCDEFGHJKMNPQRSTVWXYZ0"


def test_exported_from_public_errors_module():
    assert "ManagedResultRefusedError" in public_errors.__all__
    assert public_errors.ManagedResultRefusedError is ManagedResultRefusedError


def test_bases_and_code_are_unchanged():
    assert ManagedResultRefusedError.__bases__ == (InvalidToolResultError,)
    assert issubclass(ManagedResultRefusedError, ValueError)
    assert ManagedResultRefusedError.code == "managed_result_refused"


def test_runtime_raises_the_exported_class():
    from dendrux.runtime.submit import refuse_managed_results
    from dendrux.types import ToolResult

    forged = [ToolResult(name="reviewer", call_id="m1", payload='"x"')]
    with pytest.raises(public_errors.ManagedResultRefusedError) as info:
        refuse_managed_results(RUN, {"m1": "agent"}, forged)
    assert type(info.value) is ManagedResultRefusedError
