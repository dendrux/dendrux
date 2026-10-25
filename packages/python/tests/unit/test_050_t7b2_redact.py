"""Low-layer redaction keeps runtime compatibility and secondary stacks."""

import ast
import inspect
import traceback


def test_low_layer_redaction_and_runtime_compatibility():
    import dendrux._redact as redaction
    from dendrux.errors import ProviderAuthError
    from dendrux.loops import _helpers
    from dendrux.runtime.context import safe_exception_info as compatibility

    assert hasattr(redaction, "safe_exception_info"), "redaction must be available below runtime"
    safe_exception_info = redaction.safe_exception_info
    assert compatibility is safe_exception_info
    imports = [
        n.module
        for n in ast.walk(ast.parse(inspect.getsource(_helpers)))
        if isinstance(n, ast.ImportFrom)
    ]
    assert not any(m and m.startswith("dendrux.runtime") for m in imports)
    try:
        try:
            raise ProviderAuthError(provider="test", message="VENDOR_SENTINEL")
        except ProviderAuthError as vendor:
            raise LookupError("secondary") from vendor
    except LookupError as secondary:
        info = safe_exception_info(secondary)
        assert isinstance(info, tuple)
        formatted = "".join(traceback.format_exception(*info))
        assert "LookupError: secondary" in formatted
        assert "raise LookupError" in formatted
        assert "VENDOR_SENTINEL" not in formatted
        assert secondary.__cause__ is not None
