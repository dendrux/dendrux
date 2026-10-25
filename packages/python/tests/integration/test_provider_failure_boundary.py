"""Provider diagnostics remain bounded and typed across async workers."""

import json
import logging

import pytest

from dendrux import Agent
from dendrux.errors import ProviderError, ProviderTimeoutError, safe_exception_info
from dendrux.llm.mock import MockLLM
from dendrux.types import ErrorInfo

PRIVATE = "VENDOR_SENTINEL_2309"


@pytest.mark.parametrize("code", ["rate_limit\n", "x\r", "x\x00", "x\ud800"])
def test_provider_code_requires_a_complete_identifier(code):
    exc = ProviderError(provider="fixture", provider_code=code)
    assert exc.safe_provider_code is None
    assert code not in exc.safe_message


@pytest.mark.parametrize("field", ["provider", "model"])
@pytest.mark.parametrize("value", ["fixture\nFORGED", "fixture\x00", "x\ud800", "x" * 1000])
def test_provider_identity_is_safe_for_persisted_diagnostics(field, value):
    exc = ProviderError(**{field: value})
    info = ErrorInfo.from_error(exc).to_dict()
    assert info[field] is None, "invalid identity must be omitted, not echoed"
    assert value not in info["safe_message"]


def test_group_diagnostic_does_not_format_vendor_leaves(caplog):
    error = ExceptionGroup(PRIVATE, [ProviderTimeoutError(message=PRIVATE)])
    with caplog.at_level(logging.ERROR):
        try:
            raise error
        except ExceptionGroup:
            logging.getLogger("dendrux.fixture").error(
                "request failed", exc_info=safe_exception_info(error)
            )
    assert PRIVATE not in caplog.text


@pytest.mark.parametrize("stream", [False, True])
async def test_grouped_provider_failure_preserves_typed_handling(store, caplog, stream):
    class GroupProvider(MockLLM):
        async def complete(self, *args, **kwargs):
            raise ExceptionGroup(
                PRIVATE,
                [
                    ExceptionGroup(
                        "worker",
                        [ProviderTimeoutError(provider="fixture", model="m", message=PRIVATE)],
                    )
                ],
            )

        async def complete_stream(self, *args, **kwargs):
            await self.complete(*args, **kwargs)
            yield  # pragma: no cover

    agent = Agent(prompt="p", provider=GroupProvider([]), state_store=store, strict_tenancy=False)
    try:
        with caplog.at_level(logging.ERROR):
            if stream:
                events = [event async for event in agent.stream("hello", tenant_id="A")]
                errors = [event for event in events if event.error]
                assert errors and "timeout" in errors[-1].error
            else:
                with pytest.raises(ProviderTimeoutError):
                    await agent.run("hello", tenant_id="A")
        rows = await store.list_runs(tenant_id="A")
        assert len(rows) == 1
        record = await store.get_run(rows[0].id)
        assert record.error == "fixture timeout (HTTP n/a, code n/a)"
        assert PRIVATE not in json.dumps(record.meta)
        assert PRIVATE not in caplog.text
    finally:
        await agent.close()
