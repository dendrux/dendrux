"""Typed failures are enforced where a provider is invoked, regardless of binding."""

import json
from dataclasses import replace
from types import MethodType

import pytest
from pydantic import BaseModel

from dendrux import Agent
from dendrux.errors import ProviderTimeoutError
from dendrux.llm.mock import MockLLM
from dendrux.loops.react import ReActLoop
from dendrux.loops.single import SingleCall

PRIVATE = "PRIVATE_GROUP_BINDING_SENTINEL"


def failure():
    return ExceptionGroup(PRIVATE, [ProviderTimeoutError(provider="fixture", message=PRIVATE)])


async def fail_complete(*args, **kwargs):
    raise failure()


async def fail_stream(*args, **kwargs):
    raise failure()
    yield  # pragma: no cover


def make_provider(binding):
    class Parent(MockLLM):
        pass

    class Child(Parent):
        pass

    if binding == "mixin":

        class Mixin:
            complete = fail_complete
            complete_stream = fail_stream

        class Mixed(Mixin, MockLLM):
            pass

        return Mixed([])
    if binding == "inherited":
        Parent.complete = fail_complete
        Parent.complete_stream = fail_stream
        return Child([])
    provider = Child([])
    provider.complete = MethodType(fail_complete, provider)
    provider.complete_stream = MethodType(fail_stream, provider)
    if binding == "factory":

        def create(*args, **kwargs):
            raise failure()

        provider.complete_stream = create
    return provider


@pytest.mark.parametrize("binding", ["mixin", "inherited", "instance", "factory"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("loop_type", [ReActLoop, SingleCall])
async def test_provider_binding_preserves_typed_run_failure(
    store, caplog, binding, stream, loop_type
):
    agent = Agent(
        prompt="p",
        provider=make_provider(binding),
        loop=loop_type(),
        state_store=store,
        strict_tenancy=False,
    )
    try:
        if stream:
            async with agent.stream("go", tenant_id="A") as source:
                events = [event async for event in source]
            assert source.result.meta["error_type"] == "ProviderTimeoutError"
            assert "fixture timeout" in events[-1].error
        else:
            with pytest.raises(ProviderTimeoutError):
                await agent.run("go", tenant_id="A")
        rows = await store.list_runs(tenant_id="A")
        assert len(rows) == 1
        assert rows[0].error == "fixture timeout (HTTP n/a, code n/a)"
        assert PRIVATE not in json.dumps(rows[0].meta)
        assert PRIVATE not in caplog.text
    finally:
        await agent.close()


@pytest.mark.parametrize("binding", ["mixin", "inherited", "instance"])
async def test_structured_completion_uses_the_same_typed_boundary(store, binding):
    class Answer(BaseModel):
        answer: str

    provider = make_provider(binding)
    provider.capabilities = replace(provider.capabilities, supports_structured_output=True)
    agent = Agent(
        prompt="p",
        provider=provider,
        output_type=Answer,
        loop=SingleCall(),
        state_store=store,
        strict_tenancy=False,
    )
    try:
        with pytest.raises(ProviderTimeoutError):
            await agent.run("go", tenant_id="A")
    finally:
        await agent.close()
