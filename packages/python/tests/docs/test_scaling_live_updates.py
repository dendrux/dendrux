"""The custom transport in the scaling guide bounds slow or absent readers."""

import ast
import asyncio
from pathlib import Path

import pytest

from dendrux.store import LiveUpdates
from tests.docs._snippets import extract_snippets

REPO = Path(__file__).resolve().parents[4]
GUIDE = REPO / "docs/guides/scaling.mdx"
pytestmark = pytest.mark.skipif(not GUIDE.is_file(), reason="repository checkout only")


@pytest.mark.parametrize("listening", [False, True], ids=["no-listener", "slow-listener"])
async def test_loopback_overflow_is_bounded_and_yields_a_gap(listening):
    snippet = next(s for s in extract_snippets(GUIDE, REPO) if "class LoopbackUpdates" in s.code)
    tree = ast.parse(snippet.code)
    tree.body = [
        node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom, ast.ClassDef))
    ]
    namespace = {}
    exec(compile(tree, str(GUIDE), "exec"), namespace)
    bus = namespace["LoopbackUpdates"](max_inbound=2)
    reader = bus.receive()
    try:
        if listening:
            assert await anext(reader) is None
            await bus.send(['["first"]'])
            assert await anext(reader) == '["first"]'
        messages = [f'["run-{i}"]' for i in range(10_001)]
        await bus.send(messages)
        assert bus._queue.qsize() == 2
        assert bus._queue.maxsize == bus.max_inbound
        if not listening:
            assert await anext(reader) is None
        async with asyncio.timeout(1):
            assert await anext(reader) is LiveUpdates.GAP
            assert [await anext(reader), await anext(reader)] == messages[-2:]
            assert bus._queue.empty()
            await bus.send(['["after-gap"]'])
            assert await anext(reader) == '["after-gap"]'
            await bus.send(messages)
            assert bus._queue.qsize() == 2
            assert await anext(reader) is LiveUpdates.GAP
            assert [await anext(reader), await anext(reader)] == messages[-2:]
    finally:
        await reader.aclose()
        await bus.close()
