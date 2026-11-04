"""An oversized publication is discarded without retaining its body or losing framing."""

import asyncio

import pytest

from dendrux.db import _redis_updates as ru
from tests.unit.test_redis_updates_hardening import listen_parser, message


@pytest.mark.parametrize("path", ["url", "options"])
def test_decode_responses_cannot_override_wire_bytes(path):
    url = "redis://localhost/5"
    options = {"client_options": {"decode_responses": True}}
    if path == "url":
        url += "?decode_responses=true"
        options = {}
    with pytest.raises(ValueError, match="decode_responses"):
        ru.RedisUpdates(url, **options)


async def test_oversize_reply_is_replaced_and_next_reply_survives():
    parser, reader = listen_parser(ru.RedisUpdates("redis://localhost/5"))
    reader.feed_data(message(7901) + ru._reply(b'["good"]'))
    first = await parser.read_response()
    assert first[2] is ru._DISCARDED
    assert (await parser.read_response())[2] == b'["good"]'


async def test_fragmented_large_reply_keeps_memory_bounded():
    parser, reader = listen_parser(ru.RedisUpdates("redis://localhost/5"))
    size = 2_000_000
    reader.feed_data(b"*3\r\n$7\r\nmessage\r\n$1\r\nc\r\n$2000000\r\n")
    reading = asyncio.create_task(parser.read_response())
    try:
        for offset in range(0, size, 4096):
            reader.feed_data(b"x" * min(4096, size - offset))
            await asyncio.sleep(0)
            assert sum(map(len, parser._chunks)) + len(parser._buffer) < 100_000
            assert len(reader._buffer) < 100_000
        reader.feed_data(b"\r\n" + ru._reply(b'["good"]'))
        assert (await reading)[2] is ru._DISCARDED
        assert (await parser.read_response())[2] == b'["good"]'
    finally:
        reading.cancel()
        await asyncio.gather(reading, return_exceptions=True)


async def test_interrupted_discard_fails_connection_instead_of_replaying_partial_body():
    parser, reader = listen_parser(ru.RedisUpdates("redis://localhost/5"))
    reader.feed_data(b"*3\r\n$7\r\nmessage\r\n$1\r\nc\r\n$2000000\r\n" + b"x" * 70000)
    with pytest.raises(ConnectionError):
        async with asyncio.timeout(0.02):
            await parser.read_response()
