"""A discarded frame cannot be replayed after a later field times out."""

import asyncio

import pytest
from redis._parsers import _AsyncRESP2Parser
from redis.asyncio.connection import Connection

from dendrux.db._redis_updates import _DISCARDED, _bounded_parser


def _bulk(body):
    return b"$" + str(len(body)).encode() + b"\r\n" + body + b"\r\n"


def _parser():
    connection = Connection(parser_class=_bounded_parser(_AsyncRESP2Parser, 100), protocol=2)
    connection._reader = asyncio.StreamReader()
    parser = connection._parser
    parser.on_connect(connection)
    return parser, connection._reader, connection


async def test_oversize_nonlast_frame_refuses_replay_immediately():
    parser, stream, _ = _parser()
    stream.feed_data(b"*3\r\n" + _bulk(b"message") + _bulk(b"x" * 101))
    with pytest.raises(TimeoutError):
        async with asyncio.timeout(0.01):
            await parser.read_response()
    stream.feed_data(
        _bulk(b'["after"]') + b"*3\r\n" + _bulk(b"message") + _bulk(b"c") + _bulk(b'["next"]')
    )
    before = bytes(stream._buffer)
    with pytest.raises(ConnectionError, match="replay"):
        async with asyncio.timeout(0.1):
            await parser.read_response()
    assert bytes(stream._buffer) == before


async def test_buffered_oversize_field_advances_to_following_field():
    parser, _, _ = _parser()
    parser._buffer = b"*3\r\n" + _bulk(b"message") + _bulk(b"x" * 101) + _bulk(b'["good"]')
    assert await parser.read_response() == [b"message", _DISCARDED, b'["good"]']


async def test_new_connection_clears_poison_and_reads_normally():
    parser, stream, connection = _parser()
    stream.feed_data(b"*3\r\n" + _bulk(b"message") + _bulk(b"x" * 101))
    with pytest.raises(TimeoutError):
        async with asyncio.timeout(0.01):
            await parser.read_response()
    parser.on_disconnect()
    connection._reader = asyncio.StreamReader()
    parser.on_connect(connection)
    connection._reader.feed_data(_bulk(b"ok"))
    assert await parser.read_response() == b"ok"
