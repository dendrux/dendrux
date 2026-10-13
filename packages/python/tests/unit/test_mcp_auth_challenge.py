"""``WWW-Authenticate`` challenge capture on credential rejections.

The parsed challenge rides on ``MCPAuthenticationError.challenge`` at every
raise site (connect, discovery, tool call) and never enters the message.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import httpx2
import pytest
from mcp.server.mcpserver import MCPServer as SDKServer

from dendrux.mcp import MCPAuthChallenge, MCPAuthenticationError, MCPSource
from dendrux.mcp._client import MCPClientAdapter, parse_auth_challenge
from tests.unit.test_mcp_streamable_http_auth import _serve

SOURCE_URL = "https://mcp.example.com/mcp"
FULL_HEADER = (
    'Bearer realm="mcp", error="insufficient_scope", '
    'error_description="needs HEADER-DESCRIPTION", scope="files:read files:write", '
    'resource_metadata="https://mcp.example.com/.well-known/oauth-protected-resource"'
)


def _response(
    status: int,
    header: str | None,
    *,
    url: str = SOURCE_URL,
    body: dict[str, Any] | None = None,
) -> httpx2.Response:
    headers = {"www-authenticate": header} if header is not None else {}
    request = httpx2.Request("POST", url, json=body or {"method": "initialize"})
    return httpx2.Response(status, headers=headers, request=request)


def _rendered(error: BaseException) -> str:
    parts: list[str] = [repr(error), error.transport_detail or ""]
    node: BaseException | None = error
    while node is not None:
        parts.append(str(node))
        node = node.__cause__ or node.__context__
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def test_full_challenge_is_parsed() -> None:
    challenge = parse_auth_challenge(_response(403, FULL_HEADER))
    assert challenge == MCPAuthChallenge(
        scheme="Bearer",
        error="insufficient_scope",
        error_description="needs HEADER-DESCRIPTION",
        scope=("files:read", "files:write"),
        resource_metadata_url="https://mcp.example.com/.well-known/oauth-protected-resource",
    )


def test_unquoted_fields_and_bare_scheme() -> None:
    challenge = parse_auth_challenge(_response(401, "Bearer error=invalid_token, scope=read"))
    assert challenge is not None
    assert challenge.scheme == "Bearer"
    assert challenge.error == "invalid_token"
    assert challenge.scope == ("read",)
    assert challenge.resource_metadata_url is None

    bare = parse_auth_challenge(_response(401, "Bearer"))
    assert bare == MCPAuthChallenge(scheme="Bearer")


def test_missing_header_has_no_challenge() -> None:
    assert parse_auth_challenge(_response(401, None)) is None
    assert parse_auth_challenge(object()) is None


@pytest.mark.parametrize("header", ['"=,,', 'error=""', "=", '   , scope="" '])
def test_malformed_header_yields_empty_fields(header: str) -> None:
    challenge = parse_auth_challenge(_response(401, header))
    assert challenge is not None
    assert challenge.scheme is None
    assert challenge.error is None
    assert challenge.scope == ()
    assert challenge.resource_metadata_url is None


def test_parser_failure_is_never_an_exception() -> None:
    with patch("dendrux.mcp._client.extract_field_from_www_auth", side_effect=ValueError("boom")):
        challenge = parse_auth_challenge(_response(401, FULL_HEADER))
    assert challenge == MCPAuthChallenge()


def test_unreadable_or_non_text_header_has_no_challenge() -> None:
    class RaisingHeaders:
        def get(self, name: str) -> str:
            raise RuntimeError("headers unavailable")

    class Fake:
        def __init__(self, headers: Any) -> None:
            self.headers = headers

    assert parse_auth_challenge(Fake(RaisingHeaders())) is None
    assert parse_auth_challenge(Fake({"www-authenticate": b"Bearer"})) is None
    assert parse_auth_challenge(Fake({"www-authenticate": ""})) is None


async def test_response_without_a_request_is_not_request_rejected() -> None:
    adapter = MCPClientAdapter(MCPSource.http("demo", SOURCE_URL))
    # A bare httpx2 response raises RuntimeError on ``.request``.
    error = await _connect_with(
        adapter, httpx2.Response(401, headers={"www-authenticate": "Bearer"})
    )
    assert error.status_code == 401
    assert not error.request_rejected
    assert error.challenge == MCPAuthChallenge(scheme="Bearer")


def test_stdio_source_status_is_never_request_rejected() -> None:
    adapter = MCPClientAdapter(MCPSource.stdio("demo", ["echo"]))
    response = _response(401, "Bearer")
    failure = httpx2.HTTPStatusError("rejected", request=response.request, response=response)
    error = adapter._safe_error("connect to", failure)
    assert isinstance(error, MCPAuthenticationError)
    assert not error.request_rejected
    assert error.challenge == MCPAuthChallenge(scheme="Bearer")


def test_status_attribute_without_response_falls_back_to_observed_rejection() -> None:
    class StatusError(RuntimeError):
        status_code = 401

    adapter = MCPClientAdapter(MCPSource.http("demo", SOURCE_URL))
    adapter._last_rejection = None
    error = adapter._safe_error("connect to", StatusError("rejected"))
    assert isinstance(error, MCPAuthenticationError)
    assert error.challenge is None
    assert not error.request_rejected


# ---------------------------------------------------------------------------
# Raise sites with an observed response
# ---------------------------------------------------------------------------


class _ObservedClient:
    adapter: MCPClientAdapter
    responses: tuple[httpx2.Response, ...] = ()

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    async def __aenter__(self) -> _ObservedClient:
        for response in type(self).responses:
            await type(self).adapter._observe_response(response)
        raise RuntimeError("Server returned an error response")

    async def __aexit__(self, *exc: Any) -> None:
        return None


async def _connect_with(adapter: MCPClientAdapter, *responses: httpx2.Response) -> Any:
    _ObservedClient.adapter = adapter
    _ObservedClient.responses = responses
    with (
        patch("dendrux.mcp._client.Client", _ObservedClient),
        patch("dendrux.mcp._client.streamable_http_client", return_value=object()),
        pytest.raises(MCPAuthenticationError) as caught,
    ):
        await adapter.connect()
    return caught.value


@pytest.mark.parametrize("status", [401, 403])
async def test_connect_rejection_carries_the_challenge(status: int) -> None:
    adapter = MCPClientAdapter(MCPSource.http("demo", SOURCE_URL))
    error = await _connect_with(adapter, _response(status, FULL_HEADER))
    assert error.status_code == status
    assert error.request_rejected
    assert error.challenge is not None
    assert error.challenge.error == "insufficient_scope"
    assert error.challenge.scope == ("files:read", "files:write")
    assert "HEADER-DESCRIPTION" not in str(error)
    assert "insufficient_scope" not in str(error)
    assert "well-known" not in str(error)


async def test_connect_rejection_from_another_origin_is_not_request_rejected() -> None:
    adapter = MCPClientAdapter(MCPSource.http("demo", SOURCE_URL))
    error = await _connect_with(
        adapter, _response(401, FULL_HEADER, url="https://elsewhere.example/mcp")
    )
    assert error.status_code == 401
    assert not error.request_rejected
    assert error.challenge is not None


async def test_connect_rejection_without_header_has_no_challenge() -> None:
    adapter = MCPClientAdapter(MCPSource.http("demo", SOURCE_URL))
    error = await _connect_with(adapter, _response(401, None))
    assert error.request_rejected
    assert error.challenge is None


async def test_status_in_exception_tree_uses_that_response() -> None:
    adapter = MCPClientAdapter(MCPSource.http("demo", SOURCE_URL))
    response = _response(401, 'Bearer error="invalid_token"')
    failure = httpx2.HTTPStatusError("rejected", request=response.request, response=response)
    error = adapter._safe_error("connect to", failure)
    assert isinstance(error, MCPAuthenticationError)
    assert error.request_rejected
    assert error.challenge is not None
    assert error.challenge.error == "invalid_token"


async def test_discovery_rejection_carries_the_challenge() -> None:
    adapter = MCPClientAdapter(MCPSource.http("demo", SOURCE_URL))

    class Client:
        async def list_tools(self, **kwargs: Any) -> Any:
            await adapter._observe_response(_response(401, 'Bearer error="invalid_token"'))
            raise RuntimeError("Server returned an error response")

    adapter._client = Client()
    with pytest.raises(MCPAuthenticationError) as caught:
        await adapter.list_tools()
    assert caught.value.request_rejected
    assert caught.value.challenge is not None
    assert caught.value.challenge.error == "invalid_token"


async def test_tool_call_rejection_carries_the_challenge() -> None:
    adapter = MCPClientAdapter(MCPSource.http("demo", SOURCE_URL))

    async def call(name: str, arguments: dict[str, Any], **kwargs: Any) -> Any:
        body = {"method": "tools/call", "params": {"name": name}}
        await adapter._observe_response(_response(403, FULL_HEADER, body=body))
        raise RuntimeError("SDK rejected request")

    class Client:
        call_tool = staticmethod(call)

    adapter._client = Client()
    with pytest.raises(MCPAuthenticationError) as caught:
        await adapter.call_tool("echo", {})
    error = caught.value
    assert error.status_code == 403
    assert error.request_rejected
    assert error.challenge is not None
    assert error.challenge.error == "insufficient_scope"
    assert error.challenge.scope == ("files:read", "files:write")
    assert "HEADER-DESCRIPTION" not in str(error)


# ---------------------------------------------------------------------------
# End to end through the real SDK Streamable HTTP stack
# ---------------------------------------------------------------------------


def _challenging_app(state: dict[str, Any]) -> Any:
    server = SDKServer(name="challenge-demo")

    @server.tool()
    def echo(text: str) -> str:
        return text

    inner = server.streamable_http_app()

    async def app(scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] == "http":
            headers = dict(scope["headers"])
            if headers.get(b"authorization") != f"Bearer {state['token']}".encode():
                await send(
                    {
                        "type": "http.response.start",
                        "status": state["status"],
                        "headers": [(b"www-authenticate", state["challenge"].encode())],
                    }
                )
                await send({"type": "http.response.body", "body": b"rejected"})
                return
        await inner(scope, receive, send)

    return app


@pytest.mark.parametrize("status", [401, 403])
async def test_real_sdk_connect_rejection_carries_the_challenge(status: int) -> None:
    state = {"token": "good", "status": status, "challenge": FULL_HEADER}
    async with _serve(_challenging_app(state)) as url:
        adapter = MCPClientAdapter(
            MCPSource.http("demo", url, headers={"Authorization": "Bearer WRONG-TOKEN"})
        )
        with pytest.raises(MCPAuthenticationError) as caught:
            await adapter.connect()
        await adapter.close()
    error = caught.value
    assert error.status_code == status
    assert error.request_rejected
    assert error.challenge is not None
    assert error.challenge.scheme == "Bearer"
    assert error.challenge.error == "insufficient_scope"
    assert error.challenge.scope == ("files:read", "files:write")
    rendered = _rendered(error)
    assert "WRONG-TOKEN" not in rendered
    assert "HEADER-DESCRIPTION" not in str(error)


@pytest.mark.parametrize(
    ("status", "header", "expected_error"),
    [
        (401, 'Bearer error="invalid_token", error_description="expired"', "invalid_token"),
        (403, FULL_HEADER, "insufficient_scope"),
    ],
)
async def test_real_sdk_tool_call_rejection_carries_the_challenge(
    status: int, header: str, expected_error: str
) -> None:
    state = {"token": "one", "status": status, "challenge": header}
    async with _serve(_challenging_app(state)) as url:
        adapter = MCPClientAdapter(
            MCPSource.http("demo", url, headers={"Authorization": "Bearer one"})
        )
        await adapter.connect()
        try:
            state["token"] = "two"
            with pytest.raises(MCPAuthenticationError) as caught:
                await adapter.call_tool("echo", {"text": "hi"})
        finally:
            await adapter.close()
    error = caught.value
    assert error.status_code == status
    assert error.request_rejected
    assert error.challenge is not None
    assert error.challenge.error == expected_error
    assert expected_error not in str(error)


# ---------------------------------------------------------------------------
# No response header or body text escapes into messages, events, or logs
# ---------------------------------------------------------------------------

SENTINEL = "SENTINEL-7f3a"
SENTINEL_HEADER = (
    f'Bearer realm="{SENTINEL}-realm", error="insufficient_scope", '
    f'error_description="{SENTINEL}-description", scope="{SENTINEL}-scope", '
    f'resource_metadata="https://{SENTINEL}.example/prm"'
)


def _sentinel_app(state: dict[str, Any]) -> Any:
    server = SDKServer(name="sentinel-demo")

    @server.tool()
    def echo(text: str) -> str:
        return text

    inner = server.streamable_http_app()

    async def app(scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] == "http":
            headers = dict(scope["headers"])
            if headers.get(b"authorization") != f"Bearer {state['token']}".encode():
                await send(
                    {
                        "type": "http.response.start",
                        "status": state["status"],
                        "headers": [
                            (b"www-authenticate", SENTINEL_HEADER.encode()),
                            (b"content-type", b"application/json"),
                        ],
                    }
                )
                body = f'{{"error": "{SENTINEL}-body", "detail": "{SENTINEL}"}}'
                await send({"type": "http.response.body", "body": body.encode()})
                return
        await inner(scope, receive, send)

    return app


def _exception_text(error: BaseException) -> str:
    parts: list[str] = []
    stack: list[BaseException] = [error]
    seen: set[int] = set()
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        parts.extend([str(node), repr(node), str(getattr(node, "transport_detail", ""))])
        if isinstance(node, BaseExceptionGroup):
            stack.extend(node.exceptions)
        for linked in (node.__cause__, node.__context__):
            if linked is not None:
                stack.append(linked)
    return "\n".join(parts)


def _log_text(caplog: pytest.LogCaptureFixture) -> str:
    # httpcore2's DEBUG wire trace prints raw response (and request) headers
    # by design; it is a third-party transport log outside Dendrux's control.
    # Every other logger, Dendrux and the MCP SDK included, is checked.
    records = [r for r in caplog.records if not r.name.startswith("httpcore2")]
    assert any(r.name.startswith("dendrux") for r in records)
    return "\n".join(r.getMessage() for r in records)


def _assert_no_sentinel(error: BaseException, events: list[Any], log_text: str) -> None:
    assert SENTINEL not in _exception_text(error)
    for event in events:
        assert SENTINEL not in repr(event)
    assert SENTINEL not in log_text


@pytest.mark.parametrize("status", [401, 403])
async def test_rejection_text_never_escapes_at_connect_discovery_and_call(
    status: int, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level("DEBUG")
    state = {"token": "good", "status": status}
    async with _serve(_sentinel_app(state)) as url:
        adapter = MCPClientAdapter(
            MCPSource.http("demo", url, headers={"Authorization": "Bearer stale"})
        )
        with pytest.raises(MCPAuthenticationError) as connect_error:
            await adapter.connect()
        await adapter.close()
        failures: list[BaseException] = [connect_error.value]
        # Discovery and call: the token is revoked after the handshake.
        for operation in ("list_tools", "call_tool"):
            state["token"] = "good"
            adapter = MCPClientAdapter(
                MCPSource.http("demo", url, headers={"Authorization": "Bearer good"})
            )
            await adapter.connect()
            state["token"] = "rotated"
            try:
                with pytest.raises(MCPAuthenticationError) as caught:
                    if operation == "list_tools":
                        await adapter.list_tools()
                    else:
                        await adapter.call_tool("echo", {"text": "hi"})
            finally:
                await adapter.close()
            failures.append(caught.value)
    for error in failures:
        assert isinstance(error, MCPAuthenticationError)
        assert error.status_code == status
        assert error.request_rejected
        assert error.challenge is not None
        assert error.challenge.error_description == f"{SENTINEL}-description"
        _assert_no_sentinel(error, [], _log_text(caplog))


async def test_rejection_text_never_escapes_through_the_runtime(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from dendrux.mcp import MCPRuntime
    from dendrux.mcp._runtime import _ViewToolSource

    class Provider:
        def __init__(self) -> None:
            self.token = "stale"

        async def get_auth(self) -> dict[str, str]:
            return {"Authorization": f"Bearer {self.token}"}

        async def on_auth_rejected(self) -> None:
            self.token = "still-stale"

    class Recorder:
        def __init__(self) -> None:
            self.events: list[Any] = []

        def on_event(self, event: Any) -> None:
            self.events.append(event)

    caplog.set_level("DEBUG")
    recorder = Recorder()
    state = {"token": "good", "status": 401}
    async with _serve(_sentinel_app(state)) as url, MCPRuntime(observer=recorder) as runtime:
        # Connect-time recovery runs once and the second 401 surfaces.
        connection = runtime.bind(
            connection_key="recovering",
            credentials=Provider(),
            source=MCPSource.http("demo", url, reauthenticate_on_401=True),
        )
        with pytest.raises(MCPAuthenticationError) as connect_error:
            await connection.discover()
        assert connect_error.value.request_rejected

        # A 403 on a tool call through a managed view.
        good = runtime.bind(
            connection_key="calls",
            source=MCPSource.http("calls", url, headers={"Authorization": "Bearer good"}),
        )
        view = _ViewToolSource(good.tools())
        await view._discover()
        state["token"] = "rotated"
        state["status"] = 403
        try:
            with pytest.raises(MCPAuthenticationError) as call_error:
                await view._create_executor("echo")(text="hi")
        finally:
            await view.close()
    assert recorder.events
    for error in (connect_error.value, call_error.value):
        _assert_no_sentinel(error, recorder.events, _log_text(caplog))
    assert call_error.value.challenge is not None
    assert call_error.value.challenge.error == "insufficient_scope"
