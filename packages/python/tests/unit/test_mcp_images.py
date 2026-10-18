"""Track D1: MCP image content as parts (§5.3, §5.16, N-11, P1-8)."""

from __future__ import annotations

import base64
import json
import struct
import zlib
from typing import Any

import pytest
from mcp.types import CallToolResult, ImageContent, TextContent

from dendrux.agent import Agent
from dendrux.loops.react import ReActLoop, _execute_tool
from dendrux.mcp._client import MCPClientAdapter
from dendrux.mcp._errors import MCPToolCallError
from dendrux.mcp._result import normalize_mcp_result
from dendrux.mcp._server import create_mcp_executor
from dendrux.mcp._source import MCPSource, redact_source_value
from dendrux.media import Image, ImagePolicy, ToolOutput
from dendrux.strategies.native import NativeToolCalling
from dendrux.tool import tool
from dendrux.tools._lookups import build_tool_lookups
from dendrux.types import ImagePart, LLMResponse, ToolCall, ToolDef
from tests.unit.test_loop_images import ImageLLM, RecordingRecorder, make_png, make_runtime

PNG = make_png(seed=77)
B64 = base64.b64encode(PNG).decode()
SECRET = "HEADER-SECRET-123"


def padded_png(extra: int) -> bytes:
    """A valid PNG grown by a ``tEXt`` chunk of ``extra`` bytes."""
    body = b"tEXt" + b"k\x00" + b"x" * extra
    chunk = struct.pack(">I", len(body) - 4) + body + struct.pack(">I", zlib.crc32(body))
    return PNG[:33] + chunk + PNG[33:]


def result(*blocks: Any, structured: Any = None, is_error: bool = False) -> CallToolResult:
    return CallToolResult(content=list(blocks), structuredContent=structured, isError=is_error)


def image_block(data: str = B64, mime: str = "image/png") -> ImageContent:
    return ImageContent(data=data, mimeType=mime)


def source() -> MCPSource:
    return MCPSource.http(
        "remote", "https://mcp.example.com/c", headers={"Authorization": f"Bearer {SECRET}"}
    )


def executor_for(res: CallToolResult, *, images: Any = "parts") -> Any:
    class _Client:
        async def call_tool(self, name: str, arguments: dict[str, Any], **kw: Any) -> Any:
            return res

    adapter = MCPClientAdapter(source())
    adapter._client = _Client()  # type: ignore[assignment]
    return create_mcp_executor(
        adapter, namespace="remote", mcp_tool_name="shot", max_result_bytes=100_000, images=images
    )


# ---------------------------------------------------------------------------
# normalize_mcp_result: parts vs inline (golden, §5.16)
# ---------------------------------------------------------------------------


def test_inline_is_v030_shape_and_default() -> None:
    res = result(TextContent(text="Here"), image_block())
    expected = {
        "content": [
            {"type": "text", "text": "Here"},
            {"type": "image", "data": B64, "mimeType": "image/png"},
        ]
    }
    assert normalize_mcp_result(res) == expected
    assert normalize_mcp_result(res, images="inline") == expected


def test_parts_replaces_image_blocks_with_stubs() -> None:
    res = result(TextContent(text="Here"), image_block(), image_block(mime="image/jpeg"))
    out = normalize_mcp_result(res, images="parts")
    assert isinstance(out, ToolOutput)
    assert out.result == {
        "content": [
            {"type": "text", "text": "Here"},
            {"type": "image", "image_index": 0, "mime_type": "image/png"},
            {"type": "image", "image_index": 1, "mime_type": "image/jpeg"},
        ]
    }
    assert [img.source for img in out.images] == ["base64", "base64"]  # type: ignore[union-attr]
    assert B64 not in json.dumps(out.result)


def test_parts_without_images_is_unchanged() -> None:
    res = result(TextContent(text="a"), TextContent(text="b"))
    assert normalize_mcp_result(res, images="parts") == "a\nb"
    structured = result(structured={"x": 1})
    assert normalize_mcp_result(structured, images="parts") == {"x": 1}


def test_structured_content_keeps_images() -> None:
    res = result(image_block(), structured={"rows": 2})
    out = normalize_mcp_result(res, images="parts")
    assert isinstance(out, ToolOutput) and out.result == {"rows": 2} and len(out.images) == 1
    assert normalize_mcp_result(res) == {"rows": 2}  # inline: v0.3.0 (images discarded)


def test_agent_payload_vs_call_tool_golden() -> None:
    """The documented before/after: agent-run payload vs ``call_tool`` return."""
    res = result(TextContent(text="Captured."), image_block())
    agent_payload = json.dumps(normalize_mcp_result(res, images="parts").result)
    call_tool_value = normalize_mcp_result(res, images="inline")
    assert agent_payload == (
        '{"content": [{"type": "text", "text": "Captured."}, '
        '{"type": "image", "image_index": 0, "mime_type": "image/png"}]}'
    )
    assert call_tool_value["content"][1]["data"] == B64


# ---------------------------------------------------------------------------
# Executor redaction (cases a–d) and is_error
# ---------------------------------------------------------------------------


async def test_redaction_text_block_next_to_image() -> None:
    out = await executor_for(result(TextContent(text=f"token {SECRET}"), image_block()))()
    assert isinstance(out, ToolOutput)
    assert SECRET not in json.dumps(out.result)
    assert out.images[0].text == B64  # type: ignore[union-attr] — bytes untouched


async def test_redaction_structured_with_image() -> None:
    out = await executor_for(result(image_block(), structured={"deep": [{"k": SECRET}]}))()
    assert out.result == {"deep": [{"k": "[redacted]"}]}
    assert len(out.images) == 1


async def test_redaction_mime_type_stub() -> None:
    out = await executor_for(result(image_block(mime=f"image/png;{SECRET}")))()
    assert SECRET not in json.dumps(out.result)
    assert SECRET not in repr(out.images[0])
    assert SECRET not in (out.images[0].media_type or "")  # type: ignore[union-attr]


def test_redaction_drops_declared_type_carrying_a_credential() -> None:
    src = MCPSource.http("remote", "https://mcp.example.com/c", headers={"X-Key": "image/png"})
    red = redact_source_value(
        src, ToolOutput("x", images=[Image.from_base64(B64, media_type="image/png")])
    )
    assert red.images[0].media_type is None


def test_redact_source_value_tooloutput_branch() -> None:
    value = ToolOutput({"a": SECRET}, images=[Image.from_bytes(PNG)])
    red = redact_source_value(source(), value)
    assert red.result == {"a": "[redacted]"} and red.images == value.images


async def test_is_error_with_image_never_str_of_tooloutput() -> None:
    res = result(TextContent(text=f"bad {SECRET}"), image_block(), is_error=True)
    with pytest.raises(MCPToolCallError) as err:
        await executor_for(res)()
    message = str(err.value)
    assert "ToolOutput(" not in message and B64[:12] not in message and SECRET not in message
    assert message == "bad [redacted]\n[image omitted]"
    assert err.value.__context__ is None


async def test_call_tool_inline_executor_keeps_base64() -> None:
    out = await executor_for(result(TextContent(text="x"), image_block()), images="inline")()
    assert out["content"][1]["data"] == B64


# ---------------------------------------------------------------------------
# Lenient MCP ingest vs strict local tools (N-11)
# ---------------------------------------------------------------------------


def mcp_lookups(images: list[Image]) -> Any:
    async def run(**params: Any) -> Any:
        return ToolOutput("ok", images=images)

    tdef = ToolDef(
        name="remote__shot",
        description="",
        parameters={"type": "object", "properties": {}},
        meta={"mcp_tool_name": "shot", "source_name": "remote"},
    )
    return build_tool_lookups([], mcp_executors={"remote__shot": run}, mcp_tool_defs=[tdef])


LENIENT_INPUTS = [
    lambda: Image.from_base64("\n".join(B64[i : i + 20] for i in range(0, len(B64), 20))),
    lambda: Image.from_base64(B64, media_type="image/jpeg"),
    lambda: Image.from_base64(base64.b64encode(b"<svg xmlns='x'></svg>").decode()),
    lambda: Image.from_base64(base64.b64encode(PNG[:20] + b"\x00" * 40).decode()),
    lambda: Image.from_base64(base64.b64encode(padded_png(400)).decode()),
]


async def test_mcp_lenient_notes_and_call_succeeds() -> None:
    rt = make_runtime(ImagePolicy(max_image_bytes=300))
    images = [make() for make in LENIENT_INPUTS]
    res = await _execute_tool(
        ToolCall(name="remote__shot", params={}, id="c1"), mcp_lookups(images), image_runtime=rt
    )
    assert res.success and res.payload == '"ok"'
    assert len(res.images) == 2
    assert all(isinstance(i, ImagePart) and i.ref.media_type == "image/png" for i in res.images)
    assert res.notes == (
        "[image from MCP could not be used: unsupported_type]",
        "[image from MCP could not be used: corrupt]",
        "[image from MCP could not be used: image_bytes]",
    )


async def test_local_tool_same_inputs_strict() -> None:
    rt = make_runtime(ImagePolicy(max_image_bytes=300))
    for make in LENIENT_INPUTS[:2]:

        @tool()
        async def local() -> Any:
            """Local."""
            return ToolOutput("ok", images=[make()])  # noqa: B023

        res = await _execute_tool(
            ToolCall(name="local", params={}, id="c"), build_tool_lookups([local]), image_runtime=rt
        )
        assert not res.success


async def test_mcp_notes_only_text_message_and_run_payload_clean() -> None:
    rt = make_runtime()
    bad = Image.from_base64(base64.b64encode(b"not an image").decode())
    lookups = mcp_lookups([bad])
    res = await _execute_tool(
        ToolCall(name="remote__shot", params={}, id="c1"), lookups, image_runtime=rt
    )
    from dendrux.types import tool_result_message

    msg = tool_result_message(res)
    assert msg.parts is None and msg.content.endswith("could not be used: unsupported_type]")


async def test_mcp_image_end_to_end_through_loop() -> None:
    """MCP executor → transform sees ToolOutput → image on the wire, no secret anywhere."""
    exec_ = executor_for(result(TextContent(text=f"shot {SECRET}"), image_block()))
    tdef = ToolDef(
        name="remote__shot",
        description="",
        parameters={"type": "object", "properties": {}},
        meta={"mcp_tool_name": "shot", "source_name": "remote"},
    )
    llm = ImageLLM(
        [
            LLMResponse(
                tool_calls=[ToolCall(name="remote__shot", params={}, provider_tool_call_id="p")]
            ),
            LLMResponse(text="done"),
        ]
    )
    seen: list[Any] = []

    def transform(ctx: Any) -> Any:
        seen.append(type(ctx.result).__name__)
        return ctx.result

    agent = Agent(provider=llm, prompt="p", tool_result_transform=transform)

    async def lookups(loop: Any = None) -> Any:
        return build_tool_lookups([], mcp_executors={"remote__shot": exec_}, mcp_tool_defs=[tdef])

    agent.get_tool_lookups = lookups  # type: ignore[method-assign]
    agent.get_all_tool_defs = lambda loop=None: [tdef]  # type: ignore[method-assign]
    rec = RecordingRecorder()
    await ReActLoop().run(
        agent=agent,
        provider=llm,
        strategy=NativeToolCalling(),
        user_input="go",
        recorder=rec,
        image_runtime=make_runtime(),
    )
    assert seen == ["ToolOutput"]
    wire_tool = llm.calls[1][-1]
    assert wire_tool.images[0].data == PNG
    assert SECRET not in wire_tool.content and B64[:16] not in wire_tool.content
    traces = json.dumps([m.content for m in rec.messages])
    assert SECRET not in traces and B64[:16] not in traces
