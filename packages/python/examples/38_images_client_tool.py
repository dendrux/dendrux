"""Images from a client tool: screenshot, pause, submit, resume.

The model asks the user's app for a screenshot (a ``target="client"`` tool).
Dendrux pauses the run. The app captures the screen and sends it back with
``agent.submit_tool_results(..., ToolResult(..., images=(Image.from_bytes(png),)))``.
The submit happens on a *second* Agent over the same database, the way a web
worker that never saw the run would do it, and the model then reads the image.

By default the model is a scripted mock, so no API key is needed; the mock
cannot read pixels, it proves the pause, the stored blob and the resume. Pass
--provider to use a real vision model:

    python examples/38_images_client_tool.py
    python examples/38_images_client_tool.py --provider anthropic
    python examples/38_images_client_tool.py --provider openai
    python examples/38_images_client_tool.py --provider openai-responses
    python examples/38_images_client_tool.py --provider openrouter

Keys come from the environment (ANTHROPIC_API_KEY, OPENAI_API_KEY,
OPENROUTER_API_KEY).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import struct
import tempfile
import zlib
from pathlib import Path
from typing import TYPE_CHECKING

from dendrux import Agent, Image, tool
from dendrux.llm.mock import MockLLM
from dendrux.store import RunStore
from dendrux.types import LLMResponse, RunStatus, ToolCall, ToolResult

if TYPE_CHECKING:
    from dendrux.llm.base import LLMProvider

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "images"


def solid_png(rgb: tuple[int, int, int], size: int = 64) -> bytes:
    """A small solid-colour PNG, used when the repo fixtures are not around."""

    def chunk(kind: bytes, body: bytes) -> bytes:
        crc = zlib.crc32(kind + body) & 0xFFFFFFFF
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", crc)

    row = b"\x00" + bytes(rgb) * size
    header = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(row * size))
        + chunk(b"IEND", b"")
    )


def screenshot_bytes() -> bytes:
    """Stand-in for the app's real screen capture."""
    path = FIXTURES / "zebra42.png"
    return path.read_bytes() if path.exists() else solid_png((200, 40, 40))


@tool(target="client")
async def take_screenshot() -> str:
    """Capture the user's screen. Runs in the client app, not on the server."""
    return ""


def make_provider(name: str, *, second_half: bool = False) -> LLMProvider:
    if name == "anthropic":
        from dendrux.llm.anthropic import AnthropicProvider

        return AnthropicProvider(model="claude-haiku-4-5")
    if name == "openai":
        from dendrux.llm.openai import OpenAIProvider

        return OpenAIProvider(model="gpt-4o-mini")
    if name == "openai-responses":
        from dendrux.llm.openai_responses import OpenAIResponsesProvider

        return OpenAIResponsesProvider(model="gpt-4o-mini")
    if name == "openrouter":
        from dendrux.llm.openrouter import OpenRouterProvider

        return OpenRouterProvider(model="anthropic/claude-haiku-4.5")
    if second_half:
        return MockLLM([LLMResponse(text="The code word in the screenshot is ZEBRA42.")])
    return MockLLM(
        [LLMResponse(tool_calls=[ToolCall(name="take_screenshot", params={})])],
    )


def build_agent(provider: LLMProvider, url: str) -> Agent:
    return Agent(
        name="ScreenAgent",
        provider=provider,
        prompt="Use take_screenshot when you need to see the user's screen, then answer.",
        tools=[take_screenshot],
        database_url=url,
    )


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--provider",
        default="mock",
        choices=["mock", "anthropic", "openai", "openai-responses", "openrouter"],
    )
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        url = f"sqlite+aiosqlite:///{Path(tmp) / 'screens.db'}"

        # Process 1: start the run. It pauses when the model asks for a screenshot.
        async with build_agent(make_provider(args.provider), url) as agent:
            paused = await agent.run("Take a screenshot and tell me the code word shown in it.")
            assert paused.status is RunStatus.WAITING_CLIENT_TOOL, paused.status

        # Your HTTP layer would get the pending call from the run.paused SSE frame;
        # RunStore.get_pauses is the public way to read it back.
        async with RunStore.from_database_url(url) as store:
            call = (await store.get_pauses(paused.run_id))[-1].pending_tool_calls[0]
        print("paused, waiting for:", call["name"])

        # The app captures the screen and sends it back. The text payload is
        # JSON, like every tool result; the images ride beside it.
        result_for_model = ToolResult(
            name=call["name"],
            call_id=call["id"],
            payload=json.dumps("Captured."),
            images=(Image.from_bytes(screenshot_bytes()),),
        )

        # Process 2: a different Agent (new provider, new engine) submits and resumes.
        async with build_agent(make_provider(args.provider, second_half=True), url) as agent:
            done = await agent.submit_tool_results(paused.run_id, [result_for_model])
            print("status:", done.status.value)
            print("answer:", done.answer)
            assert done.status is RunStatus.SUCCESS

            # The blob is tenant-scoped and readable only through the run that references it.
            async with RunStore.from_database_url(url) as store:
                events = await store.get_events(paused.run_id, event_types=["run.resumed"])
                print("run.resumed:", events[0].data)
                traces = await store.get_traces(paused.run_id)
                tool_trace = next(t for t in traces if t.role == "tool")
                image_part = next(
                    p for p in (tool_trace.meta or {})["dendrux.parts"] if p["type"] == "image"
                )
                data = await store.read_image(paused.run_id, image_part["sha256"])
                print(f"read back {len(data)} bytes of {image_part['media_type']} via the run")


if __name__ == "__main__":
    asyncio.run(main())
