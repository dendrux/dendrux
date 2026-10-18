"""Images: a user image in, and a tool that returns an image.

Two things on one agent:

1. The user attaches an image to the question
   (``agent.run(["What is the code word?", Image.from_path(...)])``).
2. A server tool returns an image next to its text
   (``return ToolOutput("Rendered.", images=[Image.from_bytes(png)])``), and
   the model reads that image in the next turn.

Images are stored as content-addressed blobs (SQLite here) and the run keeps
small references to them. The script prints what was stored and what the model
was sent.

By default it runs against a scripted mock model, so it needs no API key and
proves the plumbing (the mock cannot read pixels). Pass --provider to run the
same agent against a real vision model:

    python examples/37_images.py
    python examples/37_images.py --provider anthropic
    python examples/37_images.py --provider openai
    python examples/37_images.py --provider openai-responses
    python examples/37_images.py --provider openrouter

Keys come from the environment (ANTHROPIC_API_KEY, OPENAI_API_KEY,
OPENROUTER_API_KEY). Use --image to pass your own file for the user image.
"""

from __future__ import annotations

import argparse
import asyncio
import struct
import tempfile
import zlib
from pathlib import Path
from typing import TYPE_CHECKING

from dendrux import Agent, Image, ToolOutput, tool
from dendrux.llm.mock import MockLLM
from dendrux.store import RunStore
from dendrux.types import LLMResponse, RunStatus, ToolCall

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


def load(name: str, fallback_rgb: tuple[int, int, int]) -> bytes:
    path = FIXTURES / name
    return path.read_bytes() if path.exists() else solid_png(fallback_rgb)


KIWI = load("kiwi77.png", (40, 160, 60))


@tool()
async def render_badge(label: str) -> ToolOutput:
    """Render a badge for the given label and return it as an image.

    Args:
        label: Text the badge is for.
    """
    return ToolOutput(f"Rendered a badge for {label}.", images=[Image.from_bytes(KIWI)])


def make_provider(name: str) -> LLMProvider:
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
    return MockLLM(
        [
            LLMResponse(
                text="Let me render the badge too.",
                tool_calls=[ToolCall(name="render_badge", params={"label": "kiwi"})],
            ),
            LLMResponse(text="The first image says ZEBRA42 and the badge says KIWI77."),
        ]
    )


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else None)
    parser.add_argument(
        "--provider",
        default="mock",
        choices=["mock", "anthropic", "openai", "openai-responses", "openrouter"],
    )
    parser.add_argument("--image", type=Path, help="image file for the user turn")
    args = parser.parse_args()

    provider = make_provider(args.provider)
    user_image = (
        Image.from_path(args.image)
        if args.image
        else Image.from_bytes(load("zebra42.png", (200, 40, 40)))
    )

    with tempfile.TemporaryDirectory() as tmp:
        url = f"sqlite+aiosqlite:///{Path(tmp) / 'images.db'}"
        async with Agent(
            name="ImageDemo",
            provider=provider,
            prompt=(
                "You describe images. Read the code word in the user's image, then call "
                "render_badge once and read the code word in the badge it returns."
            ),
            tools=[render_badge],
            database_url=url,
        ) as agent:
            result = await agent.run(
                [
                    "What is the code word in this image? Then render a badge for 'kiwi' "
                    "and tell me the code word in the badge.",
                    user_image,
                ]
            )
            print("status:", result.status.value)
            print("answer:", result.answer)
            assert result.status is RunStatus.SUCCESS

            # What was stored: two blobs, both referenced from this run.
            async with RunStore.from_database_url(url) as store:
                for event in await store.get_events(result.run_id, event_types=["tool.completed"]):
                    print("tool.completed:", event.data)
                traces = await store.get_traces(result.run_id)
                for trace in traces:
                    parts = (trace.meta or {}).get("dendrux.parts")
                    if parts:
                        kinds = [p["type"] for p in parts]
                        print(f"trace {trace.order_index} ({trace.role}): parts={kinds}")
                        for part in parts:
                            if part["type"] == "image":
                                info = await store.blobs.info(part["sha256"])
                                assert info is not None
                                print(
                                    f"  stored {part['media_type']} {part['width']}x"
                                    f"{part['height']} {info.byte_length} bytes, state={info.state}"
                                )

        if isinstance(provider, MockLLM):
            last = provider.wire_messages[-1]
            seen = [(m.role.value, len(m.images)) for m in last if m.images]
            print("mock model received images in:", seen)


if __name__ == "__main__":
    asyncio.run(main())
