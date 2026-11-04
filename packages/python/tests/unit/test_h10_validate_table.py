"""``Agent._validate`` precedence pinned row by row, with the full message compared exactly.

Each row builds a minimal ``Agent`` that fails one stage, or two stages at once to
show which wins. Messages are compared with ``==`` so wording and stage order are
both fixed. Pure: no provider call, no database.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from dendrux.agent import Agent
from dendrux.loops.react import ReActLoop
from dendrux.loops.single import SingleCall
from dendrux.tool import tool

SKILLS_DIR = str(Path(__file__).parent.parent / "fixtures" / "skills")
SKILL_NAMES = "['read_skill_resource', 'run_skill_script', 'use_skill']"


@tool()
async def srv(x: int) -> int:
    """Server tool."""
    return x


@tool(target="client")
async def cli(x: int) -> int:
    """Client tool."""
    return x


@tool()
async def use_skill(x: int) -> int:
    """A tool squatting on a reserved skill name."""
    return x


@tool()
async def read_skill_resource(x: int) -> int:
    """A tool squatting on a reserved skill name."""
    return x


@tool()
async def run_skill_script(x: int) -> int:
    """A tool squatting on a reserved skill name."""
    return x


def plain(x: int) -> int:
    return x


def hook(*args: Any, **kwargs: Any) -> None:
    return None


class Out(BaseModel):
    value: int


class NegativeDepth(Agent):
    max_delegation_depth = -1


def mcp() -> list[Any]:
    from dendrux.mcp import MCPServer

    return [MCPServer("fs", command=["echo"])]


def agent(**kwargs: Any) -> Agent:
    kwargs.setdefault("name", "A")
    kwargs.setdefault("prompt", "p")
    return Agent(**kwargs)


S3A = (
    "Agent 'A' uses SingleCall loop but has 1 tool(s): ['srv']. SingleCall agents cannot "
    "have tools — tool execution requires multi-turn iteration. Skills are still supported "
    "on SingleCall (delivered inline into the system prompt). Either remove the tools or "
    "switch to ReActLoop."
)
S3B = (
    "Agent 'A' uses SingleCall loop but has 1 tool_source(s). SingleCall agents cannot "
    "have tool_sources for the same reason as tools — multi-turn execution is required. "
    "Skills (inline-delivered) are still supported."
)
S3C = (
    "Agent 'A' has before_tool but uses SingleCall loop. "
    "before_tool is not supported with SingleCall (no tool calls)."
)
S4 = (
    "Agent 'A' has output_type=Out but does not use SingleCall loop. Structured output is "
    "only supported with SingleCall in this version. Either set loop=SingleCall() or "
    "remove output_type."
)
S5 = "Agent 'A' has a non-tool in its tools list: 'plain'. Decorate it with @tool()."


def reserved(name: str) -> str:
    return (
        f"Agent 'A' has a tool named '{name}' but also has skills configured. "
        f"'{name}' is reserved for skill runtime tools when skills are present."
    )


S9B = "Agent 'A' has deny=['x'] but uses SingleCall loop. deny is not supported with SingleCall."
S10B = (
    "Agent 'A' has require_approval=['x'] but uses SingleCall loop. "
    "require_approval is not supported with SingleCall."
)
S10D = (
    "Agent 'A' require_approval contains 'cli' which has target=client. "
    "require_approval only supports server tools."
)

ROWS: list[tuple[str, type[Agent], Any, str]] = [
    (
        "s1-prompt",
        Agent,
        lambda: {"prompt": ""},
        "Agent 'A' requires a prompt. Set prompt as a class attribute or pass it to the "
        "constructor.",
    ),
    (
        "s2-depth-from-class-attribute",
        NegativeDepth,
        lambda: {},
        "max_delegation_depth must be a non-negative integer or None, got -1",
    ),
    (
        "o1-prompt-before-depth",
        NegativeDepth,
        lambda: {"prompt": ""},
        "Agent 'A' requires a prompt. Set prompt as a class attribute or pass it to the "
        "constructor.",
    ),
    (
        "o1-prompt-before-structure-checks",
        Agent,
        lambda: {"prompt": "", "loop": SingleCall(), "tools": [srv], "max_iterations": 0},
        "Agent 'A' requires a prompt. Set prompt as a class attribute or pass it to the "
        "constructor.",
    ),
    (
        "o1-depth-before-structure-checks",
        NegativeDepth,
        lambda: {"loop": SingleCall(), "tools": [srv], "max_iterations": 0},
        "max_delegation_depth must be a non-negative integer or None, got -1",
    ),
    ("s3a-single-call-with-tools", Agent, lambda: {"loop": SingleCall(), "tools": [srv]}, S3A),
    (
        "s3b-single-call-with-tool-sources",
        Agent,
        lambda: {"loop": SingleCall(), "tool_sources": mcp()},
        S3B,
    ),
    (
        "s3c-single-call-with-before-tool",
        Agent,
        lambda: {"loop": SingleCall(), "before_tool": hook},
        S3C,
    ),
    (
        "o2-s3a-before-s3b",
        Agent,
        lambda: {"loop": SingleCall(), "tools": [srv], "tool_sources": mcp()},
        S3A,
    ),
    (
        "o2-s3a-before-s3c",
        Agent,
        lambda: {"loop": SingleCall(), "tools": [srv], "before_tool": hook},
        S3A,
    ),
    (
        "o2-s3b-before-s3c",
        Agent,
        lambda: {"loop": SingleCall(), "tool_sources": mcp(), "before_tool": hook},
        S3B,
    ),
    (
        "o2-s3a-before-s3b-and-s3c",
        Agent,
        lambda: {
            "loop": SingleCall(),
            "tools": [srv],
            "tool_sources": mcp(),
            "before_tool": hook,
        },
        S3A,
    ),
    ("s4-output-type-default-loop", Agent, lambda: {"output_type": Out}, S4),
    ("s4-output-type-react-loop", Agent, lambda: {"output_type": Out, "loop": ReActLoop()}, S4),
    (
        "o4-s3a-before-s4",
        Agent,
        lambda: {"loop": SingleCall(), "tools": [srv], "output_type": Out},
        S3A,
    ),
    ("s5-non-tool", Agent, lambda: {"tools": [plain]}, S5),
    (
        "o4-s4-before-s5",
        Agent,
        lambda: {"tools": [plain], "output_type": Out},
        S4,
    ),
    (
        "s6-reserved-use-skill",
        Agent,
        lambda: {"tools": [use_skill], "skills_dir": SKILLS_DIR},
        reserved("use_skill"),
    ),
    (
        "s6-reserved-read-skill-resource",
        Agent,
        lambda: {"tools": [read_skill_resource], "skills_dir": SKILLS_DIR},
        reserved("read_skill_resource"),
    ),
    (
        "s6-reserved-run-skill-script",
        Agent,
        lambda: {"tools": [run_skill_script], "skills_dir": SKILLS_DIR},
        reserved("run_skill_script"),
    ),
    (
        "s6-first-reserved-wins",
        Agent,
        lambda: {"tools": [srv, run_skill_script, use_skill], "skills_dir": SKILLS_DIR},
        reserved("run_skill_script"),
    ),
    (
        "o4-s5-before-s6",
        Agent,
        lambda: {"tools": [plain, use_skill], "skills_dir": SKILLS_DIR},
        S5,
    ),
    (
        "o4-s6-before-s7",
        Agent,
        lambda: {"tools": [use_skill], "skills_dir": SKILLS_DIR, "max_iterations": 0},
        reserved("use_skill"),
    ),
    (
        "s7-iterations-zero",
        Agent,
        lambda: {"max_iterations": 0},
        "Agent 'A' max_iterations must be >= 1, got 0.",
    ),
    (
        "s7-iterations-negative",
        Agent,
        lambda: {"max_iterations": -3},
        "Agent 'A' max_iterations must be >= 1, got -3.",
    ),
    (
        "s8-iterations-over-ceiling",
        Agent,
        lambda: {"max_iterations": 201},
        "Agent 'A' max_iterations cannot exceed 200, got 201.",
    ),
    (
        "o5-s5-before-s7",
        Agent,
        lambda: {"tools": [plain], "max_iterations": 0},
        S5,
    ),
    (
        "o5-s3a-before-s8",
        Agent,
        lambda: {"loop": SingleCall(), "tools": [srv], "max_iterations": 500},
        S3A,
    ),
    (
        "o5-s7-before-s9a",
        Agent,
        lambda: {"max_iterations": 0, "deny": ["x"]},
        "Agent 'A' max_iterations must be >= 1, got 0.",
    ),
    (
        "s9a-deny-without-tools",
        Agent,
        lambda: {"deny": ["x"]},
        "Agent 'A' has deny=['x'] but no tools.",
    ),
    (
        "s9a-deny-names-sorted",
        Agent,
        lambda: {"deny": ["b", "a"]},
        "Agent 'A' has deny=['a', 'b'] but no tools.",
    ),
    (
        "s9b-deny-single-call-with-skills-only",
        Agent,
        lambda: {"loop": SingleCall(), "skills_dir": SKILLS_DIR, "deny": ["x"]},
        S9B,
    ),
    (
        "o3-s3a-masks-s9b",
        Agent,
        lambda: {"loop": SingleCall(), "tools": [srv], "deny": ["srv"]},
        S3A,
    ),
    (
        "o3-s3b-masks-s9b",
        Agent,
        lambda: {
            "loop": SingleCall(),
            "tool_sources": mcp(),
            "skills_dir": SKILLS_DIR,
            "deny": ["x"],
        },
        S3B,
    ),
    (
        "o3-s3c-masks-s9b",
        Agent,
        lambda: {
            "loop": SingleCall(),
            "before_tool": hook,
            "skills_dir": SKILLS_DIR,
            "deny": ["x"],
        },
        S3C,
    ),
    (
        "s9c-deny-unknown-with-tools",
        Agent,
        lambda: {"tools": [srv], "deny": ["nope"]},
        "Agent 'A' deny contains unknown tool(s): ['nope']. Available tools: ['srv'].",
    ),
    (
        "s9c-deny-partial-unknown",
        Agent,
        lambda: {"tools": [srv, cli], "deny": ["srv", "b", "a"]},
        "Agent 'A' deny contains unknown tool(s): ['a', 'b']. Available tools: ['cli', 'srv'].",
    ),
    (
        "s9c-deny-unknown-with-skills-lists-skill-tools",
        Agent,
        lambda: {"skills_dir": SKILLS_DIR, "deny": ["x"]},
        f"Agent 'A' deny contains unknown tool(s): ['x']. Available tools: {SKILL_NAMES}.",
    ),
    (
        "s9c-deny-unknown-with-tools-and-skills",
        Agent,
        lambda: {"tools": [srv], "skills_dir": SKILLS_DIR, "deny": ["x"]},
        "Agent 'A' deny contains unknown tool(s): ['x']. Available tools: "
        "['read_skill_resource', 'run_skill_script', 'srv', 'use_skill'].",
    ),
    (
        "s10a-require-approval-without-tools",
        Agent,
        lambda: {"require_approval": ["x"]},
        "Agent 'A' has require_approval=['x'] but no tools.",
    ),
    (
        "s10b-require-approval-single-call-with-skills-only",
        Agent,
        lambda: {"loop": SingleCall(), "skills_dir": SKILLS_DIR, "require_approval": ["x"]},
        S10B,
    ),
    (
        "o3-s3a-masks-s10b",
        Agent,
        lambda: {"loop": SingleCall(), "tools": [srv], "require_approval": ["srv"]},
        S3A,
    ),
    (
        "o3-s3b-masks-s10b",
        Agent,
        lambda: {
            "loop": SingleCall(),
            "tool_sources": mcp(),
            "skills_dir": SKILLS_DIR,
            "require_approval": ["x"],
        },
        S3B,
    ),
    (
        "o3-s3c-masks-s10b",
        Agent,
        lambda: {
            "loop": SingleCall(),
            "before_tool": hook,
            "skills_dir": SKILLS_DIR,
            "require_approval": ["x"],
        },
        S3C,
    ),
    (
        "s10c-require-approval-unknown",
        Agent,
        lambda: {"tools": [srv], "require_approval": ["nope"]},
        "Agent 'A' require_approval contains unknown tool(s): ['nope']. Available tools: ['srv'].",
    ),
    (
        "s10c-require-approval-partial-unknown",
        Agent,
        lambda: {"tools": [srv], "require_approval": ["srv", "b", "a"]},
        "Agent 'A' require_approval contains unknown tool(s): ['a', 'b']. "
        "Available tools: ['srv'].",
    ),
    (
        "s10d-require-approval-client-tool",
        Agent,
        lambda: {"tools": [cli], "require_approval": ["cli"]},
        S10D,
    ),
    (
        "o7-s10c-before-s10d",
        Agent,
        lambda: {"tools": [cli], "require_approval": ["cli", "nope"]},
        "Agent 'A' require_approval contains unknown tool(s): ['nope']. Available tools: ['cli'].",
    ),
    (
        "o8-s10d-runs-even-with-tool-sources",
        Agent,
        lambda: {"tools": [cli], "tool_sources": mcp(), "require_approval": ["cli"]},
        S10D,
    ),
    (
        "s10e-deny-and-require-approval-overlap",
        Agent,
        lambda: {"tools": [srv], "deny": ["srv"], "require_approval": ["srv"]},
        "Agent 'A' has tools in both deny and require_approval: ['srv']. "
        "A tool cannot be both denied and require approval.",
    ),
    (
        "o7-s10d-before-s10e",
        Agent,
        lambda: {"tools": [cli], "deny": ["cli"], "require_approval": ["cli"]},
        S10D,
    ),
    (
        "o6-s9c-hides-s10c",
        Agent,
        lambda: {"tools": [srv], "deny": ["a"], "require_approval": ["b"]},
        "Agent 'A' deny contains unknown tool(s): ['a']. Available tools: ['srv'].",
    ),
    (
        "o6-s9b-hides-s10b",
        Agent,
        lambda: {
            "loop": SingleCall(),
            "skills_dir": SKILLS_DIR,
            "deny": ["x"],
            "require_approval": ["y"],
        },
        S9B,
    ),
    (
        "o6-s9a-hides-s10a",
        Agent,
        lambda: {"deny": ["x"], "require_approval": ["y"]},
        "Agent 'A' has deny=['x'] but no tools.",
    ),
    (
        "o6-s9c-hides-s10e",
        Agent,
        lambda: {"tools": [srv], "deny": ["srv", "z"], "require_approval": ["srv"]},
        "Agent 'A' deny contains unknown tool(s): ['z']. Available tools: ['srv'].",
    ),
]


@pytest.mark.parametrize(
    ("cls", "kwargs", "message"), [row[1:] for row in ROWS], ids=[row[0] for row in ROWS]
)
def test_validate_precedence_table(cls: type[Agent], kwargs: Any, message: str) -> None:
    with pytest.raises(ValueError) as caught:
        cls(**{"name": "A", "prompt": "p", **kwargs()})
    assert type(caught.value) is ValueError
    assert str(caught.value) == message


def test_table_covers_every_stage_and_overlap() -> None:
    ids = {row[0] for row in ROWS}
    assert len(ids) == len(ROWS)
    for stage in ("s1", "s2", "s3a", "s3b", "s3c", "s4", "s5", "s6", "s7", "s8"):
        assert any(i.startswith(stage + "-") for i in ids), stage
    for stage in ("s9a", "s9b", "s9c", "s10a", "s10b", "s10c", "s10d", "s10e"):
        assert any(i.startswith(stage + "-") for i in ids), stage
    for overlap in ("o1", "o2", "o3", "o4", "o5", "o6", "o7", "o8"):
        assert any(i.startswith(overlap + "-") for i in ids), overlap


@pytest.mark.parametrize(
    "kwargs",
    [
        lambda: {"max_iterations": 1},
        lambda: {"max_iterations": 200},
        lambda: {"max_delegation_depth": 0},
        lambda: {"max_delegation_depth": None},
        lambda: {"tools": [srv], "deny": ["srv"]},
        lambda: {"tools": [srv], "require_approval": ["srv"]},
        lambda: {"tools": [srv], "skills_dir": SKILLS_DIR, "deny": ["use_skill"]},
        lambda: {"tool_sources": mcp(), "deny": ["unknown_until_discovery"]},
        lambda: {"tool_sources": mcp(), "require_approval": ["unknown_until_discovery"]},
        lambda: {"loop": SingleCall(), "skills_dir": SKILLS_DIR},
        lambda: {"output_type": Out, "loop": SingleCall()},
    ],
    ids=[
        "iterations-floor",
        "iterations-ceiling",
        "depth-zero",
        "depth-none",
        "deny-known-tool",
        "require-approval-server-tool",
        "deny-skill-tool-name",
        "deny-deferred-to-discovery",
        "require-approval-deferred-to-discovery",
        "single-call-with-skills",
        "single-call-with-output-type",
    ],
)
def test_valid_configurations_are_accepted(kwargs: Any) -> None:
    agent(**kwargs())
