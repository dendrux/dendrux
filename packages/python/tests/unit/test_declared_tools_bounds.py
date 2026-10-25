"""Tool declarations inspect a bounded prefix of code-owned definitions."""

from dendrux import Agent, tool
from dendrux.runtime.runner import _emit_init_events
from tests.unit.test_init_events import RecordingRecorder


async def test_declared_names_are_sorted_and_bounded_before_event_serialization():
    functions = []
    for index in range(1005):

        def local() -> str:
            return "unused"

        local.__name__ = "x" * 65 if index == 0 else f"local_{index:04d}"
        functions.append(tool()(local))
    agent = Agent(prompt="p", tools=functions)
    recorder = RecordingRecorder()
    try:
        await _emit_init_events(agent, recorder, None, "r")
        declarations = [
            data for kind, _, data, _ in recorder.governance_events if kind == "tools.declared"
        ]
        assert len(declarations) == 1
        assert declarations[0] == {"local_tool_names": [f"local_{i:04d}" for i in range(1, 1000)]}
    finally:
        await agent.close()
