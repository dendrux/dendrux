"""Expected recovery decisions and the source-derived persisted phase inventory."""

import ast
from pathlib import Path

from dendrux.runtime import persistence

LOOP_PHASES = (
    "model_next",
    "dispatch",
    "awaiting_approval",
    "awaiting_client",
    "awaiting_input",
    "finalizing",
)
EVIDENCE = ("absent", "started", "completed", "failed", "unknown")
# Columns: absent intent, unsafe started intent, safe started intent, recorded outcome.
PHASE_ACTIONS = {
    "invalid": ("resolve", "approval_replay", "replay", "observe"),
    "gated": ("approval", "approval_replay", "replay", "observe"),
    "client": ("wait_client", "wait_client", "wait_client", "wait_client"),
    "dispatchable": ("dispatch", "approval_replay", "replay", "observe"),
    "approved": ("dispatch", "replay", "replay", "observe"),
    "rejected": ("resolve", "resolve", "resolve", "observe"),
}


# Non-phase strings occurring inside expressions at phase positions.
PHASE_POSITION_NOISE = {
    "phase",
    "call_id",
    "pending",
    "target",
    "waiting_approval",
    "waiting_client_tool",
    "waiting_human_input",
    "commentary",
    "final_answer",
}


def _strings(node):
    return {
        child.value
        for child in ast.walk(node)
        if isinstance(child, ast.Constant) and isinstance(child.value, str)
    }


def _phase_position(node):
    if isinstance(node, ast.Name):
        return node.id in {"phase", "set_phase"}
    if isinstance(node, ast.Subscript):
        return isinstance(node.slice, ast.Constant) and node.slice.value == "phase"
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "get"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "phase"
    )


def _phase_expressions(node):
    if isinstance(node, ast.Assign) and any(_phase_position(t) for t in node.targets):
        return [node.value]
    if isinstance(node, ast.AnnAssign) and _phase_position(node.target) and node.value:
        return [node.value]
    if isinstance(node, ast.Dict):
        return [
            v
            for k, v in zip(node.keys, node.values, strict=True)
            if isinstance(k, ast.Constant) and k.value == "phase"
        ]
    if isinstance(node, ast.Compare) and any(
        _phase_position(t) for t in [node.left, *node.comparators]
    ):
        return [node]
    if isinstance(node, ast.keyword) and node.arg == "set_phase":
        return [node.value]
    return []


def source_phase_inventory():
    """Scan every runtime source file, including recovery writers and SQL consumers.

    H1c should replace this literal scan with imports from runtime._loop_state;
    H1d's literal guard then owns detection of new phase literals.
    """
    found = {}
    root = Path(persistence.__file__).resolve().parent
    for path in sorted(root.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            for expression in _phase_expressions(node):
                for value in _strings(expression):
                    if value not in PHASE_POSITION_NOISE:
                        found.setdefault(value, set()).add(
                            f"{path.relative_to(root)}:{node.lineno}"
                        )
    return found
