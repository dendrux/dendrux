"""Keep backend decisions in stores and enumerate the removal/proxy seams."""

import ast
import re
from pathlib import Path


def test_runtime_has_no_backend_special_cases():
    root = Path(__file__).parents[2] / "src" / "dendrux"
    allowed = {
        ("runtime/state.py", "__getattr__", "SQLAlchemyStateStore"),
        ("runtime/state.py", "claimed_store", "isinstance(store, _ClaimedStateStore)"),
        ("runtime/state.py", "unclaimed_store", "isinstance(store, _ClaimedStateStore)"),
    }
    observed = set()
    forbidden = re.compile(r"dialect\.name|SQLAlchemyStateStore|isinstance\(.*StateStore\)?")
    for path in root.rglob("*.py"):
        relative = path.relative_to(root).as_posix()
        if relative.startswith(("runtime/stores/", "db/")):
            continue
        source = path.read_text()
        functions = [
            node
            for node in ast.walk(ast.parse(source))
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        for number, line in enumerate(source.splitlines(), 1):
            match = forbidden.search(line)
            if match is None:
                continue
            containing = [node for node in functions if node.lineno <= number <= node.end_lineno]
            function = min(containing, key=lambda node: node.end_lineno - node.lineno).name
            key = (relative, function, match.group())
            assert key in allowed, (relative, number, line)
            observed.add(key)
    assert observed == allowed
