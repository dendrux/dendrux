"""Resolve numeric API defaults statically, without importing the compared package."""

from __future__ import annotations

import ast
import operator
from datetime import timedelta
from typing import Any

import griffe

_ARITHMETIC = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul}


def _render(value: Any, seen: frozenset[str] = frozenset()) -> str:
    if not isinstance(value, griffe.Expr):
        return str(value)
    parts = []
    for token in value.iterate(flat=True):
        text = str(token)
        if isinstance(token, griffe.ExprName):
            path = token.canonical_path
            if path == "datetime":
                text = path
            elif path == "datetime.timedelta":
                text = token.name if isinstance(token.parent, griffe.ExprName) else path
            elif token.name.isupper() and path not in seen:
                try:
                    target = token.resolved
                except (AttributeError, KeyError, griffe.GriffeError):
                    target = None
                if isinstance(target, griffe.Attribute) and target.value is not None:
                    rendered = _render(target.value, seen | {path})
                    resolved = _Numbers().visit(ast.parse(rendered, mode="eval")).body
                    if _numeric_default(resolved):
                        text = f"({ast.unparse(resolved)})"
        parts.append(text)
    return "".join(parts)


class _Numbers(ast.NodeTransformer):
    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:  # noqa: N802
        node = self.generic_visit(node)
        operation = _ARITHMETIC.get(type(node.op))
        if (
            operation is not None
            and isinstance(node.left, ast.Constant)
            and isinstance(node.right, ast.Constant)
            and type(node.left.value) in (int, float)
            and type(node.right.value) in (int, float)
        ):
            return ast.Constant(operation(node.left.value, node.right.value))
        return node

    def visit_Call(self, node: ast.Call) -> ast.AST:  # noqa: N802
        node = self.generic_visit(node)
        if ast.unparse(node.func) != "datetime.timedelta":
            return node
        values = [*node.args, *(kw.value for kw in node.keywords)]
        if not all(isinstance(n, ast.Constant) and type(n.value) in (int, float) for n in values):
            return node
        try:
            duration = timedelta(
                *(ast.literal_eval(arg) for arg in node.args),
                **{
                    kw.arg: ast.literal_eval(kw.value) for kw in node.keywords if kw.arg is not None
                },
            )
        except (OverflowError, TypeError, ValueError):
            return node
        return ast.parse(repr(duration), mode="eval").body


def _numeric_literal(value: Any) -> bool:
    if isinstance(value, tuple):
        return all(_numeric_literal(item) for item in value)
    return type(value) in (int, float, bool)


def _numeric_default(node: ast.AST) -> bool:
    if isinstance(node, ast.Call) and ast.unparse(node.func) == "datetime.timedelta":
        return all(_numeric_default(arg) for arg in node.args) and all(
            kw.arg is not None and _numeric_default(kw.value) for kw in node.keywords
        )
    try:
        return _numeric_literal(ast.literal_eval(node))
    except (ValueError, TypeError):
        return False


def normalize_defaults(obj: Any) -> None:
    """Resolve constant aliases before Griffe compares old and new default expressions."""
    if isinstance(obj, griffe.Alias):
        return
    if isinstance(obj, griffe.Function):
        for param in obj.parameters:
            if param.default is not None:
                try:
                    expr = ast.parse(_render(param.default), mode="eval")
                except (SyntaxError, RecursionError):
                    continue
                resolved = _Numbers().visit(expr).body
                param.default = ast.unparse(resolved)
    for member in obj.members.values():
        normalize_defaults(member)
