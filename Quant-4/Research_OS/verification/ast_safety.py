"""Conservative static checks for common temporal and train/test leakage."""
from __future__ import annotations

import ast
from dataclasses import dataclass


@dataclass(frozen=True)
class SafetyFinding:
    code: str
    line: int
    severity: str
    message: str


class _LeakageVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.findings: list[SafetyFinding] = []

    def add(self, node: ast.AST, code: str, message: str, severity: str = "CRITICAL") -> None:
        self.findings.append(SafetyFinding(code, getattr(node, "lineno", 0), severity, message))

    def visit_Call(self, node: ast.Call) -> None:
        name = self._name(node.func)
        if name.endswith(".bfill") or name == "bfill":
            self.add(node, "PIT_BACKFILL", "backward filling can import future observations")
        if (name.endswith(".shift") or name == "shift") and node.args:
            arg = node.args[0]
            if isinstance(arg, ast.UnaryOp) and isinstance(arg.op, ast.USub):
                self.add(node, "PIT_NEGATIVE_SHIFT", "negative shift reads future rows")
        if name.endswith(".fit") or name in {"fit", "fit_transform"} or name.endswith(".fit_transform"):
            self.add(node, "FIT_SCOPE_REVIEW", "fit boundary must be proven train-only", "HIGH")
        if name.endswith(".merge") or name in {"merge", "join"} or name.endswith(".join"):
            keywords = {item.arg for item in node.keywords}
            if not ({"on", "left_on", "right_on"} & keywords):
                self.add(node, "UNKEYED_JOIN", "join has no explicit temporal/entity key", "HIGH")
        if name.endswith(".concat") or name == "concat":
            self.add(node, "CONCAT_SCOPE_REVIEW", "concatenation may mix train and evaluation sets", "HIGH")
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
            key = node.slice.value.lower()
            if any(token in key for token in ("future", "forward_return", "target", "label")):
                self.add(node, "TARGET_DERIVED_FEATURE", f"review target-like column {node.slice.value!r}", "HIGH")
        self.generic_visit(node)

    @staticmethod
    def _name(node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return f"{_LeakageVisitor._name(node.value)}.{node.attr}".lstrip(".")
        return ""


def inspect_source(source: str) -> tuple[SafetyFinding, ...]:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return (SafetyFinding("SYNTAX_ERROR", exc.lineno or 0, "CRITICAL", str(exc)),)
    visitor = _LeakageVisitor()
    visitor.visit(tree)
    return tuple(visitor.findings)
