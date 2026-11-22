"""Enforce the module-independence contract (plan §10c N2).

``datasets``, ``training``, ``inference``, ``evaluation`` are standalone: each may import
only ``common`` and ``backends`` (and itself) — never a sibling top-level module. ``common``
and ``backends`` are the shared sinks and may be imported by anyone.

This is a static scan of ``from will_it_render.X...`` / ``import will_it_render.X...`` in the
package source; it does not execute the modules (so heavy/GPU deps are irrelevant).
"""
from __future__ import annotations

import ast
import pathlib

PKG = pathlib.Path(__file__).resolve().parent.parent / "src" / "will_it_render"
SIBLINGS = {"datasets", "training", "inference", "evaluation"}
ALLOWED = {"common", "backends"}


def _module_of(path: pathlib.Path) -> str:
    return path.relative_to(PKG).parts[0]


def _imported_tops(path: pathlib.Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    tops: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            parts = node.module.split(".")
            if parts[:1] == ["will_it_render"] and len(parts) >= 2:
                tops.add(parts[1])
        elif isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split(".")
                if parts[:1] == ["will_it_render"] and len(parts) >= 2:
                    tops.add(parts[1])
    return tops


# evaluation/baselines/ is the method-comparison *bridge*: each baseline subclasses the
# inference SteeringModel and runs through the shared steering loop (and uses its grad-
# aggregate registry) by design. That intentional evaluation→inference edge is exempt; the
# mirror edge (inference reusing a baseline helper) was removed by hoisting the shared
# control-signal resolver into common. Everything else must obey the sink-only rule.
EXEMPT_PREFIXES = ("evaluation/baselines/",)


def test_no_cross_sibling_imports():
    violations = []
    for path in PKG.rglob("*.py"):
        owner = _module_of(path)
        if owner not in SIBLINGS:
            continue
        rel = path.relative_to(PKG).as_posix()
        if any(rel.startswith(pre) for pre in EXEMPT_PREFIXES):
            continue
        for imported in _imported_tops(path):
            if imported == owner or imported in ALLOWED:
                continue
            violations.append(f"{path.relative_to(PKG)} imports will_it_render.{imported}")
    assert not violations, "cross-module imports violate N2:\n  " + "\n  ".join(sorted(violations))
