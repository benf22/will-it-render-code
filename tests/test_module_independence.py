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
        mods = []
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            mods.append(node.module)
        elif isinstance(node, ast.Import):
            mods.extend(a.name for a in node.names)
        for mod in mods:
            parts = mod.split(".")
            if parts[:1] == ["will_it_render"] and len(parts) >= 2:
                tops.add(".".join(parts[1:]))   # full path after will_it_render.
    return tops


# The evaluation/baselines/ subtree is the method-comparison *bridge*: each baseline subclasses
# the inference SteeringModel and runs through the shared steering loop by design, and the
# steering machinery dispatches to it (build_baseline_steering_model). So edges BOTH into and
# out of evaluation.baselines are allowed; everything else obeys the sink-only rule (siblings
# may import only common/ and backends/).
BRIDGE_PREFIX = "evaluation/baselines/"
BRIDGE_TARGET = "evaluation.baselines"


def test_no_cross_sibling_imports():
    violations = []
    for path in PKG.rglob("*.py"):
        owner = _module_of(path)
        if owner not in SIBLINGS:
            continue
        if path.relative_to(PKG).as_posix().startswith(BRIDGE_PREFIX):
            continue  # the bridge itself may reach inference
        for imported in _imported_tops(path):
            top = imported.split(".")[0]
            if top == owner or top in ALLOWED:
                continue
            if imported == BRIDGE_TARGET or imported.startswith(BRIDGE_TARGET + "."):
                continue  # importing the baseline dispatcher/methods is the allowed bridge
            violations.append(f"{path.relative_to(PKG)} imports will_it_render.{imported}")
    assert not violations, "cross-module imports violate N2:\n  " + "\n  ".join(sorted(violations))
