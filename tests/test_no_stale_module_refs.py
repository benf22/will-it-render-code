"""Fail if tracked source references modules that were intentionally dropped from this
single-node public repo (plan §10c N6): the distributed filesystem-queue / SLURM / sweep
tooling (``infra/``, ``queue_trainer``, ``steering.sweep``, ``launch_vm``/``launch_slurm``,
``sbatch``). These would be dangling references — nothing in this repo provides them.

``torchrun`` is intentionally NOT forbidden: single-node multi-GPU DDP is in scope.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

FORBIDDEN = [
    ("queue_trainer", re.compile(r"\bqueue_trainer\b")),
    ("steering.sweep", re.compile(r"\bsteering\.sweep\b")),
    ("infra package", re.compile(r"\binfra[./]")),
    ("launch_vm", re.compile(r"\blaunch_vm\b")),
    ("launch_slurm", re.compile(r"\blaunch_slurm\b")),
    ("sbatch", re.compile(r"\bsbatch\b")),
]

ALLOWLIST = {"tests/test_no_stale_module_refs.py"}


def _tracked_py_and_md() -> list[Path]:
    out = subprocess.run(["git", "ls-files", "*.py", "*.md"], cwd=REPO,
                         capture_output=True, text=True, check=True)
    return [REPO / r for r in out.stdout.split("\n") if r and r not in ALLOWLIST]


def test_no_dropped_module_refs():
    hits = []
    for p in _tracked_py_and_md():
        try:
            text = p.read_text()
        except (UnicodeDecodeError, FileNotFoundError):
            continue
        for label, pat in FORBIDDEN:
            if pat.search(text):
                hits.append(f"{p.relative_to(REPO)}: {label}")
    assert not hits, "stale references to dropped modules:\n  " + "\n  ".join(sorted(hits))
