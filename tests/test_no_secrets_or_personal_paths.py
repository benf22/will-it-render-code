"""Fail if any personal path, cluster identifier, source-repo name, or token-shaped secret
leaks into tracked files (plan §10b R7 / §8 final audit).

Runs over ``git ls-files`` so it scans exactly what would be published. An allowlist exempts
this scanner's own pattern list and any intentionally-gated fixtures.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# (label, compiled pattern) — forbidden substrings / shapes in published content.
FORBIDDEN = [
    ("cortex path", re.compile(r"/cortex\b")),
    ("schwartz-lab path", re.compile(r"/private/schwartz-lab")),
    ("home-lab path", re.compile(r"/home/lab/fishmab1")),
    ("username fishmab1", re.compile(r"\bfishmab1\b")),
    ("username benfish", re.compile(r"\bbenfish\b")),
    ("personal email", re.compile(r"benf22@")),
    ("source repo name", re.compile(r"\bvideo_probing\b")),
    ("hf token", re.compile(r"hf_[A-Za-z0-9]{20,}")),
    ("wandb key", re.compile(r"\b[0-9a-f]{40}\b")),
]

# Files allowed to contain the pattern strings themselves (scanners / docs describing them).
ALLOWLIST = {
    "tests/test_no_secrets_or_personal_paths.py",
    "tests/README.md",
}


def _tracked_text_files() -> list[Path]:
    out = subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=True)
    files = []
    for rel in out.stdout.split("\n"):
        if not rel or rel in ALLOWLIST:
            continue
        p = REPO / rel
        if p.suffix in {".png", ".jpg", ".jpeg", ".gif", ".mp4", ".wav", ".pt", ".safetensors"}:
            continue
        files.append(p)
    return files


def test_no_forbidden_strings_in_tracked_files():
    hits = []
    for p in _tracked_text_files():
        try:
            text = p.read_text()
        except (UnicodeDecodeError, FileNotFoundError):
            continue
        for label, pat in FORBIDDEN:
            if pat.search(text):
                hits.append(f"{p.relative_to(REPO)}: {label}")
    assert not hits, "forbidden strings in tracked files:\n  " + "\n  ".join(sorted(hits))
