from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List

import torch
from safetensors.torch import save_file


_DTYPES: Dict[str, torch.dtype] = {
    "float32": torch.float32, "fp32": torch.float32,
    "float16": torch.float16, "fp16": torch.float16, "half": torch.float16,
    "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
}


def resolve_feature_dtype(name: str | None) -> torch.dtype:
    """Map a config string (e.g. 'float16') to a torch dtype. None/unknown → float32."""
    if name is None:
        return torch.float32
    key = str(name).strip().lower()
    if key not in _DTYPES:
        raise ValueError(f"Unknown feature_dtype {name!r}; choose from {sorted(_DTYPES)}")
    return _DTYPES[key]


def make_output_dir(output_root: str | Path, prompt_idx: int, *, seed: int | None = None) -> Path:
    name = f"prompt_{prompt_idx}"
    if seed is not None:
        name += f"__seed_{seed}"
    path = Path(output_root) / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_features(
    out_dir: Path,
    features: List[torch.Tensor],  # list of flat tensors, one per step
    steps: List[int],
    dtype: torch.dtype = torch.float32,
) -> Path:
    feat_tensor = torch.stack(features).to(dtype)
    steps_tensor = torch.tensor(steps, dtype=torch.long)
    path = out_dir / "features.safetensors"
    save_file({"features": feat_tensor, "steps": steps_tensor}, str(path))
    return path


class StepwiseFeatureSaver:
    """Saves each denoising step's feature tensor immediately to avoid memory buildup.

    Usage:
        saver = StepwiseFeatureSaver(out_dir)
        saver.save_step(step_idx, feature_tensor)  # called per step
        saver.finalize()  # merges per-step files into features.safetensors
    """

    def __init__(self, out_dir: Path, dtype: torch.dtype = torch.float32) -> None:
        self._out_dir = out_dir
        self._dtype = dtype
        self._saved: list[tuple[int, Path]] = []

    def save_step(self, step_idx: int, feature: torch.Tensor) -> None:
        path = self._out_dir / f"step_{step_idx:03d}.safetensors"
        save_file({"feature": feature.to(self._dtype).cpu()}, str(path))
        self._saved.append((step_idx, path))

    def finalize(self) -> Path:
        return self._out_dir


class StepwiseWordFeatureSaver:
    """Saves per-word feature tensors for each denoising step.

    Each step's file contains one tensor per target word under the key
    ``features__{word}``.  ``finalize`` stacks steps and writes the final
    safetensors plus a ``feature_metadata.json`` sidecar.

    Usage:
        saver = StepwiseWordFeatureSaver(out_dir)
        saver.save_step(step_idx, {"dog": tensor, "barking": tensor})
        saver.finalize(token_index_map)
    """

    def __init__(self, out_dir: Path, dtype: torch.dtype = torch.float32) -> None:
        self._out_dir = out_dir
        self._dtype = dtype
        self._saved: list[tuple[int, Path]] = []

    def save_step(self, step_idx: int, word_features: Dict[str, torch.Tensor]) -> None:
        tensors = {f"features__{w}": f.to(self._dtype).cpu() for w, f in word_features.items()}
        path = self._out_dir / f"step_{step_idx:03d}.safetensors"
        save_file(tensors, str(path))
        self._saved.append((step_idx, path))

    def finalize(self, token_index_map: Dict[str, List[int]]) -> Path:
        # Write metadata sidecar so training code knows which words are present
        words = list(token_index_map.keys())
        metadata = {
            "mode": "per_word",
            "words": words,
            "token_indices": {w: token_index_map[w] for w in words},
        }
        meta_path = self._out_dir / "feature_metadata.json"
        meta_path.write_text(json.dumps(metadata, indent=2))
        return self._out_dir
