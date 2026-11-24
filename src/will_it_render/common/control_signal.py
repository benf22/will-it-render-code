"""Per-frame control-signal resolution, shared by the TempoControl baseline and the
inference-side timeline steerer.

Hoisted to ``common`` so ``inference`` does not import ``evaluation.baselines`` (avoids a
cycle): the timeline steerer and the baseline both build the same ``[T]`` activation signal.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F


def resolve_control_signal(
    raw: str | Sequence[float] | None, n_frames: int, device: torch.device
) -> torch.Tensor:
    """Build the ``[T]`` control signal. ``"all_ones"`` / None → every frame active.

    An explicit list of a different length is resampled to ``n_frames`` with
    ``adaptive_max_pool1d`` (the same resampling the audio path uses), so a signal authored
    for another model's latent frame count still lines up.
    """
    if raw is None or (isinstance(raw, str) and raw.strip().lower() == "all_ones"):
        return torch.ones(n_frames, dtype=torch.float32, device=device)
    if isinstance(raw, str):
        values = [float(x) for x in raw.replace(",", " ").split()]
    else:
        values = [float(x) for x in raw]
    if not values:
        raise ValueError("control_signal is empty; use 'all_ones' or a list of values.")
    sig = torch.tensor(values, dtype=torch.float32, device=device)
    if sig.numel() != n_frames:
        sig = F.adaptive_max_pool1d(sig.view(1, 1, -1), n_frames).view(-1)
    return sig
