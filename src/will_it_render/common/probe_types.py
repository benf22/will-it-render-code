"""Shared probe data types for the will-it-render extraction pipeline.

These definitions are intentionally independent of any model library (Ovi, LTX, etc.)
so that run_extraction.py and aggregator.py can run in any Python environment.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch


class CaptureType(Enum):
    AUDIO_CA = "audio_ca"
    VIDEO_CA = "video_ca"


@dataclass
class BlockCapture:
    attn_map:     torch.Tensor  # [B, H, Lq, Lk], CPU float32, softmax applied
    text_key:     torch.Tensor  # [B, Lk, H, D], CPU float32
    capture_type: CaptureType
