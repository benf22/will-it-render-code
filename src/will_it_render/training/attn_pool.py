"""Learnable attention pooling probe.

Replaces hard averaging over heads/layers with content-conditioned softmax pooling
while keeping spatial structure (T, H, W) intact.

Data shapes through the probe:
  Input:          [B, n_blocks*n_heads, T, h, w]
  After reshape:  [B, n_blocks, n_heads, T, h, w]
  After HeadPool: [B, n_blocks, T, h, w]      (pool_dim=2 collapses n_heads)
  After LayerPool:[B, T, h, w]                 (pool_dim=1 collapses n_blocks)
  After conv:     [B, conv_ch, 1, 1, 1]
  Output:         [B, 1]                       (logit)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class AttentionPool(nn.Module):
    """Learnable attention pooling over one axis of a tensor.

    Usage:
        head_pool  = AttentionPool(n_heads)   # pools dim 2: [B, L, H, T, h, w] → [B, L, T, h, w]
        layer_pool = AttentionPool(n_blocks)  # pools dim 1: [B, L, T, h, w]    → [B, T, h, w]

    Algorithm:
        1. Move pool_dim to position 1                               [B, N, ...]
        2. Mean over all trailing dims → content summary             [B, N]
        3. Linear [N→N] + softmax → attention weights                [B, N]
        4. Broadcast weights × x, sum over dim 1                     [B, ...]
    """

    def __init__(self, n: int) -> None:
        super().__init__()
        # content-conditioned scorer; bias acts as a learned global prior
        self.score = nn.Linear(n, n, bias=True)

    def forward(self, x: torch.Tensor, pool_dim: int = 1) -> torch.Tensor:
        x = x.transpose(1, pool_dim)                               # [B, N, ...]
        N = x.shape[1]
        trailing = tuple(range(2, x.ndim))
        summary = x.mean(dim=trailing)                              # [B, N]
        weights = self.score(summary).softmax(dim=-1)               # [B, N]
        shape = (x.shape[0], N) + (1,) * len(trailing)
        out = (weights.view(shape) * x).sum(dim=1)                  # [B, ...]
        return out


class AttnPoolProbe(nn.Module):
    """Full probe: head-pool → layer-pool → lightweight spatial conv → MLP.

    Activated by ``probe_model: attn_pool`` in probing.yaml.
    Does NOT need avg_channels=True — pooling is learned internally.
    """

    def __init__(
        self,
        n_blocks: int,
        n_heads: int,
        latent_thw: tuple[int, int, int],
        conv_ch: int = 32,
        mlp_hidden: int = 64,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.head_pool = AttentionPool(n_heads)    # collapses H dim
        self.layer_pool = AttentionPool(n_blocks)  # collapses L dim
        # Lightweight 3-D conv on [B, 1, T, h, w]; AdaptiveAvgPool collapses spatial
        self.conv = nn.Sequential(
            nn.Conv3d(1, conv_ch, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.GELU(),
            nn.AdaptiveAvgPool3d(1),                               # [B, conv_ch, 1, 1, 1]
        )
        self.head = nn.Sequential(
            nn.Flatten(),                                           # [B, conv_ch]
            nn.Dropout(dropout),
            nn.Linear(conv_ch, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, 1),                              # [B, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, h, w = x.shape                                    # [B, L*H, T, h, w]
        x = x.view(B, self.n_blocks, self.n_heads, T, h, w)        # [B, L, H, T, h, w]
        x = self.head_pool(x, pool_dim=2)                          # [B, L, T, h, w]
        x = self.layer_pool(x, pool_dim=1)                         # [B, T, h, w]
        x = x.unsqueeze(1)                                         # [B, 1, T, h, w]
        x = self.conv(x)                                           # [B, conv_ch, 1, 1, 1]
        return self.head(x)                                        # [B, 1]
