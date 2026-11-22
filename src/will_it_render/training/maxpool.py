from __future__ import annotations

import torch
import torch.nn as nn


class MaxPoolProbe(nn.Module):
    """Spatial max-pool probe for per-layer attention features.

    Input: flat [B, n_blocks_heads * T * h * w]
    Pipeline:
        1. Reshape → [B, n_blocks_heads, T, h, w]
        2. Max-pool over (h, w)          → [B, n_blocks_heads, T]
        3. Flatten                        → [B, n_blocks_heads * T]
        4. Linear → GELU → Dropout → Linear → logit [B, 1]
    """

    def __init__(
        self,
        n_blocks_heads: int,
        latent_thw: tuple[int, int, int],
        mlp_hidden: int = 128,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.n_blocks_heads = n_blocks_heads
        self.latent_thw = latent_thw
        T, h, w = latent_thw
        flat_dim = n_blocks_heads * T
        self.head = nn.Sequential(
            nn.Linear(flat_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T, h, w = self.latent_thw
        B = x.shape[0]
        x = x.view(B, self.n_blocks_heads, T, h, w)
        x = x.flatten(3).max(dim=-1).values  # [B, n_blocks_heads, T]
        x = x.flatten(1)                      # [B, n_blocks_heads * T]
        return self.head(x)
