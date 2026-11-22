from __future__ import annotations

import torch
import torch.nn as nn


class ConvProbe(nn.Module):
    """Memory-efficient 3D conv probe for attention feature maps.

    Expects a flat input of shape [B, n_blocks_heads * T * h * w] and internally
    reshapes to [B, n_blocks_heads, T, h, w] before processing.

    Pipeline:
        1. Reshape flat input → [B, C, T, h, w]
        2. 1×1×1 bottleneck  — projects C (e.g. 1152) down to bottleneck_ch (e.g. 128)
        3. Two spatial-only conv stages with stride (1,2,2) — 32×32 → 16×16 → 8×8
        4. AdaptiveAvgPool3d(1)  — collapses [T, h, w] to a single vector per channel
        5. MLP head → logit
    """

    def __init__(
        self,
        n_blocks_heads: int,
        latent_thw: tuple[int, int, int],  # (T, h, w) of the stored latent
        bottleneck_ch: int = 128,
        conv_kernel_size: int = 3,
        mlp_hidden: int = 128,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.n_blocks_heads = n_blocks_heads
        self.latent_thw = latent_thw  # stored for checkpoint / repr

        # 1x1x1 channel bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv3d(n_blocks_heads, bottleneck_ch, kernel_size=1, bias=False),
            nn.BatchNorm3d(bottleneck_ch),
            nn.ReLU(inplace=True),
        )

        # Spatial-only downsampling: (1,k,k) kernel, stride (1,2,2) — temporal dim unchanged
        pad = conv_kernel_size // 2
        def _spatial_stage(ch: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv3d(ch, ch, kernel_size=(1, conv_kernel_size, conv_kernel_size), stride=(1, 2, 2), padding=(0, pad, pad), bias=False),
                nn.BatchNorm3d(ch),
                nn.ReLU(inplace=True),
            )

        self.spatial1 = _spatial_stage(bottleneck_ch)  # h,w: 32→16
        self.spatial2 = _spatial_stage(bottleneck_ch)  # h,w: 16→8

        # Global pool + MLP head
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(bottleneck_ch, mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T, h, w = self.latent_thw
        B = x.shape[0]

        # [B, D] → [B, n_blocks_heads, T, h, w]
        x = x.view(B, self.n_blocks_heads, T, h, w)

        # [B, 1152, 3, 32, 32] → [B, 128, 3, 32, 32]
        x = self.bottleneck(x)

        # [B, 128, 3, 32, 32] → [B, 128, 3, 16, 16]
        x = self.spatial1(x)

        # [B, 128, 3, 16, 16] → [B, 128, 3, 8, 8]
        x = self.spatial2(x)

        # [B, 128, 3, 8, 8] → [B, 128, 1, 1, 1] → [B, 128] → [B, 1]
        x = self.pool(x)
        return self.mlp(x)  # logit [B, 1]
