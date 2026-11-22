from __future__ import annotations

import torch
import torch.nn as nn


class ProbeMLP(nn.Module):
    def __init__(
        self,
        input_dim: int = 205_200,
        hidden_dims: tuple[int, ...] = (512, 128),
        dropout: float = 0.3,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # logit [B, 1]
