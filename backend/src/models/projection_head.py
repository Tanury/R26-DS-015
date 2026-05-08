from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ProjectionHead(nn.Module):
    def __init__(self, in_features: int, embedding_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, max(embedding_dim, in_features // 2)),
            nn.ELU(),
            nn.Linear(max(embedding_dim, in_features // 2), embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), p=2, dim=1)
