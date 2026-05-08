from __future__ import annotations

import torch
from torch import nn


def _temporal_same_padding(kernel_size: int) -> nn.ZeroPad2d:
    """Pad only the temporal axis so Conv2d keeps the sample length stable."""
    left = (kernel_size - 1) // 2
    right = kernel_size // 2
    return nn.ZeroPad2d((left, right, 0, 0))


class EEGNetEncoder(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_samples: int,
        f1: int = 8,
        d: int = 2,
        f2: int = 16,
        temporal_kernel: int = 64,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.n_channels = int(n_channels)
        self.n_samples = int(n_samples)
        self.block1 = nn.Sequential(
            _temporal_same_padding(int(temporal_kernel)),
            nn.Conv2d(1, f1, (1, temporal_kernel), padding=0, bias=False),
            nn.BatchNorm2d(f1),
            nn.Conv2d(f1, f1 * d, (n_channels, 1), groups=f1, bias=False),
            nn.BatchNorm2d(f1 * d),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )
        self.block2 = nn.Sequential(
            _temporal_same_padding(16),
            nn.Conv2d(f1 * d, f1 * d, (1, 16), padding=0, groups=f1 * d, bias=False),
            nn.Conv2d(f1 * d, f2, (1, 1), bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_samples)
            self.out_features = int(self.block2(self.block1(dummy)).flatten(1).shape[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected EEG tensor [batch, 1, channels, samples], got {tuple(x.shape)}")
        if x.shape[2] != self.n_channels or x.shape[3] != self.n_samples:
            raise ValueError(
                "EEG tensor shape does not match the trained encoder: "
                f"expected channels={self.n_channels}, samples={self.n_samples}; "
                f"got channels={x.shape[2]}, samples={x.shape[3]}"
            )
        return self.block2(self.block1(x)).flatten(1)


class EEGNetClassifier(nn.Module):
    def __init__(self, encoder: EEGNetEncoder, n_classes: int = 2):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(encoder.out_features, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(x))
