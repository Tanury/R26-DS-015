from __future__ import annotations

import torch
from torch import nn

from src.models.eegnet import EEGNetEncoder
from src.models.projection_head import ProjectionHead


class EEGEncoderModel(nn.Module):
    def __init__(self, n_channels: int, n_samples: int, config: dict):
        super().__init__()
        model_cfg = config.get("model", {})
        self.encoder = EEGNetEncoder(
            n_channels=n_channels,
            n_samples=n_samples,
            f1=int(model_cfg.get("f1", 8)),
            d=int(model_cfg.get("d", 2)),
            f2=int(model_cfg.get("f2", 16)),
            temporal_kernel=int(model_cfg.get("temporal_kernel", 64)),
                dropout=float(model_cfg.get("dropout", 0.5)),
        )
        embedding_dim = int(model_cfg.get("embedding_dim", 256))
        self.projection = ProjectionHead(
            self.encoder.out_features,
            embedding_dim=embedding_dim,
        )
        self.classifier = nn.Linear(embedding_dim, 2)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.encoder(x)
        z_eeg = self.projection(features)
        logits = self.classifier(z_eeg)
        probabilities = torch.softmax(logits, dim=1)
        return {
            "logits": logits,
            "probabilities": probabilities,
            "ad_probability": probabilities[:, 1],
            "z_eeg": z_eeg,
        }
