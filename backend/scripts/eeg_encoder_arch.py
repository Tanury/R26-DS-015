"""Model definitions needed to load a training-run `.pth` checkpoint.

Serving does not need this file — the deployed graph is TorchScript and loads without
any project code. It exists for one job: the fold checkpoints in
`r26_ds015_artifacts/checkpoints/` are plain state dicts, and reading them back
requires the class they came from.

Transcribed from the `NeuroRiskEncoder` cell of
`R26_DS_015_neuro_risk_eeg_encoder.ipynb`. It must stay byte-compatible with that
definition: `build_encoder()` loads with `strict=True` so a drift in layer names fails
loudly rather than silently producing an untrained tensor.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


class SharedProjectionHead(nn.Module):
    """Linear -> BatchNorm -> ReLU -> L2 normalize.

    The collapsed-row guard matters at inference too: an all-zero embedding is the
    fusion contract's reserved signal for "EEG absent", so a degenerate input is
    mapped to the uniform unit vector instead.
    """

    def __init__(self, in_features: int, embedding_dim: int = 256, activation: str = "relu"):
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.linear = nn.Linear(in_features, embedding_dim)
        self.norm = nn.BatchNorm1d(embedding_dim)
        self.activation = {"relu": nn.ReLU(), "identity": nn.Identity(),
                           "elu": nn.ELU()}[activation]

    def forward(self, x):
        h = self.activation(self.norm(self.linear(x)))
        z = F.normalize(h, p=2, dim=1, eps=1e-12)
        collapsed = z.norm(dim=1, keepdim=True) < 1e-6
        if bool(collapsed.any()):
            fallback = torch.full_like(z, 1.0 / (self.embedding_dim ** 0.5))
            z = torch.where(collapsed, fallback, z)
        return z


class TemporalTokenizer(nn.Module):
    """Strided conv over time: [B, F, C, T] -> [B, n_tokens, token_dim]."""

    def __init__(self, in_planes: int, n_channels: int, n_time: int, token_dim: int, patch: int):
        super().__init__()
        self.patch = max(1, min(int(patch), max(1, n_time // 8)))
        self.proj = nn.Conv1d(in_planes * n_channels, token_dim,
                              kernel_size=self.patch, stride=self.patch)
        self.n_tokens = max(1, n_time // self.patch)
        self.token_dim = token_dim

    def forward(self, x):
        b, f, c, t = x.shape
        return self.proj(x.reshape(b, f * c, t)).transpose(1, 2)


class EEGTransformer(nn.Module):
    """Lightweight patch transformer with a CLS token — the architecture that shipped."""

    def __init__(self, in_planes, n_channels, n_time, cfg, dropout):
        super().__init__()
        d_model = int(cfg["d_model"])
        self.tokenizer = TemporalTokenizer(in_planes, n_channels, n_time, d_model, int(cfg["patch"]))
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos = nn.Parameter(torch.zeros(1, self.tokenizer.n_tokens + 1, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=int(cfg["heads"]), dim_feedforward=int(cfg["ff"]),
            dropout=dropout, batch_first=True, norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(cfg["layers"]))
        self.norm = nn.LayerNorm(d_model)
        self.out_features = d_model
        self.attention_weights = None

    def forward(self, x):
        tokens = self.tokenizer(x)
        b, n, _ = tokens.shape
        tokens = torch.cat([self.cls.expand(b, -1, -1), tokens], dim=1)
        tokens = tokens + self.pos[:, : n + 1, :]
        return self.norm(self.encoder(tokens))[:, 0]


class ExtendedEEGNet(nn.Module):
    """Compact CNN over the (channel, time) plane. Kept so ablation checkpoints load."""

    def __init__(self, in_planes, n_channels, n_time, cfg, dropout):
        super().__init__()
        f1, d, f2 = int(cfg["f1"]), int(cfg["d"]), int(cfg["f2"])
        k = int(max(3, min(cfg["temporal_kernel"], max(3, n_time // 2))))
        self.block1 = nn.Sequential(
            nn.ZeroPad2d(((k - 1) // 2, k // 2, 0, 0)),
            nn.Conv2d(in_planes, f1, (1, k), bias=False),
            nn.BatchNorm2d(f1),
            nn.Conv2d(f1, f1 * d, (n_channels, 1), groups=f1, bias=False),
            nn.BatchNorm2d(f1 * d),
            nn.ELU(),
            nn.AdaptiveAvgPool2d((1, max(4, n_time // 4))),
            nn.Dropout(dropout),
        )
        k2 = 16 if n_time >= 64 else 3
        self.block2 = nn.Sequential(
            nn.ZeroPad2d(((k2 - 1) // 2, k2 // 2, 0, 0)),
            nn.Conv2d(f1 * d, f1 * d, (1, k2), groups=f1 * d, bias=False),
            nn.Conv2d(f1 * d, f2, (1, 1), bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AdaptiveAvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )
        self.out_features = f2 * 8
        self.attention_weights = None

    def forward(self, x):
        return self.block2(self.block1(x)).flatten(1)


class BiLSTMAttention(nn.Module):
    """Bidirectional LSTM with additive attention pooling. Ablation only."""

    def __init__(self, in_planes, n_channels, n_time, cfg, dropout):
        super().__init__()
        self.tokenizer = TemporalTokenizer(in_planes, n_channels, n_time,
                                           int(cfg["token_dim"]), int(cfg["patch"]))
        hidden, layers = int(cfg["hidden"]), int(cfg["layers"])
        self.lstm = nn.LSTM(self.tokenizer.token_dim, hidden, num_layers=layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.attend = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Tanh(),
                                    nn.Linear(hidden, 1))
        self.dropout = nn.Dropout(dropout)
        self.out_features = hidden * 2
        self.attention_weights = None

    def forward(self, x):
        seq, _ = self.lstm(self.tokenizer(x))
        weights = torch.softmax(self.attend(seq), dim=1)
        self.attention_weights = weights.detach().squeeze(-1)
        return self.dropout((seq * weights).sum(dim=1))


BACKBONES = {
    "eegnet": (ExtendedEEGNet, "eegnet"),
    "bilstm": (BiLSTMAttention, "bilstm"),
    "transformer": (EEGTransformer, "transformer"),
}


class NeuroRiskEncoder(nn.Module):
    """Backbone -> 256-D L2-normalized z_eeg -> independent risk head + 4-class head."""

    def __init__(self, backbone: str, in_planes: int, n_channels: int, n_time: int,
                 model_cfg: dict[str, Any], risk_conditions: list[str]):
        super().__init__()
        cls, cfg_key = BACKBONES[backbone]
        self.backbone_name = backbone
        self.embedding_dim = int(model_cfg["embedding_dim"])
        self.n_risk = int(model_cfg["n_risk"])
        self.risk_conditions = list(risk_conditions)
        self.input_shape = (in_planes, n_channels, n_time)

        self.backbone = cls(in_planes, n_channels, n_time,
                            model_cfg[cfg_key], float(model_cfg["dropout"]))
        self.head = SharedProjectionHead(self.backbone.out_features, self.embedding_dim,
                                         model_cfg.get("head_activation", "relu"))
        self.risk_head = nn.Linear(self.embedding_dim, self.n_risk)
        self.class_head = nn.Linear(self.embedding_dim, int(model_cfg["n_classes"]))

    def forward(self, x):
        z_eeg = self.head(self.backbone(x))
        risk_logits = self.risk_head(z_eeg)
        logits = self.class_head(z_eeg)
        return {
            "risk_scores": torch.sigmoid(risk_logits),
            "probabilities": torch.softmax(logits, dim=1),
            "z_eeg": z_eeg,
        }


def build_encoder(checkpoint_path, device: str = "cpu") -> tuple[NeuroRiskEncoder, dict]:
    """Rebuild one fold's encoder from its checkpoint, weights loaded strictly."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    planes, channels, times = [int(v) for v in checkpoint["input_shape"]]
    model = NeuroRiskEncoder(
        backbone=str(checkpoint["backbone"]),
        in_planes=planes, n_channels=channels, n_time=times,
        model_cfg=checkpoint["config"]["model"],
        risk_conditions=list(checkpoint["risk_conditions"]),
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval().to(device)
    return model, checkpoint
