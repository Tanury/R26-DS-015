from __future__ import annotations

import torch


def class_weights_from_labels(labels: list[int]) -> torch.Tensor:
    counts = torch.bincount(torch.tensor(labels, dtype=torch.long), minlength=2).float()
    weights = counts.sum() / (2.0 * counts.clamp_min(1.0))
    return weights
