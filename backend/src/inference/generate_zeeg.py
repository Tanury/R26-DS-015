from __future__ import annotations

import numpy as np
import torch


def aggregate_embedding(z_epochs: torch.Tensor) -> tuple[np.ndarray, float]:
    """Aggregate per-epoch z_eeg embeddings into a single subject-level embedding.

    Each epoch's embedding is already L2-normalized by the projection head.
    We average across epochs and re-normalize to produce a unit vector.

    The pre-normalization norm of the mean vector is returned as an
    embedding_consistency score: values near 1.0 indicate all epochs point in
    the same direction (stable, consistent embeddings). Values near 0 indicate
    high variance across epochs.

    Returns:
        (z_eeg, embedding_consistency)
        - z_eeg:                 L2-normalized 256D subject embedding.
        - embedding_consistency: Norm of the mean vector before final normalization
                                 (range 0-1; higher = more consistent epochs).
    """
    z = z_epochs.detach().cpu().numpy()
    mean_z = z.mean(axis=0)
    raw_norm = float(np.linalg.norm(mean_z))
    if raw_norm > 0:
        mean_z = mean_z / raw_norm
    return mean_z.astype(float), min(float(raw_norm), 1.0)
