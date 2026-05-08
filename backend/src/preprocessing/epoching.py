from __future__ import annotations

import numpy as np


def make_fixed_length_epochs(data: np.ndarray, sfreq: float, epoch_length_seconds: float, overlap: float) -> np.ndarray:
    samples_per_epoch = int(round(sfreq * epoch_length_seconds))
    step = max(1, int(round(samples_per_epoch * (1.0 - overlap))))
    if data.shape[1] < samples_per_epoch:
        return np.empty((0, data.shape[0], samples_per_epoch), dtype=data.dtype)
    starts = range(0, data.shape[1] - samples_per_epoch + 1, step)
    return np.stack([data[:, start : start + samples_per_epoch] for start in starts])
