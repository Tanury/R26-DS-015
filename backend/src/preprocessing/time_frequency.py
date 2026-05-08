from __future__ import annotations

import numpy as np


def transform_epochs_for_eegnet(epochs: np.ndarray, cfg: dict) -> tuple[np.ndarray, dict]:
    """Create the Phase 1 EEGNet input tensor and describe optional feature paths."""
    transform = str(cfg.get("feature_transform", "raw_eeg_tensor")).lower()
    if transform not in {"raw", "raw_eeg", "raw_eeg_tensor", "eegnet_raw"}:
        raise ValueError(
            "Phase 1 EEGNet expects raw EEG tensors. "
            "Use build_stft_features/build_cwt_features for exploratory features, "
            f"not feature_transform={transform!r}."
        )

    metadata = {
        "model_input": "raw_eeg_tensor",
        "standardization": "per-epoch per-channel z-score",
        "stft_available": bool(cfg.get("use_stft", False)),
        "cwt_available": bool(cfg.get("use_cwt", False)),
        "note": "STFT/CWT helpers are available for exploration; EEGNet baseline trains on raw tensors.",
    }
    return standardize_epochs(epochs).astype("float32"), metadata


def standardize_epochs(epochs: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    if epochs.shape[0] == 0:
        return epochs.astype("float32")
    mean = epochs.mean(axis=-1, keepdims=True)
    std = epochs.std(axis=-1, keepdims=True)
    return (epochs - mean) / (std + eps)


def build_stft_features(
    epochs: np.ndarray,
    sfreq: float,
    window_seconds: float = 1.0,
    overlap: float = 0.5,
) -> np.ndarray:
    """Return simple STFT power features with shape [epochs, channels, freq, frames]."""
    if epochs.shape[0] == 0:
        return np.empty((0, epochs.shape[1], 0, 0), dtype="float32")
    window = max(2, int(round(float(window_seconds) * float(sfreq))))
    hop = max(1, int(round(window * (1.0 - float(overlap)))))
    if epochs.shape[-1] < window:
        raise ValueError("STFT window is longer than the epoch length.")

    hann = np.hanning(window).astype(epochs.dtype)
    frames = []
    for start in range(0, epochs.shape[-1] - window + 1, hop):
        segment = epochs[:, :, start:start + window] * hann
        frames.append(np.abs(np.fft.rfft(segment, axis=-1)) ** 2)
    return np.stack(frames, axis=-1).astype("float32")


def build_cwt_features(epochs: np.ndarray, sfreq: float, frequencies_hz: list[float]) -> np.ndarray:
    """Placeholder for Morlet CWT features, kept explicit for Phase 2 extension."""
    raise NotImplementedError(
        "Morlet CWT feature extraction is reserved for the post-baseline extension. "
        "Phase 1 EEGNet uses raw EEG tensors to avoid changing the model input contract."
    )
