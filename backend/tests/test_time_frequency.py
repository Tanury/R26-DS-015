from __future__ import annotations

import numpy as np
import pytest

from src.preprocessing.time_frequency import (
    build_cwt_features,
    build_stft_features,
    standardize_epochs,
    transform_epochs_for_eegnet,
)


def _epochs(n: int = 4, ch: int = 8, samples: int = 500) -> np.ndarray:
    return np.random.default_rng(0).normal(size=(n, ch, samples)).astype("float32")


def test_standardize_epochs_zero_mean_per_channel():
    out = standardize_epochs(_epochs())
    assert out.shape == (4, 8, 500)
    assert abs(float(out.mean(axis=-1).mean())) < 1e-4


def test_build_stft_features_shape_and_non_negative():
    out = build_stft_features(_epochs(n=3, ch=4, samples=500), sfreq=250.0)
    assert out.shape == (3, 4, 126, 3)
    assert np.all(out >= 0)


def test_transform_epochs_for_eegnet_keeps_raw_shape():
    epochs = _epochs()
    out, summary = transform_epochs_for_eegnet(epochs, {"feature_transform": "raw_eeg_tensor"})
    assert out.shape == epochs.shape
    assert summary["model_input"] == "raw_eeg_tensor"


def test_cwt_feature_path_is_explicitly_phase2():
    with pytest.raises(NotImplementedError, match="Phase 1 EEGNet uses raw EEG tensors"):
        build_cwt_features(_epochs(), sfreq=250.0, frequencies_hz=[4, 8, 12])
