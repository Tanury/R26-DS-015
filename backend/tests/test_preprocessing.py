from __future__ import annotations

import numpy as np
import pytest

from src.preprocessing.artifact_rejection import reject_by_amplitude, signal_quality
from src.preprocessing.epoching import make_fixed_length_epochs
from src.preprocessing.time_frequency import build_stft_features, transform_epochs_for_eegnet
from src.inference.generate_zeeg import aggregate_embedding
from src.utils.seed import stable_hash_int
import torch


def test_epoching_shape():
    data = np.zeros((2, 1000), dtype="float32")
    epochs = make_fixed_length_epochs(data, sfreq=250, epoch_length_seconds=2.0, overlap=0.5)
    # 1000 samples, 500 per epoch, 250 step → starts at 0, 250, 500 → 3 epochs
    assert epochs.shape == (3, 2, 500)


def test_amplitude_rejection():
    data = np.zeros((2, 1000), dtype="float32")
    epochs = make_fixed_length_epochs(data, sfreq=250, epoch_length_seconds=2.0, overlap=0.5)
    # Inject a spike into epoch index 1
    epochs[1, 0, 0] = 300e-6  # 300µV > 150µV threshold
    clean, mask = reject_by_amplitude(epochs, threshold_uv=150)
    assert clean.shape[0] == 2
    assert mask.tolist() == [True, False, True]


def test_amplitude_rejection_handles_no_epochs():
    epochs = np.empty((0, 2, 500), dtype="float32")
    clean, mask = reject_by_amplitude(epochs, threshold_uv=150)
    assert clean.shape == epochs.shape
    assert mask.shape == (0,)


def test_signal_quality_labels():
    assert signal_quality(0.85) == "Good"
    assert signal_quality(0.65) == "Moderate"
    assert signal_quality(0.3) == "Poor"


def test_stable_hash_is_reproducible():
    assert stable_hash_int("sub-001") == stable_hash_int("sub-001")
    assert stable_hash_int("sub-001") != stable_hash_int("sub-002")


def test_transform_epochs_for_eegnet_keeps_raw_tensor_shape():
    epochs = np.random.default_rng(0).normal(size=(3, 2, 500)).astype("float32")
    transformed, summary = transform_epochs_for_eegnet(epochs, {"feature_transform": "raw_eeg_tensor"})
    assert transformed.shape == epochs.shape
    assert summary["model_input"] == "raw_eeg_tensor"


def test_stft_feature_shape():
    epochs = np.random.default_rng(0).normal(size=(2, 3, 500)).astype("float32")
    features = build_stft_features(epochs, sfreq=250, window_seconds=1.0, overlap=0.5)
    assert features.shape[0] == 2
    assert features.shape[1] == 3
    assert features.ndim == 4


def test_rejection_before_standardize_order():
    """Verify rejection threshold applies to raw-scale voltage, not z-scored values.
    After z-scoring, a 300µV spike becomes dimensionless and the µV threshold
    would be meaningless. The pipeline must reject BEFORE standardizing.
    """
    rng = np.random.default_rng(0)
    data = rng.normal(0, 10e-6, size=(4, 1250))  # 4ch, 5 seconds at 250Hz
    data[0, 100] = 500e-6  # large spike in channel 0 at sample 100
    epochs = make_fixed_length_epochs(data, sfreq=250, epoch_length_seconds=2.0, overlap=0.5)
    # Epoch containing the spike should be rejected
    clean, mask = reject_by_amplitude(epochs, threshold_uv=150.0)
    assert clean.shape[0] < epochs.shape[0], "Spiked epoch was not rejected"
    # After rejection, peak-to-peak amplitude of clean epochs should be below threshold
    assert np.ptp(clean, axis=2).max() < 150e-6


def test_aggregate_embedding_returns_unit_vector_and_consistency():
    z = torch.randn(10, 256)
    z = torch.nn.functional.normalize(z, dim=1)  # simulate projection head output
    z_agg, consistency = aggregate_embedding(z)
    assert z_agg.shape == (256,)
    norm = float(np.linalg.norm(z_agg))
    assert abs(norm - 1.0) < 1e-5, f"Aggregated z_eeg is not unit norm: {norm}"
    assert 0.0 <= consistency <= 1.0


def test_aggregate_embedding_consistency_high_when_aligned():
    """When all epoch embeddings point in the same direction, consistency should be ~1."""
    direction = torch.nn.functional.normalize(torch.randn(1, 256), dim=1)
    z = direction.repeat(10, 1)
    _, consistency = aggregate_embedding(z)
    assert consistency > 0.99, f"Expected high consistency for aligned embeddings, got {consistency}"
