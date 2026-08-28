"""Serving must reproduce training, bug for bug.

A serving/training preprocessing mismatch is silent: the tensors keep the right
shape and the model still returns confident numbers, they are just wrong. This
suite pins the one place that actually bit — per-epoch standardization.

The full end-to-end parity check (real .set -> preprocessing -> TorchScript ->
compare against the run's own held-out scores) lives in
`scripts/check_serving_parity.py`, because it needs the dataset and PyTorch. What
runs here is the arithmetic that check depends on.
"""

import numpy as np
import pytest

from app.services.eeg_preprocessing import LEGACY_EPS, standardize


@pytest.fixture
def volt_scale_epochs() -> np.ndarray:
    """Four channels spanning the amplitude range of real EEG, in Volts."""
    rng = np.random.default_rng(0)
    return np.stack([
        rng.normal(0, scale, size=(4, 512))
        for scale in (1e-6, 5e-6, 1e-5, 5e-5)
    ]).astype("float64")


def test_default_mode_is_legacy(volt_scale_epochs) -> None:
    """The deployed model predates the fix, so the safe default is what it was trained on."""
    assert np.allclose(standardize(volt_scale_epochs),
                       standardize(volt_scale_epochs, mode="legacy_eps"))


def test_legacy_mode_reproduces_the_original_arithmetic(volt_scale_epochs) -> None:
    mean = volt_scale_epochs.mean(axis=-1, keepdims=True)
    std = volt_scale_epochs.std(axis=-1, keepdims=True)
    expected = (volt_scale_epochs - mean) / (std + LEGACY_EPS)
    assert np.allclose(standardize(volt_scale_epochs, mode="legacy_eps"), expected, atol=1e-6)


def test_legacy_under_normalises_low_amplitude_channels(volt_scale_epochs) -> None:
    """Documents *why* the fix exists, and pins the size of the distortion.

    A channel at std 1e-6 V comes out at ~0.5 variance while one at 5e-5 V comes
    out at ~0.98 — amplitude information the z-score is meant to remove. Recording
    amplitude varies by site, already a confound in this cohort.
    """
    per_channel = standardize(volt_scale_epochs, mode="legacy_eps").std(axis=-1)
    assert per_channel.min() < 0.6, "expected the smallest channel to be badly shrunk"
    assert per_channel.max() > 0.95
    assert per_channel.max() - per_channel.min() > 0.3


def test_exact_mode_gives_unit_variance_at_every_amplitude(volt_scale_epochs) -> None:
    per_channel = standardize(volt_scale_epochs, mode="zscore_exact").std(axis=-1)
    assert np.allclose(per_channel, 1.0, atol=1e-3)


def test_exact_mode_is_scale_invariant(volt_scale_epochs) -> None:
    assert np.allclose(
        standardize(volt_scale_epochs, mode="zscore_exact"),
        standardize(volt_scale_epochs * 50.0, mode="zscore_exact"),
        atol=1e-4,
    )


def test_modes_diverge_enough_to_change_a_prediction(volt_scale_epochs) -> None:
    """Guards the assumption behind versioning this at all.

    On a real held-out control the two modes moved the PD risk score from 0.026 to
    0.848. If they ever became equivalent, versioning would be dead weight — this
    fails loudly in that case rather than leaving the machinery unexplained.
    """
    legacy = standardize(volt_scale_epochs, mode="legacy_eps")
    exact = standardize(volt_scale_epochs, mode="zscore_exact")
    assert np.abs(legacy - exact).max() > 0.1


def test_flat_channels_are_safe_in_both_modes() -> None:
    epochs = np.zeros((2, 3, 128), dtype="float64")
    epochs[0, 0, :] = np.linspace(-1e-5, 1e-5, 128)
    for mode in ("legacy_eps", "zscore_exact"):
        out = standardize(epochs, mode=mode)
        assert np.isfinite(out).all(), f"{mode} produced non-finite values"


def test_unknown_mode_is_rejected(volt_scale_epochs) -> None:
    with pytest.raises(ValueError):
        standardize(volt_scale_epochs, mode="whatever")


def test_bundle_declaring_an_unknown_mode_fails_loudly(monkeypatch) -> None:
    """A bundle must never be served under a normalisation nobody recognises."""
    pytest.importorskip("torch")
    from app.core.exceptions import EegBundleError
    from app.services import eeg_model_loader as loader

    loader.reset_cache()
    real_load = loader.load_eeg_assets.__wrapped__

    import joblib

    original = joblib.load

    def patched(path, *args, **kwargs):
        bundle = original(path, *args, **kwargs)
        if isinstance(bundle, dict):
            bundle = {**bundle, "standardization": "nonsense"}
        return bundle

    monkeypatch.setattr(joblib, "load", patched)
    with pytest.raises(EegBundleError, match="standardization"):
        real_load()
    loader.reset_cache()
