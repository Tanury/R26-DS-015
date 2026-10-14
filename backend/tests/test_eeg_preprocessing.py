"""Signal-path unit tests.

These run without PyTorch — MNE and SciPy are enough — and cover the parts whose
failures are silent: standardization that leaves amplitude information behind, and
epoching arithmetic that produces the right shape from the wrong windows.
"""

import numpy as np
import pytest

from app.services.eeg_preprocessing import (  # noqa: F401
    _taper_join,
    make_epochs,
    reject_by_amplitude,
    signal_grade,
    standardize,
)


def test_standardize_gives_unit_variance_at_volt_scale() -> None:
    """Regression: `x / (std + 1e-6)` on Volt-scale EEG under-normalises badly.

    A per-channel std of 1e-6 V would come out at 0.5 rather than 1.0, and one at
    5e-5 V at 0.98 — an amplitude-dependent shrink that leaves exactly the
    information z-scoring is supposed to remove. Recording amplitude varies by
    site, which is already a confound in this cohort.
    """
    rng = np.random.default_rng(0)
    epochs = np.stack([
        rng.normal(0, scale, size=(4, 512))
        for scale in (1e-6, 5e-6, 1e-5, 5e-5)
    ]).astype("float64")

    out = standardize(epochs, mode="zscore_exact")
    per_channel_std = out.std(axis=-1)

    assert np.allclose(per_channel_std, 1.0, atol=1e-3), (
        f"channels not unit-variance: {per_channel_std.ravel()[:6]}"
    )
    assert np.allclose(out.mean(axis=-1), 0.0, atol=1e-5)


def test_standardize_handles_flat_channels_without_dividing_by_zero() -> None:
    epochs = np.zeros((2, 3, 128), dtype="float64")
    epochs[0, 0, :] = np.linspace(-1e-5, 1e-5, 128)   # one live channel
    out = standardize(epochs, mode="zscore_exact")
    assert np.isfinite(out).all()
    assert np.allclose(out[0, 1], 0.0), "a flat channel must standardize to zeros"
    assert abs(float(out[0, 0].std()) - 1.0) < 1e-3


def test_standardize_is_scale_invariant() -> None:
    """The same signal recorded at two gains must standardize identically."""
    rng = np.random.default_rng(7)
    base = rng.normal(0, 1e-6, size=(2, 8, 256))
    assert np.allclose(standardize(base, mode="zscore_exact"),
                       standardize(base * 50.0, mode="zscore_exact"), atol=1e-4)


def test_epoching_window_count_and_overlap() -> None:
    data = np.arange(128 * 2560, dtype="float64").reshape(128, 2560)
    epochs = make_epochs(data, sfreq=256.0, seconds=4.0, overlap=0.5)
    # 1024-sample windows stepping by 512 across 2560 samples -> 4 windows.
    assert epochs.shape == (4, 128, 1024)
    # Consecutive windows must overlap by exactly half.
    assert np.array_equal(epochs[0, :, 512:], epochs[1, :, :512])


def test_epoching_returns_empty_when_recording_is_too_short() -> None:
    short = np.zeros((128, 100), dtype="float64")
    assert make_epochs(short, 256.0, 4.0, 0.5).shape[0] == 0


def test_amplitude_rejection_uses_raw_volt_scale() -> None:
    epochs = np.zeros((3, 4, 256), dtype="float64")
    epochs[0] = 50e-6 * np.sin(np.linspace(0, 6, 256))      # ~100 uV p2p, keep
    epochs[1] = 400e-6 * np.sin(np.linspace(0, 6, 256))     # ~800 uV p2p, drop
    epochs[2] = 10e-6 * np.sin(np.linspace(0, 6, 256))      # ~20 uV p2p, keep
    kept = reject_by_amplitude(epochs, threshold_uv=150.0)
    assert kept.shape[0] == 2


def test_taper_join_suppresses_the_seam_between_segments() -> None:
    """Pre-epoched PD recordings are reassembled; a butt join injects a step."""
    segments = np.ones((3, 2, 128), dtype="float64")
    segments[1] *= -1.0                       # worst case: sign flip at every join
    joined = _taper_join(segments, sfreq=256.0, ramp_ms=40.0)
    assert joined.shape == (2, 384)
    butt = np.concatenate(list(segments), axis=-1)
    assert np.abs(np.diff(joined, axis=-1)).max() < np.abs(np.diff(butt, axis=-1)).max()


@pytest.mark.parametrize(
    ("ratio", "expected"),
    [(0.95, "Good"), (0.80, "Good"), (0.65, "Moderate"), (0.50, "Moderate"), (0.30, "Poor")],
)
def test_signal_grade_thresholds(ratio: float, expected: str) -> None:
    assert signal_grade(ratio) == expected


def test_rejection_diagnosis_names_the_noisy_electrodes() -> None:
    """A rejection must say which channels failed, not just that it failed.

    Amplitude rejection drops an epoch when its *worst* channel exceeds the limit,
    so a couple of detached electrodes can fail an otherwise usable recording.
    Naming them is the difference between "too noisy" and something the uploader
    can fix.
    """
    from app.services.eeg_preprocessing import _rejection_diagnosis

    rng = np.random.default_rng(3)
    channels = [f"A{i}" for i in range(1, 9)]
    epochs = rng.normal(0, 20e-6, size=(30, 8, 1024))     # ~120 uV p2p, fine
    epochs[:, 2, :] *= 40                                  # A3 detached
    epochs[:, 5, :] *= 12                                  # A6 noisy

    message, diagnostics = _rejection_diagnosis(
        epochs, channels, reject_uv=150.0, min_clean=20, n_surviving=0)

    assert "A3" in message, "the worst channel must be named in the message"
    assert diagnostics["reason"] == "amplitude_rejection"
    assert diagnostics["epochs_generated"] == 30
    assert diagnostics["channels_over_threshold"] >= 2
    worst = [entry["channel"] for entry in diagnostics["worst_channels"]]
    assert worst[0] == "A3", f"expected A3 ranked worst, got {worst[:3]}"
    assert "A6" in worst
    assert diagnostics["median_epoch_peak_to_peak_uv"] > 150.0


def test_rejection_diagnosis_handles_a_too_short_recording() -> None:
    from app.services.eeg_preprocessing import _rejection_diagnosis

    message, diagnostics = _rejection_diagnosis(
        np.empty((0, 8, 1024)), [f"A{i}" for i in range(1, 9)], 150.0, 20, 0)
    assert diagnostics["reason"] == "too_short"
    assert "shorter than one" in message
