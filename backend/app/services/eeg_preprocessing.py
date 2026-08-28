"""EEG preprocessing — the notebook's Stages 1-4, as a service.

Mirrors `R26_DS_015_neuro_risk_eeg_encoder.ipynb` exactly, because a mismatch
between training and serving preprocessing is silent and fatal: the tensors would
still be the right shape.

Stage 1  5th-order Butterworth 0.5-40 Hz, applied BEFORE resampling to 256 Hz so
         content above the new Nyquist cannot alias into the analysis band.
Stage 2  FastICA with the three documented rejection criteria.
Stage 3  4 s epochs, 50% overlap, 150 uV peak-to-peak rejection on the RAW voltage
         scale (before z-scoring, where a uV threshold would be meaningless).
Stage 4  Hybrid STFT + Morlet CWT band-power tensor.

MNE is imported lazily so that importing this module on a deployment without it
does not break Tier 1.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np

from app.core.exceptions import EegIngestError, EegQualityError


BIOSEMI_128 = [f"{bank}{i}" for bank in "ABCD" for i in range(1, 33)]
FRONTAL_CHANNELS = [c for c in BIOSEMI_128 if c.startswith(("C", "D"))]
DROP_PREFIXES = ("EXG", "Status", "GSR", "Erg", "Resp", "Plet", "Temp")
REGION_OF_BANK = {
    "A": "posterior", "B": "right_lateral", "C": "left_lateral", "D": "frontal_central",
}


def _mne():
    try:
        import mne

        mne.set_log_level("ERROR")
        return mne
    except ImportError as exc:
        raise EegIngestError(
            "MNE is required to process EEG recordings but is not installed."
        ) from exc


def load_recording(set_path: Path) -> tuple[Any, str]:
    """Read an EEGLAB .set as raw, or reassemble it if it stores epochs."""
    mne = _mne()
    try:
        return mne.io.read_raw_eeglab(set_path, preload=True, verbose=False), "continuous"
    except FileNotFoundError as exc:
        raise EegIngestError(
            "The .set header is present but its .fdt data file is missing. "
            "Upload both files together."
        ) from exc
    except TypeError:
        pass  # file holds epochs, not a continuous recording

    try:
        epochs = mne.io.read_epochs_eeglab(set_path, verbose=False)
    except FileNotFoundError as exc:
        raise EegIngestError(
            "The .set header is present but its .fdt data file is missing."
        ) from exc
    except Exception as exc:
        raise EegIngestError(f"Unreadable EEGLAB file: {exc}") from exc

    sfreq = float(epochs.info["sfreq"])
    continuous = _taper_join(epochs.get_data().astype("float64"), sfreq)
    info = mne.create_info(list(epochs.ch_names), sfreq, ch_types="eeg")
    return mne.io.RawArray(continuous, info, verbose=False), "pre_epoched"


def _taper_join(segments: np.ndarray, sfreq: float, ramp_ms: float = 40.0) -> np.ndarray:
    """Concatenate pre-epoched segments with a cosine ramp at each join.

    The segments were artifact-rejected independently, so consecutive ones are not
    physiologically continuous. Butt-joining injects a step discontinuity, which is
    broadband energy the filter and the model would both read as signal.
    """
    n_seg, n_ch, n_time = segments.shape
    ramp_len = max(2, min(int(round(sfreq * ramp_ms / 1000.0)), n_time // 4))
    window = np.ones(n_time, dtype="float32")
    ramp = 0.5 * (1 - np.cos(np.linspace(0, np.pi, ramp_len, dtype="float32")))
    window[:ramp_len] = ramp
    window[-ramp_len:] = ramp[::-1]
    return (segments * window[None, None, :]).transpose(1, 0, 2).reshape(n_ch, n_seg * n_time)


def select_channels(raw: Any, channel_order: list[str]) -> Any:
    """Pick and reorder the configured channels; channel k must always mean electrode k."""
    keep = [c for c in raw.ch_names if not c.startswith(DROP_PREFIXES)]
    available = {c.strip().upper(): c for c in keep}
    picks, missing = [], []
    for wanted in channel_order:
        actual = available.get(wanted.upper())
        (picks.append(actual) if actual else missing.append(wanted))
    if missing:
        raise EegIngestError(
            f"{len(missing)} required channel(s) absent, e.g. {missing[:6]}. "
            f"This module expects the BioSemi 128 montage (A1-D32); "
            f"the file provides {len(raw.ch_names)} channels."
        )
    out = raw.copy().pick(picks)
    out.reorder_channels(picks)
    out.rename_channels(dict(zip(picks, channel_order)))
    return out


def apply_fastica(raw: Any, config: dict[str, Any]) -> tuple[Any, list[dict[str, Any]]]:
    """FastICA with the proposal's three criteria; every exclusion is logged with its score."""
    from scipy.signal import welch
    from scipy.stats import kurtosis as scipy_kurtosis

    mne = _mne()
    from mne.preprocessing import ICA

    n_components = int(min(config.get("ica_n_components", 25), len(raw.ch_names) - 1))
    ica = ICA(
        n_components=n_components, method="fastica",
        random_state=int(config.get("ica_random_state", 42)),
        max_iter=int(config.get("ica_max_iter", 200)),
    )
    ica.fit(raw, picks="eeg", verbose=False)

    frontal_picks = [i for i, c in enumerate(raw.ch_names) if c in set(FRONTAL_CHANNELS)]
    frontal = raw.get_data(picks=frontal_picks or None).mean(axis=0)
    sources = ica.get_sources(raw).get_data()
    sfreq = float(raw.info["sfreq"])

    freqs, psd = welch(sources, fs=sfreq, nperseg=min(sources.shape[-1], int(sfreq * 2)), axis=-1)
    in_band = (freqs >= 0.5) & (freqs <= 40.0)
    high = (freqs > 20.0) & (freqs <= 40.0)

    kurt_max = float(config.get("ica_kurtosis_threshold", 5.0))
    frontal_min = float(config.get("ica_frontal_corr_threshold", 0.40))
    hf_min = float(config.get("ica_hf_power_ratio_threshold", 0.55))

    scored = []
    for i, component in enumerate(sources):
        total = float(psd[i][in_band].sum())
        corr = np.corrcoef(component, frontal)[0, 1]
        entry = {
            "component": i,
            "kurtosis": float(scipy_kurtosis(component, fisher=True, bias=False)),
            "frontal_corr": float(abs(corr)) if np.isfinite(corr) else 0.0,
            "hf_power_ratio": float(psd[i][high].sum() / total) if total > 0 else 0.0,
        }
        criteria = []
        if entry["kurtosis"] > kurt_max:
            criteria.append("kurtosis")
        if entry["frontal_corr"] >= frontal_min:
            criteria.append("frontal_bipolar")
        if entry["hf_power_ratio"] >= hf_min:
            criteria.append("high_frequency")
        entry["criteria"] = criteria
        entry["severity"] = (
            max(entry["kurtosis"], 0.0) / kurt_max
            + entry["frontal_corr"] / frontal_min
            + entry["hf_power_ratio"] / hf_min
        )
        scored.append(entry)

    flagged = sorted(
        (e for e in scored if e["criteria"]), key=lambda e: e["severity"], reverse=True
    )[: int(config.get("ica_max_exclude", 6))]
    if not flagged:
        return raw, []

    cleaned = raw.copy()
    ica.exclude = sorted(e["component"] for e in flagged)
    ica.apply(cleaned, verbose=False)
    return cleaned, [
        {k: e[k] for k in ("component", "criteria", "kurtosis", "frontal_corr", "hf_power_ratio")}
        for e in flagged
    ]


def make_epochs(data: np.ndarray, sfreq: float, seconds: float, overlap: float) -> np.ndarray:
    samples = int(round(sfreq * seconds))
    step = max(1, int(round(samples * (1.0 - overlap))))
    if data.shape[1] < samples:
        return np.empty((0, data.shape[0], samples), dtype="float32")
    starts = range(0, data.shape[1] - samples + 1, step)
    return np.stack([data[:, s:s + samples] for s in starts]).astype("float32")


def reject_by_amplitude(epochs: np.ndarray, threshold_uv: float) -> np.ndarray:
    if epochs.shape[0] == 0:
        return epochs
    peak_to_peak = np.ptp(epochs, axis=2).max(axis=1)
    return epochs[peak_to_peak <= threshold_uv * 1e-6]


LEGACY_EPS = 1e-6
STANDARDIZATION_MODES = ("legacy_eps", "zscore_exact")


def standardize(epochs: np.ndarray, mode: str = "legacy_eps") -> np.ndarray:
    """Per-epoch per-channel z-score, in whichever variant the model was trained with.

    `zscore_exact` is the correct one: flat channels handled by an explicit mask,
    every other channel scaled to exactly unit variance.

    `legacy_eps` reproduces `x / (std + 1e-6)`. That is subtly wrong — EEG here is
    in Volts, where a per-channel std runs 1e-6 to 1e-5, so an additive 1e-6 is the
    same order as the signal and shrinks low-amplitude channels far more than
    high-amplitude ones (0.50x at std=1e-6 versus 0.98x at std=5e-5), leaving
    amplitude information the z-score is meant to remove.

    Both exist because **serving must match training, bug included**. A model
    trained under `legacy_eps` and served under `zscore_exact` sees inputs it was
    never fitted on: on a low-amplitude recording that flips a healthy control from
    PD 0.026 to PD 0.848. The mode is therefore read from the model bundle, not
    chosen here, and defaults to `legacy_eps` for bundles that predate the fix.
    Retraining is what switches a deployment to `zscore_exact`.
    """
    if mode not in STANDARDIZATION_MODES:
        raise ValueError(f"Unknown standardization mode {mode!r}")
    if epochs.shape[0] == 0:
        return epochs.astype("float32")

    mean = epochs.mean(axis=-1, keepdims=True)
    std = epochs.std(axis=-1, keepdims=True)
    if mode == "legacy_eps":
        return ((epochs - mean) / (std + LEGACY_EPS)).astype("float32")

    scale = np.where(std > 0, std, 1.0)
    normalized = (epochs - mean) / scale
    return np.where(std > 0, normalized, 0.0).astype("float32")


def signal_grade(clean_ratio: float) -> str:
    return "Good" if clean_ratio >= 0.8 else ("Moderate" if clean_ratio >= 0.5 else "Poor")


def _rejection_diagnosis(
    epochs: np.ndarray,
    channel_order: list[str],
    reject_uv: float,
    min_clean: int,
    n_surviving: int,
) -> tuple[str, dict[str, Any]]:
    """Explain a rejection failure in terms the uploader can act on.

    Amplitude rejection drops an epoch when its *worst* channel exceeds the
    threshold, so a handful of detached electrodes can fail an otherwise usable
    recording. Naming them turns "too noisy" into something fixable.
    """
    if epochs.shape[0] == 0:
        return (
            "The recording is shorter than one 4-second epoch, so there is nothing "
            "to assess.",
            {"reason": "too_short", "epochs_generated": 0},
        )

    per_epoch_uv = np.ptp(epochs, axis=2).max(axis=1) * 1e6
    per_channel_uv = np.ptp(epochs, axis=2).max(axis=0) * 1e6
    noisy = [(channel_order[i], float(per_channel_uv[i]))
             for i in np.argsort(per_channel_uv)[::-1]
             if per_channel_uv[i] > reject_uv]

    median_uv = float(np.median(per_epoch_uv))
    diagnostics: dict[str, Any] = {
        "reason": "amplitude_rejection",
        "epochs_generated": int(epochs.shape[0]),
        "epochs_surviving": int(n_surviving),
        "epochs_required": int(min_clean),
        "threshold_uv": float(reject_uv),
        "median_epoch_peak_to_peak_uv": round(median_uv, 1),
        "max_epoch_peak_to_peak_uv": round(float(per_epoch_uv.max()), 1),
        "channels_over_threshold": len(noisy),
        "total_channels": len(channel_order),
        "worst_channels": [{"channel": name, "peak_to_peak_uv": round(value, 1)}
                           for name, value in noisy[:8]],
    }

    message = (
        f"Only {n_surviving} of {epochs.shape[0]} epochs survived the "
        f"{reject_uv:g} uV peak-to-peak limit (minimum {min_clean} needed). "
        f"Typical epoch amplitude is {median_uv:,.0f} uV, about "
        f"{median_uv / reject_uv:.0f}x the limit."
    )
    if noisy:
        worst = ", ".join(f"{name} ({value:,.0f} uV)" for name, value in noisy[:3])
        message += (
            f" {len(noisy)} of {len(channel_order)} channels exceed it, worst: {worst}. "
            "That pattern usually means a few detached or bridged electrodes rather "
            "than a bad recording overall — repairing or interpolating them upstream "
            "would likely make this subject assessable."
        )
    else:
        message += " The noise is spread across all channels rather than isolated to a few."
    return message, diagnostics


def preprocess(
    set_path: Path,
    config: dict[str, Any],
    progress: Callable[[int, str], None] | None = None,
) -> dict[str, Any]:
    """Run Stages 1-3 and return standardized epochs plus a quality report."""
    def report(pct: int, label: str) -> None:
        if progress:
            progress(pct, label)

    channel_order = list(config.get("channel_order") or BIOSEMI_128)
    target_sfreq = float(config.get("sampling_rate_hz", 256))
    epoch_seconds = float(config.get("epoch_length_seconds", 4.0))
    overlap = float(config.get("epoch_overlap", 0.5))
    reject_uv = float(config.get("amplitude_reject_uv", 150.0))
    min_clean = int(config.get("min_clean_epochs", 20))
    max_epochs = int(config.get("max_epochs", 80))
    # Serving must reproduce the normalisation the model was fitted under; see
    # standardize(). Bundles that predate the fix do not declare it.
    standardization = str(config.get("standardization", "legacy_eps"))

    report(5, "Reading recording")
    raw, source_kind = load_recording(set_path)
    original_sfreq = float(raw.info["sfreq"])
    duration_s = float(raw.n_times / original_sfreq)

    report(15, "Selecting channels")
    raw.pick("eeg", exclude="bads")
    raw = select_channels(raw, channel_order)
    try:
        raw.set_montage("biosemi128", on_missing="ignore", verbose=False)
    except Exception:
        pass

    report(25, "Band-pass filtering")
    filtered = raw.copy().filter(
        l_freq=0.5, h_freq=40.0, method="iir",
        iir_params=dict(order=5, ftype="butter", output="sos"), verbose=False,
    )
    if abs(float(filtered.info["sfreq"]) - target_sfreq) > 1e-6:
        filtered.resample(target_sfreq, verbose=False)

    ica_rejections: list[dict[str, Any]] = []
    if config.get("use_ica", True):
        report(40, "FastICA artifact removal")
        filtered, ica_rejections = apply_fastica(filtered, config)

    report(75, "Epoching and rejection")
    raw_scale = make_epochs(
        filtered.get_data(picks="eeg"), float(filtered.info["sfreq"]), epoch_seconds, overlap
    )
    total = int(raw_scale.shape[0])
    surviving = reject_by_amplitude(raw_scale, reject_uv)
    n_surviving = int(surviving.shape[0])

    if n_surviving < min_clean:
        raise EegQualityError(
            *_rejection_diagnosis(raw_scale, channel_order, reject_uv, min_clean, n_surviving)
        )

    if n_surviving > max_epochs:
        idx = np.linspace(0, n_surviving - 1, max_epochs).round().astype(int)
        surviving = surviving[np.unique(idx)]

    epochs = standardize(surviving, mode=standardization)
    clean_ratio = float(n_surviving / total) if total else 0.0
    report(85, "Preprocessing complete")

    return {
        "epochs": epochs,
        "quality": {
            "epochs_used": int(epochs.shape[0]),
            "total_epochs_generated": total,
            "clean_epoch_ratio": round(clean_ratio, 4),
            "grade": signal_grade(clean_ratio),
            "ica_components_removed": len(ica_rejections),
            "ica_rejections": ica_rejections,
            "channels": int(epochs.shape[1]),
            "sampling_rate_hz": float(filtered.info["sfreq"]),
            "source_kind": source_kind,
            "standardization": standardization,
            "warnings": (
                ["Recording was stored pre-epoched; 1 s segments were reassembled with "
                 "tapered joins and are not physiologically continuous."]
                if source_kind == "pre_epoched" else []
            ),
        },
        "input_metadata": {
            "original_sampling_rate_hz": original_sfreq,
            "duration_seconds": round(duration_s, 1),
        },
    }
