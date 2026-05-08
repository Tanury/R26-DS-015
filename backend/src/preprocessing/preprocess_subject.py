from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

from src.data.load_bids import find_subject_eeg_file, read_raw_eeg
from src.preprocessing.artifact_rejection import (
    apply_fastica_if_enabled,
    reject_by_amplitude,
    signal_quality,
)
from src.preprocessing.epoching import make_fixed_length_epochs
from src.preprocessing.filter_eeg import filter_raw
from src.preprocessing.time_frequency import transform_epochs_for_eegnet
from src.utils.config_loader import load_config, resolve_data_file, resolve_path
from src.utils.logger import get_logger
from src.utils.seed import set_seed, stable_hash_int

LOGGER = get_logger(__name__)


def _canonical_channel_name(name: str) -> str:
    return name.strip().replace(".", "").upper()


def _apply_channel_selection(raw, prep: dict, subject_id: str):
    mode = str(prep.get("channel_selection", "auto")).lower()
    if mode not in {"selected", "fixed", "common_19"}:
        return raw, {
            "mode": mode,
            "applied": False,
            "channel_order": list(raw.ch_names),
            "missing_channels": [],
            "note": "Using all available EEG channels after MNE bad-channel exclusion.",
        }

    selected = [str(ch) for ch in prep.get("selected_channels", [])]
    if not selected:
        raise ValueError("preprocessing.selected_channels must be set when channel_selection is selected.")

    aliases = {
        _canonical_channel_name(k): str(v)
        for k, v in prep.get("channel_aliases", {}).items()
    }
    available = {_canonical_channel_name(ch): ch for ch in raw.ch_names}
    picks = []
    missing = []
    resolved_order = []
    for wanted in selected:
        key = _canonical_channel_name(wanted)
        actual = available.get(key)
        if actual is None and key in aliases:
            actual = available.get(_canonical_channel_name(aliases[key]))
        if actual is None:
            missing.append(wanted)
        else:
            picks.append(actual)
            resolved_order.append(wanted)

    if missing:
        raise ValueError(
            f"{subject_id}: missing required EEG channels for fixed order: {missing}. "
            f"Available channels: {raw.ch_names}"
        )

    selected_raw = raw.copy().pick(picks, exclude=())
    selected_raw.reorder_channels(picks)
    selected_raw.rename_channels({actual: wanted for actual, wanted in zip(picks, resolved_order)})
    return selected_raw, {
        "mode": mode,
        "applied": True,
        "channel_order": resolved_order,
        "missing_channels": [],
        "note": "Fixed selected channel order enforced before preprocessing.",
    }


def _plot_raw(raw, out_path: Path, seconds: float) -> None:
    data = raw.get_data(picks="eeg")[: min(8, len(raw.ch_names))]
    sfreq = raw.info["sfreq"]
    n = min(data.shape[1], int(seconds * sfreq))
    t = np.arange(n) / sfreq
    plt.figure(figsize=(10, 5))
    offset = 0.0
    for channel in data[:, :n]:
        plt.plot(t, channel + offset, linewidth=0.7)
        offset += np.nanstd(channel) * 6 + 1e-5
    plt.xlabel("Seconds")
    plt.ylabel("Channels offset")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=140)
    plt.close()


def _synthetic_raw(subject_id: str, label: int, channels: int, samples: int, sfreq: float):
    rng = np.random.default_rng(stable_hash_int(subject_id))
    default_19 = [
        "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
        "F7", "F8", "T3", "T4", "T5", "T6", "Fz", "Cz", "Pz",
    ]
    ch_names = default_19[:channels] if channels <= len(default_19) else [
        *default_19,
        *[f"EEG{i:02d}" for i in range(channels - len(default_19))],
    ]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    t = np.arange(samples) / sfreq
    alpha = np.sin(2 * np.pi * (8.0 if label == 1 else 10.0) * t)
    data = rng.normal(0, 12e-6, size=(channels, samples)) + alpha * (8e-6 if label == 1 else 5e-6)
    return mne.io.RawArray(data, info, verbose=False)


def preprocess_raw(
    raw,
    cfg: dict,
    subject_id: str,
    input_file: str,
    label: int | None = None,
    synthetic: bool = False,
) -> dict:
    paths = cfg["paths"]
    prep = cfg["preprocessing"]
    figures_dir = resolve_path(paths["figures_dir"])
    epochs_dir = resolve_path(paths["processed_epochs_dir"])
    logs_dir = resolve_path(paths["preprocessing_logs_dir"])
    epochs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    raw.pick("eeg", exclude="bads")
    raw, channel_summary = _apply_channel_selection(raw, prep, subject_id)
    number_of_channels = len(raw.ch_names)
    actual_sampling_rate_hz = float(raw.info["sfreq"])
    duration_seconds = float(raw.n_times / raw.info["sfreq"])

    if prep.get("montage"):
        try:
            raw.set_montage(prep["montage"], on_missing="ignore", verbose=False)
        except Exception:
            LOGGER.warning("Could not set montage %s for %s", prep["montage"], subject_id)

    raw_plot = figures_dir / f"{subject_id}_raw_eeg_plot.png"
    _plot_raw(raw, raw_plot, float(prep.get("plot_seconds", 10.0)))

    # Fix: filter BEFORE resampling to avoid aliasing.
    # Resampling from 500→250 Hz without filtering first would alias
    # frequencies 125-250 Hz into the 0-40 Hz signal band.
    filtered = filter_raw(raw, prep["low_freq_hz"], prep["high_freq_hz"])
    target_sampling_rate_hz = float(prep["target_sampling_rate_hz"])
    if abs(float(filtered.info["sfreq"]) - target_sampling_rate_hz) > 1e-6:
        filtered.resample(target_sampling_rate_hz, verbose=False)
    filtered, ica_summary = apply_fastica_if_enabled(filtered, prep, subject_id)

    filtered_plot = figures_dir / f"{subject_id}_filtered_eeg_plot.png"
    _plot_raw(filtered, filtered_plot, float(prep.get("plot_seconds", 10.0)))

    # Epoch on raw-amplitude data, then reject by peak-to-peak amplitude,
    # then standardize. Order matters: rejection must happen before z-scoring
    # because z-scored amplitudes are dimensionless (threshold of 150µV is meaningless
    # after standardization).
    epochs_raw_scale = make_fixed_length_epochs(
        filtered.get_data(picks="eeg"),
        filtered.info["sfreq"],
        float(prep["epoch_length_seconds"]),
        float(prep["epoch_overlap"]),
    )
    total_epochs = int(epochs_raw_scale.shape[0])

    # Reject on raw voltage scale using peak-to-peak amplitude.
    clean_epochs_raw, clean_mask = reject_by_amplitude(
        epochs_raw_scale, float(prep["amplitude_reject_uv"])
    )

    # Standardize/transform only the clean epochs to keep rejection on raw voltage.
    clean_epochs, transform_summary = transform_epochs_for_eegnet(clean_epochs_raw, prep)

    clean_epochs_used = int(clean_epochs.shape[0])
    rejected = int(total_epochs - clean_epochs_used)
    clean_ratio = float(clean_epochs_used / total_epochs) if total_epochs else 0.0
    quality = signal_quality(clean_ratio)

    min_clean = int(prep["min_clean_epochs"])
    meets_min_clean_epochs = clean_epochs_used >= min_clean
    if clean_epochs_used < min_clean:
        input_status = "Poor Signal"
        LOGGER.warning(
            "%s: only %d clean epochs (min required: %d). Marked as Poor Signal.",
            subject_id, clean_epochs_used, min_clean,
        )
    else:
        input_status = "Valid EEG input"

    # Warn clearly when synthetic data is being used
    if synthetic:
        input_status = "SYNTHETIC — no real EEG file found. Results are not clinically valid."
        LOGGER.warning(
            "SYNTHETIC DATA in use for %s — results are NOT clinically valid.", subject_id
        )

    epoch_path = epochs_dir / f"{subject_id}_epochs.npz"
    np.savez_compressed(
        epoch_path,
        epochs=clean_epochs,
        label=-1 if label is None else int(label),
        subject_id=subject_id,
        sfreq=float(filtered.info["sfreq"]),
        channels=np.array(filtered.ch_names),
        channel_order=np.array(filtered.ch_names),
    )

    summary = {
        "input_metadata": {
            "input_file": input_file,
            "file_format": Path(input_file).suffix,
            "input_type": "Raw multichannel resting-state closed-eyes EEG",
            "subject_id": subject_id,
            "number_of_channels": number_of_channels,
            "raw_input_channel_support": "configurable; target support 32 to 128 EEG channels",
            "phase1_selected_channels": number_of_channels,
            "phase1_channel_order": list(filtered.ch_names),
            "supported_channel_range": "Phase 1 uses fixed selected channels; future target support is configurable 32 to 128 EEG channels.",
            "actual_sampling_rate_hz": actual_sampling_rate_hz,
            "duration_seconds": duration_seconds,
            "input_status": input_status,
            "synthetic_data": synthetic,  # Fix: always surfaced in output JSON
        },
        "preprocessing_summary": {
            "pipeline": [
                "Channel selection and montage setup",
                "Band-pass filtering 0.5-40 Hz (applied BEFORE resampling to prevent aliasing)",
                "Resample to target sampling rate",
                "Optional FastICA artifact cleaning when enabled",
                "Artifact/noise rejection by peak-to-peak amplitude threshold (on raw-scale epochs)",
                "Epoch segmentation with 50% overlap",
                "Raw EEG tensor transformation with z-score standardization (applied AFTER rejection)",
                "Model-ready EEG tensor transformation for EEGNet baseline",
            ],
            "epoch_note": "Epochs are used as EEG signal windows. They are not treated as independent subjects.",
            "channel_summary": channel_summary,
            "ica_summary": ica_summary,
            "feature_transform_summary": transform_summary,
            "min_clean_epochs_required": min_clean,
            "meets_min_clean_epochs": meets_min_clean_epochs,
            "total_epochs_generated": total_epochs,
            "clean_epochs_used": clean_epochs_used,
            "rejected_epochs": rejected,
            "rejection_rate": float(rejected / total_epochs) if total_epochs else 1.0,
            "clean_epoch_ratio": clean_ratio,
            "signal_quality": quality,
        },
        "visual_outputs": {
            "raw_eeg_plot": str(raw_plot),
            "filtered_eeg_plot": str(filtered_plot),
        },
        "epoch_file": str(epoch_path),
    }
    with (logs_dir / f"{subject_id}_preprocessing_log.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def preprocess_subject(subject_id: str, config_path: str = "configs/preprocessing.yaml") -> dict:
    cfg = load_config(config_path)
    set_seed(int(cfg["project"].get("seed", 42)))
    if cfg.get("_force_preprocess"):
        _remove_subject_outputs(cfg, subject_id)

    label = None
    label_file = resolve_path(cfg["paths"]["labels_dir"]) / "ad_hc_subjects.csv"
    if label_file.exists():
        labels_df = pd.read_csv(label_file)
        match = labels_df[labels_df["subject_id"] == subject_id]
        if len(match):
            label = int(match.iloc[0]["label"])

    synthetic = False
    try:
        eeg_file = find_subject_eeg_file(resolve_path(cfg["paths"]["dataset_root"]), subject_id)
        raw = read_raw_eeg(eeg_file)
        input_file = str(eeg_file)
    except FileNotFoundError:
        if not cfg.get("data", {}).get("synthetic_fallback", True):
            raise
        synthetic = True
        LOGGER.warning(
            "Using SYNTHETIC fallback for %s — real EEG file not found.", subject_id
        )
        raw = _synthetic_raw(
            subject_id,
            0 if label is None else label,
            int(cfg.get("data", {}).get("synthetic_channels", 19)),
            int(cfg.get("data", {}).get("synthetic_samples", 30000)),
            250.0,
        )
        input_file = f"{subject_id}_synthetic.edf"

    return preprocess_raw(raw, cfg, subject_id, input_file, label, synthetic=synthetic)


def _remove_subject_outputs(cfg: dict, subject_id: str) -> None:
    paths = cfg["paths"]
    candidates = [
        resolve_path(paths["processed_epochs_dir"]) / f"{subject_id}_epochs.npz",
        resolve_path(paths["preprocessing_logs_dir"]) / f"{subject_id}_preprocessing_log.json",
        resolve_path(paths["figures_dir"]) / f"{subject_id}_raw_eeg_plot.png",
        resolve_path(paths["figures_dir"]) / f"{subject_id}_filtered_eeg_plot.png",
    ]
    for path in candidates:
        if path.exists():
            path.unlink()


def _remove_all_preprocessing_outputs(cfg: dict) -> None:
    for directory_key, patterns in {
        "processed_epochs_dir": ["*_epochs.npz"],
        "preprocessing_logs_dir": ["*_preprocessing_log.json"],
        "figures_dir": ["*_raw_eeg_plot.png", "*_filtered_eeg_plot.png"],
    }.items():
        directory = resolve_path(cfg["paths"][directory_key])
        if not directory.exists():
            continue
        for pattern in patterns:
            for path in directory.glob(pattern):
                path.unlink()


def _exclude_poor_signal_subjects(cfg: dict, results: list[dict]) -> None:
    """Remove subjects with too few clean epochs from Phase 1 train/eval labels."""
    label_file = resolve_data_file(cfg, "label_file", "labels_dir")
    if not label_file.exists():
        return

    min_clean = int(cfg["preprocessing"].get("min_clean_epochs", 1))
    poor_rows = []
    for result in results:
        summary = result.get("preprocessing_summary", {})
        clean_epochs = int(summary.get("clean_epochs_used", 0))
        subject_id = str(result.get("input_metadata", {}).get("subject_id", ""))
        if subject_id and clean_epochs < min_clean:
            poor_rows.append(
                {
                    "subject_id": subject_id,
                    "clean_epochs_used": clean_epochs,
                    "min_clean_epochs": min_clean,
                    "signal_quality": summary.get("signal_quality", "Poor"),
                    "exclusion_reason": (
                        "Excluded from Phase 1 training/evaluation because preprocessing "
                        "produced fewer clean epochs than min_clean_epochs."
                    ),
                }
            )

    if not poor_rows:
        return

    labels = pd.read_csv(label_file)
    poor_subjects = {row["subject_id"] for row in poor_rows}
    removed = labels[labels["subject_id"].isin(poor_subjects)].copy()
    kept = labels[~labels["subject_id"].isin(poor_subjects)].copy()
    if len(removed) == 0:
        return

    labels_dir = resolve_path(cfg["paths"]["labels_dir"])
    labels_dir.mkdir(parents=True, exist_ok=True)
    excluded_path = labels_dir / "excluded_poor_signal_subjects.csv"
    poor_df = pd.DataFrame(poor_rows)
    poor_df = poor_df.merge(
        removed,
        on="subject_id",
        how="left",
        suffixes=("", "_label_file"),
    )
    poor_df.to_csv(excluded_path, index=False)
    kept.to_csv(label_file, index=False)

    epochs_dir = resolve_path(cfg["paths"]["processed_epochs_dir"])
    for subject_id in poor_subjects:
        epoch_path = epochs_dir / f"{subject_id}_epochs.npz"
        if epoch_path.exists():
            epoch_path.unlink()

    LOGGER.warning(
        "Excluded %d poor-signal subject(s) from %s: %s. Details: %s",
        len(removed),
        label_file,
        ", ".join(sorted(poor_subjects)),
        excluded_path,
    )


def preprocess_all(config_path: str = "configs/training.yaml", force: bool = False) -> list[dict]:
    cfg = load_config(config_path)
    if force:
        _remove_all_preprocessing_outputs(cfg)
        cfg["_force_preprocess"] = True
    label_file = resolve_data_file(cfg, "label_file", "labels_dir")
    if label_file.exists():
        subjects = pd.read_csv(label_file)["subject_id"].tolist()
    else:
        n = int(cfg["data"].get("synthetic_subjects_per_class", 6))
        subjects = (
            [f"sub-synth-hc-{i:03d}" for i in range(n)] +
            [f"sub-synth-ad-{i:03d}" for i in range(n)]
        )
    if not force:
        results = [preprocess_subject(sid, config_path) for sid in subjects]
        _exclude_poor_signal_subjects(cfg, results)
        return results

    results = []
    for sid in subjects:
        subject_cfg_path = config_path
        # preprocess_subject reloads config, so remove per-subject outputs here too.
        _remove_subject_outputs(cfg, sid)
        results.append(preprocess_subject(sid, subject_cfg_path))
    _exclude_poor_signal_subjects(cfg, results)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/training.yaml")
    parser.add_argument("--subject_id")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.all:
        preprocess_all(args.config, force=args.force)
    else:
        if args.force:
            cfg = load_config(args.config)
            _remove_subject_outputs(cfg, args.subject_id or "sub-001")
        preprocess_subject(args.subject_id or "sub-001", args.config)


if __name__ == "__main__":
    main()
