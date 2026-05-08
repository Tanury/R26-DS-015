from __future__ import annotations

from pathlib import Path

import mne


def read_raw_eeg(path: str | Path):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".edf":
        return mne.io.read_raw_edf(path, preload=True, verbose=False)
    if suffix == ".bdf":
        return mne.io.read_raw_bdf(path, preload=True, verbose=False)
    if suffix == ".fif":
        return mne.io.read_raw_fif(path, preload=True, verbose=False)
    if suffix == ".set":
        return mne.io.read_raw_eeglab(path, preload=True, verbose=False)
    if suffix == ".vhdr":
        return mne.io.read_raw_brainvision(path, preload=True, verbose=False)
    raise ValueError(f"Unsupported EEG file format: {path.suffix}")


def find_subject_eeg_file(dataset_root: str | Path, subject_id: str) -> Path:
    root = Path(dataset_root)
    subject_dir = root / subject_id
    candidates = (
        list(subject_dir.rglob("*.edf"))
        + list(subject_dir.rglob("*.bdf"))
        + list(subject_dir.rglob("*.set"))
        + list(subject_dir.rglob("*.vhdr"))
        + list(subject_dir.rglob("*.fif"))
    )
    if not candidates:
        raise FileNotFoundError(f"No supported EEG file found for {subject_id} under {subject_dir}")
    return sorted(candidates)[0]
