from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.utils.config_loader import load_config, resolve_path
from src.utils.file_utils import write_json


def validate_preprocessed_shapes(config_path: str = "configs/training.yaml") -> dict:
    cfg = load_config(config_path)
    epochs_dir = resolve_path(cfg["paths"]["processed_epochs_dir"])
    expected_channels = cfg["preprocessing"].get("selected_channels")
    min_epochs = int(cfg["preprocessing"].get("min_clean_epochs", 1))

    files = sorted(epochs_dir.glob("*_epochs.npz"))
    rows = []
    ok = True
    reference_shape = None
    reference_channels = None

    for path in files:
        with np.load(path, allow_pickle=True) as data:
            epochs = data["epochs"]
            channels = [str(ch) for ch in data.get("channel_order", data.get("channels", []))]

        subject_ok = True
        messages = []
        shape = tuple(int(v) for v in epochs.shape)
        if reference_shape is None:
            reference_shape = shape[1:]
        elif shape[1:] != reference_shape:
            subject_ok = False
            messages.append(f"shape {shape[1:]} does not match reference {reference_shape}")

        if reference_channels is None:
            reference_channels = channels
        elif channels != reference_channels:
            subject_ok = False
            messages.append("channel order does not match reference")

        if expected_channels and channels != list(expected_channels):
            subject_ok = False
            messages.append("channel order does not match preprocessing.selected_channels")

        if shape[0] < min_epochs:
            subject_ok = False
            messages.append(f"clean epochs {shape[0]} is below min_clean_epochs {min_epochs}")

        rows.append({
            "file": str(path),
            "subject_id": path.name.replace("_epochs.npz", ""),
            "shape": list(shape),
            "channels": channels,
            "passed": subject_ok,
            "messages": messages,
        })
        ok = ok and subject_ok

    if not files:
        ok = False

    result = {
        "passed": bool(ok),
        "epochs_dir": str(epochs_dir),
        "files_checked": len(files),
        "reference_epoch_shape": list(reference_shape) if reference_shape else None,
        "reference_channel_order": reference_channels,
        "expected_channel_order": list(expected_channels) if expected_channels else None,
        "min_clean_epochs": min_epochs,
        "subjects": rows,
    }
    write_json(resolve_path(cfg["paths"]["reports_dir"]) / "preprocessed_shape_validation.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/training.yaml")
    args = parser.parse_args()
    result = validate_preprocessed_shapes(args.config)
    print(f"Preprocessed shape validation: {'PASS' if result['passed'] else 'FAIL'}")
    print(f"Files checked: {result['files_checked']}")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
