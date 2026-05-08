from __future__ import annotations

import pandas as pd

from src.utils.config_loader import load_config, resolve_data_file


def load_demo_subjects(config_path: str = "configs/training.yaml", include_synthetic: bool = False) -> list[str]:
    cfg = load_config(config_path)
    split_file = resolve_data_file(cfg, "split_file", "splits_dir")
    active_fold = int(cfg["data"].get("active_fold", 0))
    if split_file.exists():
        splits = pd.read_csv(split_file)
        val_subjects = (
            splits[
                (splits["fold"].astype(int) == active_fold) &
                (splits["split"].astype(str) == "val")
            ]["subject_id"]
            .astype(str)
            .tolist()
        )
        if not include_synthetic:
            val_subjects = [sid for sid in val_subjects if not sid.startswith("sub-synth")]
        if val_subjects:
            return val_subjects

    label_file = resolve_data_file(cfg, "label_file", "labels_dir")
    if label_file.exists():
        subjects = pd.read_csv(label_file)["subject_id"].astype(str).tolist()
        if not include_synthetic:
            subjects = [sid for sid in subjects if not sid.startswith("sub-synth")]
        if subjects:
            return subjects

    if include_synthetic:
        return ["sub-synth-hc-000", "sub-synth-ad-000"]
    return ["sub-001"]
