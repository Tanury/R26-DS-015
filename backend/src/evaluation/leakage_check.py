from __future__ import annotations

import argparse

import pandas as pd

from src.utils.config_loader import load_config, resolve_data_file, resolve_path
from src.utils.file_utils import write_json


def check_no_subject_leakage(split_file: str) -> dict:
    df = pd.read_csv(split_file)
    results = []
    ok = True
    for fold, part in df.groupby("fold"):
        train_subjects = set(part.loc[part["split"] == "train", "subject_id"])
        val_subjects = set(part.loc[part["split"] == "val", "subject_id"])
        overlap = sorted(train_subjects & val_subjects)
        ok = ok and len(overlap) == 0
        results.append({"fold": int(fold), "overlap_subjects": overlap, "passed": len(overlap) == 0})
    return {
        "split_strategy": "Subject-level split",
        "group_key": "subject_id",
        "no_epoch_leakage": bool(ok),
        "fold_results": results,
        "important_note": "Epochs from the same subject are never placed in both training and testing sets.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/training.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    split_path = resolve_data_file(cfg, "split_file", "splits_dir")
    result = check_no_subject_leakage(str(split_path))
    write_json(resolve_path(cfg["paths"]["reports_dir"]) / "leakage_check.json", result)
    if not result["no_epoch_leakage"]:
        raise SystemExit("Subject leakage detected.")


if __name__ == "__main__":
    main()
