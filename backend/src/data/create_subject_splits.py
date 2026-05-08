from __future__ import annotations

import argparse

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.utils.config_loader import load_config, resolve_data_file, resolve_path
from src.utils.file_utils import write_json
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)


def create_subject_splits(config_path: str = "configs/training.yaml") -> dict:
    cfg = load_config(config_path)
    label_file = resolve_data_file(cfg, "label_file", "labels_dir")
    split_file = resolve_data_file(cfg, "split_file", "splits_dir")
    split_file.parent.mkdir(parents=True, exist_ok=True)

    labels = pd.read_csv(label_file)
    labels = labels[labels["label"].isin([0, 1])].copy()
    labels = labels.sort_values("subject_id").reset_index(drop=True)

    n_splits = int(cfg["data"].get("n_splits", 5))
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=cfg["project"]["seed"])
    rows = []
    for fold, (train_idx, val_idx) in enumerate(
        splitter.split(labels, labels["label"], groups=labels["subject_id"])
    ):
        for idx in train_idx:
            row = labels.iloc[idx].to_dict()
            row.update({"fold": fold, "split": "train"})
            rows.append(row)
        for idx in val_idx:
            row = labels.iloc[idx].to_dict()
            row.update({"fold": fold, "split": "val"})
            rows.append(row)

    split_df = pd.DataFrame(rows)
    split_df.to_csv(split_file, index=False)

    fold_summaries = []
    for fold, part in split_df.groupby("fold"):
        summary = {"fold": int(fold)}
        for split_name, split_part in part.groupby("split"):
            summary[split_name] = {
                "subjects": int(split_part["subject_id"].nunique()),
                "hc": int((split_part["label"] == 0).sum()),
                "ad": int((split_part["label"] == 1).sum()),
            }
        fold_summaries.append(summary)

    payload = {
        "split_strategy": "Subject-level split",
        "cross_validation": "StratifiedGroupKFold",
        "group_key": "subject_id",
        "no_epoch_leakage": True,
        "folds": fold_summaries,
    }
    write_json(resolve_path(cfg["paths"]["splits_dir"]) / "split_summary.json", payload)
    LOGGER.info("Wrote subject splits to %s", split_file)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/training.yaml")
    args = parser.parse_args()
    create_subject_splits(args.config)


if __name__ == "__main__":
    main()
