from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.utils.config_loader import load_config, resolve_path
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)

LABEL_MAP = {
    "C": 0,
    "HC": 0,
    "CONTROL": 0,
    "HEALTHY CONTROL": 0,
    "A": 1,
    "AD": 1,
    "ALZHEIMER": 1,
    "ALZHEIMER'S DISEASE": 1,
    "F": -1,
    "FTD": -1,
    "FRONTOTEMPORAL DEMENTIA": -1,
}

LABEL_NAME = {
    0: "Healthy Control",
    1: "Alzheimer's EEG Pattern",
    -1: "Frontotemporal Dementia",
}

# Minimum expected subject counts — warn if actual counts fall far below these
_MIN_EXPECTED_ADHC = 10
_EXPECTED_DS004504_COUNTS = {
    "Healthy Control": 29,
    "Alzheimer's EEG Pattern": 36,
    "Frontotemporal Dementia excluded": 23,
    "Phase 1 total": 65,
}


def _find_group_column(df: pd.DataFrame) -> str:
    candidates = ["Group", "group", "diagnosis", "Diagnosis", "participant_group", "condition"]
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"Could not find diagnosis/group column. Columns: {list(df.columns)}")


def _normalize_subject_id(value: str) -> str:
    text = str(value)
    return text if text.startswith("sub-") else f"sub-{text}"


def prepare_labels(config_path: str = "configs/preprocessing.yaml") -> dict:
    cfg = load_config(config_path)
    dataset_root = resolve_path(cfg["paths"]["dataset_root"])
    labels_dir = resolve_path(cfg["paths"]["labels_dir"])
    labels_dir.mkdir(parents=True, exist_ok=True)

    participants_tsv = dataset_root / "participants.tsv"
    if not participants_tsv.exists():
        raise FileNotFoundError(
            f"Missing {participants_tsv}. Download ds004504 before preparing labels."
        )

    df = pd.read_csv(participants_tsv, sep="\t")
    if "participant_id" not in df.columns:
        raise ValueError("participants.tsv must contain participant_id.")

    group_col = _find_group_column(df)
    df["subject_id"] = df["participant_id"].map(_normalize_subject_id)
    df["group_code"] = df[group_col].astype(str).str.strip()
    df["group_norm"] = df["group_code"].str.upper()
    df["label"] = df["group_norm"].map(LABEL_MAP)

    unknown = df[df["label"].isna()]
    if len(unknown) > 0:
        raise ValueError(f"Unknown group values: {sorted(unknown['group_code'].unique())}")

    df["label"] = df["label"].astype(int)
    df["label_name"] = df["label"].map(LABEL_NAME)

    processed = df[["subject_id", "participant_id", "group_code", "label", "label_name"]].copy()
    ad_hc = processed[processed["label"].isin([0, 1])].copy()
    ftd = processed[processed["label"].eq(-1)].copy()
    ftd["exclusion_reason"] = "Excluded from Phase 1 because FTD and AD are clinically distinct."

    # Fix: sanity check on subject counts — warn if dataset appears truncated or partial
    n_hc = int((ad_hc["label"] == 0).sum())
    n_ad = int((ad_hc["label"] == 1).sum())
    n_ftd = int(len(ftd))
    if len(ad_hc) < _MIN_EXPECTED_ADHC:
        LOGGER.warning(
            "Only %d AD/HC subjects found (HC=%d, AD=%d). Expected ~65 for ds004504. "
            "Check that the dataset downloaded correctly.",
            len(ad_hc), n_hc, n_ad,
        )
    LOGGER.info("Labels prepared — AD: %d  HC: %d  FTD excluded: %d", n_ad, n_hc, n_ftd)

    counts = {
        "Healthy Control": n_hc,
        "Alzheimer's EEG Pattern": n_ad,
        "Frontotemporal Dementia excluded": n_ftd,
        "Phase 1 total": int(len(ad_hc)),
    }
    count_check_passed = counts == _EXPECTED_DS004504_COUNTS
    if not count_check_passed:
        LOGGER.warning(
            "ds004504 expected counts are AD=36, HC=29, FTD=23, Phase 1 total=65; "
            "found AD=%d, HC=%d, FTD=%d, Phase 1 total=%d. "
            "This is acceptable for a subset smoke test but not for final Phase 1 reporting.",
            n_ad, n_hc, n_ftd, len(ad_hc),
        )

    processed.to_csv(labels_dir / "participants_processed.csv", index=False)
    ad_hc.to_csv(labels_dir / "ad_hc_subjects.csv", index=False)
    ftd.to_csv(labels_dir / "excluded_ftd_subjects.csv", index=False)

    mapping = {
        "training_classes": {"0": "Healthy Control", "1": "Alzheimer's EEG Pattern"},
        "excluded_groups": [
            {
                "code": "F",
                "label": "Frontotemporal Dementia",
                "reason": "Excluded from Phase 1 because FTD and AD are clinically distinct.",
            }
        ],
        "counts": counts,
        "expected_counts": _EXPECTED_DS004504_COUNTS,
        "count_check_passed": count_check_passed,
    }
    with (labels_dir / "label_mapping.json").open("w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)

    return mapping


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/preprocessing.yaml")
    args = parser.parse_args()
    prepare_labels(args.config)


if __name__ == "__main__":
    main()
