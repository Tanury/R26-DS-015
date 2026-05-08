from __future__ import annotations

import pandas as pd

from src.evaluation.leakage_check import check_no_subject_leakage


def test_leakage_check_passes_subject_level_split(tmp_path):
    split_file = tmp_path / "splits.csv"
    pd.DataFrame(
        [
            {"subject_id": "sub-001", "fold": 0, "split": "train"},
            {"subject_id": "sub-002", "fold": 0, "split": "val"},
            {"subject_id": "sub-003", "fold": 0, "split": "train"},
        ]
    ).to_csv(split_file, index=False)
    result = check_no_subject_leakage(str(split_file))
    assert result["no_epoch_leakage"] is True


def test_leakage_check_fails_overlap(tmp_path):
    split_file = tmp_path / "splits.csv"
    pd.DataFrame(
        [
            {"subject_id": "sub-001", "fold": 0, "split": "train"},
            {"subject_id": "sub-001", "fold": 0, "split": "val"},
        ]
    ).to_csv(split_file, index=False)
    result = check_no_subject_leakage(str(split_file))
    assert result["no_epoch_leakage"] is False
