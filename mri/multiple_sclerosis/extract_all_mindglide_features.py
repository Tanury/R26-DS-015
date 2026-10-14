"""
extract_all_mindglide_features.py


Extracts ALL of MindGlide's region volumes (not just Lesion) from the
segmentation files already computed and cached on disk by
validate_mindglide_on_ms3seg.py and check_mindglide_on_lemon_hc.py.

This does NOT re-run MindGlide's segmentation (the slow GPU inference
step) -- it just reads the .nii.gz segmentation files that are already
sitting in the work directories from earlier runs and computes volumes
for every region using MindGlide's own volumes_dataframe() helper,
which is pure local post-processing (no network, no GPU needed). This
should take a couple of minutes for 150 subjects, not another hour.

Usage:
    python3 extract_all_mindglide_features.py \\
        --ms3seg-work mindglide_validation_work --ms3seg-csv mindglide_validation_full.csv \\
        --lemon-work mindglide_lemon_work --lemon-csv mindglide_lemon_hc_50.csv \\
        --out mindglide_all_features.csv

If a cached segmentation file is missing for some subject (e.g. that
subject errored out during the original run), that subject is skipped
with a warning rather than crashing the whole extraction.
"""

import argparse
from pathlib import Path

import pandas as pd
from mindglide import volumes_dataframe


def extract_for_cohort(work_dir: Path, ids: list, id_suffix_pattern: str, label: int) -> pd.DataFrame:
    """ids: list of subject/patient IDs. id_suffix_pattern: how the cached
    file is named, e.g. '{id}_mindglide_seg.nii.gz'."""
    rows = []
    for subj_id in ids:
        seg_path = work_dir / id_suffix_pattern.format(id=subj_id)
        if not seg_path.exists():
            print(f"  ⚠️  No cached segmentation for {subj_id} at {seg_path} -- skipping")
            continue
        vol_df = volumes_dataframe(str(seg_path))
        row = {"id": subj_id, "label": label}
        for _, r in vol_df.iterrows():
            row[r["Region_Name"]] = r["Volume_mm3"]
        rows.append(row)
    return pd.DataFrame(rows)


def _filter_successful(df: pd.DataFrame, id_col: str) -> list:
    """Return the list of IDs that completed without error. Handles the case
    where the 'error' column doesn't exist at all (every subject succeeded)."""
    if "error" in df.columns:
        df = df[df["error"].isna()]
    return df[id_col].tolist()


def main(ms3seg_work: str, ms3seg_csv: str, lemon_work: str, lemon_csv: str, out_csv: str) -> None:
    ms_df = pd.read_csv(ms3seg_csv, dtype={"patient_id": str})
    ms_ids = _filter_successful(ms_df, "patient_id")
    hc_df = pd.read_csv(lemon_csv, dtype={"subject_id": str})
    hc_ids = _filter_successful(hc_df, "subject_id")

    print(f"Extracting all region volumes for {len(ms_ids)} MS3SEG patients "
          f"from cached segmentations in {ms3seg_work}...")
    ms_features = extract_for_cohort(
        Path(ms3seg_work), ms_ids, "{id}_mindglide_seg.nii.gz", label=1
    )

    print(f"\nExtracting all region volumes for {len(hc_ids)} LEMON subjects "
          f"from cached segmentations in {lemon_work}...")
    hc_features = extract_for_cohort(
        Path(lemon_work), hc_ids, "{id}_mindglide_seg.nii.gz", label=0
    )

    combined = pd.concat([ms_features, hc_features], ignore_index=True)
    combined.to_csv(out_csv, index=False)

    print(f"\nExtracted {len(combined)} subjects x {combined.shape[1]-2} region-volume features")
    print(f"Columns: {[c for c in combined.columns if c not in ('id', 'label')]}")
    print(f"Saved to: {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ms3seg-work", default="mindglide_validation_work",
                         help="Work dir from validate_mindglide_on_ms3seg.py containing cached .nii.gz segmentations")
    parser.add_argument("--ms3seg-csv", required=True, help="mindglide_validation_full.csv (for the patient ID list)")
    parser.add_argument("--lemon-work", default="mindglide_lemon_work",
                         help="Work dir from check_mindglide_on_lemon_hc.py containing cached .nii.gz segmentations")
    parser.add_argument("--lemon-csv", required=True, help="mindglide_lemon_hc_50.csv (for the subject ID list)")
    parser.add_argument("--out", default="mindglide_all_features.csv")
    args = parser.parse_args()
    main(args.ms3seg_work, args.ms3seg_csv, args.lemon_work, args.lemon_csv, args.out)