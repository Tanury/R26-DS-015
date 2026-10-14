"""
batch_preprocess_thickness.py
==============================
R26-DS-015 — eye/ms preprocessing

Runs vol_loader + mat_loader + thickness.py across all 35 subjects and
caches the result as a CSV: one row per subject, 8 thickness features
(microns) + label. This is the thickness-only pipeline (see project
discussion: with n=35, a 33M-parameter 3D CNN branch would badly overfit,
whereas RNFL/GCIP thinning is a well-established, literature-validated
MS biomarker on its own -- same reasoning as the MRI tabular-classifier
decision).

Usage:
    python3 batch_preprocess_thickness.py \\
        "/Users/tanuridissanayaka/Research Datasets/MS OCT/OCT_Manual_Delineations-2018_June_29" \\
        --out thickness_features.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from vol_loader import load_vol
from mat_loader import load_boundaries
from thickness import compute_thickness_features, central_bscan_window, LAYER_NAMES


def batch_process(root: str, out_csv: str, central_window: int = 10) -> pd.DataFrame:
    root_path = Path(root)
    vol_dir = root_path / "vol"
    mat_dir = root_path / "delineation"

    vol_files = sorted(vol_dir.glob("*.vol"))
    print(f"Found {len(vol_files)} .vol files\n")

    rows = []
    failed = []

    for vol_path in vol_files:
        subject_id = vol_path.stem  # e.g. "hc01_spectralis_macula_v1_s1_R"
        mat_path = mat_dir / f"{subject_id}.mat"

        try:
            vol = load_vol(str(vol_path))
            bd = load_boundaries(str(mat_path), size_x=vol.meta["size_x"])
            feats_whole = compute_thickness_features(bd.boundaries, scale_y_mm=vol.meta["scale_y"])
            window = central_bscan_window(vol.meta["n_bscans"], window=central_window)
            feats_central = compute_thickness_features(
                bd.boundaries, scale_y_mm=vol.meta["scale_y"], bscan_indices=window
            )
        except Exception as e:
            failed.append(f"[{subject_id}] {e}")
            continue

        # short_id e.g. "hc01" / "ms14" -- strip the rest of the filename
        short_id = subject_id.split("_")[0]
        label = 0 if short_id.startswith("hc") else 1  # 0 = Healthy, 1 = MS

        row = {
            "subject_id": short_id,
            "label": label,
            "group": "Healthy" if label == 0 else "MS",
            "mat_format": bd.source_format,
        }
        for name, val in zip(feats_whole.layer_names, feats_whole.subject_vector):
            row[f"thickness_{name}_um"] = val
        for name, val in zip(feats_central.layer_names, feats_central.subject_vector):
            row[f"thickness_{name}_central_um"] = val
        rows.append(row)

        print(f"  {short_id:6s} ({row['group']:7s}) -- whole: "
              f"{', '.join(f'{n}={v:.1f}' for n, v in zip(feats_whole.layer_names, feats_whole.subject_vector))}")

    df = pd.DataFrame(rows)
    df = df.sort_values("subject_id").reset_index(drop=True)
    df.to_csv(out_csv, index=False)

    print(f"\nProcessed {len(df)}/{len(vol_files)} subjects successfully.")
    print(f"  Healthy: {(df['label'] == 0).sum()}   MS: {(df['label'] == 1).sum()}")
    print(f"Saved to: {out_csv} (columns: whole-scan '_um' and central-window '_central_um' features)")

    if failed:
        print(f"\n⚠️  {len(failed)} subjects FAILED (not in output):")
        for f in failed:
            print(f"  {f}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", help="Path to OCT_Manual_Delineations-2018_June_29 folder")
    parser.add_argument("--out", default="thickness_features.csv")
    parser.add_argument("--central-window", type=int, default=10,
                         help="Number of central B-scans to average for the foveal-restricted feature set")
    args = parser.parse_args()
    batch_process(args.dataset_root, args.out, central_window=args.central_window)